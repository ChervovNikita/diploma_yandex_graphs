"""Pure custody/selector contracts; injected digest/selector support stdlib QA."""
from replay_gate import require, ARMS, EXPECTED_ARMS, validate_selector


def best_candidate(rows, select):
    best = None
    for row in rows:
        best, _ = select(best, candidate_id=row["candidate_id"], hits50=row["hits50"], order=row["order"])
    return best


def validate_valid_receipt(receipt):
    require(receipt["positive_queries"] == 60084 and receipt["negative_queries"] == 100000 and receipt["all_query_rows_complete"] is True and receipt["encoder_calls"] == 1 and receipt["query_batches"] == [1, 1] and receipt["graph"] == "complete_TRAIN_only" and receipt["serving_pool"] == "mean_raw_logits", "Original full VALID receipt differs")
    require(set(receipt["score_digests"]) == {"positive", "negative"}, "Per-model score digest missing")


def validate_journal_payload(payload, journal, complete, identity, unit, seed, select):
    require(payload["schema"] == "ncnc-pilot-own-trusted-state-v1" and payload["identity"] == identity and payload["unit"] == unit and payload["seed"] == seed, "Own journal payload differs")
    state = payload["state"]
    require(state["epoch"] == journal["epoch"] == 100 and state["phase"] == "complete" and state["fresh_scientific_initialization"] is True and state["resource_state_donor"] is False, "Own journal is incomplete/donor-derived")
    count = 4 if unit == "native_bank4" else 1
    require(len(state["fits"]) == count, "Journal unique-fit denominator differs")
    for record, metadata in zip(state["fits"], complete["member_metadata"]):
        for key in ("fit_seed", "member", "initial_state_sha256", "initial_rng_sha256"):
            require(record[key] == metadata[key], "Journal fit identity/initial state/RNG differs")
        require([e["epoch"] for e in record["epochs"]] == list(range(1, 101)), "Journal all100 epochs missing/duplicated")
        candidates = []
        for epoch, sealed in zip(record["epochs"], metadata["epochs"]):
            require(epoch["start_rng_sha256"] == sealed["start_rng_sha256"] and epoch["end_rng_sha256"] == sealed["end_rng_sha256"] and epoch["train"]["stream"] == sealed["stream"] and epoch["train"]["full_batches"] == epoch["train"]["optimizer_steps"] == 17, "Journal stream/RNG disagrees with COMPLETE")
            validate_valid_receipt(epoch["VALID"])
            candidates.append({"candidate_id": "epoch_" + str(epoch["epoch"]), "order": epoch["epoch"], "hits50": epoch["private_hits50"]})
        require(record["best"] == best_candidate(candidates, select) and record["best_state"] is not None and record["current"] is not None, "Journal individual selector differs from original strict improvement/first tie")
    if count == 4:
        candidates = state["ensemble_candidates"]
        require([r["order"] for r in candidates] == list(range(1, 102)), "All101 ordered ensemble candidates required")
        require([r["candidate_id"] for r in candidates] == ["same_epoch_" + str(i) for i in range(1, 101)] + ["individual_validation_best_bank"], "Ensemble candidate identity differs")
        extra = candidates[-1]
        require(len(extra["extra_VALID_evaluations"]) == 4 and extra["member_individual_best_epochs"] == [r["best"]["order"] for r in state["fits"]], "Candidate101 four extra evaluations/best bank omitted")
        for receipt in extra["extra_VALID_evaluations"]:
            validate_valid_receipt(receipt)
        ordered = [{"candidate_id": r["candidate_id"], "order": r["order"], "hits50": r["private_hits50"]} for r in candidates]
        require(state["ensemble_best"] == best_candidate(ordered, select) and len(state["ensemble_best_states"]) == 4, "Ensemble selected state differs")
    else:
        require(state["ensemble_best"] is None and state["ensemble_best_states"] is None and state["ensemble_candidates"] == [], "Unexpected independent-bank candidates")
    return state


def selected_reference(state, arm):
    if arm == ARMS[1]:
        selection, states = state["ensemble_best"], state["ensemble_best_states"]
        if selection["order"] == 101:
            digests = [r["score_digests"] for r in state["ensemble_candidates"][-1]["extra_VALID_evaluations"]]
        else:
            digests = [r["epochs"][selection["order"] - 1]["VALID"]["score_digests"] for r in state["fits"]]
    else:
        record = state["fits"][0]
        selection, states = record["best"], [record["best_state"]]
        digests = [record["epochs"][selection["order"] - 1]["VALID"]["score_digests"]]
    return selection, states, digests


def validate_selected_payload(payload, cell, state, identity, digest):
    unit, seed, arm = cell["unit"], cell["base_seed"], cell["arm"]
    require(arm in EXPECTED_ARMS[unit] and payload["schema"] == "ncnc-pilot-validation-selected-checkpoint-v1" and payload["identity"] == identity and payload["unit"] == unit and payload["seed"] == seed and payload["arm"] == arm, "Selected checkpoint family/unit/seed/arm differs")
    require(payload["serving_pool"] == "mean_raw_logits" and payload["validation_graph"] == "complete_TRAIN_only" and payload["eventual_test_graph"] == "native_TRAIN_plus_VALID" and payload["fresh_scientific_initialization"] is True and payload["resource_state_donor"] is False, "Selected checkpoint graph/pool/donor flags differ")
    validate_selector(payload["selection"], arm)
    expected_selection, expected_states, score_digests = selected_reference(state, arm)
    require(payload["selection"] == cell["selection"] == expected_selection, "Selected checkpoint selector differs from own journal")
    saved = payload["states"]
    require(len(saved) == len(expected_states) == (4 if arm == ARMS[1] else 1), "Selected snapshot denominator differs")
    require([digest(s) for s in saved] == [digest(s) for s in expected_states], "Selected full snapshot differs from own journal")
    if arm == ARMS[1] and expected_selection["order"] == 101:
        require([digest(s) for s in saved] == [digest(r["best_state"]) for r in state["fits"]], "Candidate101 is not four individual best states")
    for snapshot in saved:
        require(set(snapshot) == {"models", "optimizer", "rng", "flags"}, "Selected complete snapshot tree differs")
        require(len(snapshot["models"]) == len(snapshot["flags"]) == (2 if unit in ("native_bank4", "native70") else 1), "Selected model/flag tree cardinality differs")
    # N64 always points to member0's individual best. I4 may use another epoch.
    return saved, score_digests


def public_cells(cells, *, full_pass=False):
    return [{"arm": c["arm"], "base_seed": c["base_seed"], "status": c["status"], "replayed_VALID_hits50": c.get("replayed_VALID_hits50") if full_pass else None, "selected_hits50_equal": c.get("selected_hits50_equal") if full_pass else None} for c in cells]


def summaries(gate, cells):
    """Original complete-family endpoints only, after all25 replay slots pass."""
    import statistics
    require(len(cells) == 25 and all(c["status"] == "PASS" for c in cells), "No successful-subset summary")
    values = {(c["arm"], c["base_seed"]): c["replayed_VALID_hits50"] for c in cells}
    require(all(values[(c["arm"], c["base_seed"])] == c["served_VALID_hits50"] for c in gate["lock"]["cells"]), "Replay disagrees with original locked selection")
    pairs = [(candidate, control) for candidate in ARMS[2:4] for control in (ARMS[0], ARMS[1], ARMS[4])]
    pairs += [(ARMS[1], ARMS[0]), (ARMS[4], ARMS[0]), (ARMS[1], ARMS[4])]
    exploratory = []
    for candidate, control in pairs:
        differences = [values[(candidate, seed)] - values[(control, seed)] for seed in range(5)]
        exploratory.append({"candidate": candidate, "control": control, "base_seeds": list(range(5)), "paired_VALID_hits50_differences": differences, "mean": statistics.mean(differences), "sample_sd": statistics.stdev(differences), "range": [min(differences), max(differences)], "scope": "exploratory_descriptive_allfive"})
    return {"primary_development_pilot": gate["lock"]["primary_development_pilot"], "arm_summaries": gate["lock"]["arm_summaries"], "exploratory_baseline_comparisons": exploratory, "selection_identities": [{"arm": c["arm"], "base_seed": c["base_seed"], "selection": c["selection"]} for c in gate["cells"]], "VALID_selection_optimism": True, "training_randomness_conditional_on_fixed_graph_time_split": True, "TEST_execution_authorized": False, "novelty_or_baseline_superiority_claim": False}
