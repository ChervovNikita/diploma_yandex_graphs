"""Read-only interpretation after both fixed GCN/GAT companion families close."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ARMS = ("ordinary_M1", "ordinary_genuine_I4", "factorized_allmap_M1",
        "factorized_allmap_genuine_I4", "shared4_unchanged", "separable_equal_size", "exchange")
O1, O4, F1, F4, S, D, X = ARMS
BINDINGS = {
    "GCN": {"decision_sha256": "b6a6bcef59c0ca0c3e16c12775a98153b5685782272ada4fa53587ebe7dc4da3",
            "config_sha256": "aeee4950c3db650ff62c5b225f92c766539e5721e9f9c1dc362dfd149854dea1"},
    "GAT": {"decision_sha256": "75ad1b2344a21da97b1a4efde666f16f9413a8558de40936e1413b422aa4be60",
            "config_sha256": "3a5925a2c4b15aa225a98265a5153cb27948919f0ce723bf4203b473871f71d2"}}
STATES = ("no_alternative_wrong", "aggregation_only_correct", "alternative_lost", "alternative_served")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired(values):
    values = [float(v) for v in values]
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = 4.302652729911275 * sd / math.sqrt(3)
    flipped = [statistics.mean(a * b for a, b in zip(signs, values)) for signs in itertools.product((-1, 1), repeat=3)]
    return {"seed_deltas": values, "mean": mean, "sample_sd": sd, "min": min(values), "max": max(values),
            "descriptive_95pct_t_interval_df2": [mean - half, mean + half],
            "nonnegative_seed_count": sum(v >= 0 for v in values), "positive_seed_count": sum(v > 0 for v in values),
            "exact_sign_flip_two_sided_p": sum(abs(v) >= abs(mean) for v in flipped) / 8}


def logsumexp(a, axis):
    maximum = a.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(a - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def score(a):
    raw, y = a["raw_logits"].astype(np.float64), a["y"]
    log_probs = raw - logsumexp(raw, -1)[..., None]
    member_nll = -np.take_along_axis(log_probs, y[None, :, None], axis=2).squeeze(2)
    pool_nll = -(logsumexp(log_probs, 0) - math.log(len(raw)))[np.arange(len(y)), y]
    member_correct, pool_correct = ~a["member_errors"], ~a["pooled_errors"]
    coverage = member_correct.any(0)
    lost, aggregation_only = coverage & ~pool_correct, ~coverage & pool_correct
    counts = {"nodes": len(y), "pooled_correct": int(pool_correct.sum()), "coverage": int(coverage.sum()),
              "lost_correct_alternatives": int(lost.sum()), "aggregation_only_correct": int(aggregation_only.sum()),
              "unavailable_alternatives": int((~coverage).sum()), "served_alternatives": int((coverage & pool_correct).sum())}
    require(counts["pooled_correct"] == counts["coverage"] - counts["lost_correct_alternatives"] + counts["aggregation_only_correct"], "Coverage decomposition failed")
    classes = []
    for label in range(10):
        mask = y == label
        classes.append({"class": label, "nodes": int(mask.sum()), "pooled_accuracy_pct": float(pool_correct[mask].mean() * 100),
                        "pooled_nll": float(pool_nll[mask].mean()), "member_accuracy_pct": (member_correct[:, mask].mean(1) * 100).tolist(),
                        "member_nll": member_nll[:, mask].mean(1).tolist()})
    quality = {"pooled_accuracy_pct": float(pool_correct.mean() * 100), "pooled_nll": float(pool_nll.mean()),
               "member_accuracy_pct": (member_correct.mean(1) * 100).tolist(), "member_nll": member_nll.mean(1).tolist(),
               "mean_member_accuracy_pct": float(member_correct.mean() * 100),
               "worst_member_accuracy_pct": float(member_correct.mean(1).min() * 100)}
    return {"quality": quality, "counts": counts, "classes": classes,
            "P": pool_correct, "V": coverage, "C": member_correct, "state": 2 * coverage.astype(np.int64) + pool_correct.astype(np.int64)}


def comparison(a, b, labels):
    repair, harm = ~b["P"] & a["P"], b["P"] & ~a["P"]
    new_coverage, removed_coverage = a["V"] & ~b["V"], b["V"] & ~a["V"]
    changes = {k: a["counts"][k] - b["counts"][k] for k in ("pooled_correct", "coverage", "lost_correct_alternatives", "aggregation_only_correct")}
    require(changes["pooled_correct"] == changes["coverage"] - changes["lost_correct_alternatives"] + changes["aggregation_only_correct"], "Paired count identity failed")
    require(changes["pooled_correct"] == int(repair.sum() - harm.sum()), "Repair/harm identity failed")
    return {"repairs": int(repair.sum()), "harms": int(harm.sum()), "count_decomposition_delta": changes,
            "new_coverage": int(new_coverage.sum()), "removed_coverage": int(removed_coverage.sum()),
            "new_coverage_served": int((new_coverage & a["P"]).sum()), "new_coverage_lost": int((new_coverage & ~a["P"]).sum()),
            "repair_causes": {"new_member_alternative": int((repair & a["V"] & ~b["V"]).sum()),
                              "existing_member_alternative_served": int((repair & a["V"] & b["V"]).sum()),
                              "aggregation_only": int((repair & ~a["V"]).sum())},
            "harm_causes": {"available_alternative_lost": int((harm & a["V"]).sum()),
                            "removed_member_alternative": int((harm & ~a["V"] & b["V"]).sum()),
                            "lost_aggregation_only": int((harm & ~a["V"] & ~b["V"]).sum())},
            "transition_rows_reference_columns_candidate": np.bincount(b["state"] * 4 + a["state"], minlength=16).reshape(4, 4).tolist(),
            "class_repairs": [int((repair & (labels == k)).sum()) for k in range(10)],
            "class_harms": [int((harm & (labels == k)).sum()) for k in range(10)]}


def closed_owner(path):
    owner = json.loads(path.read_text())
    require(owner.get("exit_code") == 0 and owner.get("timed_out") is False, "Successful finite owner exit required")
    for field in ("direct_child_wait", "child_pid_absent", "owned_cuda_pid_absent", "complete_family", "scientific_success"):
        require(owner.get(field) is True, "Owner closure missing: " + field)
    require(owner.get("TEST_access") is False, "Owner must declare no TEST access")
    return owner


def closed_family(backbone, output, owner_path, owner):
    bound = BINDINGS[backbone]
    complete_path, decision_path = output / "COMPLETE_FAMILY.json", owner_path.parent / "DECISION.md"
    config_path = owner_path.parent / "CONFIG.json"
    require(owner.get("complete_sha256") == digest(complete_path), "Owner must bind this exact complete report")
    require(digest(decision_path) == bound["decision_sha256"], "Frozen backbone DECISION.md identity mismatch")
    require(digest(config_path) == bound["config_sha256"], "Fixed companion config identity mismatch")
    complete = json.loads(complete_path.read_text())
    require(complete.get("complete") is True and complete.get("TEST_access") is False, "Complete no-TEST family required")
    require(complete.get("groups") == complete.get("expected_groups") == 21 and complete.get("fit_units") == complete.get("expected_fit_units") == 39, "Exact21/39 family required")
    cfg = json.loads(config_path.read_text())
    require(cfg["backbone"] == backbone and json.loads((output / "CONFIG.json").read_text()) == cfg, "Exact configured backbone required")
    require(complete["config_sha256"] == bound["config_sha256"], "Complete report must bind the fixed config")
    seeds, rows = cfg["seeds"], complete["results"]
    require(len(seeds) == len(set(seeds)) == 3 and len(rows) == 21, "Three distinct paired seeds required")
    index = {(r["seed"], r["arm"]): r for r in rows}
    require(len(index) == 21 and set(index) == {(s, a) for s in seeds for a in ARMS}, "Exact all-arm roster required")
    require(sum(len(r["fits"]) for r in rows) == 39 and all(r["serving"] == "probability_mean" for r in rows), "39 fit units and frozen probability serving required")
    for r in rows:
        members = 4 if "genuine_I4" in r["arm"] else 1
        require([u["member"] for u in r["fits"]] == list(range(members)), "Missing independently selected acquisition unit")
    operations = {k: sum(u["operation_counts"][k] for r in rows for u in r["fits"]) for k in rows[0]["fits"][0]["operation_counts"]}
    require(operations == complete["operation_counts"], "Complete operation-count aggregation mismatch")
    return dict(backbone=backbone, output=output, owner_path=owner_path, owner=owner, complete_path=complete_path,
                decision_path=decision_path, complete=complete, cfg=cfg, seeds=seeds, index=index, operations=operations)


def selected(output, seeds):
    arrays, archive_hashes, ids, labels = {}, {}, None, None
    keys = {"ids", "y", "raw_logits", "probability_mean", "member_errors", "pooled_errors"}
    for seed in seeds:
        for arm in ARMS:
            path = output / f"{arm}_seed{seed}" / "selected_VALID.npz"
            with np.load(path, allow_pickle=False) as archive:
                require(set(archive.files) == keys, "Exact selected VALID keys required")
                a = {k: archive[k].copy() for k in keys}
            members = 1 if arm in (O1, F1) else 4
            require(a["ids"].shape == a["y"].shape == (5274,) and a["ids"].dtype == a["y"].dtype == np.int64, "Exact VALID roles required")
            require(a["raw_logits"].shape == (members, 5274, 10) and a["raw_logits"].dtype == np.float32, "Original float32 member logits required")
            require(a["probability_mean"].shape == (5274, 10) and a["probability_mean"].dtype == np.float32, "Native float32 serving probabilities required")
            require(a["member_errors"].shape == (members, 5274) and a["pooled_errors"].shape == (5274,) and a["member_errors"].dtype == a["pooled_errors"].dtype == np.bool_, "Exact saved error flags required")
            require(np.isfinite(a["raw_logits"]).all() and np.isfinite(a["probability_mean"]).all(), "Finite selected outputs required")
            if ids is None:
                ids, labels = a["ids"], a["y"]
                require(len(np.unique(ids)) == 5274 and ids.min() >= 0 and ids.max() < 11701 and set(labels.tolist()) == set(range(10)), "Ordered ten-class VALID identity required")
            require(np.array_equal(ids, a["ids"]) and np.array_equal(labels, a["y"]), "All21 archives must have identical ordered IDs/labels")
            require(np.array_equal(a["probability_mean"].argmax(1) != labels, a["pooled_errors"]), "Saved probability decisions and errors disagree")
            arrays[seed, arm], archive_hashes[str(path)] = a, digest(path)
    return arrays, archive_hashes, ids, labels


def analyze(context, selected_arrays):
    backbone, output = context["backbone"], context["output"]
    owner_path, owner, complete_path, complete = (context[k] for k in ("owner_path", "owner", "complete_path", "complete"))
    seeds, index, operations = (context[k] for k in ("seeds", "index", "operations"))
    arrays, archive_hashes, ids, labels = selected_arrays
    scores = {key: score(value) for key, value in arrays.items()}  # All21 identities checked before scoring.
    groups, aggregates = [], {}
    for arm in ARMS:
        arm_rows = []
        for seed in seeds:
            a = scores[seed, arm]
            row = {"arm": arm, "seed": seed, **{k: a[k] for k in ("quality", "counts", "classes")},
                   "selected_steps": [u["selected_step"] for u in index[seed, arm]["fits"]], "costs": index[seed, arm]["costs"],
                   "versus": {ref: comparison(a, scores[seed, ref], labels) for ref in ((S, D, F4, O4, O1, F1) if arm == S else (S, D, F4))}}
            groups.append(row)
            arm_rows.append(row)
        total = {k: sum(r["counts"][k] for r in arm_rows) for k in arm_rows[0]["counts"]}
        aggregates[arm] = {"summed_counts_over_three_seed_readouts": total,
                           "mean_accuracy_pct": statistics.mean(r["quality"]["pooled_accuracy_pct"] for r in arm_rows),
                           "mean_nll": statistics.mean(r["quality"]["pooled_nll"] for r in arm_rows),
                           "acquisition_seconds_sum": sum(r["costs"]["acquisition_seconds"] for r in arm_rows),
                           "selected_readout_seconds_sum": sum(r["costs"]["selected_serving_readout_seconds"] for r in arm_rows),
                           "inference_parameter_bytes_per_seed": [r["costs"]["inference_parameter_bytes"] for r in arm_rows]}

    contrasts = {}
    for candidate, reference in ((X, S), (X, D), (X, O4), (X, F4), (X, O1), (X, F1), (S, F4), (S, O4), (S, O1), (S, F1), (F1, O1), (F4, O4)):
        a, b = [scores[s, candidate] for s in seeds], [scores[s, reference] for s in seeds]
        contrasts[candidate + "_minus_" + reference] = {
            "accuracy_pp": paired(100 * (u["counts"]["pooled_correct"] - v["counts"]["pooled_correct"]) / 5274 for u, v in zip(a, b)),
            "nll": paired(u["quality"]["pooled_nll"] - v["quality"]["pooled_nll"] for u, v in zip(a, b)),
            "mean_member_accuracy_pp": paired(100 * (u["C"].sum() / len(u["C"]) - v["C"].sum() / len(v["C"])) / 5274 for u, v in zip(a, b))}
    stat = lambda ref, metric: contrasts[X + "_minus_" + ref][metric]
    xs, xd, member, nll = stat(S, "accuracy_pp"), stat(D, "accuracy_pp"), stat(S, "mean_member_accuracy_pp"), stat(S, "nll")
    member_count_delta = sum(int(scores[s, X]["C"].sum()) - int(scores[s, S]["C"].sum()) for s in seeds)
    development = {"mean_accuracy_vs_shared_at_least_0_2pp": xs["mean"] >= .2,
                   "mean_accuracy_vs_separable_at_least_0_1pp": xd["mean"] >= .1,
                   "shared_all_seed_nonnegative_and_two_positive": xs["nonnegative_seed_count"] == 3 and xs["positive_seed_count"] >= 2,
                   "separable_all_seed_nonnegative_and_two_positive": xd["nonnegative_seed_count"] == 3 and xd["positive_seed_count"] >= 2,
                   "mean_member_vs_shared_nonnegative_on_average": member_count_delta >= 0,
                   "mean_member_vs_shared_each_seed_at_least_minus_0_1pp": min(member["seed_deltas"]) >= -.1,
                   "mean_nll_deterioration_vs_shared_at_most_0_02": nll["mean"] <= .02,
                   "each_seed_nll_deterioration_vs_shared_at_most_0_05": max(nll["seed_deltas"]) <= .05}
    advantage = {ref: {"mean_accuracy_gain_at_least_0_2pp": stat(ref, "accuracy_pp")["mean"] >= .2,
                       "all_seed_accuracy_nonnegative": stat(ref, "accuracy_pp")["nonnegative_seed_count"] == 3} for ref in (O4, F4)}
    singles = {ref: sum(scores[s, X]["counts"]["pooled_correct"] - scores[s, ref]["counts"]["pooled_correct"] for s in seeds) > 0 for ref in (O1, F1)}
    development_pass = all(development.values())
    advantage_pass = development_pass and all(all(v.values()) for v in advantage.values()) and all(singles.values())

    baseline_stat = lambda ref, metric: contrasts[S + "_minus_" + ref][metric]
    baseline_banks = {ref: {"mean_accuracy_gain_at_least_0_2pp": baseline_stat(ref, "accuracy_pp")["mean"] >= .2,
                            "all_seed_accuracy_nonnegative": baseline_stat(ref, "accuracy_pp")["nonnegative_seed_count"] == 3,
                            "at_least_two_positive_seed_gains": baseline_stat(ref, "accuracy_pp")["positive_seed_count"] >= 2} for ref in (O4, F4)}
    baseline_singles = {ref: sum(scores[s, S]["counts"]["pooled_correct"] - scores[s, ref]["counts"]["pooled_correct"] for s in seeds) > 0 for ref in (O1, F1)}
    baseline_nll = {ref: {"mean_nll_deterioration_at_most_0_02": baseline_stat(ref, "nll")["mean"] <= .02,
                          "each_seed_nll_deterioration_at_most_0_05": max(baseline_stat(ref, "nll")["seed_deltas"]) <= .05} for ref in (O4, F4)}
    baseline_accuracy_pass = all(all(v.values()) for v in baseline_banks.values()) and all(baseline_singles.values())
    baseline_nll_pass = all(all(v.values()) for v in baseline_nll.values())

    complementarity = []
    for seed in seeds:
        p = {arm: scores[seed, arm]["P"] for arm in (S, D, X, F4)}
        rd, rx, hd, hx = ~p[S] & p[D], ~p[S] & p[X], p[S] & ~p[D], p[S] & ~p[X]
        complementarity.append({"seed": seed, "shared_repairs": int((rd & rx).sum()), "separable_only_repairs": int((rd & ~rx).sum()),
                                "exchange_only_repairs": int((rx & ~rd).sum()), "shared_harms": int((hd & hx).sum()),
                                "separable_only_harms": int((hd & ~hx).sum()), "exchange_only_harms": int((hx & ~hd).sum()),
                                "separable_only_repairs_on_factorI4_errors": int((rd & ~rx & ~p[F4]).sum()),
                                "exchange_only_repairs_on_factorI4_errors": int((rx & ~rd & ~p[F4]).sum())})
    persistence = {kind: int(np.logical_and.reduce([(~scores[s, S]["P"] & scores[s, X]["P"]) if kind == "exchange_repairs_all_three" else (scores[s, S]["P"] & ~scores[s, X]["P"]) for s in seeds]).sum()) for kind in ("exchange_repairs_all_three", "exchange_harms_all_three")}
    interaction = {"accuracy_pp": paired(100 * ((scores[s, F4]["counts"]["pooled_correct"] - scores[s, O4]["counts"]["pooled_correct"]) - (scores[s, F1]["counts"]["pooled_correct"] - scores[s, O1]["counts"]["pooled_correct"])) / 5274 for s in seeds),
                   "negative_nll": paired((scores[s, O4]["quality"]["pooled_nll"] - scores[s, F4]["quality"]["pooled_nll"]) - (scores[s, O1]["quality"]["pooled_nll"] - scores[s, F1]["quality"]["pooled_nll"]) for s in seeds)}
    report = {"complete": True, "backbone": backbone, "companion_configurations": BINDINGS, "groups": 21, "fit_units": 39, "TEST_access": False, "seeds": seeds,
              "input_bindings": {"analysis_source_sha256": digest(Path(__file__)), "owner_end_sha256": digest(owner_path),
                                 "complete_family_sha256": digest(complete_path), "decision_sha256": BINDINGS[backbone]["decision_sha256"],
                                 "training_source_sha256": complete["source_sha256"], "config_sha256": complete["config_sha256"],
                                 "ordered_VALID_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "ordered_VALID_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(), "selected_archives": archive_hashes},
              "owner_seconds": owner["seconds"], "cost_overlap": owner["overlap"], "operation_counts": operations,
              "groups_detail": groups, "aggregates": aggregates, "paired_contrasts": contrasts, "transition_state_order": STATES,
              "connector_repair_complementarity": complementarity, "same_node_persistence": persistence,
              "factorization_by_independent_ensemble_interaction": interaction,
              "frozen_decision": {"development_criteria": development, "development_advance": development_pass,
                                  "independent_bank_advantage_criteria": advantage, "positive_mean_against_each_single": singles,
                                  "desired_ensemble_quality_advantage": advantage_pass,
                                  "baseline_accuracy_transfer_criteria": baseline_banks,
                                  "baseline_positive_mean_against_each_single": baseline_singles,
                                  "baseline_accuracy_transfer": baseline_accuracy_pass,
                                  "baseline_proper_loss_criteria": baseline_nll,
                                  "baseline_proper_loss_protection": baseline_nll_pass,
                                  "baseline_accuracy_with_proper_loss_protection": baseline_accuracy_pass and baseline_nll_pass,
                                  "single_criterion_interpretation": "Exceed both singles means positive mean paired accuracy, as prospectively clarified by root; every seed delta remains reported."},
              "interpretation_limits": ["Both native backbones were prospectively fixed companions; neither outcome selects, drops or retunes the other.",
                  "Baseline accuracy transfer and proper-loss protection are separately reported; accuracy alone is not general quality improvement.",
                  "Encountered development VALID after checkpoint selection; unused confirmation is still required.",
                  "Three paired seeds measure optimization variation on this graph, not uncertainty across graphs/splits.",
                  "The t interval is descriptive; exact sign-flip minimum p is 0.125 one-sided or 0.25 two-sided. No node-IID or bootstrap replication claim.",
                  "Numerical decision gates alone do not establish reference competence or learning-curve resolution; root reviews those scopes separately.",
                  "Coverage is an oracle diagnostic; pooled correctness also includes aggregation-only correct answers.",
                  "Positive interaction cannot rescue poorer final quality. D/X are alternatives, with no measured D×X or exchange-without-factorization factorial.",
                  "Costs are observed acquisition/readout scopes; genuine I4 was acquired sequentially, with no isolated four-body serving-memory claim."]}
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("gcn-output", "gcn-owner-end", "gat-output", "gat-owner-end", "report"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    require(not report_path.exists(), "Report must be a new file")
    outputs = {"GCN": Path(args.gcn_output), "GAT": Path(args.gat_output)}
    owner_paths = {"GCN": Path(args.gcn_owner_end), "GAT": Path(args.gat_owner_end)}
    require(outputs["GCN"].resolve() != outputs["GAT"].resolve(), "Two distinct companion outputs required")
    owners = {b: closed_owner(owner_paths[b]) for b in BINDINGS}  # Both successful closures first.
    contexts = {b: closed_family(b, outputs[b], owner_paths[b], owners[b]) for b in BINDINGS}
    require({k: v for k, v in contexts["GCN"]["cfg"].items() if k != "backbone"} ==
            {k: v for k, v in contexts["GAT"]["cfg"].items() if k != "backbone"}, "Companion configs must differ only by backbone")
    global np
    import numpy as np
    payloads = {b: selected(outputs[b], contexts[b]["seeds"]) for b in BINDINGS}
    require(np.array_equal(payloads["GCN"][2], payloads["GAT"][2]) and np.array_equal(payloads["GCN"][3], payloads["GAT"][3]), "Both companions require the exact same ordered VALID IDs/labels")
    families = {b: analyze(contexts[b], payloads[b]) for b in BINDINGS}  # All42 identities checked before scoring.
    report = {"complete": True, "backbones": list(BINDINGS), "groups": 42, "fit_units": 78, "TEST_access": False,
              "prospective_companion_bindings": BINDINGS, "analysis_source_sha256": digest(Path(__file__)), "families": families,
              "scope": "Fixed GCN/GAT development transfer on the same encountered WikiCS graph; both outcomes retained, no selected winner or unused confirmation."}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    summary = {b: {k: families[b]["frozen_decision"][k] for k in ("development_advance", "desired_ensemble_quality_advantage", "baseline_accuracy_transfer", "baseline_proper_loss_protection", "baseline_accuracy_with_proper_loss_protection")} for b in BINDINGS}
    print(json.dumps({"report": str(report_path), "complete": True, "families": summary}))


if __name__ == "__main__":
    main()
