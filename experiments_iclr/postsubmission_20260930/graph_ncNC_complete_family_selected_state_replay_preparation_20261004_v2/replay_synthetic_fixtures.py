"""Trusted fabricated tensors/states for the separately admitted qualification."""
from pathlib import Path
from time import perf_counter
import copy
from replay_gate import require, descriptor, EXPECTED_ARMS, UNITS, ARMS
from replay_run import atomic_json


def local(path):
    return {**descriptor(path), "path": path.name}


def mixed_flags(model):
    parts = model if isinstance(model, tuple) else (model,)
    for part in parts:
        part.train(True)
        modules = list(part.named_modules())
        if len(modules) > 1:
            modules[-1][1].training = False


def build(context, root, api, mods, device, counters):
    """40 actual tiny engineering updates; fabricated100epoch metadata, not fits."""
    import torch
    from pilot_train import train_batch
    model, state_api, evaluate, data_api = (api[n] for n in ("pilot_model", "pilot_state", "pilot_evaluate", "pilot_data"))
    metric = evaluate.evaluator(context)
    root.mkdir(mode=0o700)
    identity = {**context["original_identity"], "family_id": "FABRICATED_NCNC_REPLAY_QUALIFICATION_ONLY",
                "family_lock_output_directory": str(root / "lock"), "synthetic_only": True}
    fixture_context = {**context, "identity": identity}
    generator = torch.Generator(device="cpu").manual_seed(91234)
    pairs = torch.tensor([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,0),(0,1)], device=device)
    raw = pairs.T.repeat_interleave(2, dim=1)
    raw[:, 1::2] = pairs.T.flip(0)
    probe = torch.tensor([(i, j) for i in range(16) for j in range(16) if i != j], device=device)
    data = {"x": torch.randn(16, 128, generator=generator).to(device), "pairs": pairs, "raw_edge_index": raw,
            "valid_positive": probe, "valid_negative": probe, "fabricated_inputs_only": True}
    negatives = torch.tensor([(i % 16, (i + 5) % 16) for i in range(13)], device=device)
    indices = torch.tensor([0, 2, 5, 9], device=device)
    units = {}
    updates = 0
    for unit in UNITS:
        for seed in range(5):
            began = perf_counter()
            records = []
            for member in range(4 if unit == UNITS[0] else 1):
                fit_seed = seed + 5 * member if unit == UNITS[0] else seed
                pair = model.make_native(mods, fit_seed, 64 if unit == UNITS[0] else 70, device) if unit in (UNITS[0], UNITS[3]) else model.make_factorized(mods, seed, device)
                instance, optimizer = pair
                initial = state_api.snapshot(instance, optimizer)
                mode = "private" if unit == UNITS[1] else "pooled_after_clamp" if unit == UNITS[2] else None
                start_rng = state_api.rng_digest(state_api.rng_state())
                model.train_flag(instance, True)
                train_batch(instance, optimizer, data, mods, negatives, indices, mode=mode)
                updates += 1
                counters["actual_engineering_updates"] = updates
                require(bool(optimizer.state), "Engineering Adam state was not populated")
                end_rng = state_api.rng_digest(state_api.rng_state())
                mixed_flags(instance)
                individual = state_api.snapshot(instance, optimizer)
                served = individual
                if unit == UNITS[0] and member == 0:
                    model.train_flag(instance, True)
                    train_batch(instance, optimizer, data, mods, negatives, indices)
                    updates += 1
                    counters["actual_engineering_updates"] = updates
                    mixed_flags(instance)
                    served = state_api.snapshot(instance, optimizer)
                    require(state_api.state_digest(individual) != state_api.state_digest(served), "N64/I4 fabricated selected snapshots did not differ")
                records.append({"fit_seed": fit_seed, "member": member, "initial_state_sha256": state_api.state_digest(initial),
                                "initial_rng_sha256": state_api.rng_digest(initial["rng"]), "individual": individual, "served": served,
                                "start_rng": start_rng, "end_rng": end_rng})
                del instance, optimizer, pair, initial
            units[(unit, seed)] = {"records": records, "engineering_wall_seconds": perf_counter() - began}
    require(updates == 40, "Engineering update denominator differs")
    for seed in range(5):
        p, q = (units[(u, seed)]["records"][0] for u in UNITS[1:3])
        require(p["initial_state_sha256"] == q["initial_state_sha256"] and p["initial_rng_sha256"] == q["initial_rng_sha256"] and p["start_rng"] == q["start_rng"] and p["end_rng"] == q["end_rng"], "Actual fabricated F4 twin engineering RNG pairing differs")
    scoring_calls = 0
    def score(unit, seed, member, saved, current_data):
        nonlocal scoring_calls
        fit_seed = seed + 5 * member if unit == UNITS[0] else seed
        instance, optimizer = model.make_native(mods, fit_seed, 64 if unit == UNITS[0] else 70, device) if unit in (UNITS[0], UNITS[3]) else model.make_factorized(mods, seed, device)
        state_api.restore_snapshot(instance, optimizer, saved)
        require(state_api.state_digest(state_api.snapshot(instance, optimizer)) == state_api.state_digest(saved), "Fabricated complete restoration model/Adam/RNG/flags differs")
        before = state_api.rng_digest(state_api.rng_state())
        scoring_calls += 1
        counters["reference_score_valid_calls"] = scoring_calls
        output = evaluate.score_valid(instance, current_data, mods, mode="private" if unit == UNITS[1] else "pooled_after_clamp" if unit == UNITS[2] else None)
        require(state_api.rng_digest(state_api.rng_state()) == before, "Fabricated reference scoring changed RNG")
        del instance, optimizer
        return output
    probe_rows = [score(UNITS[0], 0, m, r["served"], data) for m, r in enumerate(units[(UNITS[0], 0)]["records"])]
    probe_positive, _ = evaluate.mean_native_scores(probe_rows)
    require(float(probe_positive.max()) > float(probe_positive.min()), "Fabricated probe is degenerate; qualification must fail, not repair")
    high, low = int(probe_positive.argmax()), int(probe_positive.argmin())
    alternatives = torch.nonzero((probe_positive > probe_positive.min()) & (probe_positive < probe_positive.max())).flatten()
    alternate = int(alternatives[0]) if len(alternatives) else int(torch.nonzero(probe_positive > probe_positive.min()).flatten()[-1])
    positive_ids = torch.tensor([high, alternate], device=device).repeat(30042)
    data["valid_positive"] = probe[positive_ids]
    data["valid_negative"] = probe[low:low + 1].repeat(100000, 1)
    data["valid_negative"][131] = probe[high]
    data["valid_negative"][-1] = probe[alternate]
    tensor_path = root / "FABRICATED_TENSORS.pt"
    state_api.atomic_torch(tensor_path, {"schema": "ncnc-replay-owned-fabricated-tensors-v1", "fabricated_inputs_only": True,
                                        "tensors": {k: v.detach().cpu() for k, v in data.items() if torch.is_tensor(v)}})
    tensor_digests = {k: data_api.tensor_sha(v) for k, v in data.items() if torch.is_tensor(v)}
    inputs, pins, cells, arm_values = [], [], [], {}
    for (unit, seed), item in sorted(units.items()):
        target = root / (unit + "_" + str(seed))
        target.mkdir(mode=0o700)
        references = [score(unit, seed, m, r["individual"], data) for m, r in enumerate(item["records"])]
        served_rows = list(references)
        if unit == UNITS[0]:
            served_rows[0] = score(unit, seed, 0, item["records"][0]["served"], data)
        fit_records, metadata = [], []
        for member, (record, reference) in enumerate(zip(item["records"], references)):
            quality = evaluate.hits50(metric, reference[0], reference[1])
            best = {"candidate_id": "epoch_1", "order": 1, "hits50": quality}
            epochs, meta_epochs = [], []
            for epoch in range(1, 101):
                stream = {"negative_draw_sha256": data_api.tensor_sha(negatives), "permutation_sha256": "FABRICATED_REPEATED_METADATA",
                          "dropped_tail_sha256": "FABRICATED_REPEATED_METADATA", "full_batches": 17,
                          "supervised_records": 1114112, "dropped_tail_records": 64940, "negative_rows_drawn": 1179052}
                valid = served_rows[member][2] if unit == UNITS[0] and epoch == 2 else reference[2]
                epochs.append({"epoch": epoch, "start_rng_sha256": record["start_rng"], "end_rng_sha256": record["end_rng"],
                               "train": {"stream": stream, "full_batches": 17, "optimizer_steps": 17}, "VALID": valid,
                               "private_hits50": quality, "fabricated_history_row_not_executed": True})
                meta_epochs.append({"epoch": epoch, "start_rng_sha256": record["start_rng"], "end_rng_sha256": record["end_rng"], "stream": stream,
                                    "full_batches": 17, "optimizer_steps": 17, "positive_queries": 60084, "negative_queries": 100000,
                                    "train_wall_seconds": 0., "valid_wall_seconds": 0., "fabricated_history_row_not_executed": True})
            fit_records.append({**{k: record[k] for k in ("fit_seed", "member", "initial_state_sha256", "initial_rng_sha256")},
                                "current": record["served"], "best": best, "best_state": record["individual"], "epochs": epochs})
            metadata.append({**{k: record[k] for k in ("fit_seed", "member", "initial_state_sha256", "initial_rng_sha256")}, "epochs": meta_epochs})
        state = {"epoch": 100, "phase": "complete", "fits": fit_records, "ensemble_best": None, "ensemble_best_states": None,
                 "ensemble_candidates": [], "cuda_visible_devices": context["release"]["cuda_visible_devices"],
                 "fresh_scientific_initialization": True, "resource_state_donor": False, "fabricated_metadata_only_not100epochs": True}
        selections = {}
        if unit == UNITS[0]:
            bank_quality = evaluate.hits50(metric, *evaluate.mean_native_scores(served_rows))
            individual_bank_quality = evaluate.hits50(metric, *evaluate.mean_native_scores(references))
            require(bank_quality == 1. if seed == 0 else True, "Fabricated seed0 bank did not retain all selected highs over its strict shared-pool threshold")
            # If another seed has zero Hits, order2 cannot strictly beat order1.
            # Its I4 may select candidate101; seed0 must exercise differing epoch2.
            candidates = [{"order": e, "candidate_id": "same_epoch_" + str(e), "private_hits50": bank_quality if e == 2 else 0.,
                           "fabricated_history_row_not_executed": True} for e in range(1, 101)]
            candidates.append({"order": 101, "candidate_id": "individual_validation_best_bank", "private_hits50": individual_bank_quality,
                               "extra_VALID_evaluations": [r[2] for r in references], "member_individual_best_epochs": [1] * 4,
                               "fabricated_history_row_not_executed": True})
            best = None
            for candidate in candidates:
                best, _ = mods["design"].select_validation_candidate(best, candidate_id=candidate["candidate_id"], hits50=candidate["private_hits50"], order=candidate["order"])
            selected_states = [r["individual"] for r in item["records"]] if best["order"] == 101 else [r["served"] for r in item["records"]]
            if best["order"] == 1:
                # At zero first ties, store the individual states and their epoch1 receipts.
                selected_states = [r["individual"] for r in item["records"]]
                best["hits50"] = individual_bank_quality
                require(individual_bank_quality == 0., "Fabricated first-tie bank contract differs")
            state.update(ensemble_best=best, ensemble_best_states=selected_states, ensemble_candidates=candidates)
            require(seed != 0 or best["order"] == 2, "Seed0 must independently select N64 epoch1 and I4 epoch2")
            selections[ARMS[0]] = (fit_records[0]["best"], [item["records"][0]["individual"]])
            selections[ARMS[1]] = (best, selected_states)
        else:
            selections[EXPECTED_ARMS[unit][0]] = (fit_records[0]["best"], [item["records"][0]["individual"]])
        state_api.write_journal(target, fixture_context, unit, seed, state)
        checkpoints = {arm: state_api.write_selected(target, fixture_context, arm, unit, seed, selector, snapshots)
                       for arm, (selector, snapshots) in selections.items()}
        complete = {"schema": "ncnc-pilot-complete-unit-v1", "identity": identity, "unit": unit, "base_seed": seed, "epochs": 100,
                    "unique_fits": len(fit_records), "completed_optimizer_steps": 1700 * len(fit_records), "scientific_initialization": "fresh",
                    "resource_state_donor": False, "test_file_opened": False, "all_official_VALID_rows_complete": True,
                    "independent_candidate_count": 101 if unit == UNITS[0] else 0, "candidate101_extra_VALID_evaluations": 4 if unit == UNITS[0] else 0,
                    "selected_checkpoints": checkpoints, "selection_artifacts": {a: local(target / ("PRIVATE_SELECTION_" + a + ".json")) for a in checkpoints},
                    "member_metadata": metadata, "fabricated_contract_metadata_not_scientific_completion": True}
        atomic_json(target / "COMPLETE.json", complete)
        atomic_json(target / "ATTEMPTS.json", {"schema": "ncnc-pilot-inclusive-attempts-v1", "identity": identity,
                    "attempts": [{"attempt": 1, "stage": "fit", "unit": unit, "base_seed": seed, "status": "COMPLETE",
                                  "observed_wall_seconds": item["engineering_wall_seconds"], "inclusive_wall_seconds": item["engineering_wall_seconds"],
                                  "fabricated_contract_attempt_metadata": True}]})
        physical = root / (unit + "_" + str(seed) + "_PHYSICAL.json")
        atomic_json(physical, {"unit": unit, "base_seed": seed, "output_directory": str(target), "exit_code": 0, "fabricated_contract_terminal_not_process_exit": True})
        inputs.append({"unit": unit, "base_seed": seed, "output_directory": str(target), "disposition": "COMPLETE", "complete_sha256": descriptor(target / "COMPLETE.json")["sha256"]})
        pins.append({"unit": unit, "base_seed": seed, "output_directory": str(target), "disposition": "COMPLETE", "terminal": local(target / "COMPLETE.json"),
                     "physical_terminal": descriptor(physical), "attempts": local(target / "ATTEMPTS.json"), "journal": local(target / "JOURNAL.json")})
        for arm, (selector, _) in selections.items():
            arm_values[(arm, seed)] = selector["hits50"]
            cells.append({"arm": arm, "base_seed": seed, "status": "COMPLETE", "served_VALID_hits50": selector["hits50"], "selection": selector,
                          "checkpoint": {"output_directory": str(target), **checkpoints[arm]}})
        del state, fit_records, references, served_rows
    require(scoring_calls == 44, "Fabricated reference scoring denominator differs")
    import statistics
    diffs = [arm_values[(ARMS[2], s)] - arm_values[(ARMS[3], s)] for s in range(5)]
    lock = {"schema": "ncnc-pilot-full-family-lock-v1", "identity": identity, "status": "COMPLETE_FAMILY_LOCKED", "unique_fits_planned": 35,
            "unique_fits_completed": 35, "served_arm_seed_cells": 25, "complete_served_cells": 25, "test_file_opened": False,
            "TEST_execution_authorized": False, "no_success_only_subset_summary": True, "terminal_failures": [], "inputs": inputs, "cells": cells,
            "arm_summaries": {a: {"mean_VALID_hits50": statistics.mean(arm_values[(a, s)] for s in range(5)), "sample_sd": statistics.stdev(arm_values[(a, s)] for s in range(5))} for a in ARMS},
            "primary_development_pilot": {"paired_VALID_hits50_differences": diffs, "mean_difference": statistics.mean(diffs), "sample_sd": statistics.stdev(diffs),
                                        "range": [min(diffs), max(diffs)], "sign_count": {"private_greater": sum(d > 0 for d in diffs), "equal": sum(d == 0 for d in diffs), "pooled_greater": sum(d < 0 for d in diffs)},
                                        "fabricated_only_no_scientific_inference": True}, "fabricated_metadata_only": True}
    lock_root = root / "lock"
    lock_root.mkdir(mode=0o700)
    atomic_json(lock_root / "FAMILY_LOCK.json", lock)
    lock_pin = descriptor(lock_root / "FAMILY_LOCK.json")
    for pin in pins:
        target = Path(pin["output_directory"])
        atomic_json(target / "FAMILY_CLOSURE.json", {"schema": "ncnc-pilot-immutable-family-closure-v1", "identity": identity,
                    "family_lock_path": lock_pin["path"], "family_lock_sha256": lock_pin["sha256"], "status": lock["status"], "resume_or_seed_replacement_permitted": False})
        pin["closure"] = local(target / "FAMILY_CLOSURE.json")
    return {"context": {**context, "identity": identity, "release": {**context["release"], "family_lock": lock_pin, "unit_custody": pins},
                        "synthetic_tensor_custody": descriptor(tensor_path), "synthetic_tensor_digests": tensor_digests},
            "data": data, "engineering_updates": updates, "reference_score_valid_calls": scoring_calls,
            "N64_I4_seed0_distinct_selected_snapshots": True, "fixture_root": root}
