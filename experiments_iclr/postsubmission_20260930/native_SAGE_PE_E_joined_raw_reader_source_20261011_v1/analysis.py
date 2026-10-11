"""Complete joined PE/E raw reader; stdlib until all18-bank custody closes.

No calibration, Torch/model import, model forward or checkpoint deserialize.
The sealed E preflight and F score/contrast utilities remain unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

SEEDS = [7301, 7403, 7507]
E_ARMS = ["BASE", "E4", "N4", "equipped_M1", "equipped_I4"]
PE_ARM = "PE4_gamma1_F"
ARMS = E_ARMS + [PE_ARM]
E_READER_SHA = "f17dc5ac8e0f85add18e920decd867c650a050d522bc549f71be20cd7ce93c1f"
F_MATH_SHA = "b2d4deb6e7a1b9c6da5597e851a42caf11ece728f13b95ebd4a3871b01941ef0"
WIDTH = {"members": 4, "gamma": 1, "width": 59, "parameters": 215508,
         "BASE_parameter_cap": 216674, "next_width": 60, "next_parameters": 221560}
PROTOCOL = {
    "schema_version": 1, "study_kind": "native_SAGE_PE_E_joined_raw",
    "seeds": SEEDS, "arms": ARMS, "E_banks": 15, "E_bundles": 24, "E_body_records": 24,
    "PE_banks": 3, "PE_bundles": 3, "PE_private_body_records": 12,
    "joined_banks": 18, "joined_bundles": 27, "joined_body_records": 36,
    "PE_width_contract": WIDTH, "E75_terminal_and_reader_close_before_join": True,
    "new_calibration": False, "model_forwards_or_selected_state_deserialize": False,
    "raw_authority": "archived float32 member/pooled error flags",
    "separate_contextual_screens": ["BASE_minus_PE4_gamma1_F", "E4_minus_PE4_gamma1_F"],
    "raw_mean_gain_pp": 0.2, "all_three_nonnegative": True, "at_least_two_positive": True,
    "E_primary_rule_modified": False, "fullwidth_equipped_I4_decisive_quality_reference_retained": True,
    "PE_role": "fixed storage-matched context, not decisive quality replacement",
    "uncertainty": "descriptive paired95pct t intervals df2 over3 optimizer repeats on one encountered graph; no significance claim",
    "TEST_access": False, "tuning": False, "historical_cross_runtime_score_reuse": False,
    "primitive_or_acceptance_claim": False,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def current(path):
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}


def receipt(item):
    require(isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"}, "Exact root receipt required")
    p = Path(item["path"]).resolve()
    require(current(p) == item, "Root-bound file changed")
    return p


def read(path):
    return json.loads(Path(path).read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preflight(spec):
    require(spec.get("schema_version") == 1 and spec.get("study_kind") == PROTOCOL["study_kind"]
            and isinstance(spec.get("study_id"), str) and spec["study_id"], "Exact joined study identity required")
    require(receipt(spec["reader_source"]) == Path(__file__).resolve(), "Joined reader source differs")
    require(read(receipt(spec["protocol"])) == PROTOCOL, "Prospectively frozen joined raw protocol differs")
    pe = spec["PE"]; pe_root = Path(pe["family_root"]).resolve()
    complete_path = receipt(pe["complete_banks"]); complete = read(complete_path)
    require(complete_path == pe_root / "COMPLETE_PE_BANKS.json", "PE completion path differs")
    expected = {"complete": True, "fixed_seed_roster": SEEDS, "completed_seed_block": SEEDS,
                "fixed_arm_roster": [PE_ARM], "groups": 3, "optimizer_acquisition_bundles": 3,
                "native_body_fit_records": 12, "expected_groups": 3, "expected_bundles": 3,
                "expected_body_records": 12, "width_contract": WIDTH, "TEST_access": False,
                "comparison_interpreted": False}
    require(all(complete.get(k) == v for k, v in expected.items()), "Entire fixed PE3-bank/12-member roster must close")
    e_spec_path = receipt(spec["E_input_spec"]); e_spec = read(e_spec_path)
    require(e_spec["reader_source"]["sha256"] == E_READER_SHA and e_spec["math_reader"]["sha256"] == F_MATH_SHA,
            "Exact sealed E preflight/F math sources required")
    e_reader = load("joined_exact_E_reader", receipt(e_spec["reader_source"]))
    e = e_reader.preflight(e_spec)  # Original all15-bank/24-bundle/24-body admission unchanged.
    require(e["seeds"] == SEEDS and e["protocol"]["arms"] == E_ARMS, "E roster differs")
    e_summary_path = receipt(spec["E_complete_summary"]); e_summary = read(e_summary_path)
    require(all(e_summary.get(k) == v for k, v in {
        "complete": True, "study_kind": "native_SAGE_endpoint_VALUE", "banks": 15,
        "optimizer_acquisition_bundles": 24, "native_body_fit_records": 24,
        "fixed_seed_roster": SEEDS, "fixed_arm_roster": E_ARMS,
        "all75_calibration_attempts_terminal_before_interpretation": True,
        "raw_authority_preserved": True, "TEST_access": False,
        "input_spec_sha256": spec["E_input_spec"]["sha256"], "reader_source_sha256": E_READER_SHA}.items()),
        "E reader and75 fixed scalar endpoints must close before the join")
    require(e_summary["calibration_costs"]["attempted_scalar_fits"] == 75
            and e_summary["calibration_costs"]["successful_scalar_fits"] + e_summary["calibration_costs"]["failed_scalar_fits"] == 75,
            "Retain every E scalar endpoint including failures")
    e_owner = read(receipt(spec["E_reader_owner_end"]))
    require(all(e_owner.get(k) == v for k, v in {"exit_code": 0, "timed_out": False, "complete_analysis": True,
        "summary": spec["E_complete_summary"], "native_fits_added": 0, "native_forward_calls": 0, "TEST_access": False}.items()),
        "Successful actual E reader owner closure required")
    e_report_path = receipt(e_summary["full_report"])
    pe_release = read(receipt(pe["release"])); cfg_path = receipt(pe["config"]); cfg = read(cfg_path)
    require(pe_release.get("enabled") is True and pe_release.get("mode") == "fit"
            and pe_release["config"] == pe["config"] and Path(pe_release["output"]).resolve() == pe_root
            and pe_release.get("historical_cross_runtime_score_reuse") is False, "Actual fixed PE fit release required")
    require(read(pe_root / "CONFIG.json") == cfg and complete["config_sha256"] == pe["config"]["sha256"], "Entire saved PE config differs")
    fixed = {"stage": "F", "K": None, "k_selection_receipt": None, "seeds": SEEDS, "depth": 2,
             "hidden": 128, "dropout": .2, "learning_rate": .001, "max_updates": 1000,
             "patience_updates": 300, "member_seed_stride": 1000003, "dropout_seed_offset": 3000007}
    require(all(cfg.get(k) == v for k, v in fixed.items()), "Fixed native PE acquisition differs")
    source_paths = {k: receipt(v) for k, v in pe_release["sources"].items()}
    source_sha = {str(source_paths[k]): v["sha256"] for k, v in pe_release["sources"].items()}
    require(complete["source_sha256"] == source_sha and read(receipt(pe["source_hashes"])) == {
        "config_sha256": pe["config"]["sha256"], "files": source_sha}, "Exact actual PE source custody differs")
    for k in ("common_routes", "factors", "models", "run_base", "run_common", "sage_conv"):
        require(e["source_sha256"].get(str(source_paths[k])) == pe_release["sources"][k]["sha256"], "Same native/operator source required: " + k)
    require(set(pe_release["roles"]) == {"train", "valid"}, "Exact PE safe roles required")
    for k, item in pe_release["roles"].items():
        require(receipt(item) == e["role_paths"][k] and item == e["safe_roles"][k], "Fresh same ordered-role custody differs")
    admission = read(receipt(pe_release["host_runtime_admission"]))
    require(admission["runtime_fingerprint_sha256"] == e["runtime"]["runtime_fingerprint_sha256"]
            and admission["hostname"] == e["runtime"]["hostname"] and admission["gpu_uuid"] == e["runtime"]["gpu_uuid"]
            and admission["config_sha256"] == pe["config"]["sha256"], "Fresh same physical/runtime PE/E required")
    consumer_path = receipt(pe["consumer_complete"]); consumer = read(consumer_path)
    require(consumer_path == pe_root / "CONSUMER_COMPLETE.json" and all(consumer.get(k) == v for k, v in {
        "status": "COMPLETE", "mode": "fit", "banks": 3, "optimizer_bundles": 3, "private_member_records": 12,
        "comparison_interpreted": False, "historical_cross_runtime_score_reuse": False, "TEST_access": False}.items()),
        "Finite PE consumer must close every bank")
    pe_owner = read(receipt(pe["owner_end"]))
    require(all(pe_owner.get(k) == v for k, v in {"mode": "fit", "success": True, "exit_code": 0,
        "timed_out": False, "child_pid_absent": True, "owned_cuda_pid_absent": True,
        "direct_child_wait": True, "record": pe["consumer_complete"], "TEST_access": False}.items()),
        "Actual successful PE external owner closure required")
    snapshot = read(receipt(pe["snapshot"]))
    require(snapshot.get("schema_version") == 1 and snapshot.get("made_after_complete_owner_closure") is True
            and snapshot.get("complete_banks") == pe["complete_banks"] and snapshot.get("owner_end") == pe["owner_end"]
            and len(snapshot.get("banks", [])) == 3, "Root snapshot of all PE results/checkpoints/archives before arrays required")
    records, archives, checkpoints = {}, [], []
    for row in snapshot["banks"]:
        seed = row["seed"]; folder = pe_root / f"{PE_ARM}_seed{seed}"
        require(seed in SEEDS and seed not in records and row["arm"] == PE_ARM, "Every fixed PE bank once")
        result = read(receipt(row["bank_result"])); unit = read(receipt(row["unit_result"]))
        require(Path(row["bank_result"]["path"]).resolve() == folder / "RESULT.json"
                and Path(row["unit_result"]["path"]).resolve() == folder / "member0" / "RESULT.json"
                and result in complete["results"] and result["fits"] == [unit], "Complete bank and unit metadata differ")
        require(all(result.get(k) == v for k, v in {"arm": PE_ARM, "seed": seed, "stage": "F", "K": None,
            "architecture": "packed_gamma1", "serving": "probability_mean", "optimizer_acquisition_bundles": 1,
            "native_bodies_fitted": 4, "native_trajectories_per_forward": 4, "selection_scope": "coherent_bank"}.items()), "PE bank semantics differ")
        require(all(unit.get(k) == v for k, v in {"member": 0, "architecture": "packed_gamma1", "factorized": False,
            "learning_objective": "own_only", "within_route_supcon": False, "preclassifier_width": 59,
            "native_bodies_fitted": 4, "optimizer_acquisition_bundles": 1, "selection_scope": "coherent_bank",
            "packed_width_contract": WIDTH}.items()), "Unchanged F objective/selector/private body contract required")
        selected, steps = unit["selected_step"], unit["completed_updates"]
        require(type(selected) is int and type(steps) is int and 1 <= selected <= steps <= 1000, "Fixed acquisition/selected step required")
        ops = unit["operation_counts"]
        require(all(ops.get(k) == v for k, v in {"updates": steps, "backwards": steps, "adam_steps": steps,
            "train_fullgraph_native_trajectories": 4 * steps, "valid_selection_fullgraph_native_trajectories": 4 * steps,
            "selected_valid_fullgraph_native_trajectories": 4, "native_graph_block_calls": 8 * (2 * steps + 1),
            "train_probability_pool_backward_evaluations": 0, "supcon_batched_gram_calls": 0,
            "packed_mean_calls": 2 * (2 * steps + 1), "grouped_affine_calls": 10 * (2 * steps + 1),
            "member_dropout_calls": 20 * steps,
            "edge_channel_values": e["DATA_SHAPES"]["edges"] * 236 * 2 * (2 * steps + 1)}.items()), "Actual complete PE work counters differ")
        require(unit["costs"]["parameters"] == result["costs"]["parameters"] == 215508 and len(unit["body_fit_records"]) == 4, "Complete private parameter budget differs")
        for m, b in enumerate(unit["body_fit_records"]):
            require(b["body_index"] == m and b["native_initialization_seed"] == seed + 1000003 * m
                and b["factorized"] is False and b["packed_member_axis"] == 0 and b["packed_member_index"] == m
                and b["selected_step"] == selected and b["selection_scope"] == "coherent_bank"
                and b["route_factor_dropout_seeds"] == [{"route": m, "factor_seed": None, "dropout_seed": seed + 1000003 * m + 3000007}], "Independent native starts/dropout and coherent selection differ")
        cp = receipt(row["checkpoint"]); archive = receipt(row["archive"])
        require(cp == (folder / "member0" / "selected.pt").resolve() == Path(unit["selected_state"]).resolve()
                and row["checkpoint"]["bytes"] == unit["costs"]["selected_checkpoint_bytes"]
                and archive == (folder / "selected_VALID.npz").resolve(), "Exact selected state/archive snapshot required")
        checkpoints.append(dict(row["checkpoint"], arm=PE_ARM, seed=seed))
        archives.append(dict(row["archive"], arm=PE_ARM, seed=seed, members=4))
        records[seed, PE_ARM] = result
    require(len(records) == 3 and len(complete["results"]) == 3, "All3 PE banks before array import")
    e_report = read(e_report_path)
    require(e_report["frozen_policy"] == e_summary["frozen_policy"] and e_report["raw_authority_preserved"] is True
            and e_report["all75_calibration_attempts_terminal_before_interpretation"] is True
            and len(e_report["calibration_records"]) == 15
            and all(len(r["fits"]) == 5 and all(f["status"] in ("finite_fixed_endpoint", "failed_retained") for f in r["fits"]) for r in e_report["calibration_records"]),
            "Retain completed E policy and all75 terminal outcomes unchanged")
    return {"E": e, "E_report": e_report, "E_summary": e_summary, "PE_records": records,
            "PE_archives": archives, "PE_checkpoints": checkpoints, "PE_consumer": consumer,
            "PE_owner": pe_owner, "source_snapshot": snapshot, "spec": spec}


def descriptive(value):
    if isinstance(value, dict):
        return {k: descriptive(v) for k, v in value.items() if k != "exact_sign_flip_two_sided_p"}
    if isinstance(value, list):
        return [descriptive(v) for v in value]
    return value


def level_summary(core, values):
    result = descriptive(core.paired(values))
    result["seed_values"] = result.pop("seed_deltas")
    result.pop("nonnegative_seed_count"); result.pop("positive_seed_count")
    result["uncertainty_scope"] = PROTOCOL["uncertainty"]
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input-spec", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); started = time.perf_counter(); spec_path = Path(args.input_spec).resolve(); spec = read(spec_path)
    context = preflight(spec)
    core = load("joined_exact_F_math", context["E"]["math_reader_path"])
    # All metadata/checkpoint/archive snapshots and original E readout close first.
    import numpy as np
    require(np.__version__ == context["E_report"]["numerical_reader_runtime"]["numpy"], "Same admitted CPU readout runtime required")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    arrays, ids, labels = core.payloads(np, context["E"])  # Original E15 count unchanged.
    for item in context["PE_archives"]:
        require(current(item["path"]) == {k: item[k] for k in ("path", "bytes", "sha256")}, "PE archive changed after entire custody gate")
        with np.load(item["path"], allow_pickle=False) as f:
            require(set(f.files) == {"ids", "y", "raw_logits", "probability_mean", "member_errors", "pooled_errors"}, "Exact PE selected archive schema required")
            a = {k: f[k].copy() for k in f.files}
        require(np.array_equal(a["ids"], ids) and np.array_equal(a["y"], labels)
                and a["raw_logits"].dtype == a["probability_mean"].dtype == np.float32
                and a["raw_logits"].shape == (4, 5274, 10) and a["probability_mean"].shape == (5274, 10)
                and a["member_errors"].dtype == a["pooled_errors"].dtype == np.bool_
                and a["member_errors"].shape == (4, 5274) and a["pooled_errors"].shape == (5274,)
                and np.isfinite(a["raw_logits"]).all() and np.isfinite(a["probability_mean"]).all()
                and np.array_equal(a["pooled_errors"], a["probability_mean"].argmax(-1) != labels), "Ordered-role/schema/raw-authority PE archive differs")
        arrays[item["seed"], PE_ARM] = a
    require(len(arrays) == 18, "All18 selected archive schemas before joined scoring")
    records = dict(context["E"]["records"]); records.update(context["PE_records"])
    old = {(r["seed"], r["arm"]): r["raw"] for r in context["E_report"]["arm_details"]}
    values, details = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            a = arrays[seed, arm]; x = a["raw_logits"].astype(np.float64)
            log_p = x - core.logsumexp(np, x, -1)[..., None]
            score = core.score(np, log_p, a["probability_mean"], a["member_errors"], a["pooled_errors"], labels)
            selected, counts = records[seed, arm]["valid"], score["counts"]
            require(counts["pooled_correct"] == round(selected["accuracy"] * 5274)
                    and counts["coverage"] == selected["correct_alternative_count"]
                    and counts["lost_correct_alternatives"] == selected["coverage_lost_in_pooling"]
                    and counts["member_correct"] == [round(v * 5274) for v in selected["member_accuracy"]], "Raw archived integer decisions differ from selected records")
            if arm in E_ARMS:
                require(counts == old[seed, arm]["counts"], "Original E raw count changed in joined reconstruction")
            values[seed, arm] = score
            details.append({"arm": arm, "seed": seed, "raw": core.public_score(score),
                            "archived_member_flags_vs_raw_argmax_disagreements": int(((a["raw_logits"].argmax(-1) != labels) != a["member_errors"]).sum()),
                            "actual_costs": records[seed, arm]["costs"]})
    contrasts = {arm + "_minus_" + PE_ARM: core.contrast(np, values, arm, PE_ARM, labels, SEEDS, "complete_PE_E_raw_procedures") for arm in E_ARMS}
    screens = {}
    cost_fields = ("parameters", "inference_parameter_bytes", "acquisition_seconds", "selected_restore_seconds",
                   "selected_serving_readout_seconds", "acquisition_restore_selected_readout_seconds")
    for arm in E_ARMS:
        key = arm + "_minus_" + PE_ARM
        contrasts[key]["diagnostic_cost_deltas"] = {k: core.paired(records[s, arm]["costs"][k] - records[s, PE_ARM]["costs"][k] for s in SEEDS) for k in cost_fields}
    for arm in ("BASE", "E4"):
        delta = contrasts[arm + "_minus_" + PE_ARM]["quality_deltas"]["pooled_accuracy_pct"]
        screens[arm + "_minus_" + PE_ARM] = {"paired_raw_accuracy_pp": delta,
            "mean_at_least_0_2_pp": delta["mean"] >= .2, "all_three_nonnegative": delta["nonnegative_seed_count"] == 3,
            "at_least_two_positive": delta["positive_seed_count"] >= 2,
            "pass": delta["mean"] >= .2 and delta["nonnegative_seed_count"] == 3 and delta["positive_seed_count"] >= 2}
    aggregates = [{"arm": a, "quality": {k: level_summary(core, (values[s, a]["quality"][k] for s in SEEDS)) for k in core.METRICS},
                   "costs": core.scalar_costs([records[s, a] for s in SEEDS])} for a in ARMS]
    report = {"complete": True, "study_id": spec["study_id"], "study_kind": PROTOCOL["study_kind"],
        "protocol": PROTOCOL, "banks": 18, "optimizer_bundles": 27, "body_records": 36,
        "all_rosters_snapshotted_before_arrays": True, "E75_and_reader_closed_before_join": True,
        "original_E_raw_count_unchanged": True, "raw_error_flags_authoritative": True,
        "new_calibration": 0, "model_forwards": 0, "selected_state_deserializations": 0, "TEST_access": False,
        "original_E_frozen_policy_unchanged": context["E_summary"]["frozen_policy"],
        "original_E_calibration_costs_and_failures": context["E_summary"]["calibration_costs"],
        "separate_PE_contextual_raw_screens": descriptive(screens), "arm_aggregates": aggregates,
        "arm_details": details, "raw_contrasts": descriptive(contrasts),
        "observed_scope_costs": {"E_preparation": context["E"]["preparation"],
            "E_family_observed_seconds": context["E"]["entire_family_observed_seconds"],
            "PE_external_owner_seconds": context["PE_owner"]["seconds"], "PE_consumer": context["PE_consumer"],
            "joined_raw_reader_seconds": time.perf_counter() - started,
            "joined_process_lifetime_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "scope": "Nested timers retained; sums do not establish isolated speed or equal runtime. PE remains storage-matched context; full-width I4 stays decisive."},
        "custody": {"input_spec": current(spec_path), "E_input_spec": spec["E_input_spec"],
            "E_complete_summary": spec["E_complete_summary"], "E_reader_owner_end": spec["E_reader_owner_end"],
            "PE": spec["PE"], "reader_source": current(__file__)},
        "uncertainty_scope": PROTOCOL["uncertainty"], "primitive_or_acceptance_claim": False}
    path = output / "COMPLETE_JOINED_RAW_REPORT.json"; path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = {k: report[k] for k in ("complete", "study_id", "study_kind", "banks", "optimizer_bundles", "body_records",
        "original_E_raw_count_unchanged", "E75_and_reader_closed_before_join", "separate_PE_contextual_raw_screens",
        "original_E_frozen_policy_unchanged", "new_calibration", "model_forwards", "selected_state_deserializations", "TEST_access")}
    summary["full_report"] = current(path)
    (output / "COMPLETE_JOINED_RAW_SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
