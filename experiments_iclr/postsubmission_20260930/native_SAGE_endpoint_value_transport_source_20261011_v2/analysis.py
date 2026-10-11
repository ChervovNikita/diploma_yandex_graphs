"""Complete endpoint VALUE reader; reuse qualified F readout/math utilities.

Stdlib until all15 banks/24 bundles/24 native bodies, owner, source/runtime/role,
selected checkpoint/archive custody close. No model forward/deserialization.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import statistics
import sys

ARMS = ("BASE", "E4", "N4", "equipped_M1", "equipped_I4")
PRIMARY_REFERENCES = ("BASE", "equipped_I4")
SPECS = {"BASE": ("shared", 4, 1, 1, "none"),
         "E4": ("shared", 4, 1, 1, "endpoint"),
         "N4": ("shared", 4, 1, 1, "node"),
         "equipped_M1": ("single", 1, 1, 1, "endpoint"),
         "equipped_I4": ("single", 4, 4, 4, "endpoint")}


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def current(path):
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}


def read(path):
    return json.loads(Path(path).read_text())


def preflight(spec):
    contracts = load_source("endpoint_readout_contracts", Path(__file__).with_name("contracts.py"))
    receipt = contracts.receipt
    require(spec.get("schema_version") == 1 and spec.get("study_kind") == "native_SAGE_endpoint_VALUE"
            and isinstance(spec.get("study_id"), str) and spec["study_id"],
            "Exact root endpoint-VALUE reader identity required")
    require(receipt(spec["reader_source"]) == Path(__file__).resolve(), "Reader source differs")
    require(read(receipt(spec["protocol"])) == contracts.PROTOCOL, "Fixed prospective protocol differs")
    cfg_path = receipt(spec["config"]); cfg = read(cfg_path)
    lock = contracts.validate(cfg, action="fixed_family_fit",
                              executing_source=cfg["source_receipts"]["training_family"]["path"])
    root = Path(spec["family_root"]).resolve()
    require(root.is_dir() and read(root / "CONFIG.json") == cfg, "Saved/submitted entire config differs")
    complete_path = receipt(spec["complete_family"])
    require(complete_path == root / "COMPLETE_FAMILY.json", "Complete family path differs")
    complete = read(complete_path)
    expected = {"complete": True, "full_same_runtime_stage_complete": True,
        "stage": "F", "K": None, "fixed_seed_roster": contracts.SEEDS,
        "completed_seed_block": contracts.SEEDS, "fixed_arm_roster": list(ARMS),
        "full_expected_groups": 15, "full_expected_optimizer_acquisition_bundles": 24,
        "full_expected_native_body_fit_records": 24, "full_native_trajectories_per_roster_forward": 51,
        "groups": 15, "fit_units": 24, "optimizer_acquisition_bundles": 24,
        "native_body_fit_records": 24, "expected_groups": 15, "expected_fit_units": 24,
        "expected_native_body_fit_records": 24, "TEST_access": False,
        "study_kind": "native_SAGE_endpoint_VALUE", "baseline_policy": "fresh_required"}
    require(all(complete.get(key) == value for key, value in expected.items()), "Entire finite native family must close before arrays")
    require(complete.get("source_sha256") == lock["source_sha256"]
        and complete.get("safe_role_sha256") == lock["safe_role_sha256"]
        and complete.get("learning_lock") == lock and complete.get("protocol") == contracts.PROTOCOL
        and complete.get("config_sha256") == spec["config"]["sha256"], "Frozen complete source/runtime/role/config custody differs")
    require(read(root / "PREPARATION.json") == complete.get("preparation")
            and complete.get("finite_family_limits") == cfg["family_limits"],
            "Preparation cost and finite entire-family budget custody differs")
    require(read(root / "SOURCE_HASHES.json") == {"config_sha256": spec["config"]["sha256"], "files": lock["source_sha256"]}, "Saved source digest record differs")
    runtime = lock["root_runtime_admission"]
    require(runtime["hostname"] == socket.gethostname() and complete.get("runtime_admission") == runtime,
            "Same admitted physical/runtime/source identity required")
    owner = read(receipt(spec["owner_end"]))
    required_owner = {"status": "COMPLETE_EXIT_0", "exit_code": 0,
        "study_kind": "native_SAGE_endpoint_VALUE", "config_sha256": spec["config"]["sha256"],
        "training_source_sha256": cfg["source_receipts"]["training_family"]["sha256"],
        "runtime_fingerprint_sha256": runtime["runtime_fingerprint_sha256"],
        "hostname": runtime["hostname"], "gpu_uuid": runtime["gpu_uuid"],
        "banks": 15, "optimizer_bundles": 24, "native_body_records": 24,
        "complete_family_sha256": spec["complete_family"]["sha256"], "TEST_access": False}
    require(all(owner.get(key) == value for key, value in required_owner.items()), "Successful entire owner closure required")
    shapes = read(root / "DATA_SHAPES.json")
    require(shapes.get("nodes") == 11701 and shapes.get("TRAIN_rows") == 580 and shapes.get("VALID_rows") == 5274
            and not shapes.get("qualification_never_deserializes_VALID") and shapes.get("TEST_access") is False,
            "Actual whole factual graph/role metadata required")
    records, archives, checkpoints, bodies = {}, [], [], []
    for result in complete["results"]:
        seed, arm = result.get("seed"), result.get("arm")
        require(seed in contracts.SEEDS and arm in SPECS and (seed, arm) not in records, "Every fixed bank exactly once")
        architecture, members, nbody, bundles, operation = SPECS[arm]
        folder = root / (arm + "_seed" + str(seed))
        require(read(folder / "RESULT.json") == result, "Bank RESULT differs from complete record")
        require(result.get("architecture") == architecture and result.get("serving") == "probability_mean"
                and result.get("optimizer_acquisition_bundles") == bundles and result.get("native_bodies_fitted") == nbody
                and result.get("native_trajectories_per_forward") == members and result.get("value_operation") == operation,
                "Bank body/route/ownership/operator semantics differ")
        fits = result.get("fits")
        require(isinstance(fits, list) and len(fits) == bundles, "Entire bank optimizer roster required")
        for member, unit in enumerate(fits):
            require(read(folder / ("member" + str(member)) / "RESULT.json") == unit, "Fit RESULT differs")
            selected, steps = unit.get("selected_step"), unit.get("completed_updates")
            require(type(selected) is int and type(steps) is int and 1 <= selected <= steps <= 1000
                    and unit.get("member") == member and unit.get("architecture") == architecture
                    and unit.get("factorized") is True and unit.get("learning_objective") == "own_only"
                    and unit.get("within_route_supcon") is False and unit.get("value_operation") == operation,
                    "Fixed native constructor/own CE/stopping/operator contract differs")
            routes = 4 if architecture == "shared" else 1
            operations = unit["operation_counts"]
            expected_counts = {"updates": steps, "backwards": steps, "adam_steps": steps,
                "train_fullgraph_native_trajectories": routes * steps,
                "valid_selection_fullgraph_native_trajectories": routes * steps,
                "selected_valid_fullgraph_native_trajectories": routes,
                "native_graph_block_calls": 2 * routes * (2 * steps + 1),
                "train_probability_pool_backward_evaluations": 0,
                "supcon_batched_gram_calls": 0}
            require(all(operations.get(key) == value for key, value in expected_counts.items()), "Native work counters differ")
            require(unit.get("selection_scope") == ("coherent_bank" if routes == 4 else "own_body"), "Native selector scope differs")
            value_roles = unit["value_parameter_roles"]
            expected_added = 0 if operation == "none" else 2 * (8224 + 544 * routes)
            require(value_roles["total"] == expected_added
                and value_roles["shared_builder_parameters"] == (0 if operation == "none" else 16448)
                and value_roles["private_builder_factor_parameters"] == (0 if operation == "none" else 1088 * routes),
                "Actual registered VALUE parameter count differs")
            require(unit["costs"]["parameters"] == (216674 if arm == "BASE" else 237474 if routes == 4 else 225600),
                    "Actual native body/full operator parameter count differs")
            native_member = 0 if architecture == "shared" else member
            body_rows = unit.get("body_fit_records")
            require(isinstance(body_rows, list) and len(body_rows) == 1, "One actual body per optimizer bundle")
            body = body_rows[0]
            expected_seeds = [{"route": row, "factor_seed": seed + 1000003 * row + 2000003,
                               "dropout_seed": seed + 1000003 * row + 3000007}
                              for row in (range(4) if architecture == "shared" else (member,))]
            require(body.get("body_index") == member and body.get("native_initialization_seed") == seed + 1000003 * native_member
                    and body.get("route_factor_dropout_seeds") == expected_seeds
                    and body.get("selected_step") == selected, "Genuine/native factor/dropout constructor law differs")
            checkpoint = receipt(unit["selected_state_receipt"])
            require(checkpoint == (folder / ("member" + str(member)) / "selected.pt").resolve()
                and checkpoint == Path(unit["selected_state"]).resolve()
                and unit["costs"]["selected_checkpoint_bytes"] == unit["selected_state_receipt"]["bytes"], "Selected checkpoint custody differs")
            checkpoints.append(dict(unit["selected_state_receipt"], arm=arm, seed=seed, member=member))
            bodies.append(dict(body, arm=arm, seed=seed))
            transport = unit["transport_operation_counts"]
            total = transport["total"]
            require(all(operations.get(key) == total.get(key, 0) for key in contracts.VALUE_COUNT_KEYS), "Actual transport total counter differs")
            if operation == "none":
                require(not any(transport.values()) and unit.get("value_installations") == [], "BASE must retain native messages")
            else:
                rows = shapes["edges"] if operation == "endpoint" else 11701
                conv_calls = 2 * routes * (2 * steps + 1)
                expected_value = {"value_conv_calls": conv_calls, "value_builder_calls": 2 * conv_calls,
                    "value_builder_rows": 2 * rows * conv_calls,
                    "value_builder_MACs": 8192 * rows * conv_calls,
                    "value_pair_matrix_products": rows * conv_calls,
                    "value_pair_matrix_MACs": 64 * rows * conv_calls,
                    "value_message_rows": shapes["edges"] * conv_calls,
                    "value_message_application_MACs": 512 * shapes["edges"] * conv_calls}
                require(all(total.get(key) == value for key, value in expected_value.items()), "Actual full-graph VALUE work differs")
                require(len(unit.get("value_installations", [])) == 2
                        and all(row.get("native_map_parameter_identity_preserved") is True
                                and row.get("conditioning") == operation for row in unit["value_installations"]),
                        "Preserved native ownership and two hooked factual layers required")
                require(transport["TRAIN"]["value_conv_calls"] == 2 * routes * steps
                        and transport["VALID_selector"]["value_conv_calls"] == 2 * routes * steps
                        and transport["selected_VALID"]["value_conv_calls"] == 2 * routes,
                        "TRAIN/selection/selected work scopes differ")
        archive = receipt(result["selected_VALID_archive"])
        require(archive == (folder / "selected_VALID.npz").resolve(), "Selected archive path differs")
        archives.append(dict(result["selected_VALID_archive"], arm=arm, seed=seed, members=members))
        records[seed, arm] = result
    require(set(records) == {(seed, arm) for seed in contracts.SEEDS for arm in ARMS}
            and len(archives) == 15 and len(checkpoints) == 24 and len(bodies) == 24,
            "Every bank, independent fit and exact selected body must close")
    math_path = receipt(spec["math_reader"])
    require(spec["math_reader"]["sha256"] == "b2d4deb6e7a1b9c6da5597e851a42caf11ece728f13b95ebd4a3871b01941ef0",
            "Pin the reused exact F readout utility source")
    return {"seeds": contracts.SEEDS, "records": records, "archives": archives,
        "checkpoints": checkpoints, "body_records": bodies,
        "role_paths": {key: Path(value["path"]).resolve() for key, value in cfg["safe_roles"].items()},
        "protocol": contracts.PROTOCOL, "math_reader_path": math_path,
        "runtime": runtime, "source_sha256": lock["source_sha256"],
        "safe_roles": cfg["safe_roles"], "complete_family": spec["complete_family"],
        "owner_end": spec["owner_end"], "config": spec["config"], "DATA_SHAPES": shapes,
        "preparation": complete["preparation"],
        "entire_family_observed_seconds": complete["entire_family_observed_seconds"],
        "finite_family_limits": complete["finite_family_limits"]}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def analyze(core, np, raw, calibrated, arrays, labels, context):
    seeds = context["seeds"]
    PROTOCOL = context["protocol"]
    raw_contrasts = {a + "_minus_" + b: core.contrast(np, raw, a, b, labels, seeds) for a in ARMS for b in ARMS if a != b}
    calibrated_contrasts = {a + "_minus_" + b: core.contrast(np, calibrated, a, b, labels, seeds) for a in ARMS for b in ARMS if a != b}
    for a in ARMS:
        for b in ARMS:
            if a == b:
                continue
            cost_deltas = {k: core.paired(context["records"][s, a]["costs"][k] - context["records"][s, b]["costs"][k] for s in seeds)
                           for k in ("parameters", "inference_parameter_bytes", "acquisition_seconds", "selected_restore_seconds", "selected_serving_readout_seconds", "acquisition_restore_selected_readout_seconds")}
            raw_contrasts[a + "_minus_" + b]["diagnostic_cost_deltas"] = cost_deltas
            calibrated_contrasts[a + "_minus_" + b]["diagnostic_native_cost_deltas"] = cost_deltas
    same_state = {}
    for arm in ARMS:
        values = {(s, "raw"): raw[s, arm] for s in seeds}
        values.update({(s, "calibrated"): calibrated[s, arm] for s in seeds})
        same_state[arm] = core.contrast(np, values, "calibrated", "raw", labels, seeds, "same_selected_state_calibration")
    gates = {}
    for ref in PRIMARY_REFERENCES:
        key = "E4_minus_" + ref
        accuracy = raw_contrasts[key]["quality_deltas"]["pooled_accuracy_pct"]
        raw_gate = {"paired_raw_accuracy_pp": accuracy, "mean_at_least_0_2_pp": accuracy["mean"] >= 0.2,
                    "all_three_nonnegative": accuracy["nonnegative_seed_count"] == 3,
                    "at_least_two_positive": accuracy["positive_seed_count"] >= 2}
        raw_gate["pass"] = all(raw_gate[k] for k in ("mean_at_least_0_2_pp", "all_three_nonnegative", "at_least_two_positive"))
        cal = calibrated_contrasts[key]
        finite = cal["status"] == "finite_complete"
        confidence = {"finite_complete": finite, "paired_calibrated_NLL_deterioration": cal["quality_deltas"]["pooled_nll"] if finite else None,
                      "paired_calibrated_accuracy_pp": cal["quality_deltas"]["pooled_accuracy_pct"] if finite else None,
                      "calibrated_accuracy_is_reported_not_an_extra_gate": True}
        confidence["mean_NLL_deterioration_at_most_0_02"] = finite and confidence["paired_calibrated_NLL_deterioration"]["mean"] <= 0.02
        confidence["each_seed_NLL_deterioration_at_most_0_05"] = finite and confidence["paired_calibrated_NLL_deterioration"]["max"] <= 0.05
        confidence["pass"] = finite and confidence["mean_NLL_deterioration_at_most_0_02"] and confidence["each_seed_NLL_deterioration_at_most_0_05"]
        gates[ref] = {"raw_accuracy": raw_gate, "confidence_protection": confidence}
    policy = {"candidate": "E4", "required_references": list(PRIMARY_REFERENCES), "contrasts": gates,
              "raw_accuracy_screen_pass": all(g["raw_accuracy"]["pass"] for g in gates.values()),
              "calibrated_confidence_protection_pass": all(g["confidence_protection"]["pass"] for g in gates.values()),
              "member_quality_diagnostic_not_veto": True, "costs_diagnostic_not_veto": True,
              "no_K_choice": True, "no_automatic_confirmation_launch": True,
              "favorable_findings_preserved_on_failure": True, "scope": PROTOCOL["scope"]}
    policy["joint_accuracy_confidence_development_clue"] = policy["raw_accuracy_screen_pass"] and policy["calibrated_confidence_protection_pass"]
    policy["accuracy_only_clue_with_confidence_tradeoff"] = policy["raw_accuracy_screen_pass"] and not policy["calibrated_confidence_protection_pass"]
    endpoint = raw_contrasts["E4_minus_N4"]["quality_deltas"]["pooled_accuracy_pct"]
    attribution = {"paired_raw_accuracy_pp": endpoint, "mean_at_least_0_1_pp": endpoint["mean"] >= 0.1,
                   "all_three_nonnegative": endpoint["nonnegative_seed_count"] == 3,
                   "at_least_two_positive": endpoint["positive_seed_count"] >= 2}
    attribution["pass"] = all(attribution[key] for key in ("mean_at_least_0_1_pp", "all_three_nonnegative", "at_least_two_positive"))
    policy["separate_endpoint_attribution"] = attribution
    policy["raw_M1_contrast_reported"] = "E4_minus_equipped_M1"
    aggregates = []
    for readout, values in (("raw", raw), ("calibrated", calibrated)):
        for arm in ARMS:
            rs = [values[s, arm] for s in seeds]
            native = [context["records"][s, arm] for s in seeds]
            if any(r is None for r in rs):
                aggregate = {"status": "failed_calibration_retained", "missing_seed_readouts": [s for s, r in zip(seeds, rs) if r is None], "acquisition_costs": core.scalar_costs(native)}
            else:
                aggregate = {"status": "finite_complete", "mean_quality": {k: statistics.mean(r["quality"][k] for r in rs) for k in core.METRICS},
                             "summed_three_seed_readout_counts": {k: sum(r["counts"][k] for r in rs) for k in rs[0]["counts"] if k != "member_correct"},
                             "acquisition_costs": core.scalar_costs(native), "count_scope": "Three dependent readouts of the same 5274 development nodes."}
            aggregates.append({"readout": readout, "arm": arm, "aggregate": aggregate})
    diagnostics = []
    for seed in seeds:
        masks = {arm: core.strict_mask(np, arrays[seed, arm]["raw_logits"], labels) for arm in ARMS}
        for arm in ARMS:
            a, r = arrays[seed, arm], raw[seed, arm]
            raw64 = a["raw_logits"].astype(np.float64)
            log_p = raw64 - core.logsumexp(np, raw64, -1)[..., None]
            d = {"arm": arm, "seed": seed, "current_bank_strict_rival": core.cohort(np, masks[arm], r["P"], labels),
                 "BASE_strict_rival_resolved": int((masks["BASE"] & ~masks[arm]).sum()),
                 "new_strict_rivals_vs_BASE": int((~masks["BASE"] & masks[arm]).sum()),
                 "fixed_native_reference_strict_rival_cohorts_raw": {ref: core.cohort(np, mask, r["P"], labels) for ref, mask in masks.items()},
                 "fixed_native_reference_strict_rival_cohorts_calibrated": None if calibrated[seed, arm] is None else {ref: core.cohort(np, mask, calibrated[seed, arm]["P"], labels) for ref, mask in masks.items()},
                 "FP64_raw_pool_argmax_disagreements_with_archived_errors": int(((np.exp(log_p).mean(0).argmax(-1) != labels) != a["pooled_errors"]).sum()),
                 "raw_logit_argmax_disagreements_with_archived_member_errors": int(((a["raw_logits"].argmax(-1) != labels) != a["member_errors"]).sum()),
                 "stable_NLL_minus_original_reported": r["quality"]["pooled_nll"] - context["records"][seed, arm]["valid"]["nll"],
                 "scope": "Fresh native VALUE selected-state cohorts; not old allocation anchors. Cross-procedure repairs include learning/stopping/selection; zero current strict-rival repairs is not zero repair of reference failures."}
            diagnostics.append(d)
    return policy, aggregates, raw_contrasts, calibrated_contrasts, same_state, diagnostics



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = Path(args.input_spec).resolve()
    input_bytes = input_path.read_bytes()
    spec = json.loads(input_bytes)
    context = preflight(spec)
    core = load_source("endpoint_reused_F_reader_math", context["math_reader_path"])
    core.ARMS = ARMS  # Local loaded utility namespace only; preserved source bytes.
    # No numerical library or selected-array import before every native gate.
    import numpy as np
    import torch
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays, ids, labels = core.payloads(np, context)
    endpoints, calibration_records, calibration_costs, folds = core.calibrate(np, torch, arrays, ids, labels, context["seeds"], output)
    # All seventy-five attempts are terminal before scoring, contrasts or gates.
    raw, calibrated, arm_details = {}, {}, []
    for arm in ARMS:
        for seed in context["seeds"]:
            a = arrays[seed, arm]
            raw64 = a["raw_logits"].astype(np.float64)
            log_p = raw64 - core.logsumexp(np, raw64, -1)[..., None]
            raw[seed, arm] = core.score(np, log_p, a["probability_mean"], a["member_errors"], a["pooled_errors"], labels)
            counts, selected = raw[seed, arm]["counts"], context["records"][seed, arm]["valid"]
            require(counts["pooled_correct"] == round(selected["accuracy"] * 5274)
                    and counts["coverage"] == selected["correct_alternative_count"]
                    and counts["lost_correct_alternatives"] == selected["coverage_lost_in_pooling"]
                    and counts["member_correct"] == [round(v * 5274) for v in selected["member_accuracy"]],
                    "Raw archived integer decisions differ from complete selected records")
            endpoint = endpoints[seed, arm]
            if endpoint is None:
                calibrated[seed, arm] = None
            else:
                member_log, pool_log = endpoint
                calibrated[seed, arm] = core.score(np, member_log, np.exp(pool_log), a["member_errors"], pool_log.argmax(-1) != labels, labels)
            arm_details.append({"arm": arm, "seed": seed, "raw": core.public_score(raw[seed, arm]),
                                "calibrated": core.public_score(calibrated[seed, arm]),
                                "calibrated_member_argmax_disagreements_with_raw_flags": None if endpoint is None else int(((endpoint[0].argmax(-1) != labels) != a["member_errors"]).sum()),
                                "native_selected_record": context["records"][seed, arm]})
    policy, aggregates, raw_contrasts, cal_contrasts, effects, diagnostics = analyze(core, np, raw, calibrated, arrays, labels, context)
    for r in calibration_records:
        r["selected_readout"] = core.public_score(calibrated[r["seed"], r["arm"]])
    closure = {k: v for k, v in context.items() if k not in ("records", "role_paths", "math_reader_path")}
    closure["safe_roles"] = context["safe_roles"]
    closure["reused_math_source"] = spec["math_reader"]
    report = {"complete": True, "study_id": spec["study_id"], "stage": "F", "K": None, "study_kind": "native_SAGE_endpoint_VALUE", "TEST_access": False,
              "banks": 15, "optimizer_acquisition_bundles": 24, "native_body_fit_records": 24,
              "fixed_seed_roster": context["seeds"], "fixed_arm_roster": list(ARMS),
              "entire_same_runtime_roster_admitted_before_arrays": True,
              "all75_calibration_attempts_terminal_before_interpretation": True,
              "raw_authority_preserved": True, "protocol": context["protocol"], "frozen_policy": policy,
              "custody": closure, "folds": folds, "calibration_costs": calibration_costs,
              "arm_aggregates": aggregates, "arm_details": arm_details, "diagnostics": diagnostics,
              "calibration_records": calibration_records, "raw_contrasts": raw_contrasts,
              "calibrated_contrasts": cal_contrasts, "calibration_minus_raw_same_selected_state": effects,
              "input_spec_sha256": hashlib.sha256(input_bytes).hexdigest(), "reader_source_sha256": digest(__file__),
              "numerical_reader_runtime": {"torch": torch.__version__, "numpy": np.__version__, "CPU_threads": torch.get_num_threads(), "calibration_device": "cpu", "calibration_dtype": "float64"},
              "limits": ["Favorable raw findings and confidence failures remain visible; no desired verdict or silent winner selection.",
                         "Costs and member quality are diagnostics, not independent vetoes on served-pool utility.",
                         "Native VALUE changes and receiver capacity are compared through full fitted procedures; independent ownership/optimizer histories differ.",
                         "Native selection precedes OOF temperature fitting; encountered development, not unused whole-pipeline confirmation.",
                         "No K, historical anchors, TEST, model deserialization/forward, grid, retry or final refit.",
                         "Three seeds have descriptive paired uncertainty; no independent-node significance, primitive novelty or manuscript acceptance claim."]}
    write_json(output / "COMPLETE_REPORT.json", report)
    documents = [("SAGE_ENDPOINT_VALUE_POLICY_SUMMARY.json", {"complete": True, "stage": "F", "K": None, "study_kind": "native_SAGE_endpoint_VALUE", "TEST_access": False,
                   "banks": 15, "optimizer_acquisition_bundles": 24, "native_body_fit_records": 24,
                   "frozen_policy": policy, "calibration_costs": calibration_costs})]
    for category, rows in (("SAGE_ENDPOINT_VALUE_ARM_AGGREGATES", aggregates), ("SAGE_ENDPOINT_VALUE_ARM_DETAILS", arm_details),
                           ("SAGE_ENDPOINT_VALUE_DIAGNOSTICS", diagnostics), ("CALIBRATION_FIT_DETAILS", calibration_records),
                           ("SAGE_ENDPOINT_VALUE_raw_full_procedure_contrasts", [{"key": k, "contrast": v} for k, v in raw_contrasts.items()]),
                           ("SAGE_ENDPOINT_VALUE_calibrated_full_procedure_contrasts", [{"key": k, "contrast": v} for k, v in cal_contrasts.items()]),
                           ("SAGE_ENDPOINT_VALUE_calibration_minus_raw_same_selected_state", [{"key": k, "contrast": v} for k, v in effects.items()]),
                           ("CHECKPOINT_CUSTODY", context["checkpoints"]), ("ARCHIVE_CUSTODY", context["archives"]),
                           ("NATIVE_BODY_CUSTODY", context["body_records"]), ("OWNER_CUSTODY", [context["owner_end"]])):
        documents += core.chunk_documents(category, rows)
    partitions = []
    for name, value in documents:
        write_json(output / name, value)
        partitions.append(current(output / name))
    summary = {"complete": True, "study_id": spec["study_id"], "stage": "F", "K": None, "study_kind": "native_SAGE_endpoint_VALUE", "TEST_access": False,
               "banks": 15, "optimizer_acquisition_bundles": 24, "native_body_fit_records": 24,
               "fixed_seed_roster": context["seeds"], "fixed_arm_roster": list(ARMS),
               "entire_same_runtime_roster_admitted_before_arrays": True,
               "all75_calibration_attempts_terminal_before_interpretation": True,
               "calibration_costs": calibration_costs, "frozen_policy": policy,
               "folds": folds, "full_report": current(output / "COMPLETE_REPORT.json"),
               "partitions": partitions, "input_spec_sha256": report["input_spec_sha256"],
               "reader_source_sha256": report["reader_source_sha256"], "raw_authority_preserved": True}
    write_json(output / "COMPLETE_ANALYSIS_SUMMARY.json", summary)
    print(json.dumps({"complete": True, "banks": 15, "bundles": 24, "body_records": 24,
                      "calibration_attempts": 75, "raw_screen": policy["raw_accuracy_screen_pass"],
                      "confidence_screen": policy["calibrated_confidence_protection_pass"],
                      "TEST_access": False}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
