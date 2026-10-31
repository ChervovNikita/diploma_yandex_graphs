"""Stdlib plan/shape checks only; no tensors, scientific outputs or execution."""
import hashlib
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def load(name):
    return json.loads((HERE / name).read_text())


def main():
    plan = load("PROTOCOL.json")
    inputs = load("INPUT_CONTRACT.json")
    resource = load("RESOURCE_PLAN.json")
    pins = load("SOURCE_PINS.json")
    checks = []

    def check(name, truth):
        if not truth:
            raise ValueError(name)
        checks.append({"check": name, "passed": True})

    scope = plan["study_scope"]
    arms = scope["arms"]
    tasks = scope["tasks"]
    seeds = scope["seeds"]
    roles = scope["checkpoint_roles"]
    members = scope["members"]
    shapes = inputs["task_shapes"]
    check("postlaunch not primary preregistration", plan["timing_provenance"]["prospective_diagnostic_planned_after_study_launch"] is True and plan["timing_provenance"]["preregistered_primary_endpoint"] is False)
    check("analyst numerical access absent", plan["timing_provenance"]["analyst_scientific_outcomes_accessed"] is False and plan["timing_provenance"]["analyst_training_graph_arrays_or_targets_or_logits_or_checkpoints_accessed"] is False)
    check("execution and promotion disabled", plan["execution_authorized"] is False and plan["diagnostic_implementation_present"] is False and plan["automatic_scientific_promotion_authorized"] is False and plan["paper_change_authorized_by_this_plan"] is False)
    check("root prerequisites missing", all(v is None for k, v in plan["later_prerequisites"].items() if k.endswith("_ref") or k == "exact_output_root"))
    check("root execution qualification false", all(plan["later_prerequisites"][k] is False for k in ("all30_fits_and_replays_and_family_qualified", "root_diagnostic_authorized", "execute_now")))
    check("both tasks fixed", tasks == ["Reddit2", "ogbn-products"])
    check("all five arms fixed", arms == ["exact_member", "shared_mean_message", "detached_mean_message", "single_base64", "single_parameters"])
    check("three seeds fixed", seeds == [17, 29, 43])
    check("roles fixed and separate", roles == ["final", "selected"])
    check("member shapes fixed", members == dict(zip(arms, [4, 4, 4, 1, 1])))
    expected = [{"task": t, "seed": s, "arm": a, "cell_id": shapes[t]["path_prefix"] + "_seed%d_%s" % (s, a)} for t in tasks for s in seeds for a in arms]
    check("canonical complete30 cell set/order", scope["cells"] == expected and len(expected) == 30)
    check("frozen fits/replays/updates", scope["fit_count"] == scope["replay_count_required"] == 30 and scope["epochs"] == 300 and scope["fixed_training_updates"] == 9000)
    expected_pairs = [{"reference": a, "candidate": b} for a, b in itertools.combinations(arms, 2)]
    check("all ten oriented comparisons", plan["cross_arm_comparisons"]["all_pairs"] == expected_pairs and len(expected_pairs) == 10)
    check("emphasis fixed independently of outcomes", plan["cross_arm_comparisons"]["emphasized_pairs"] == [{"reference": "exact_member", "candidate": "shared_mean_message"}, {"reference": "shared_mean_message", "candidate": "detached_mean_message"}])
    check("no test or new forward", scope["test_target_or_test_evaluation_authorized"] is False and scope["training_or_new_forward_or_tuning_authorized"] is False)
    check("no new best member selector", plan["member_analysis"]["new_member_selection_authorized"] is False)
    bins = plan["degree_analysis"]["bins"]
    check("fixed exhaustive nonoverlapping degree boundaries", [(b["lower_inclusive"], b["upper_inclusive"]) for b in bins] == [(0, 0), (1, 4), (5, 16), (17, 64), (65, 256), (257, None)])
    check("all bins/overall retained", plan["degree_analysis"]["include_overall"] is True and plan["output_contract"]["all_rows_including_empty_strata_required"] is True)
    check("no adaptive degree changes/causal claims", plan["degree_analysis"]["bins_may_change_after_outcomes"] is False and plan["degree_analysis"]["causal_or_unseen_graph_population_claims_authorized"] is False)
    groups = len(bins) + 1
    blocks = len(tasks) * len(seeds) * len(roles)
    expected_rows = {"arm_summary": blocks * len(arms) * groups,
                     "cross_arm_decisions": blocks * len(expected_pairs) * groups,
                     "correct_member_count": blocks * sum(m + 1 for m in members.values()) * groups,
                     "member_pair": blocks * sum(m * (m - 1) // 2 for m in members.values()) * groups,
                     "seed_summary": len(tasks) * len(roles) * len(expected_pairs) * groups}
    check("complete aggregate row arithmetic", plan["output_contract"]["expected_rows"] == expected_rows)
    summary = plan["seed_summary"]
    check("descriptive three-seed interval fixed", summary["n_required"] == 3 and summary["critical_value"] == 4.302652729911275 and summary["truncate_interval_to_feasible_range"] is False)
    check("no node pseudoreplication or decision claims", summary["node_or_member_pair_bootstrap_or_hypothesis_test_authorized"] is False and summary["generalization_or_acceptance_claims_authorized"] is False)
    check("no numerical outputs supplied", plan["output_contract"]["result_or_fake_illustration_present_now"] is False and plan["output_contract"]["node_ids_logits_labels_node_level_predictions_export_authorized"] is False)
    check("no input payload access authorized", inputs["binary_access_authorized_now"] is False and inputs["no_input_payload_file_inspected_for_plan"] is True)
    check("logit input shapes", [(shapes[t]["validation_nodes"], shapes[t]["classes"]) for t in tasks] == [(23699, 41), (39323, 47)])
    check("correct frozen width controls", [shapes[t]["parameter_control_width"] for t in tasks] == [66, 65])
    float_bytes = sum(len(seeds) * len(roles) * sum((members[a] + 1) * shapes[t]["validation_nodes"] * shapes[t]["classes"] * 4 for a in arms) for t in tasks)
    ids_bytes = sum(len(seeds) * len(roles) * len(arms) * shapes[t]["validation_nodes"] * 8 for t in tasks)
    target_bytes = sum(shapes[t]["validation_nodes"] * 16 for t in tasks)
    degree_bytes = sum(x["record_copied_without_payload_access"]["payload_bytes"] for x in inputs["degree_inputs"].values())
    payload = resource["input_payload_bytes"]
    check("float payload arithmetic", float_bytes == payload["float32_member_and_pooled_logits"] == 1285847040)
    check("logit ID payload arithmetic", ids_bytes == payload["int64_node_ids_in60logit_inputs"] == 15125280)
    check("target payload arithmetic", target_bytes == payload["int64_validation_targets_and_IDs"] == 1008352)
    check("degree payload metadata arithmetic", degree_bytes == payload["int64_degree_array_payload"] == 21455952)
    check("total declared payload arithmetic", float_bytes + ids_bytes + target_bytes + degree_bytes == payload["total_payload_without_serialization_overhead"] == 1323436624)
    check("64 existing binary inputs; no access now", resource["counts"]["logit_input_files"] == 60 and resource["counts"]["numerical_input_files_total"] == 64)
    cap = resource["cap_proposal"]
    check("bounded CPU/disk proposal only", cap["CPU_threads"] == 4 and cap["host_process_RSS_bytes"] == 2 * 1024**3 and cap["whole_supervised_process_wall_seconds"] == 600 and cap["new_output_disk_bytes"] == 128 * 1024**2 and resource["resource_authorized"] is False)
    check("zero GPU/new learning", cap["GPU_count"] == cap["new_training_updates"] == cap["new_model_forward_calls"] == cap["duplicate_logit_or_target_payload_exports"] == 0)
    proxy = resource["engineering_component_proxy"]
    check("proxy10shape coverage", [(x["task"], x["arm"]) for x in proxy["input_rows"]] == [(t, a) for t in tasks for a in arms])
    check("all proxy samples engineering only", all(x["timing_samples"] == 3 and x["proposed_inputs"] == 6 for x in proxy["input_rows"]))
    load_seconds = sum(6 * x["worst_cached_temporary_logit_load_seconds"] for x in proxy["input_rows"])
    hash_seconds = sum(6 * x["worst_one_full_temporary_logit_hash_seconds"] for x in proxy["input_rows"])
    metric_seconds = sum(6 * x["artificial_full_metric_call_seconds"] for x in proxy["input_rows"])
    check("load hash metric proxy arithmetic", abs(load_seconds - proxy["60_cached_temporary_logit_loads_seconds"]) < 1e-12 and abs(hash_seconds - proxy["60_one_full_temporary_logit_hash_passes_seconds"]) < 1e-12 and abs(metric_seconds - proxy["60_artificial_full_metric_calls_seconds"]) < 1e-12)
    check("component margin arithmetic", abs(1.25 * (load_seconds + 2 * hash_seconds + metric_seconds) - proxy["component_proxy_seconds"]) < 1e-12)
    check("component is not a runtime guarantee", proxy["total_diagnostic_runtime_bound"] is False and proxy["measured_new_diagnostic_runtime"] is False)
    check("new allocation unapproved; training unchanged", resource["accounting"]["reserve_not_granted_now"] is True and resource["accounting"]["modify_frozen_training_resources_estimand_or_thresholds"] is False and resource["accounting"]["root_allocation_reconciliation_ref"] is None)
    check("21 unique source/metadata pins", len(pins["pins"]) == len({p["path"] for p in pins["pins"]}) == 21)
    for pin in pins["pins"]:
        relative = Path(pin["path"])
        check("allowed source/engineering ref kind: " + pin["path"], relative.suffix in {".py", ".json"} and not relative.is_absolute() and ".." not in relative.parts and relative.parts[0] in {"shared_message_predictive_v3", "shared_message_predictive_v2", "shared_message_metric_cost_v1", "propagation_cost_engineering_v1"})
        path = (PHASE / relative).resolve()
        check("source/engineering ref remains scoped: " + pin["path"], path.is_relative_to(PHASE.resolve()))
        raw = path.read_bytes()
        check("source/engineering pin unchanged: " + pin["path"], hashlib.sha256(raw).hexdigest() == pin["sha256"] and len(raw) == pin["bytes"])
    for task, entry in inputs["degree_inputs"].items():
        manifest_path = PHASE / entry["prepared_manifest"]["path"]
        manifest = json.loads(manifest_path.read_text())
        check("opaque degree record exact: " + task, entry["record_copied_without_payload_access"] == manifest["arrays"]["degree"] and entry["binary_payload_opened_stat_or_hashed_now"] is False)
    return {"schema": "gnnm-shared-message-decision-analysis-plan-checks-v1", "status": "passed_source_metadata_only", "checks": checks, "check_count": len(checks), "source_metadata_pin_count": len(pins["pins"]), "numerical_payloads_opened_stat_or_hashed": False, "Torch_imported": False, "execution_GPU_SSH_or_scientific_evidence_accessed": False, "diagnostic_authorized": False, "scope": "Shape, count, metadata pin and disabled-gate checks; not a diagnostic implementation or scientific audit."}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
