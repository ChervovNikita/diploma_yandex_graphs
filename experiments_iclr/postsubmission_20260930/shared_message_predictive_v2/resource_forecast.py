"""Stdlib conservative cost arithmetic from all ten completed preflight rows."""
import json
import math
from .artifact_plan import forecast as artifact_forecast
from .contract import ARMS, EPOCHS, PHASE, SEEDS, TASKS, TIMING, audit_manifest, audit_runtime_import_receipt, checked, completed


def forecast(rows, fixed_costs):
    """No admission is issued. Root independently qualifies all inputs/results.

    Three measured cost samples are too few for a confidence bound. Use worst
    observed time times a fixed1.25 engineering margin; hash every checkpoint
    and logit four times (creation, replay audit, family lock, separate lock audit). Count every
    scientific CPU/IO second conservatively as GPU allocation wall time.
    """
    if len(rows) != 10 or {(r["task"], r["arm"]) for r in rows} != {(t, a) for t in TASKS for a in ARMS}:
        raise ValueError("All ten complete-task arm measurements required")
    required_fixed = {"validation_export", "prepared_input_and_operator_work", "source_protocol_runtime_audits", "supervisor_startup_manifest_json_overhead", "prior_GPU_wall_seconds"}
    if set(fixed_costs) != required_fixed or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in fixed_costs.values()):
        raise ValueError("Actual/bounded full fixed-cost terms must be explicit")
    specifications, breakdown = {}, []
    allocated, maximum_memory = sum(fixed_costs.values()), 0
    def max_kind(values, kind):
        selected = [r["seconds"] for r in values if r["kind"] == kind]
        if len(selected) != 3 or any(not math.isfinite(v) or v < 0 for v in selected):
            raise ValueError("Three finite complete artifact costs required")
        return max(selected)
    for row in rows:
        value = json.loads(checked(row["report"]).read_text())
        audit_manifest(checked(row["report"]).parent)
        chain = completed(PHASE / row["supervisor_directory"], row["outer_start"])
        if value.get("live") != chain["live"]:
            raise ValueError("Preflight completed source/outer evidence differs")
        audit_runtime_import_receipt(value, require_generated=True)
        if (value.get("schema") != "gnnm-shared-message-bounded-cost-preflight-v2" or value.get("status") != "completed"
                or value.get("task") != row["task"] or value.get("arm") != row["arm"]
                or value.get("retained_weights_or_scores") is not False or value.get("validation_target_export_or_access") is not False
                or value.get("test_target_access") is not False):
            raise ValueError("Not completed bounded train-only costs")
        if chain["gpu_uuid"] != rows[0].get("gpu_uuid") or row.get("gpu_uuid") != chain["gpu_uuid"]:
            raise ValueError("Every cost row must bind the same actual frozen GPU")
        if value["execution"] != row["execution"] or value["request"] != row["request"]:
            raise ValueError("Cost request/execution binding differs")
        task, arm = row["task"], row["arm"]
        arrays = ("three_complete_update_seconds", "three_validation_forward_CPU_slice_seconds", "three_CPU_label_free_reduction_seconds", "three_checkpoint_CPU_transfer_seconds")
        if any(len(value[k]) != 3 or any(not math.isfinite(v) or v < 0 for v in value[k]) for k in arrays):
            raise ValueError("Bounded cost repetitions missing")
        update, validation, CPU, transfer = (max(value[k]) for k in arrays)
        checkpoint_write = max_kind(value["temporary_artifact_writes"], "checkpoint")
        checkpoint_hash = max_kind(value["temporary_artifact_full_hashes"], "checkpoint")
        checkpoint_load = max_kind(value["temporary_artifact_loads"], "checkpoint")
        logit_write = max_kind(value["temporary_artifact_writes"], "validation_logits")
        logit_hash = max_kind(value["temporary_artifact_full_hashes"], "validation_logits")
        logit_load = max_kind(value["temporary_artifact_loads"], "validation_logits")
        checkpoints = EPOCHS // 5 + 2
        timings = value["pooled_inference_seconds"]
        if len(timings) != 5 or any(not math.isfinite(v) or v <= 0 for v in timings):
            raise ValueError("Complete forward timing costs missing")
        terms = {"fresh_fit_and_replay_input_and_setup": 2 * (value["input_load_seconds"] + value["model_operator_setup_transfer_seconds"]),
                 "training_updates": EPOCHS * update,
                 "validation_forward_transfers_and_CPU_metrics": (EPOCHS // 5) * (validation + CPU),
                 "all_improved_initial_final_checkpoints_transfer_write_hash": checkpoints * (transfer + checkpoint_write + 4 * checkpoint_hash),
                 "fit_and_replay_four_logit_writes_and_hashes": 4 * (logit_write + 4 * logit_hash),
                 "selected_and_final_numerical_replays_and_metrics": 2 * (validation + 2 * CPU + checkpoint_load + logit_load),
                 "trained_inference_warmups_and_samples": (TIMING["warmups"] + TIMING["samples"]) * max(timings)}
        total = len(SEEDS) * 1.25 * sum(terms.values())
        allocated += total
        maximum_memory = max(maximum_memory, value["memory"]["peak_reserved_bytes"])
        specifications.setdefault(task, {"validation_nodes": value["validation_nodes"], "classes": value["classes"], "arms": {}})
        if specifications[task]["validation_nodes"] != value["validation_nodes"] or specifications[task]["classes"] != value["classes"]:
            raise ValueError("Complete official validation shape differs across arms")
        specifications[task]["arms"][arm] = value["checkpoint_payload_bytes"]
        breakdown.append({"task": task, "arm": arm, "per_seed_raw_terms_seconds": terms,
                          "all_three_seeds_with125percent_margin_seconds": total,
                          "actual_preflight_supervised_seconds": chain["seconds"], "preflight_gpu_uuid": chain["gpu_uuid"]})
    byte_plan = artifact_forecast(specifications)
    return {"schema": "gnnm-shared-message-expanded-resource-estimate-v2", "fit_count": 30, "updates": 9000,
            "total_GPU_wall_seconds_forecast": allocated, "peak_reserved_bytes": maximum_memory,
            "scientific_artifact_bytes_bound": byte_plan["scientific_artifact_bytes_bound"],
            "prior_and_export_disk_bytes_still_require_root_actual_ledger": True,
            "fixed_costs": fixed_costs, "all10_breakdown": breakdown, "artifact_plan": byte_plan,
            "CPU_proxy_limitation": "label-free fixed-class reductions, not supervised validation metrics; root must qualify this conservative cost proxy",
            "forecast_ceiling_seconds": 16200, "wall_cap_seconds": 21600,
            "within_forecast_ceiling": allocated <= 16200, "margin_is_engineering_not_confidence_bound": True,
            "source_only_resource_admission_or_scientific_launch_established": False}
