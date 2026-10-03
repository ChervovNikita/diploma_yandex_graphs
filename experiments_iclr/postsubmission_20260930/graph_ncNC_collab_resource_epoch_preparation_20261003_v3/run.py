"""Root-admitted entry point; no launch, installation, or acquisition helper."""
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import time
from guards import preflight, import_prototype, write_json, require
from qualification_status import (commit_qualification, invalidate_qualification,
                                  record_failure, qualification_succeeded, validate_accounting)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("gpu_parity", "resource_epoch"))
    parser.add_argument("--root-admission", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    # Do not move any Torch, NumPy, pandas, model or data import above this gate.
    context = preflight(args.root_admission, args.stage, args.output_directory)
    receipt = {"schema": "ncnc-collab-TRAIN-resource-qualification-v2", "stage": args.stage,
        "UTC": datetime.now(timezone.utc).isoformat(), "status": "FAILED",
        "root_admission_sha256": context["admission_sha256"],
        "preparation_manifest_sha256": context["manifest_sha256"],
        "prototype_manifest_sha256": context["bindings"]["prototype_manifest_sha256"],
        "engineering_rng_identity": context["admission"]["engineering_rng_identity"],
        "engineering_rng_seed": context["admission"]["engineering_rng_seed"],
        "node_count": 235868, "train_records": 1179052, "native_batch_size": 65536,
        "preflight_seconds": context["preflight_seconds"], "custody_hash_seconds": context["custody_hash_seconds"],
        "heldout_split_payloads_opened": False, "predictive_metrics_computed": False,
        "scientific_result_or_launch_admitted": False, "serving_pool": "mean_raw_logits",
        "all_required_checks_passed": False, "accounting_complete": False, "stage_checks_passed": False}
    meter = None
    stage_progress = {}
    try:
        from runtime import qualify_runtime, set_engineering_rng, Meter
        device, sampler, runtime_identity = qualify_runtime(context)
        receipt["runtime_identity"] = runtime_identity
        meter = Meter()
        prototype, graph_ops, reference, selected_state = import_prototype(context["prototype"])
        from train_only_data import load_train_only, transfer_data, tensor_sha
        receipt["current_stage"] = "cold_complete_TRAIN_feature_graph_load"
        data = load_train_only(context, meter)
        receipt["data_tensor_digests"] = data["digests"]
        data = transfer_data(data, device, meter)
        full_graph = meter.call("initial_complete_TRAIN_graph_qualification", graph_ops.Graph.from_pairs, data["pairs"], 235868)
        require(len(full_graph.row) == 1935264, "Complete acquired TRAIN topology count differs")
        def graph_digest():
            import torch
            return tensor_sha(torch.stack((full_graph.row, full_graph.col)))
        meter.counts["explicit_device_to_host_tensor_bytes"] += 2 * len(full_graph.row) * 8
        require(meter.call("complete_graph_hash_and_host_transfer", graph_digest) == context["bindings"]["coalesced_graph_sha256"], "Complete TRAIN graph custody digest differs")
        receipt["complete_graph_directed_entries"] = len(full_graph.row)
        del full_graph
        set_engineering_rng(context["admission"])
        receipt["current_stage"] = args.stage
        try:
            if args.stage == "gpu_parity":
                import gpu_parity
                candidate_status = gpu_parity.run(context, prototype, graph_ops, reference, data, device, sampler, meter, stage_progress)
            else:
                import resource_epoch
                candidate_status = resource_epoch.run(context, prototype, graph_ops, reference, selected_state, data, device, sampler, meter, stage_progress)
        finally:
            # Stage-local numerical success never commits top-level success.
            receipt.update({key: value for key, value in stage_progress.items()
                            if key not in ("status", "all_required_checks_passed", "accounting_complete")})
            receipt["stage_checks_passed"] = stage_progress.get("all_required_checks_passed") is True
        receipt["current_stage"] = "final_accounting_and_CUDA_sync"
        final_accounting = validate_accounting(meter.receipt())
        receipt["final_process_accounting"] = final_accounting
        if "setup_accounting_before_twin_peak_resets" in receipt:
            receipt["setup_or_GPU_parity_accounting"] = validate_accounting(receipt.pop("setup_accounting_before_twin_peak_resets"))
        else:
            receipt["setup_or_GPU_parity_accounting"] = final_accounting
        commit_qualification(receipt, candidate_status)
        receipt["current_stage"] = "complete"
    except Exception as error:
        # Concrete runtime condition only; no rescue batch/candidate/graph cap.
        invalidate_qualification(receipt, error, "runtime_or_required_accounting")
        if meter is not None:
            try:
                receipt["diagnostic_process_accounting_after_failure"] = meter.receipt()
            except Exception as accounting_error:
                record_failure(receipt, accounting_error, "diagnostic_process_accounting_after_failure")
    finally:
        # The final metadata write cannot include its own completed duration.
        receipt["total_process_seconds_including_stdlib_admission"] = time.perf_counter() - started
    try:
        # Publish the final receipt only after the complete JSON write succeeds.
        # A failed write/rename can leave a pending file, never a final pass file.
        pending = context["output"] / "QUALIFICATION_PENDING.json"
        write_json(pending, receipt)
        pending.replace(context["output"] / "QUALIFICATION.json")
    except Exception as publication_error:
        invalidate_qualification(receipt, publication_error, "final_metadata_write")
        print("RESOURCE_QUALIFICATION_METADATA_WRITE_FAILED=" + type(publication_error).__name__
              + ": " + str(publication_error)[:1000], flush=True)
    print("RESOURCE_QUALIFICATION_STATUS=" + receipt["status"], flush=True)
    return 0 if qualification_succeeded(receipt) else 1


if __name__ == "__main__":
    sys.exit(main())
