"""Disabled exact two-GPU once-only ordinary pair supervisor; no scoring.

One own child and one own_pool child, each on its distinct qualified UUID.
Retain both complete2300 endpoints or fail; never launch the serial V4 twice.
"""
import time
STARTED = time.monotonic()
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import traceback
from types import SimpleNamespace

SOURCE_RELEASED = True
TARGET = "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930"
OBJECTIVES = ("own", "own_pool")
UPDATES, PREFIX = 2300, 16
PROCESS = ("amazon_ordinary_shared_bank_two_gpu_scheduling_preparation_20261006_v1/process_supervisor.py", "6106ee06d4764293cf58e77870f96a157bcc23f442936258a1d982c63dd740cf")
CHILD_RELEASED_SHA256 = "369f03496fd3f597da29df3dd1271747ebb8a889f3dcad8ed4e026d06cc676be"
CHILD_RELEASED_BYTES = 40364
COMMON_DESCRIPTORS = {
    "common_checkpoint": {"path": "amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/initial.pt", "bytes": 109671738, "sha256": "2e0e9b44767abc12b3dc896986ea2d4faeaa667c8682b95929f927d10718154f"},
    "common_origin_run": {"path": "amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/RUN.json", "bytes": 6402, "sha256": "2e18a77081437e47c0c4a0d59cb07b2a15500f660066cbed1c980f613c028ca5"}}


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): digest.update(block)
    return digest.hexdigest()


def load_process(root):
    path = root / PROCESS[0]; require(sha(path) == PROCESS[1], "Reviewed process helper source differs")
    name = "_ordinary_two_gpu_process_supervisor"
    require(name not in sys.modules, "Fresh pair supervisor required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    require(module.SOURCE_RELEASED is False, "Process engineering helper stays disabled")
    return module


def qualification(root, ordinary, process, registry, job):
    qrow, srow, wrow = (job[key] for key in ("normal_runtime_context_qualification", "qualification_scope", "qualification_supervisor_receipt"))
    result, scope, supervised = (json.loads(ordinary.bound(root, row).read_text()) for row in (qrow, srow, wrow))
    require(result["status"] == "PASS_NORMAL_ORDINARY_FIRST_ORDER_STREAMED_RESOURCE_ONLY"
        and result["worker_sha256"] == process.QUALIFIER[1] and result["ordinary_source_pin"] == list(process.ORDINARY)
        and result["root_scope_sha256"] == srow["sha256"] and result["implementation"] == "streamed_exact"
        and all(result[key] is True for key in ("ordinary_first_order_supported", "both_objectives_checked",
            "same_checkpoint_context_and_RNG_checked", "coupled_pool_gradient_and_Adam_update_parity_checked", "stochastic_member_RNG_replay_checked"))
        and scope["CUDA_VISIBLE_DEVICES"] == job["CUDA_VISIBLE_DEVICES"] and scope["qualifier_worker_sha256"] == process.QUALIFIER[1]
        and scope["qualifier_supervisor_worker_sha256"] == PROCESS[1]
        and all(scope[key] == row for key, row in COMMON_DESCRIPTORS.items())
        and result["input_identity"] == ordinary.INPUT_IDENTITY and not result["restoration_errors"]
        and result["model_fits"] == 0 and result["persistent_updates"] == 0,
        "Exact successful normal per-UUID qualification required before either fit")
    require(supervised["status"] == "PASS_OWNED_QUALIFIER_WHOLE_CHILD_CLOSED"
        and supervised["supervisor_worker_sha256"] == PROCESS[1] and supervised["scope_sha256"] == srow["sha256"]
        and supervised["qualifier_result"] == qrow and supervised["child_exit_code"] == 0 and supervised["wait4_closed"] is True
        and len(supervised["children"]) == 1 and supervised["children"][0]["CUDA_VISIBLE_DEVICES"] == job["CUDA_VISIBLE_DEVICES"]
        and supervised["children"][0]["wait4_closed"] is True and not supervised["cleanup_errors"]
        and job["qualification_supervisor_exit_code"] == 0, "Exact immutable qualifier external wait4 closure required")
    process.resource_closure(supervised["children"][0], result, scope["resource_limits"])
    native = next(check["result"] for check in result["checks"] if check["name"] == "native_common400_two_discarded_update_replays_per_objective")
    durations = [r["streamed_update_elapsed_seconds"] for kind in OBJECTIVES for r in native["objectives"][kind]["discarded_replay_resources"]]
    require(len(durations) == 4 and all(type(d) in (int, float) and math.isfinite(d) and d > 0 for d in durations), "Actual measured streamed update durations required")
    maximum, basis = max(durations), job["fit_cap_basis"]
    require(basis["qualification_RESULT_sha256"] == qrow["sha256"] and basis["native_streamed_max_update_seconds"] == maximum
        and basis["projected2300_update_seconds"] == maximum * UPDATES
        and type(basis["root_frozen_elapsed_margin_seconds"]) in (int, float)
        and math.isfinite(basis["root_frozen_elapsed_margin_seconds"]) and basis["root_frozen_elapsed_margin_seconds"] > 0
        and job["resource_limits"]["max_elapsed_seconds"] >= maximum * UPDATES + basis["root_frozen_elapsed_margin_seconds"],
        "Fit time caps must use actual native qualified update evidence with root-frozen margin")
    python = result["runtime"]["python_resolved"]
    require(Path(python).is_absolute() and str(Path(python).resolve()) == python and Path(python).is_file(), "Exact qualified normal interpreter required")
    return result, python


def endpoint(root, process, output, row, kind, update):
    expected = kind + "_" + str(update) + ".pt"
    require(row["path"] == expected, "Only fixed prefix/last endpoint names permitted")
    descriptor = process.immutable_descriptor(root, output / expected)
    require({k: descriptor[k] for k in ("bytes", "sha256")} == {k: row[k] for k in ("bytes", "sha256")}, "Frozen checkpoint descriptor differs")
    return descriptor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true"); parser.add_argument("--source-root")
    parser.add_argument("--registry"); parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_TWO_GPU_ONCE_ONLY_ORDINARY_PAIR", "SOURCE_RELEASED": SOURCE_RELEASED, "numeric_imports": False})); return
    require(SOURCE_RELEASED is True, "Disabled pair preparation has no fitting authority")
    require(args.source_root == TARGET and socket.gethostname() == "peptide", "Exact literal77 pair target required")
    require(args.registry and args.output, "Exact immutable paired registry and new assigned output required")
    root = Path(args.source_root).resolve(); process = load_process(root); ordinary = process.load_ordinary(root)
    root, output, registry_path = ordinary.deliberate_paths(SimpleNamespace(source_root=args.source_root, output=args.output, admission=args.registry))
    require(registry_path.stat().st_mode & 0o222 == 0, "Immutable root once-only paired registry required")
    registry = json.loads(registry_path.read_text())
    require(registry["schema"] == "root_ordinary_two_gpu_once_only_registry_v1" and registry["root_fit_authorized"] is True
        and isinstance(registry["pair_id"], str) and registry["pair_id"]
        and registry["paired_supervisor_required"] is True and registry["automatic_retry"] is False
        and registry["source_review"]["approved"] is True and registry["source_review"]["process_helpers_approved"] is True
        and registry["supervisor_worker_sha256"] == sha(__file__) and registry["worker_sha256"] == CHILD_RELEASED_SHA256
        and registry["protocol_sha256"] == sha(Path(__file__).parent / "PROTOCOL.json")
        and registry["objectives"] == list(OBJECTIVES) and registry["implementation"] == "streamed_exact"
        and registry["continuation_updates"] == UPDATES and registry["diagnostic_prefix"] == PREFIX
        and registry["external_watchdog_required"] is True and registry["A_scoring"] is False and registry["VALID_TEST_access"] is False
        and all(registry[key] == row for key, row in COMMON_DESCRIPTORS.items()), "Exact released, reviewed fixed pair root admission required")
    worker_row = registry["released_worker"]
    require(worker_row["sha256"] == CHILD_RELEASED_SHA256 and worker_row["bytes"] == CHILD_RELEASED_BYTES, "Exact flag-only one-objective worker required")
    worker = ordinary.bound(root, worker_row)
    common_path, origin_path = (ordinary.bound(root, registry[key]) for key in ("common_checkpoint", "common_origin_run"))
    require(common_path.parent == origin_path.parent, "Same immutable pilot common origin required")
    jobs = registry["jobs"]
    require(len(jobs) == 2 and [j["job_id"] for j in jobs] == list(OBJECTIVES)
        and [j["objective"] for j in jobs] == list(OBJECTIVES)
        and len({j["CUDA_VISIBLE_DEVICES"] for j in jobs}) == 2
        and all(isinstance(j["CUDA_VISIBLE_DEVICES"], str) and j["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
            and "," not in j["CUDA_VISIBLE_DEVICES"] for j in jobs), "Exactly own/own_pool on distinct assigned physical GPU UUIDs required")
    require(output == process.new_path(root, registry["supervisor_output_relative"]), "Assigned paired supervisor output differs")
    outputs, pythons = {}, {}
    grace, reap_timeout = registry["termination_grace_seconds"], registry["termination_reap_timeout_seconds"]
    for job in jobs:
        process.validate_caps(job["resource_limits"], job["external_watchdog_seconds"], grace, reap_timeout)
        require(type(job["minimum_initial_cuda_free_bytes"]) is int and job["minimum_initial_cuda_free_bytes"] > 0,
            "Frozen qualified GPU free floor required")
        outputs[job["job_id"]] = process.new_path(root, job["output_relative"])
        _, pythons[job["job_id"]] = qualification(root, ordinary, process, registry, job)
    require(len(set(outputs.values()) | {output}) == 3, "Three distinct fresh assigned outputs required")
    owner = ordinary.CreatedOutput(output, root); token = owner.creator_token
    receipt = {"status": "RUNNING_TWO_GPU_ORDINARY_PAIR", "supervisor_worker_sha256": sha(__file__),
        "child_worker_sha256": CHILD_RELEASED_SHA256, "process_helper_sha256": PROCESS[1], "pair_id": registry["pair_id"],
        "paired_registry_sha256": sha(registry_path), "protocol_sha256": registry["protocol_sha256"],
        "common_checkpoint": registry["common_checkpoint"], "common_origin_run": registry["common_origin_run"],
        "A_scoring": False, "VALID_TEST_access": False, "comparison_complete": False,
        "numeric_imports_in_supervisor": False, "launch_attempts": dict.fromkeys(OBJECTIVES, 0), "children": [], "outcomes": {}}
    def save():
        receipt["supervisor_elapsed_seconds"] = time.monotonic() - STARTED
        ordinary.atomic(owner.verify(token) / "RESULT.json", receipt)
    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, frame): raise InterruptedError("Pair supervisor interrupted: " + str(signum))
    code = 1
    try:
        for signum in old_handlers: signal.signal(signum, interrupted)
        for job in jobs:
            kind = job["job_id"]
            require(receipt["launch_attempts"][kind] == 0, "No duplicate objective launch")
            receipt["launch_attempts"][kind] = 1; save()
            process.launch(root, owner, token, kind, pythons[kind], worker,
                ["--execute-authorized", "--source-root", TARGET, "--public-b-dir", str(root / ordinary.PUBLIC_B_RELATIVE),
                 "--admission", str(registry_path), "--objective", kind, "--output", str(outputs[kind])],
                dict(os.environ, CUDA_VISIBLE_DEVICES=job["CUDA_VISIBLE_DEVICES"]), receipt["children"])
            save()
        while not all(c["wait4_closed"] for c in receipt["children"]):
            for child, job in zip(receipt["children"], jobs):
                process.watch(child, job["external_watchdog_seconds"], grace, reap_timeout)
                if child["wait4_closed"]:
                    require(child["child_exit_code"] == 0 and not child["external_timeout"], "Assigned objective failed; keep partial work and fail pair")
            save(); time.sleep(0.25)
        for child, job in zip(receipt["children"], jobs):
            kind, child_output = job["job_id"], outputs[job["job_id"]]
            result_row = process.immutable_descriptor(root, child_output / "RESULT.json")
            result = json.loads(ordinary.bound(root, result_row).read_text())
            process.resource_closure(child, result, job["resource_limits"])
            require(result["status"] == "ONE_ASSIGNED_ORDINARY_REFERENCE_COMPLETE_A_CLOSED"
                and result["assigned_objective_complete"] is True and result["comparison_complete"] is False
                and result["pair_id"] == registry["pair_id"] and result["paired_registry_sha256"] == sha(registry_path)
                and result["worker_sha256"] == CHILD_RELEASED_SHA256 and result["assigned_objective"] == kind
                and result["CUDA_VISIBLE_DEVICES"] == job["CUDA_VISIBLE_DEVICES"] and not result["restoration_errors"]
                and result["A_scoring"] is False and result["VALID_TEST_access"] is False
                and set(result["endpoints"]) == {kind} and set(result["prefixes"]) == {kind}, "Exact successful assigned child result required")
            counts = result["operation_accounting"]
            expected = {"callback_attempts": 8 * UPDATES, "backward_attempts": 4 * UPDATES,
                "small_logit_reverse_attempts": UPDATES, "optimizer_attempts": UPDATES, "completed_updates": UPDATES}
            other = next(k for k in OBJECTIVES if k != kind)
            require(all(counts[k][kind] == value and counts[k][other] == 0 for k, value in expected.items()), "Complete assigned work and zero unassigned work required")
            run_row = process.immutable_descriptor(root, child_output / "RUN.json")
            run = json.loads(ordinary.bound(root, run_row).read_text())
            require(run["common_checkpoint"] == registry["common_checkpoint"] and run["common_origin_run"] == registry["common_origin_run"]
                and run["protocol_sha256"] == registry["protocol_sha256"] and run["implementation"] == "streamed_exact"
                and run["input_identity"] == ordinary.INPUT_IDENTITY and run["paired_registry_sha256"] == sha(registry_path)
                and run["pair_id"] == registry["pair_id"] and run["assigned_objective"] == kind,
                "Both outcomes must share exact protocol/common/context registry")
            receipt["outcomes"][kind] = {"RESULT": result_row, "RUN": run_row,
                "prefix": endpoint(root, process, child_output, result["prefixes"][kind], kind, PREFIX),
                "endpoint": endpoint(root, process, child_output, result["endpoints"][kind], kind, UPDATES),
                "operation_accounting": counts, "CUDA_VISIBLE_DEVICES": job["CUDA_VISIBLE_DEVICES"]}
            save()
        require(set(receipt["outcomes"]) == set(OBJECTIVES) and receipt["launch_attempts"] == dict.fromkeys(OBJECTIVES, 1), "Both complete once-only outcomes retained")
        require(sha(registry_path) == receipt["paired_registry_sha256"] and sha(worker) == CHILD_RELEASED_SHA256
            and sha(root / PROCESS[0]) == PROCESS[1] and all(sha(root / row["path"]) == row["sha256"] for row in COMMON_DESCRIPTORS.values()),
            "Registry/source/common inputs changed")
        receipt.update(status="TWO_GPU_ORDINARY_PAIR_COMPLETE_A_CLOSED", comparison_complete=True,
            total_training_callbacks=16 * UPDATES, total_native_backward_APIs=8 * UPDATES,
            total_small_logit_reverses=2 * UPDATES, total_Adam_steps=2 * UPDATES,
            common_acquisition_rerun_callbacks=0, serving_callbacks=0, heldout_scoring=0)
        code = 0
    except BaseException as error:
        receipt.update(status="FAIL_TWO_GPU_ORDINARY_PAIR", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        for signum in old_handlers: signal.signal(signum, signal.SIG_IGN)
        errors = process.cleanup(receipt["children"], grace, reap_timeout); receipt["cleanup_errors"] = errors
        if errors: receipt.update(status="FAIL_TWO_GPU_ORDINARY_PAIR", comparison_complete=False); code = 1
        for signum, handler in old_handlers.items(): signal.signal(signum, handler)
        if code: receipt["comparison_complete"] = False
        try: save(); owner.freeze(token)
        except BaseException as error:
            code = 1; receipt.update(status="FAIL_TWO_GPU_ORDINARY_PAIR", comparison_complete=False, output_finalization_error=str(error))
            try: save(); owner.freeze(token)
            except BaseException as recovery_error: receipt["output_failure_receipt_error"] = str(recovery_error)
    print(json.dumps({"status": receipt["status"], "comparison_complete": receipt["comparison_complete"], "error": receipt.get("error")}))
    raise SystemExit(code)


if __name__ == "__main__": main()
