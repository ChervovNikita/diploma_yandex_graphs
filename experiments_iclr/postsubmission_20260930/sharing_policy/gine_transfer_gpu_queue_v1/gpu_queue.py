"""Serialize every complete six-cell probe and full replay on physical GPU0."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from queue_common import HERE, PHASE, REPO, PARENT, authorization, check_parent_sources, confined, idle_guard, phase_path, plan, sha, source_hashes, write_json
from qualify_inputs import qualification_value, request_binding

CELLS = [(seed, fraction) for fraction in (0.1, 1.0) for seed in (0, 1, 2)]


def qualification_binding(qualified):
    value = plan()
    request = request_binding(HERE / "QUALIFICATION_REQUEST_authorized1gpu_run01.json")
    if (qualified.get("schema") != "gnnm-correct-allocation-GPU-input-qualification-v1"
            or qualified.get("status") != "qualified" or qualified.get("source_sha256") != source_hashes()
            or qualified.get("qualified_gpu_uuid") != value["authorized_gpu_uuid"]
            or qualified.get("request_sha256") != sha(HERE / "QUALIFICATION_REQUEST_authorized1gpu_run01.json")
            or qualified.get("qualifier_source_sha256") != sha(HERE / "qualify_inputs.py")
            or set(qualified.get("prerequisites", {})) != {"chemistry", "mapping", "synthetic", "real128", "export", "freeze", "cuda"}):
        raise ValueError("Missing exact corrected-allocation input qualification")
    for name in ("bundle", "protocol"):
        if (qualified.get(name + "_phase_relative") != value["new_authorized_" + name + "_phase_relative"]
                or qualified.get(name + "_sha256") != value["new_authorized_" + name + "_sha256"]):
            raise ValueError("Qualified path/digest differs from the frozen newly authorized PLAN")
    if qualified.get("export_provenance_phase_relative") != request["export_provenance"]:
        raise ValueError("Qualification omits the exact newly authorized export provenance")
    for role, prerequisite in qualified["prerequisites"].items():
        if (prerequisite.get("supervisor_phase_relative") != request["supervisors"][role]
                or prerequisite.get("report_phase_relative") != request["reports"].get(role)):
            raise ValueError("Qualification substitutes a different prerequisite role/path:" + role)


def compare_requalification(qualified, fresh):
    dynamic = {"idle_no_context_gate", "initial_scope_idle_gate_before_data_digest_read"}
    stored = {key: value for key, value in qualified.items() if key not in dynamic}
    current = {key: value for key, value in fresh.items() if key not in dynamic}
    if stored != current:
        raise ValueError("Stored qualification differs from fresh complete semantic requalification")


def queue(args):
    qualification_path = confined(args.qualification)
    qualified = json.loads(qualification_path.read_text())
    qualification_binding(qualified)
    initial_idle = idle_guard(qualified["qualified_gpu_uuid"])
    fresh = qualification_value(HERE / "QUALIFICATION_REQUEST_authorized1gpu_run01.json", initial_idle)
    compare_requalification(qualified, fresh)
    bundle, protocol_path = (phase_path(qualified[key]) for key in ("bundle_phase_relative", "protocol_phase_relative"))
    if sha(bundle) != qualified["bundle_sha256"] or sha(protocol_path) != qualified["protocol_sha256"]:
        raise ValueError("Qualified input bytes changed")
    protocol = json.loads(protocol_path.read_text())
    check_parent_sources(protocol)
    interpreter = Path(os.path.abspath(args.python))
    if REPO not in interpreter.parents:
        raise ValueError("Use existing authorized repository interpreter")
    output = confined(args.output)
    if output.exists():
        raise ValueError("Refuse overwriting/resuming a queue")
    output.mkdir(parents=True)
    run_root = PARENT / "authorized_gpu_probes_v1" / output.name
    if run_root.exists():
        raise ValueError("Refuse overwriting parent probe outputs")
    write_json(output / "queue_lock.json", {"cells": [{"seed": s, "label_fraction": f} for s, f in CELLS],
               "qualification_sha256": sha(qualification_path), "source_sha256": source_hashes(),
               "bundle_sha256": sha(bundle), "protocol_sha256": sha(protocol_path),
               "maximum_parallel_cells": 1, "physical_gpu": 0, "CUDA_VISIBLE_DEVICES": "0",
               "GPU_uuid": qualified["qualified_gpu_uuid"], "initial_idle_guard": initial_idle,
               "fresh_complete_semantic_requalification": True,
               "source_and_recipe_unchanged": True, "outcome_scheduling": False, "automatic_retry": False,
               "start_UTC": datetime.now(timezone.utc).isoformat()})
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    guard = PHASE / plan()["allocation_guard_phase_relative"]
    results, begin = [], time.perf_counter()
    def event(value):
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"UTC": datetime.now(timezone.utc).isoformat(), **value}, allow_nan=False) + "\n")
    try:
        for seed, fraction in CELLS:
            label = f"seed{seed}_label{int(100 * fraction)}"
            fit = run_root / label
            stage_records = {}
            for stage in ("probe", "replay"):
                idle = idle_guard(qualified["qualified_gpu_uuid"])
                event({"cell": label, "stage": stage + "_idle_guard", "guard": idle})
                supervisor = output / "supervisors" / (label + "_" + stage)
                replay_output = run_root / "replays" / (label + ".json")
                child = [str(interpreter), str(PARENT / "probe.py"), stage, "--train-bundle", str(bundle)]
                if stage == "probe":
                    child += ["--protocol", str(protocol_path), "--seed", str(seed), "--label-fraction", str(fraction),
                              "--device", "cuda:0", "--physical-gpu", "0", "--output", str(fit)]
                else:
                    child += ["--run", str(fit), "--output", str(replay_output)]
                command = [str(interpreter), str(guard), "--output", str(supervisor), "--require-idle-gpu", "--", *child]
                event({"cell": label, "stage": stage + "_started"})
                with (output / (label + "_" + stage + "_launcher.log")).open("x") as launcher:
                    completed = subprocess.run(command, env=environment, stdout=launcher, stderr=subprocess.STDOUT, check=False)
                event({"cell": label, "stage": stage + "_terminal", "exit_code": completed.returncode})
                if completed.returncode != 0:
                    raise RuntimeError("Supervised " + stage + " failed for " + label + "; exact failure retained, no retry")
                record = authorization(supervisor, qualified["qualified_gpu_uuid"], "--output", fit if stage == "probe" else replay_output)
                stage_records[stage] = {"supervisor_phase_relative": str(supervisor.relative_to(PHASE)), **record}
                idle_guard(qualified["qualified_gpu_uuid"])
            if not (fit / "selector_lock.json").exists():
                raise ValueError("Completed probe lacks required selector lock")
            replay = json.loads(replay_output.read_text())
            if replay["status"] != "passed" or replay["selector_lock_changed"] or replay["validation_or_test_loaded"]:
                raise ValueError("Full checkpoint/decision replay did not pass")
            results.append({"seed": seed, "label_fraction": fraction, "stages": stage_records,
                            "selector_lock_sha256": sha(fit / "selector_lock.json"),
                            "full_replay_sha256": sha(replay_output), "parent_run_phase_relative": str(fit.relative_to(PHASE))})
    except Exception as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                   "completed_cells": results, "wall_seconds": time.perf_counter() - begin,
                   "remaining_cells_not_qualified": True, "retry_or_outcome_truncation": False})
        raise
    write_json(output / "queue_result.json", {"status": "all_six_full_probes_and_replays_passed", "results": results,
               "wall_seconds": time.perf_counter() - begin,
               "summed_full_probe_process_seconds": sum(r["stages"]["probe"]["seconds"] for r in results),
               "summed_full_replay_process_seconds": sum(r["stages"]["replay"]["seconds"] for r in results),
               "preparation_export_gate_freeze_costs": "retain separate required corrected-allocation supervisor times",
               "maximum_parallel_cells": 1, "predictive_validation_or_test_scored": False,
               "predictive_superiority_established": False, "all_six_required_locks_and_replays": True})
    print(json.dumps({"status": "all_six_full_probes_and_replays_passed", "queue_result_sha256": sha(output / "queue_result.json")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    queue(parser.parse_args())


if __name__ == "__main__":
    main()
