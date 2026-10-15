"""Finite three-fit BN follow-up; parent24 must finish before any new child."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() \
            != ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise ValueError("The dedicated one-GPU route is required before project reads")
    root = Path(args.project_root).resolve()
    spec_path, output = Path(args.spec).resolve(), Path(args.output).resolve()
    if not all(p.is_relative_to(root) for p in (spec_path, output, Path(__file__).resolve())):
        raise ValueError("Own repository paths required")
    raw = spec_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.spec_sha256:
        raise ValueError("Frozen follow-up descriptor changed")
    spec = json.loads(raw)
    runtime = Path(spec["source_runtime"]).resolve()
    if not runtime.is_relative_to(root):
        raise ValueError("Own runtime required")
    sys.path.insert(0, str(runtime))
    from common import atomic_json, digest, require_allocation, utc
    from run_pilot import cost_manifest
    from run_fit import validate_job
    require_allocation()
    if spec["schema"] != "bounded-native-bn-followup-v1" or spec["new_fits"] != 3 \
            or len(spec["jobs"]) != 3 or spec["TEST_access"] is not False \
            or spec["full_campaign_GO"] is not False:
        raise ValueError("Only the exact3-fit closed-TEST follow-up is admitted")
    parent = Path(spec["parent_pilot_directory"]).resolve()
    if not parent.is_relative_to(root):
        raise ValueError("Own immutable parent controls required")
    parent_spec = Path(spec["parent_SOURCE_SPEC"]).resolve()
    if not parent_spec.is_relative_to(root) or hashlib.sha256(parent_spec.read_bytes()).hexdigest() != spec["parent_SPEC_sha256"]:
        raise ValueError("Parent pilot descriptor changed")
    plan = json.loads((parent/"PLAN.json").read_text())
    parent_value = json.loads(parent_spec.read_text())
    if plan["fit_count"] != 24 or plan["cohorts"] != spec["parent_cohorts"] \
            or plan["pilot_hash"] != digest(parent_value) \
            or digest(json.loads((parent/"PILOT.json").read_text())) != digest(parent_value):
        raise ValueError("Parent controls differ")
    for job in spec["jobs"]:
        validate_job(job)
        if job["procedure"] != "gnnm_private_bn_state_v3":
            raise ValueError("No arbitrary method dispatch")
        source = Path(job["session_transform"]["path"]).resolve()
        banks = source.with_name("banks.py")
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() \
                != job["session_transform"]["sha256"] or hashlib.sha256(banks.read_bytes()).hexdigest() \
                != job["method_settings"]["banks_sha256"]:
            raise ValueError("Prepared BN source differs from this freeze")
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "new_fits": 3, "models": 0,
                          "training_steps": 0, "parent_terminal_required_before_launch": True,
                          "TEST_accessed": False}))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/"START.json", {"UTC": utc(), "pid": os.getpid(), "SPEC_sha256": args.spec_sha256,
                "new_fits": 3, "automatic_retries": 0, "TEST_accessed": False,
                "source": {"path": str(Path(__file__)), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    start = time.monotonic()
    while not (parent/"QUEUE_RESULT.json").exists():
        if time.monotonic()-start >= spec["maximum_wait_seconds"]:
            atomic_json(output/"END.json", {"status": "parent_wait_cap", "new_fits_started": 0})
            return 1
        atomic_json(output/"PROGRESS.json", {"status": "waiting_for_parent24", "UTC": utc(), "new_fits_started": 0})
        time.sleep(30)
    terminal = json.loads((parent/"QUEUE_RESULT.json").read_text())
    if terminal["status"] != "complete":
        atomic_json(output/"END.json", {"status": "parent_failed_no_launch", "parent": terminal, "new_fits_started": 0})
        return 1
    # Exactly one new GPU-owning child; the parent's training children are all
    # terminal. The prior frozen older-worker reservation remains available.
    paths = {}; completed = []
    (output/"jobs").mkdir(); (output/"fits").mkdir()
    for index, job in enumerate(spec["jobs"]):
        token = str(index).zfill(3)+"_"+digest(job)[:16]
        path, fit = output/"jobs"/(token+".json"), output/"fits"/token
        atomic_json(path, job); paths[job["id"]] = {"job": path, "output": fit}
    for job in spec["jobs"]:
        require_allocation()
        record = paths[job["id"]]
        if record["output"].exists():
            raise ValueError("No overwrite or automatic scientific retry")
        command = [sys.executable, "-B", "-u", str(runtime/"run_fit.py"), "--project-root", str(root),
                   "--job", str(record["job"]), "--output", str(record["output"])]
        with (output/("fit_"+digest(job)[:16]+".log")).open("x") as log:
            child = subprocess.run(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT,
                                   env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                                            MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1"))
        atomic_json(output/"COSTS.json", cost_manifest(spec["jobs"], paths))
        result_path = record["output"]/"RESULT.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        completed.append({"id": job["id"], "exit_code": child.returncode, "status": result.get("status", "missing_terminal")})
        atomic_json(output/"PROGRESS.json", {"status": "running", "UTC": utc(), "outcomes": completed})
        if child.returncode or result.get("status") != "complete":
            atomic_json(output/"END.json", {"status": "failed_no_retry", "outcomes": completed, "TEST_accessed": False})
            return 1
    atomic_json(output/"END.json", {"status": "complete", "outcomes": completed, "TEST_accessed": False,
                "comparison_pending": True, "no_prediction_quality_claim": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
