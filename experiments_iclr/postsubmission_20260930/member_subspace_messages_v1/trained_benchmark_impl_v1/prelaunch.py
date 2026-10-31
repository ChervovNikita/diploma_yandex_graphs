"""Read-only completed-family/cap gate; prints one child argv, never launches."""
import sys
import os
sys.path[:] = [p for p in sys.path if p and os.path.isabs(p)
               and any(os.path.commonpath([p, root]) == root for root in (sys.prefix, sys.base_prefix))]
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--task", choices=("Reddit2", "ogbn-products"))
    p.add_argument("--seed", type=int, choices=(17, 29, 43))
    p.add_argument("--consume-completed-family", action="store_true")
    args = p.parse_args()
    begin = time.perf_counter()
    request = json.loads(args.request.read_text())
    if request.get("root_authorized") is not True:
        raise ValueError("Unapproved draft; no launch plan issued")
    raw = (HERE / "SOURCE_MANIFEST_v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != request.get("source_manifest_sha256"):
        raise ValueError("Request source seal differs")
    manifest = json.loads(raw)
    row = next(r for r in manifest["files"] if r["relative_path"] == "runner.py")
    path = HERE / "runner.py"
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Benchmark runner source changed")
    name = "_trained_benchmark_prelaunch_runner_v1"
    spec = importlib.util.spec_from_file_location(name, path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[name] = runner
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), runner.__dict__)
    runner.request_authorized(request)
    guard, state = runner.bootstrap(request)
    request_ref = guard.ref(args.request)
    if args.consume_completed_family:
        if args.task is not None or args.seed is not None:
            p.error("Complete consumer accepts no selected subset")
        report = runner.audit_completed_family(request_ref, request, guard, state)
        report["consumer_audit_wall_seconds_to_charge"] = time.perf_counter() - begin
        print(json.dumps(report, sort_keys=True))
        return
    if args.task is None or args.seed is None:
        p.error("Prelaunch requires one --task --seed")
    scientific, _, _, _, prior = runner.admission(guard.checked(request_ref), request, guard)
    selected = next(r for r in runner.children() if r["task"] == args.task and r["seed"] == args.seed)
    for key in ("outer_directory", "supervisor_directory", "output_directory"):
        if (PHASE / selected[key]).exists():
            raise ValueError("Fresh child only; no retry/overlap allowed")
    runner.Budget(request_ref, request, prior, selected, scientific)
    route = json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
    interpreter = str(Path(route["repository"]) / ".venv/bin/python")
    inner = [interpreter, str(HERE / "runner.py"), "--request", str(guard.checked(request_ref)),
             "--supervisor", str(PHASE / selected["supervisor_directory"]), "--task", args.task, "--seed", str(args.seed)]
    argv = [interpreter, str(PHASE / "protocols/bounded_run_v1.py"), "--output", str(PHASE / selected["outer_directory"]),
            "--request", str(guard.checked(request_ref)), "--cap-seconds", "600", "--grace-seconds", "5", "--", interpreter,
            str(PHASE / "protocols/run_authorized_v2.py"), "--output", str(PHASE / selected["supervisor_directory"]),
            "--require-idle-gpu", "--", *inner]
    print(json.dumps(dict(schema="gnnm-trained-inference-prelaunch-plan-v1", request=request_ref, task=args.task, seed=args.seed,
                         complete_family_consumer_passed=True, full_remaining_caps_fit=True, launch_performed=False, argv=argv,
                         prelaunch_wall_seconds_to_charge=time.perf_counter()-begin,
                         coordinator_cost_reserve_must_already_be_in_root_prior=True), sort_keys=True))


if __name__ == "__main__":
    main()
