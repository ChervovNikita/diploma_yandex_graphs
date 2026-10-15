"""Prepare and sequentially execute frozen paired pilots in fresh processes.

Each native body has its own fit/selector. M1 aliases body0; I4 aggregation adds
no training. At least three explicit independent trials are required. Parent
owns GPU scheduling; this command does not start a second GPU worker.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from controllers import controller
from common import atomic_json, digest, inside, receipt, require_allocation, utc


def build_jobs(spec, root=None):
    if spec.get("schema") != "frozen-native-comparison-pilot-v1" or spec.get("TEST_access") is not False:
        raise ValueError("Frozen validation-only comparison pilot schema required")
    trials = spec["trials"]
    if len(trials) < 3 or len({int(t["seed"]) for t in trials}) != len(trials):
        raise ValueError("At least three explicitly independently seeded representative trials required")
    if len({(str(t.get("split", 0)), int(t["seed"])) for t in trials}) != len(trials):
        raise ValueError("Trials must be explicit paired seed/split entries")
    policy_input = spec if spec["family"] != "link" else {
        **spec, "procedure": "native_single", "seed": int(trials[0]["seed"]),
        "split": trials[0].get("split", 0), "body": 0}
    policy, _, _ = controller(policy_input, root)
    offsets = spec.get("native_seed_offsets", spec["recipe"].get("native_seed_offsets"))
    if not isinstance(offsets, list) or len(offsets) != 4 or offsets[0] != 0 \
            or len(set(offsets)) != 4 or any(type(v) is not int or v < 0 for v in offsets):
        raise ValueError("Four declared independent offsets with body0 offset0 required")
    methods = spec.get("methods", [{"id": "gnnm_boundary_native", "base_procedure": "gnnm_boundary_native"}])
    if policy.track == "source_reference":
        if methods:
            raise ValueError("Source-reference pilot is native M1 only; set methods:[]")
    elif not any(m["id"] == "gnnm_boundary_native" for m in methods):
        raise ValueError("Keep base boundary GNNM among the new method controls")
    if len({m["id"] for m in methods}) != len(methods):
        raise ValueError("Repeated methods are not extra scientific trials")
    jobs, cohorts = [], []
    for trial in trials:
        split, seed = trial.get("split", 0), int(trial["seed"])
        cohort = spec["id"] + "/split-" + str(split) + "/seed-" + str(seed)
        base = {"schema": "native-full-epoch-fit-v1", "cohort_id": cohort,
                "family": spec["family"], "recipe": copy.deepcopy(spec["recipe"]),
                "callback_module": spec["callback_module"], "data_root": spec["data_root"],
                "policy": copy.deepcopy(spec["policy"]), "TEST_access": False,
                "split": split, "device": "cuda:0", "torch_threads": spec.get("torch_threads", 2)}
        if spec.get("callback_sha256"):
            base["callback_sha256"] = spec["callback_sha256"]
        for key in ("link_policy_module", "link_policy_sha256", "source_rng_emulation"):
            if key in spec:
                base[key] = copy.deepcopy(spec[key])
        native_ids = []
        for body, offset in enumerate(offsets[:1] if policy.track == "source_reference" else offsets):
            job = {**copy.deepcopy(base), "id": cohort + "/native-body-" + str(body),
                   "procedure": "native_single", "base_procedure": "native_single", "seed": seed + offset, "body": body}
            jobs.append(job); native_ids.append(job["id"])
        method_ids = []
        for method in methods:
            job = {**copy.deepcopy(base), "id": cohort + "/method-" + method["id"],
                   "procedure": method["id"], "base_procedure": method.get("base_procedure", "gnnm_boundary_native"),
                   "seed": seed, "body": None}
            for key in ("session_transform", "extension_factory", "session_extension", "method_settings"):
                if key in method:
                    job[key] = copy.deepcopy(method[key])
            jobs.append(job); method_ids.append(job["id"])
        cohorts.append({"id": cohort, "native_fit_ids": native_ids, "M1_fit_id": native_ids[0],
                        "method_fit_ids": method_ids, "I4_required": policy.track != "source_reference"})
    if spec["family"] == "link":
        for job in jobs:
            controller(job, root)  # validate every declared native seed/method role
    return jobs, cohorts


def cost_manifest(jobs, paths):
    rows = []
    for job in jobs:
        folder = paths[job["id"]]["output"]
        terminal = folder / "RESULT.json"
        candidates = sorted((folder / "history").glob("before_resume_*_RESULT.json")) if (folder / "history").is_dir() else []
        if terminal.is_file():
            candidates.append(terminal)
        attempts = [json.loads(path.read_text()) for path in candidates]
        actual = [r.get("optimizer_calls_this_invocation") for r in attempts]
        exact = [c for c in actual if isinstance(c, dict)]
        row = {"job_id": job["id"], "status": attempts[-1]["status"] if attempts else "not_terminal",
               "terminal_attempt_receipts": [receipt(path) for path in candidates],
               "recorded_invocations": len(attempts),
               "observed_optimizer_attempts": sum(sum(c["attempts"]) for c in exact),
               "observed_optimizer_completed": sum(sum(c["completed"]) for c in exact),
               "known_invocation_wall_seconds": sum(float(r.get("invocation_wall_seconds", 0)) for r in attempts),
               "optimizer_work_unknown_invocations": len(actual) - len(exact),
               "abrupt_termination_costs_may_be_unknown": any(bool(r.get("resumed_checkpoint")) for r in attempts),
               "completed_epochs": attempts[-1].get("completed_epochs") if attempts else None,
               "selected_epoch": attempts[-1].get("selected_epoch") if attempts else None}
        rows.append(row)
    return {"schema": "bounded-native-pilot-actual-cost-v1", "jobs": rows,
            "planned_fits": len(jobs), "complete_fits": sum(r["status"] == "complete" for r in rows),
            "failed_fits": sum(r["status"] == "failed" for r in rows),
            "native_no_selected_state_fits": sum(r["status"] == "no_selected_state" for r in rows),
            "not_terminal_fits": sum(r["status"] == "not_terminal" for r in rows),
            "observed_optimizer_attempts": sum(r["observed_optimizer_attempts"] for r in rows),
            "observed_optimizer_completed": sum(r["observed_optimizer_completed"] for r in rows),
            "known_invocation_wall_seconds_sum": sum(r["known_invocation_wall_seconds"] for r in rows),
            "wall_sum_is_worker_cost_not_elapsed_pilot_time": True,
            "no_automatic_retries": True, "TEST_accessed": False}


def run(spec, root, output, prepare_only=False, resume=False):
    root, output = Path(root).resolve(), inside(output, root)
    jobs, cohorts = build_jobs(spec, root)
    if output.exists() and not resume:
        raise ValueError("Preserve earlier pilots; choose a fresh output or explicit --resume")
    output.mkdir(parents=True, exist_ok=resume)
    spec_path = output / "PILOT.json"
    if resume:
        if not spec_path.is_file() or digest(json.loads(spec_path.read_text())) != digest(spec):
            raise ValueError("Resume cannot change the frozen question/settings/controls")
    else:
        atomic_json(spec_path, spec)
    job_root, fit_root = output / "jobs", output / "fits"
    job_root.mkdir(exist_ok=True); fit_root.mkdir(exist_ok=True)
    paths = {}
    for index, job in enumerate(jobs):
        token = str(index).zfill(3) + "_" + digest(job)[:16]
        path = job_root / (token + ".json")
        if path.exists() and digest(json.loads(path.read_text())) != digest(job):
            raise ValueError("Existing own fit descriptor changed")
        if not path.exists():
            atomic_json(path, job)
        paths[job["id"]] = {"job": path, "output": fit_root / token}
    record = {"schema": "frozen-native-comparison-pilot-run-v1", "pilot_id": spec["id"],
              "pilot_hash": digest(spec), "fit_count": len(jobs), "trials": spec["trials"],
              "cohorts": cohorts, "jobs": [{"id": j["id"], "job_hash": digest(j),
                   "job": str(paths[j["id"]]["job"]), "output": str(paths[j["id"]]["output"])} for j in jobs],
              "status": "prepared", "TEST_accessed": False, "full_campaign_GO": False,
              "automatic_retries": 0, "source_spec_receipt": receipt(spec_path)}
    atomic_json(output / "PLAN.json", record)
    if prepare_only:
        return record
    record.update(status="running", started_UTC=utc())
    atomic_json(output / "PROGRESS.json", record)
    for job in jobs:
        binding = paths[job["id"]]
        terminal = binding["output"] / "RESULT.json"
        if terminal.is_file():
            previous = json.loads(terminal.read_text())
            if previous.get("job_hash") != digest(job):
                raise ValueError("Existing output belongs to another fit")
            if previous.get("status") in ("complete", "no_selected_state"):
                continue
        command = [sys.executable, "-B", "-u", str(Path(__file__).with_name("run_fit.py")),
                   "--project-root", str(root), "--job", str(binding["job"]), "--output", str(binding["output"])]
        if binding["output"].exists():
            recovery = binding["output"] / "recovery.pt"
            if not resume or not recovery.is_file():
                raise ValueError("Failed/partial fit requires explicit resume and an authentic completed-epoch recovery")
            command += ["--resume", str(recovery)]
        with (output / ("fit_" + digest(job)[:16] + "_" + str(__import__('time').time_ns()) + ".log")).open("x") as log:
            child = subprocess.run(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if child.returncode != 0:
            record.update(status="failed", failed_fit=job["id"], failure="Child failed; no automatic scientific retry")
            atomic_json(output / "COSTS.json", cost_manifest(jobs, paths))
            atomic_json(output / "RESULT.json", record)
            return record
        atomic_json(output / "COSTS.json", cost_manifest(jobs, paths))
    for index, cohort in enumerate(cohorts):
        if not cohort["I4_required"]:
            continue
        bank_output = output / ("I4_" + str(index).zfill(3))
        if resume and (bank_output / "RESULT.json").is_file():
            continue
        members = [str(paths[jid]["output"] / "selected_VALID.pt") for jid in cohort["native_fit_ids"]]
        command = [sys.executable, "-B", str(Path(__file__).with_name("aggregate_selected.py")),
                   "--project-root", str(root), "--members", *members, "--output", str(bank_output)]
        child = subprocess.run(command, cwd=root, stdin=subprocess.DEVNULL)
        if child.returncode != 0:
            record.update(status="failed", failure="Selected-bank aggregation failed", failed_cohort=cohort["id"])
            atomic_json(output / "COSTS.json", cost_manifest(jobs, paths))
            atomic_json(output / "RESULT.json", record)
            return record
    record.update(status="complete", finished_UTC=utc(),
                  note="Validation-only exploratory results; no heldout claim or giant-campaign GO")
    atomic_json(output / "COSTS.json", cost_manifest(jobs, paths))
    atomic_json(output / "RESULT.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only:
        require_allocation()
    root = Path(args.project_root).resolve()
    spec = json.loads(inside(args.spec, root).read_text())
    result = run(spec, root, args.output, args.prepare_only, args.resume)
    print(json.dumps({k: result.get(k) for k in ("status", "pilot_id", "fit_count", "failed_fit")}))
    sys.exit(0 if result["status"] in ("prepared", "complete") else 1)


if __name__ == "__main__":
    main()
