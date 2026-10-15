"""Frozen nine-fit preparation, sequential admission, and saved-scalar reporting.

Default operation is source-only preflight. Launch requires a separate root-owned
review/admission receipt. This module never imports a model or tensor library.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

HOST = "anogena-2-0"
GPU = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_pin(pin):
    raw = Path(pin["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("Frozen input changed: " + pin["path"])
    return raw


def guard():
    if socket.gethostname() != HOST or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True, timeout=10).splitlines() != [GPU]:
        raise ValueError("Dedicated allocation guard required before project reads")


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def atomic(path, value):
    temporary = Path(path).with_name(Path(path).name + ".tmp-" + str(os.getpid()))
    write_new(temporary, value)
    temporary.replace(path)


def jobs(spec):
    result = []
    for seed in spec["seeds"]:
        for method in spec["methods"]:
            job = copy.deepcopy(spec["common_job"])
            job.update(seed=seed, procedure=method["id"], method_settings=copy.deepcopy(method["settings"]),
                       cohort_id=spec["parent_cohort_prefix"] + "/split-0/seed-" + str(seed),
                       id=spec["id"] + "/seed-" + str(seed) + "/" + method["id"])
            result.append(job)
    return result


def fit_directory(output, index, job):
    return Path(output) / "fits" / (str(index).zfill(3) + "_" + digest(job)[:16])


def validate_structure(spec):
    from session_extension import METHODS, settings_for
    if spec["schema"] != "molbace-bn-geometry-correction-study-v1" or spec["new_fits"] != 9 \
            or spec["seeds"] != [101, 202, 303] or spec["TEST_access"] is not False \
            or spec["full_campaign_GO"] is not False or spec["campaign_roster_additions"] != 0:
        raise ValueError("Only the exact closed-TEST nine-fit study is admitted")
    if [m["id"] for m in spec["methods"]] != list(METHODS) or any(
            m["settings"] != settings_for(m["id"]) for m in spec["methods"]):
        raise ValueError("Frozen method list/settings differ")
    planned = jobs(spec)
    if len(planned) != 9 or [digest(j) for j in planned] != spec["job_sha256"]:
        raise ValueError("Frozen nine-job descriptors differ")
    controls = {(r["seed"], *r["cell"]) for r in spec["reused_controls"]}
    expected = {(s, 0, g, c) for s in spec["seeds"] for g in (0, 1) for c in (0, 1)} \
        | {(s, 1, 0, 0) for s in spec["seeds"]}
    if len(spec["reused_controls"]) != 15 or controls != expected:
        raise ValueError("Exactly fifteen complete reused factorial controls required")
    return planned


def scalar_result(result):
    if result.get("status") != "complete":
        return None
    valid = result["selected_VALID"]
    if valid["role"] != "VALID" or valid["records"] != 151 \
            or valid["complete_source_VALID"] is not True or valid["TEST_accessed"] is not False:
        raise ValueError("Whole original VALID scalar required")
    values = {"AUC_percent": 100 * float(valid["metrics"]["rocauc"]), "BCE": float(valid["loss"])}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite selected result")
    return values


def preflight(spec):
    planned = validate_structure(spec)
    root = Path(spec["project_root"]).resolve()
    pins = spec["source_pins"] + spec["data_role_pins"]
    for pin in pins:
        if not Path(pin["path"]).resolve().is_relative_to(root):
            raise ValueError("Project source/data custody required")
        read_pin(pin)
    for path in spec["absent_role_overrides"]:
        if Path(path).exists():
            raise ValueError("A new role override would change the frozen CSV input route")
    parent = json.loads(read_pin(spec["parent_specs"][0]))
    bn = json.loads(read_pin(spec["parent_specs"][1]))
    immutable = ("schema", "family", "recipe", "callback_module", "callback_sha256", "data_root",
                 "policy", "TEST_access", "split", "device", "torch_threads", "base_procedure", "body")
    if any(any(j[k] != bn["jobs"][0][k] for k in immutable) for j in planned) \
            or planned[0]["policy"] != parent["policy"]:
        raise ValueError("Native data/optimizer/recipe/controller route changed from completed controls")
    for control in spec["reused_controls"]:
        result = json.loads(read_pin(control["authority"]))
        if result["job_id"] != control["job_id"] or result["seed"] != control["seed"] \
                or result["status"] != "complete" or scalar_result(result) != control["selected"]:
            raise ValueError("Completed control identity/scalar differs")
    for pin in spec["retained_evidence"]:
        read_pin(pin)
    # The original validator imports only controller/bookkeeping, not torch.
    sys.path.insert(0, spec["source_runtime"])
    from run_fit import validate_job
    for job in planned:
        validate_job(job)
    return planned


def collect(spec, output):
    rows = []
    outcome_path = Path(output) / "OUTCOMES.json"
    admitted = {r["job_id"]: r for r in json.loads(outcome_path.read_text())} if outcome_path.exists() else {}
    for control in spec["reused_controls"]:
        row = copy.deepcopy(control)
        row.update(origin="reused", status="complete", values=row.pop("selected"))
        rows.append(row)
    methods = {m["id"]: m for m in spec["methods"]}
    for index, job in enumerate(jobs(spec)):
        fit = fit_directory(output, index, job)
        row = {"job_id": job["id"], "seed": job["seed"], "cell": methods[job["procedure"]]["cell"],
               "origin": "new", "status": "not_started", "values": None,
               "expected_RESULT": str(fit / "RESULT.json")}
        row["admission_outcome"] = admitted.get(job["id"], {"status": "not_started"})
        row["status"] = row["admission_outcome"]["status"]
        result_path = fit / "RESULT.json"
        if result_path.exists():
            raw = result_path.read_bytes()
            result = json.loads(raw)
            expected_binding = copy.deepcopy(spec["expected_runtime_binding"])
            expected_binding["extension_sha256"] = job["session_transform"]["sha256"]
            if result["job_hash"] != digest(job) or result["job_id"] != job["id"] \
                    or result["seed"] != job["seed"] or result["policy"] != job["policy"] \
                    or result.get("TEST_accessed") is not False:
                raise ValueError("Saved result differs from its frozen job")
            if result["status"] == "complete" and result["runtime_binding"] != expected_binding:
                raise ValueError("Completed result used different native/runtime sources")
            row.update(status=result["status"], values=scalar_result(result),
                       authority={"path": str(result_path), "sha256": hashlib.sha256(raw).hexdigest()})
            for key in ("selected_epoch", "completed_epochs", "stop_reason", "convergence_established",
                        "invocation_wall_seconds", "optimizer_calls", "resources", "failure", "close_error"):
                row[key] = result.get(key)
        elif fit.exists():
            row["status"] = "started_without_terminal_result"
            progress = fit / "PROGRESS.json"
            row["last_saved_progress"] = json.loads(progress.read_text()) if progress.exists() else None
            row["unreceipted_cost"] = "unknown; retain log/curve/recovery and do not impute zero"
        rows.append(row)
    return rows


def contrast_definitions():
    # Cells are (normalization private=1, geometry, private correction).
    definitions = {}
    for n, label in ((0, "shared"), (1, "private")):
        definitions.update({
            label + "_geometry_at_c0": {(n, 1, 0): 1, (n, 0, 0): -1},
            label + "_geometry_at_c1": {(n, 1, 1): 1, (n, 0, 1): -1},
            label + "_correction_at_g0": {(n, 0, 1): 1, (n, 0, 0): -1},
            label + "_correction_at_g1": {(n, 1, 1): 1, (n, 1, 0): -1},
            label + "_both_vs_neither": {(n, 1, 1): 1, (n, 0, 0): -1},
            "J_" + label: {(n, 1, 1): 1, (n, 1, 0): -1, (n, 0, 1): -1, (n, 0, 0): 1}})
    for g in (0, 1):
        for c in (0, 1):
            definitions[f"BN_effect_g{g}_c{c}"] = {(1, g, c): 1, (0, g, c): -1}
    definitions["J_private_minus_J_shared"] = {
        **definitions["J_private"], **{k: -v for k, v in definitions["J_shared"].items()}}
    return definitions


def summarize(rows, seeds):
    by_cell = {(r["seed"], *r["cell"]): r for r in rows}
    contrasts = []
    for name, coefficients in contrast_definitions().items():
        for metric in ("AUC_percent", "BCE"):
            values, missing = [], []
            for seed in seeds:
                required = [by_cell.get((seed, *cell)) for cell in coefficients]
                if any(row is None or row["values"] is None for row in required):
                    values.append({"seed": seed, "value": None})
                    missing.append(seed)
                else:
                    values.append({"seed": seed, "value": sum(
                        weight * by_cell[(seed, *cell)]["values"][metric]
                        for cell, weight in coefficients.items())})
            numeric = [r["value"] for r in values if r["value"] is not None]
            contrasts.append({"name": name, "metric": metric, "per_seed": values,
                              "missing_seeds": missing, "mean": statistics.mean(numeric) if not missing else None,
                              "sample_SD": statistics.stdev(numeric) if not missing else None,
                              "positive_negative_zero": [sum(v > 0 for v in numeric), sum(v < 0 for v in numeric),
                                                         sum(v == 0 for v in numeric)],
                              "sign_count_n": len(numeric), "aggregate_requires_all_three_seeds": True})
    return contrasts


def cell_summaries(rows, seeds):
    result = []
    for n in (0, 1):
        for g in (0, 1):
            for c in (0, 1):
                cell = [n, g, c]
                selected = [r for r in rows if r["cell"] == cell]
                complete = len(selected) == len(seeds) and all(r["values"] is not None for r in selected)
                metrics = {}
                for metric in ("AUC_percent", "BCE"):
                    values = [r["values"][metric] for r in selected if r["values"] is not None]
                    metrics[metric] = {"mean": statistics.mean(values) if complete else None,
                                       "sample_SD": statistics.stdev(values) if complete else None}
                result.append({"cell": cell, "complete_seeds": sum(r["values"] is not None for r in selected),
                               "expected_seeds": seeds, "metrics": metrics})
    return result


def report(spec, output, destination):
    rows = collect(spec, output)
    value = {"schema": "molbace-bn-geometry-correction-report-v1", "all_24_cells": rows,
             "all_planned_new_outcomes": [r for r in rows if r["origin"] == "new"],
             "contrasts": summarize(rows, spec["seeds"]), "scope": spec["analysis"],
             "cell_summaries": cell_summaries(rows, spec["seeds"]),
             "retained_M1_I4_context": spec["retained_M1_I4_context"],
             "complete_new_fits": sum(r["origin"] == "new" and r["status"] == "complete" for r in rows),
             "new_model_forwards": 0, "new_optimizer_steps": 0, "TEST_accessed": False,
             "cost_rule": "Per-fit measured costs retained above; reused controls incur zero new training, not zero historical cost."}
    write_new(destination, value)
    return value


def launch(spec, spec_sha256, output, authority_path):
    authority = json.loads(Path(authority_path).read_text())
    if authority.get("schema") != "root-reviewed-nine-fit-admission-v1" \
            or authority.get("SPEC_sha256") != spec_sha256 or authority.get("approved") is not True \
            or authority.get("new_fits") != 9 or authority.get("maximum_concurrent_fits") != 1 \
            or authority.get("allocation_GPU_UUID") != GPU or authority.get("resource_admission_approved") is not True:
        raise ValueError("Separate exact-study review/GO and resource admission receipt required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "jobs").mkdir(); (output / "fits").mkdir()
    planned = jobs(spec)
    outcomes = [{"job_id": j["id"], "status": "not_admitted"} for j in planned]
    write_new(output / "START.json", {"SPEC_sha256": spec_sha256, "authority": authority,
              "pid": os.getpid(), "UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "automatic_retries": 0, "maximum_concurrent_fits": 1, "TEST_accessed": False})
    try:
        for index, job in enumerate(planned):
            guard()
            job_path = output / "jobs" / (str(index).zfill(3) + ".json")
            write_new(job_path, job)
            fit = fit_directory(output, index, job)
            outcomes[index]["status"] = "admitted"
            atomic(output / "OUTCOMES.json", outcomes)
            command = [sys.executable, "-B", "-u", str(Path(spec["source_runtime"]) / "run_fit.py"),
                       "--project-root", spec["project_root"], "--job", str(job_path), "--output", str(fit)]
            with (output / ("fit_" + str(index).zfill(3) + ".log")).open("x") as handle:
                child = subprocess.run(command, cwd=spec["project_root"], stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                                                MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1"))
            result_path = fit / "RESULT.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            outcomes[index].update(exit_code=child.returncode, status=result.get("status", "missing_terminal"))
            atomic(output / "OUTCOMES.json", outcomes)
            if child.returncode or result.get("status") != "complete":
                break
    finally:
        atomic(output / "OUTCOMES.json", outcomes)
        report(spec, output, output / "REPORT.json")
    complete = all(row["status"] == "complete" for row in outcomes)
    write_new(output / "END.json", {"status": "complete" if complete else "stopped_no_retry",
              "outcomes": outcomes, "TEST_accessed": False, "full_campaign_GO": False})
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--mode", choices=("preflight", "report", "launch"), default="preflight")
    parser.add_argument("--output")
    parser.add_argument("--report-path")
    parser.add_argument("--launch-authority")
    args = parser.parse_args()
    guard()
    spec = json.loads(read_pin({"path": args.spec, "sha256": args.spec_sha256}))
    root = Path(spec["project_root"]).resolve()
    for value in (args.spec, args.output, args.report_path, args.launch_authority, __file__):
        if value and not Path(value).resolve().is_relative_to(root):
            raise ValueError("All study paths must remain inside the project")
    preflight(spec)
    if args.mode == "preflight":
        print(json.dumps({"status": "source_preflight_passed", "new_fits_planned": 9, "reused_cells": 15,
                          "models": 0, "optimizer_steps": 0, "TEST_accessed": False, "launched": False}))
        return 0
    if not args.output:
        parser.error("--output is required for report or launch")
    if args.mode == "report":
        if not args.report_path:
            parser.error("--report-path must name a new report file")
        report(spec, args.output, args.report_path)
        return 0
    if not args.launch_authority:
        parser.error("--launch-authority is required; this preparation grants no launch")
    return launch(spec, args.spec_sha256, args.output, args.launch_authority)


if __name__ == "__main__":
    raise SystemExit(main())
