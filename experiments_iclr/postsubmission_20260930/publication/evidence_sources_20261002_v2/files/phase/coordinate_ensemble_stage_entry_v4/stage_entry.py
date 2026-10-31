#!/usr/bin/env python3
"""Exact-request bridge for acquire/qualify/validation-fit stages; no test mode.

Source preparation only. Runtime uses stdlib and invokes the independently frozen
runner as an argv list. Existing whole-process supervision owns the final cap.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ENTRY = Path(__file__).resolve()
PHASE = ENTRY.parents[1]
EXECUTION_ROOT = PHASE / "coordinate_ensemble_execution_root_v1"
REPO = PHASE.parents[1] if PHASE.parent.name == "experiments_iclr" else PHASE.parent
INTERPRETER = REPO / ".venv/bin/python"
DESTINATION = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
REPOSITORY = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
GPU_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
GPU_IDLE_SETTLING_SECONDS = 15.0
GPU_METADATA_INTERVAL_SECONDS = 0.5
ROUTE = PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json"
PROTECTED = [ENTRY, PHASE / "protocols/bounded_run_v1.py",
             PHASE / "protocols/run_authorized_v2.py", PHASE / "protocols/run_logged.py",
             PHASE / "protocols/repo_env.sh", ROUTE]
CACHE_NAMES = ["TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR",
               "CONDA_PKGS_DIRS", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
               "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
               "MPLCONFIGDIR", "HF_HOME", "HF_DATASETS_CACHE", "NUMBA_CACHE_DIR", "WANDB_DIR"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def confined(value, must_file=False):
    require(isinstance(value, str) and Path(value).is_absolute(), "Require absolute phase paths")
    raw = Path(value)
    require(".." not in raw.parts, "Parent traversal is forbidden")
    path = Path(os.path.abspath(value))
    require(path != PHASE and path.is_relative_to(PHASE), "Path must be strictly inside phase")
    part = PHASE
    for name in path.relative_to(PHASE).parts:
        part = part / name
        require(not part.is_symlink(), "Symlink phase paths are forbidden")
    resolved = path.resolve()
    require(resolved == path, "Phase path resolves to another location")
    if must_file:
        require(path.is_file(), "Required phase file is absent")
    return path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def verify_binding(binding):
    require(set(binding) == {"path", "sha256"}, "Binding requires exactly path/SHA256")
    path = confined(binding["path"], must_file=True)
    expected = binding["sha256"]
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid SHA256")
    require(sha(path) == expected, "Hash mismatch: " + path.name)
    return path


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def absolute_values_confined(value):
    if isinstance(value, dict):
        for child in value.values():
            absolute_values_confined(child)
    elif isinstance(value, list):
        for child in value:
            absolute_values_confined(child)
    elif isinstance(value, str) and value.startswith("/"):
        confined(value)


def phase_layout(protocol):
    roots = {}
    for mode in ("acquire", "qualify", "fit"):
        definition = protocol["phases"][mode]
        root = confined(definition["root"])
        require(root != EXECUTION_ROOT and root.is_relative_to(EXECUTION_ROOT),
                "Runner phase roots must be under the new coordinate execution namespace")
        names = ["cache_root", "output_root"]
        if mode == "acquire":
            names += ["pyg_root", "public_root", "sealed_root"]
        values = [confined(definition[name]) for name in names]
        require(all(path != root and path.is_relative_to(root) for path in values), "Phase child escapes root")
        for i, left in enumerate(values):
            for right in values[i + 1:]:
                require(not (left == right or left.is_relative_to(right) or right.is_relative_to(left)),
                        "Phase data/output/cache roots overlap")
        roots[mode] = root
    for i, left in enumerate(roots.values()):
        for right in list(roots.values())[i + 1:]:
            require(not (left == right or left.is_relative_to(right) or right.is_relative_to(left)),
                    "Acquire/qualify/fit roots must be disjoint")
    lock = confined(protocol["runtime"]["gpu_lock_file"])
    require(lock.is_relative_to(roots["acquire"]), "GPU lock must be in common acquisition root")
    return roots


def gpu_guard(receipt):
    begun = time.monotonic()
    deadline = begun + GPU_IDLE_SETTLING_SECONDS
    receipt.update({"maximum_settling_seconds": GPU_IDLE_SETTLING_SECONDS,
                    "metadata_interval_seconds": GPU_METADATA_INTERVAL_SECONDS,
                    "query_fields": ["index", "uuid", "name", "memory.used", "memory.total", "utilization.gpu"],
                    "samples": [], "status": "checking"})
    try:
        while True:
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Authorized GPU did not become idle within 15 seconds")
            tick = time.monotonic()
            query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True, capture_output=True,
                                   check=True, timeout=min(10.0, remaining))
            rows = [line for line in query.stdout.splitlines() if line.strip()]
            sample = {"query_started_seconds": tick - begun,
                      "query_finished_seconds": time.monotonic() - begun,
                      "gpu_rows": [[part.strip() for part in line.split(",")] for line in rows]}
            receipt["samples"].append(sample)
            require(len(rows) == 1, "Require exactly one physical visible GPU")
            row = sample["gpu_rows"][0]
            require(len(row) == 6 and row[0] == "0" and row[1] == GPU_UUID and
                    row[2] == "NVIDIA A100-SXM4-80GB" and int(row[4]) == 81920,
                    "Authorized exact GPU identity differs")
            memory, utilization = int(row[3]), int(row[5])
            require(0 <= memory <= 100, "Authorized GPU memory is occupied or invalid before child")
            require(0 <= utilization <= 100, "Authorized GPU utilization metadata is invalid")
            require(time.monotonic() <= deadline, "GPU metadata query exhausted the 15-second settling limit")
            if utilization == 0:
                receipt["status"] = "idle"
                return {"index": row[0], "uuid": row[1], "name": row[2], "memory_used_MiB": memory,
                        "memory_total_MiB": int(row[4]), "utilization_percent": utilization}
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Authorized GPU did not become idle within 15 seconds")
            until_next_sample = max(0.0, tick + GPU_METADATA_INTERVAL_SECONDS - time.monotonic())
            time.sleep(min(until_next_sample, remaining))
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - begun


def plan(request, request_path, supervisor):
    require(request["schema"] == "coordinate-ensemble-stage-entry-request-v4" and
            request["root_admitted"] is True, "Require explicit root stage admission")
    require(request["mode"] in {"acquire", "qualify", "fit"}, "Test/preflight modes are not admitted")
    require(request["phase_root"] == str(PHASE) and request["gpu_uuid"] == GPU_UUID,
            "Request phase/GPU differs from pinned authorized identity")
    mode = request["mode"]
    require(str(REPO) == REPOSITORY and Path.cwd().resolve() == REPO,
            "Require exact authorized remote repository cwd")
    require(Path(os.path.abspath(sys.executable)) == INTERPRETER, "Require exact repository venv entry point")
    require(os.environ.get("GNNM_PHASE_ROOT") == str(PHASE) and
            os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" and
            os.environ.get("GNNM_SSH_DESTINATION") == DESTINATION, "Require existing corrected-route wrapper")
    for name in CACHE_NAMES:
        confined(os.environ.get(name))
    exposure = request["cuda_visible_devices"]
    require(exposure in {"0", GPU_UUID} and os.environ.get("CUDA_VISIBLE_DEVICES") == exposure,
            "Require exact one-device CUDA exposure in parent")
    protected = {}
    for binding in request["protected_files"]:
        path = verify_binding(binding)
        require(path not in protected, "Duplicate protected source")
        protected[path] = binding["sha256"]
    require(all(path in protected for path in PROTECTED), "Bind bridge, all supervisors, wrapper and route sources")
    route = read_json(ROUTE)
    require(route["ssh_destination"] == DESTINATION and route["repository"] == REPOSITORY and
            route["visible_gpu_count"] == 1 and route["repository_pwd_matches"] is True and
            route["forbidden_destination_reconnect_allowed"] is False, "Corrected route binding differs")
    allocation_path = verify_binding(request["allocation_evidence"])
    allocation = read_json(allocation_path)
    require(allocation["gpu_uuid"] == GPU_UUID and allocation["visible_gpu_count"] == 1 and
            allocation["ssh_destination"] == DESTINATION and allocation["child_exit_code"] == 0,
            "Retained successful allocation supervision does not bind the authorized GPU")
    protocol_path = verify_binding(request["protocol"])
    sources_path = verify_binding(request["sources"])
    runner_path = verify_binding(request["runner"])
    expected_runner = PHASE / "coordinate_ensemble_runner_v3/runner.py"
    require(runner_path == expected_runner, "Only the independently frozen stage-specific runner is admitted")
    protocol, sources = read_json(protocol_path), read_json(sources_path)
    require(protocol["schema_version"] == sources["schema_version"] == 1 and
            protocol["frozen"] is True and protocol["study_kind"] == "new_study" and
            protocol["source_manifest_sha256"] == request["sources"]["sha256"], "Frozen protocol/source binding differs")
    require(protocol["test_policy"] == "sealed_independent_confirmation_only" and
            protocol["preprocessing"] == {"features": "none", "edges": "as_pyg"}, "Study target/preprocessing contract differs")
    if mode == "acquire":
        require(protocol["runtime"]["device"] == "cpu", "Acquisition requires a CPU-only protocol")
    absolute_values_confined(protocol)
    roots = phase_layout(protocol)
    source_bindings = []
    seen = set()
    for entry in sources["files"]:
        binding = {"path": entry["path"], "sha256": entry["sha256"]}
        path = verify_binding(binding)
        require(path not in seen, "Duplicate manifest source path")
        seen.add(path)
        source_bindings.append(binding)
    require(runner_path in seen and confined(sources["model_entry"], True) in seen,
            "Manifest must bind exact runner and model entry")
    require(any(e["role"] == "author_model_dependency" for e in sources["files"]) and
            any(e["role"] == "author_dataset_loader" for e in sources["files"]), "Required reviewed dependencies absent")
    environment_path = confined(str(supervisor / "environment.json"), True)
    command_path = confined(str(supervisor / "command.json"), True)
    environment, command = read_json(environment_path), read_json(command_path)
    own_argv = [str(INTERPRETER), str(ENTRY), "--request", str(request_path), "--supervisor", str(supervisor)]
    require(command["argv"] == own_argv and command["shell"] is False and command["cwd"] == str(REPO),
            "Inner supervisor did not record this exact entry argv")
    require(environment["repository"] == str(REPO) and environment["phase"] == str(PHASE) and
            environment["script_sha256"] == protected[ENTRY] and environment["python_bytecode_disabled"] is True and
            environment["supervisor_sha256"] == protected[PHASE / "protocols/run_logged.py"] and
            environment["wrapper_sha256"] == protected[PHASE / "protocols/repo_env.sh"] and
            environment["CUDA_VISIBLE_DEVICES"] == exposure, "Inner supervisor environment/source differs")
    for value in environment["cache_and_temp"].values():
        if value == str(PHASE):
            continue
        confined(value)
    bound_path = confined(os.environ.get("GNNM_BOUND_START_JSON"), True)
    bound = read_json(bound_path)
    require(bound["schema"] == "gnnm-whole-process-bound-start-v1" and bound["phase"] == str(PHASE) and
            bound["cwd"] == str(REPO) and bound["ssh_destination"] == DESTINATION and
            bound["inner_supervisor_directory"] == str(supervisor) and bound["child_argv"] == own_argv and
            bound["root_request"]["path"] == str(request_path.relative_to(PHASE)) and
            bound["root_request"]["sha256"] == sha(request_path) and bound["root_request_child_flag"] == "--request",
            "Whole-process supervisor did not bind this request and child")
    for key, path in [("source_sha256", PHASE / "protocols/bounded_run_v1.py"),
                      ("authorization_source_sha256", PHASE / "protocols/run_authorized_v2.py"),
                      ("logging_source_sha256", PHASE / "protocols/run_logged.py"),
                      ("wrapper_sha256", PHASE / "protocols/repo_env.sh"), ("research_script_sha256", ENTRY)]:
        require(bound[key] == protected[path], "Whole-process source hash differs")
    require(bound["whole_cap_seconds"] == request["whole_cap_seconds"] and
            0 < bound["normal_child_budget_seconds"] < bound["whole_cap_seconds"] <= 28800,
            "Whole-process cap differs from request")
    timeout = request.get("per_operation_timeout_seconds", 1200)
    require(type(timeout) in {int, float} and math.isfinite(timeout) and 0 < timeout <= 1800,
            "Each exact operation must have a finite timeout <=1800 seconds")
    output = confined(request["output"])
    require(output != EXECUTION_ROOT and output.is_relative_to(EXECUTION_ROOT),
            "Stage receipts must be inside the new coordinate execution namespace")
    require(not output.exists() and not output.is_relative_to(supervisor) and not supervisor.is_relative_to(output),
            "Bridge output must be new and separate from supervisor")
    require(not (output.is_relative_to(bound_path.parent) or bound_path.parent.is_relative_to(output)),
            "Bridge output must be separate from outer evidence")
    for root in roots.values():
        require(not (root == output or root.is_relative_to(output) or output.is_relative_to(root)),
                "Bridge receipt directory must be separate from runner data/cache/output roots")
    operations = request["operations"]
    require(isinstance(operations, list) and operations, "An exact nonempty operation list is required")
    require(len(operations) * (timeout + GPU_IDLE_SETTLING_SECONDS) + 3 <= bound["normal_child_budget_seconds"],
            "Operation and GPU settling budgets exceed whole bound")
    planned, identities = [], set()
    for operation in operations:
        dataset = operation["dataset"]
        require(dataset in {"CoauthorCS", "AmazonPhoto"} and dataset in protocol["datasets"], "Dataset not frozen")
        argv = [str(INTERPRETER), str(runner_path), mode,
                "--protocol", str(protocol_path), "--protocol-sha256", request["protocol"]["sha256"],
                "--sources", str(sources_path), "--sources-sha256", request["sources"]["sha256"], "--dataset", dataset]
        bindings = []
        if mode == "acquire":
            require(set(operation) == {"dataset", "argv"}, "Acquire cannot carry a fit/cell/target argument")
            suffix = dataset
            for key in ("pyg_root", "public_root", "sealed_root"):
                require(not (confined(protocol["phases"][mode][key]) / dataset).exists(), "Acquisition dataset target already exists")
        else:
            require(set(operation) == {"dataset", "argv", "cell", "data_manifest"}, "Only exact qualification cell fields allowed")
            cell = operation["cell"]
            require(set(cell) == {"split", "seed", "arm", "recipe"} and type(cell["seed"]) is int,
                    "Qualification needs one exact typed cell")
            require(all(isinstance(cell[name], str) and re.fullmatch(r"[A-Za-z0-9_-]+", cell[name])
                        for name in ("split", "arm", "recipe")), "Unsafe qualification cell identifier")
            matches = [c for c in protocol["cells"] if c["dataset"] == dataset and
                       all(c[name] == cell[name] for name in cell)]
            require(len(matches) == 1, "Qualification cell must occur exactly once in frozen protocol")
            recipe = protocol["recipes"][cell["recipe"]]
            if mode == "qualify":
                require(protocol["qualification"]["complete_updates"] == 100 and recipe["eval_every"] == 10,
                        "Qualification must be 100 complete updates and ten train-only evaluations")
            else:
                require(recipe["min_epochs"] == 1000 and recipe["max_epochs"] == 2000 and
                        recipe["patience"] == 200 and recipe["eval_every"] == 10 and
                        recipe["checkpoint_metric"] == "validation_nll" and recipe["checkpoint_ties"] == "earliest",
                        "Validation fits require the fixed competent horizon and checkpoint rule")
                require(cell["split"] in {"calibration", "core0", "core1", "core2"} and
                        cell["seed"] in {17, 29, 43}, "Validation fit split/seed not prospective")
                if cell["split"] == "calibration":
                    require(cell["arm"] == "single" and cell["seed"] == 17 and
                            recipe["optimizer"]["lr"] in {3e-5, 3e-4},
                            "Calibration admits only the matched single and two frozen learning rates")
            require(protocol["runtime"]["device"] == "cuda:0", "Qualification must use the single exposed CUDA device")
            data_path = verify_binding(operation["data_manifest"])
            require(data_path == confined(str(confined(protocol["phases"]["acquire"]["public_root"]) /
                                             dataset / "data_manifest.json")),
                    "Qualification may consume only the canonical public acquisition manifest")
            bindings.append(operation["data_manifest"])
            require(protocol["data_bindings"][dataset]["data_manifest_sha256"] == operation["data_manifest"]["sha256"],
                    "Qualification acquisition-manifest binding differs")
            argv += ["--split", cell["split"], "--seed", str(cell["seed"]), "--arm", cell["arm"], "--recipe", cell["recipe"],
                     "--data-manifest", str(data_path), "--data-manifest-sha256", operation["data_manifest"]["sha256"]]
            suffix = f"{dataset}__{cell['split']}__seed{cell['seed']}__{cell['arm']}__{cell['recipe']}"
        require(isinstance(operation["argv"], list) and operation["argv"] == argv,
                "JSON argv must exactly equal reconstructed stage-only runner argv")
        require(suffix not in identities and Path(suffix).name == suffix and suffix not in {".", ".."}, "Duplicate/unsafe exact operation")
        identities.add(suffix)
        runner_output = confined(str(confined(protocol["phases"][mode]["output_root"]) / suffix))
        require(not runner_output.exists(), "Runner output must be new; no resume/overwrite")
        planned.append({"dataset": dataset, "argv": argv, "output": runner_output, "bindings": bindings})
    start_time = datetime.fromisoformat(bound["start_UTC"])
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    require(0 <= elapsed < bound["normal_child_budget_seconds"], "Whole-bound start time is invalid or exhausted")
    all_bindings = list(request["protected_files"]) + source_bindings + [request[name] for name in
                    ("allocation_evidence", "protocol", "sources", "runner")]
    return output, planned, all_bindings, timeout, time.monotonic() + bound["normal_child_budget_seconds"] - elapsed, bound_path


class InterruptedStage(Exception):
    pass


def interrupted(signum, frame):
    raise InterruptedStage("signal " + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--supervisor", required=True)
    args = parser.parse_args()
    request_path, supervisor = confined(args.request, True), confined(args.supervisor)
    request_sha = sha(request_path)
    request = read_json(request_path)
    output, operations, bindings, timeout, deadline, bound_path = plan(request, request_path, supervisor)
    output.mkdir(parents=True)
    write_new(output / "START.json", {"schema": "coordinate-ensemble-stage-entry-start-v1",
              "UTC": datetime.now(timezone.utc).isoformat(), "request_sha256": request_sha,
              "request": str(request_path), "supervisor": str(supervisor), "bound_start_sha256": sha(bound_path),
              "bridge_sha256": sha(ENTRY), "mode": request["mode"], "operation_count": len(operations),
              "parent_CUDA_VISIBLE_DEVICES": request["cuda_visible_devices"],
              "child_CUDA_VISIBLE_DEVICES": "" if request["mode"] == "acquire" else request["cuda_visible_devices"],
              "gpu_idle_settling": {"maximum_seconds": GPU_IDLE_SETTLING_SECONDS,
                                    "metadata_interval_seconds": GPU_METADATA_INTERVAL_SECONDS,
                                    "memory_used_MiB_maximum": 100, "utilization_percent_required": 0,
                                    "occupied_memory_rejected_immediately": True},
              "validation_fit_admitted": request["mode"] == "fit", "test_admitted": False})
    records, gpu_guards, error, active, active_record = [], [], None, None, None
    begun = time.monotonic()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for number, operation in enumerate(operations):
            require(sha(request_path) == request_sha, "Request changed after admission")
            for binding in bindings + operation["bindings"]:
                verify_binding(binding)
            require(time.monotonic() + timeout + GPU_IDLE_SETTLING_SECONDS + 2 < deadline,
                    "Insufficient whole-bound runway for exact operation and GPU settling")
            guard_receipt = {"schema": "coordinate-ensemble-stage-entry-gpu-guard-v1", "number": number}
            guard_path = output / f"operation_{number:03d}_GPU_GUARD.json"
            gpu_guards.append(guard_receipt)
            try:
                gpu = gpu_guard(guard_receipt)
            finally:
                write_new(guard_path, guard_receipt)
            require(time.monotonic() + timeout + 2 < deadline, "GPU guard exhausted operation runway")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = "" if request["mode"] == "acquire" else request["cuda_visible_devices"]
            log = output / f"operation_{number:03d}.log"
            record = {"number": number, "argv": operation["argv"], "shell": False,
                      "runner_output": str(operation["output"]), "gpu_before_child": gpu,
                      "gpu_guard_receipt": str(guard_path), "gpu_guard_receipt_sha256": sha(guard_path),
                      "child_CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"]}
            write_new(output / f"operation_{number:03d}_START.json", record)
            tick = time.monotonic()
            active_record = record
            with log.open("x") as stream:
                active = subprocess.Popen(operation["argv"], cwd=str(REPO), env=environment,
                                          stdout=stream, stderr=subprocess.STDOUT, shell=False,
                                          start_new_session=False)
                record["pid"] = active.pid
                record["exit_code"] = active.wait(timeout=timeout)
                active = None
            record["seconds"] = time.monotonic() - tick
            record["log_sha256"] = sha(log)
            records.append(record)
            active_record = None
            write_new(output / f"operation_{number:03d}_TERMINAL.json", record)
            require(record["exit_code"] == 0, "Exact runner operation returned nonzero")
            require((operation["output"] / "artifacts.json").is_file(), "Runner terminal artifact manifest absent")
            if request["mode"] in {"qualify", "fit"}:
                report = read_json(operation["output"] / "cell_report.json")
                require(report["mode"] == request["mode"] and
                        report["test_labels_read"] is False and report["test_scores_or_logits_written"] is False,
                        "Runner mode/test-isolation receipt differs")
                if request["mode"] == "qualify":
                    require(report["status"] == "QUALIFICATION_COMPLETE" and report["completed_updates"] == 100,
                            "Qualification update completion differs")
                else:
                    require(report["status"] == "FIT_COMPLETE" and
                            1000 <= report["completed_updates"] <= 2000 and
                            type(report["selected_epoch"]) is int and
                            1 <= report["selected_epoch"] <= report["completed_updates"] and
                            report["selected_validation"] is not None,
                            "Complete validation-fit receipt differs; convergence still needs scientific assessment")
            for binding in bindings + operation["bindings"]:
                verify_binding(binding)
        require(sha(request_path) == request_sha, "Request changed during stage")
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            if active is not None and active.poll() is None:
                active.terminate()
                try:
                    active.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    active.kill()
                    active.wait(timeout=1)
        except BaseException as cleanup_exc:
            error = {"type": type(cleanup_exc).__name__, "message": str(cleanup_exc),
                     "prior_error": error}
        if active_record is not None:
            active_record.update({"exit_code": active.returncode if active is not None else None,
                                  "seconds": time.monotonic() - tick, "interrupted": True,
                                  "log_sha256": sha(log) if log.is_file() else None})
            records.append(active_record)
            record_path = output / f"operation_{number:03d}_TERMINAL.json"
            if not record_path.exists():
                write_new(record_path, active_record)
        try:
            unchanged = sha(request_path) == request_sha
        except OSError:
            unchanged = False
        terminal = {"schema": "coordinate-ensemble-stage-entry-terminal-v1",
                    "UTC": datetime.now(timezone.utc).isoformat(), "complete": error is None and unchanged and len(records) == len(operations),
                    "error": error, "completed_operation_count": sum(r["exit_code"] == 0 for r in records),
                    "requested_operation_count": len(operations), "records": records, "gpu_guards": gpu_guards,
                    "request_unchanged": unchanged, "elapsed_seconds": time.monotonic() - begun,
                    "START_sha256": sha(output / "START.json"), "validation_fit_admitted": request["mode"] == "fit", "test_admitted": False,
                    "termination_owner": "Existing bounded supervisor kills the complete inherited process group"}
        write_new(output / "TERMINAL.json", terminal)
    print(json.dumps({"stage_output": str(output), "complete": terminal["complete"],
                      "completed_operation_count": terminal["completed_operation_count"]}), flush=True)
    raise SystemExit(0 if terminal["complete"] else 1)


if __name__ == "__main__":
    main()
