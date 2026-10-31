"""Admission/closure and wall/disk guards, importing only Python stdlib."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ENGINEERING_PIN = "139cf316ff00bde3b1dad7adeb6154cdc660a322262e1ee72afdb6502943c96d"
MECHANICS_PIN = "b9f76667aaf2479ee5cd8816877ebc650c99380189c4869545519e753fdf63b9"
DESTINATION = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"


def within(path, root=PHASE):
    root, path = Path(root).resolve(), Path(os.path.abspath(path))
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Path must remain strictly inside its declared packet")
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Symlink evidence/output paths are forbidden")
    return path.resolve()


def sha(path):
    path = within(path)
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(part)
    after = path.stat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if signature(before) != signature(after):
        raise ValueError("Evidence changed while hashing")
    return value.hexdigest()


def checked(ref):
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Exact phase-relative path and SHA256 required")
    path = within(PHASE / ref["path"])
    if str(path.relative_to(PHASE)) != ref["path"] or sha(path) != ref["sha256"]:
        raise ValueError("Noncanonical or changed evidence")
    return path


def write(path, value):
    with within(path, HERE).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_seal(wanted=None):
    from .import_guard import verify_packet
    actual = sha(HERE / "SOURCE_MANIFEST_v1.json")
    verify_packet(actual if wanted is None else wanted)
    return actual


def completed(directory):
    directory = within(directory)
    a, c, e, command = [json.loads((directory / name).read_text()) for name in
                       ("authorization.json", "completion.json", "environment.json", "command.json")]
    if (a.get("schema") != "gnnm-authorized-allocation-supervision-v2"
            or a.get("ssh_destination") != DESTINATION or a.get("visible_gpu_count") != 1
            or a.get("idle_required") is not True
            or a.get("gpu_name") != "NVIDIA A100-SXM4-80GB" or a.get("child_exit_code") != 0
            or c.get("exit_code") != 0 or a.get("inner_completion_sha256") != sha(directory / "completion.json")
            or a.get("allocation_guard_sha256") != sha(PHASE / "protocols/run_authorized_v2.py")
            or a.get("route_record_sha256") != sha(PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json")):
        raise ValueError("Wrong, incomplete or failed authorized supervision")
    for key, name in (("environment_sha256", "environment.json"), ("command_sha256", "command.json"), ("log_sha256", "stdout_stderr.log")):
        if c[key] != sha(directory / name):
            raise ValueError("Completed supervisor evidence changed")
    route = json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
    argv = command.get("argv", [])
    if (e.get("repository") != route["repository"] or command.get("cwd") != route["repository"]
            or e.get("supervisor_sha256") != sha(PHASE / "protocols/run_logged.py")
            or e.get("wrapper_sha256") != sha(PHASE / "protocols/repo_env.sh")
            or len(argv) < 2 or e.get("script_sha256") != sha(within(argv[1]))):
        raise ValueError("Completed repository/interpreter entrypoint source differs")
    if argv[0] != str(Path(route["repository"]) / ".venv/bin/python") or command.get("shell") is not False:
        raise ValueError("Exact repository interpreter and argument-list launch required")
    live = dict(command_sha256=sha(directory / "command.json"),
        environment_sha256=sha(directory / "environment.json"), route_sha256=sha(PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json"),
        supervisor_relative_to_project=str(directory.relative_to(PHASE)), entrypoint_sha256=e["script_sha256"],
        completed_chain_status="pending_until_child_exit")
    return dict(gpu_uuid=a["gpu_uuid"], seconds=c["seconds"], argv=argv, live=live)


def expected_cli(path, operation, supervisor):
    route = json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
    return [str(Path(route["repository"]) / ".venv/bin/python"), str(HERE / "qualify.py"), operation,
            "--request", str(within(path, HERE)), "--supervisor", str(within(supervisor))]


def outer_start(path, request_path, admitted, supervisor, argv):
    path = within(path)
    start = json.loads(path.read_text())
    route = json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
    expected = [argv[0], str(PHASE / "protocols/run_authorized_v2.py"), "--output", str(within(supervisor)),
                "--require-idle-gpu", "--", *argv]
    if (path.name != "START.json" or start.get("schema") != "gnnm-whole-process-bound-start-v1"
            or start.get("root_request") != dict(path=str(within(request_path, HERE).relative_to(PHASE)), sha256=sha(request_path))
            or start.get("root_request_child_flag") != "--request"
            or start.get("child_argv") != argv or start.get("argv") != expected
            or start.get("inner_supervisor_directory") != str(within(supervisor))
            or start.get("outer_output") != str(path.parent) or start.get("cwd") != route["repository"]
            or start.get("phase") != str(PHASE) or start.get("ssh_destination") != DESTINATION
            or start.get("shell") is not False or start.get("start_new_session") is not True
            or start.get("whole_cap_seconds") != admitted["wall_cap_seconds"]
            or not 0 < start.get("termination_grace_seconds", 0) < start["whole_cap_seconds"]
            or start.get("normal_child_budget_seconds") != start["whole_cap_seconds"] - start["termination_grace_seconds"]):
        raise ValueError("Exact external process-group deadline START required")
    for key, relative in (("source_sha256", "protocols/bounded_run_v1.py"),
                          ("authorization_source_sha256", "protocols/run_authorized_v2.py"),
                          ("logging_source_sha256", "protocols/run_logged.py"),
                          ("wrapper_sha256", "protocols/repo_env.sh"),
                          ("research_script_sha256", "shared_message_quality_v2/qualify.py")):
        if start.get(key) != sha(PHASE / relative):
            raise ValueError("Outer deadline source differs: " + relative)
    if datetime.fromisoformat(start["start_UTC"]).tzinfo is None:
        raise ValueError("External start time needs an explicit timezone")
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path))


def outer_completed(ref, request_path, admitted, supervisor, argv):
    terminal_path = checked(ref)
    terminal = json.loads(terminal_path.read_text())
    start_ref = outer_start(terminal_path.parent / "START.json", request_path, admitted, supervisor, argv)
    if (terminal_path.name != "TERMINAL.json" or terminal.get("schema") != "gnnm-whole-process-bound-terminal-v1"
            or terminal.get("START_sha256") != start_ref["sha256"] or terminal.get("complete") is not True
            or terminal.get("root_request_unchanged") is not True or terminal.get("child_exit_code") != 0
            or terminal.get("within_whole_cap") is not True
            or terminal.get("timed_out") is not False or terminal.get("interruption") is not None
            or type(terminal.get("whole_supervised_seconds")) not in (int, float)
            or not math.isfinite(terminal["whole_supervised_seconds"])
            or not 0 <= terminal["whole_supervised_seconds"] <= admitted["wall_cap_seconds"]
            or terminal.get("stdout_stderr_sha256") != sha(terminal_path.parent / "stdout_stderr.log")):
        raise ValueError("Successful completed external whole-process deadline envelope required")
    names = {"authorization.json", "completion.json", "command.json", "environment.json", "stdout_stderr.log"}
    if set(terminal.get("inner_evidence", {})) != names:
        raise ValueError("Outer terminal must bind exact inner evidence set")
    for name in names:
        file = within(Path(supervisor) / name)
        if terminal["inner_evidence"][name] != dict(sha256=sha(file), bytes=file.stat().st_size):
            raise ValueError("Outer terminal/inner evidence changed: " + name)
    return start_ref


def synthetic_chain(value, manifest_sha):
    synthetic_path = checked(value["synthetic_request"])
    synthetic, synthetic_sha = request(synthetic_path, "synthetic")
    gate_path = checked(value["synthetic_gate"])
    output = within(PHASE / synthetic["output_relative_to_phase"], HERE)
    if gate_path != output / "RESULT.json":
        raise ValueError("Synthetic result must be at its exact request-prescribed output")
    supervisor = within(PHASE / value["synthetic_supervisor_directory"])
    chain = completed(supervisor)
    argv = expected_cli(synthetic_path, "synthetic", supervisor)
    if chain["argv"] != argv:
        raise ValueError("Completed synthetic CLI/request/supervisor differs")
    outer_ref = outer_completed(value["synthetic_outer_terminal"], synthetic_path, synthetic, supervisor, argv)
    gate = json.loads(gate_path.read_text())
    start = json.loads((output / "START.json").read_text())
    live = dict(chain["live"], outer_start=outer_ref)
    if (gate.get("schema") != "gnnm-shared-message-synthetic-checks-v2" or gate.get("status") != "passed"
            or gate.get("source_manifest_sha256") != manifest_sha or gate.get("request_sha256") != synthetic_sha
            or gate.get("live") != live or start.get("live") != live
            or start.get("operation") != "synthetic" or start.get("request_sha256") != synthetic_sha
            or start.get("source_manifest_sha256") != manifest_sha
            or gate.get("device") != "cuda:0" or gate.get("gpu", {}).get("uuid") != value["gpu_uuid"]
            or chain["gpu_uuid"] != value["gpu_uuid"] or synthetic["gpu_uuid"] != value["gpu_uuid"]
            or gate.get("runtime_policy") != value["runtime_policy"] or synthetic["runtime_policy"] != value["runtime_policy"]
            or gate.get("observed_runtime_policy") != value["runtime_policy"]
            or gate.get("equivalence_tolerance") != value["equivalence_tolerance"]
            or synthetic["equivalence_tolerance"] != value["equivalence_tolerance"]
            or type(gate.get("check_count")) is not int or gate["check_count"] != len(gate.get("checks", []))
            or gate["check_count"] < 1 or not all(row.get("passed") is True for row in gate["checks"])
            or gate.get("memory_cap_bytes") != synthetic["memory_cap_bytes"]
            or type(gate.get("peak_memory", {}).get("peak_reserved_bytes")) is not int
            or not 0 <= gate["peak_memory"]["peak_reserved_bytes"] <= min(synthetic["memory_cap_bytes"], value["memory_cap_bytes"])
            or not 0 <= gate["peak_memory"].get("peak_allocated_bytes", -1) <= gate["peak_memory"]["peak_reserved_bytes"]):
        raise ValueError("Synthetic request/result/source/runtime/tolerance/GPU/memory chain differs")
    artifact = json.loads((output / "artifact_manifest.json").read_text())
    names = {"START.json", "RESULT.json"}
    if (artifact.get("schema") != "gnnm-shared-message-engineering-artifacts-v2"
            or set(artifact.get("files", {})) != names
            or {p.name for p in output.iterdir()} != names | {"artifact_manifest.json"}):
        raise ValueError("Exact synthetic START/RESULT/artifact output set required")
    for name in names:
        file = within(output / name)
        if artifact["files"][name] != dict(bytes=file.stat().st_size, sha256=sha(file)):
            raise ValueError("Synthetic artifact manifest chain changed")
    from .import_guard import verify_packet
    policy, tree = verify_packet(manifest_sha)
    if gate.get("runtime_import_tree") != tree or not gate.get("imported_phase_modules"):
        raise ValueError("Synthetic exact runtime source/import receipt missing")
    for row in gate["imported_phase_modules"]:
        if row.get("sha256") != policy["importable_sha256"].get(row.get("relative_to_phase")):
            raise ValueError("Synthetic imported phase source receipt differs")
    return dict(request_sha256=synthetic_sha, result_sha256=sha(gate_path),
        artifact_manifest_sha256=sha(output / "artifact_manifest.json"), outer_terminal=value["synthetic_outer_terminal"])


def request(path, operation):
    from .counts import TASKS, RECIPE
    path = within(path, HERE)
    value = json.loads(path.read_text())
    manifest_sha = verify_seal(value.get("source_manifest_sha256"))
    if (value.get("schema") != "gnnm-shared-message-root-engineering-request-v2"
            or value.get("operation") != operation or value.get("source_manifest_sha256") != manifest_sha
            or value.get("root_authorized_train_only_engineering") is not True
            or value.get("scientific_training_authorized") is not False
            or value.get("quality_scoring_authorized") is not False
            or value.get("heldout_labels_authorized") is not False
            or value.get("recipe") != RECIPE or value.get("seed") != 17):
        raise ValueError("Explicit source-bound training-only engineering request required")
    if (any(type(value[k]) not in (int, float) or not math.isfinite(value[k]) for k in ("wall_cap_seconds", "prior_allocation_seconds"))
            or not 0 < value["wall_cap_seconds"] <= 1800 or not 0 <= value["prior_allocation_seconds"] < 28800):
        raise ValueError("Bounded actual remaining allocation required")
    if value["prior_allocation_seconds"] + value["wall_cap_seconds"] > 28800:
        raise ValueError("Qualification would exceed total allocation budget")
    if (type(value["memory_cap_bytes"]) is not int or type(value["disk_cap_bytes"]) is not int
            or not 0 < value["memory_cap_bytes"] <= 60 * 2**30 or not 0 < value["disk_cap_bytes"] <= 64 * 2**20):
        raise ValueError("Explicit bounded engineering memory/metadata disk required")
    if operation == "real":
        if value.get("tasks") != list(TASKS) or set(value.get("engineering_phases", {})) != set(TASKS):
            raise ValueError("Both complete tasks required; no smaller-data fallback")
        if type(value["warmups"]) is not int or not 1 <= value["warmups"] <= 5 or type(value["samples"]) is not int or not 3 <= value["samples"] <= 10:
            raise ValueError("Bounded declared warmups and samples required")
        for ref in value["engineering_phases"].values():
            checked(ref)
    policy = value["runtime_policy"]
    if (policy.get("TF32") is not False or policy.get("AMP") is not False or policy.get("device") != "cuda:0"
            or type(policy.get("CPU_threads")) is not int or policy["CPU_threads"] < 1
            or type(policy.get("deterministic_algorithms")) is not bool
            or not {"torch", "numpy", "torch-geometric"}.issubset(policy.get("package_versions", {}))
            or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != policy["CUBLAS_WORKSPACE_CONFIG"]
            or (policy["deterministic_algorithms"] and policy["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"))):
        raise ValueError("Runtime freeze must precede third-party imports")
    if set(value["equivalence_tolerance"]) != {"atol", "rtol"} or any(type(value["equivalence_tolerance"][key]) not in (int, float) or not math.isfinite(value["equivalence_tolerance"][key])
           or not 0 <= value["equivalence_tolerance"][key] <= 1e-3 for key in ("atol", "rtol")):
        raise ValueError("Finite bounded numerical equivalence tolerance required")
    within(PHASE / value["output_relative_to_phase"], HERE)
    if operation == "real":
        value["_synthetic_chain"] = synthetic_chain(value, manifest_sha)
    return value, sha(path)


class Budget:
    def __init__(self, value, live):
        self.request, self.live = value, live
        outer = json.loads(checked(live["outer_start"]).read_text())
        self.started = datetime.fromisoformat(outer["start_UTC"])
        if self.started.tzinfo is None:
            raise ValueError("Timezone-aware outer start required")
        self.output = within(PHASE / value["output_relative_to_phase"], HERE)
        if self.output.exists():
            raise ValueError("Preserve previous engineering outputs; new exclusive directory required")

    def check(self):
        seconds = (datetime.now(timezone.utc) - self.started).total_seconds()
        if seconds >= self.request["wall_cap_seconds"] or self.request["prior_allocation_seconds"] + seconds >= 28800:
            raise RuntimeError("Engineering supervised wall/allocation budget exhausted")
        if self.output.exists() and sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file()) > self.request["disk_cap_bytes"]:
            raise RuntimeError("Engineering metadata disk cap exceeded")

    def seconds(self):
        return (datetime.now(timezone.utc) - self.started).total_seconds()
