"""Root-gated CPU diagnostic for a qualified complete30 validation family.

Source packet only until a separately reviewed, resource-approved request exists.
No inference, checkpoint loading, provider access, test labels or GPU calls.
"""
import os
import sys
sys.dont_write_bytecode = True
sys.path[:] = [p for p in sys.path if p and os.path.isabs(p)
               and any(os.path.commonpath([p, root]) == root
                       for root in (sys.prefix, sys.base_prefix))]
import argparse
import array
import ast
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import statistics
import struct

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
PLAN_ROOT = "shared_message_decision_analysis_plan_v1"
PLAN_SEAL_SHA = "6821b679d544bec89391c106dbfb5d0c2916f0d9f63c01ce61a0ad99ade86b79"
PLAN_MANIFEST_SHA = "7b115196efa93a5789898996944d9efd167aef7899814a4bc9e7dd78a80082ee"
BOUND_SHA = "b096966c1570da0a998dbb9fb464a9ad71490b74d4ba3406b751390637d19a23"
DESTINATION = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
RUNTIME = {"torch_version": "2.1.2+cu118", "CPU_threads": 4,
           "device": "cpu", "scientific_imports": ["torch"], "GPU_calls": 0}
CAPS = {"whole_wall_seconds": 600, "host_RSS_bytes": 2**31,
        "new_output_disk_bytes": 128 * 2**20, "CPU_threads": 4, "GPU_count": 0}
MAX_LOGIT_FILE_BYTES = 64 * 2**20
MAX_TARGET_FILE_BYTES = 2 * 2**20
MAX_METADATA_BYTES = 16 * 2**20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def confined(value, root=PHASE):
    root = Path(root).resolve()
    path = Path(os.path.abspath(value))
    require(path != root and path.is_relative_to(root), "Path outside declared scope")
    current = root
    for name in path.relative_to(root).parts:
        current /= name
        require(not current.is_symlink(), "Symlink path forbidden")
    return path.resolve()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path, budget=None):
    path = confined(path)
    before = path.stat()
    signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            hasher.update(chunk)
            if budget:
                budget.check()
    after = path.stat()
    require(signature == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Input changed during hashing")
    return hasher.hexdigest()


def checked(ref, budget=None, max_bytes=None):
    require(isinstance(ref, dict) and set(ref) in ({"path", "sha256"}, {"path", "sha256", "bytes"}), "Exact input reference required")
    relative = Path(ref["path"])
    require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == ref["path"], "Noncanonical reference")
    path = confined(PHASE / relative)
    size = path.stat().st_size
    if max_bytes is not None:
        require(size <= max_bytes, "Input exceeds declared file-size bound")
    if "bytes" in ref:
        require(type(ref["bytes"]) is int and ref["bytes"] == size, "Input size differs")
    require(isinstance(ref["sha256"], str) and sha(path, budget) == ref["sha256"], "Input hash differs")
    return path


def reference(path):
    path = confined(path)
    return {"path": str(path.relative_to(PHASE)), "sha256": sha(path)}


def same_ref(a, b):
    return isinstance(a, dict) and isinstance(b, dict) and a.get("path") == b.get("path") and a.get("sha256") == b.get("sha256")


def read_json(ref, budget=None):
    return json.loads(checked(ref, budget, MAX_METADATA_BYTES).read_text())


def write_json(path, value):
    with confined(path, HERE).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_plan_and_source(request):
    seal_path = confined(PHASE / PLAN_ROOT / "SEAL.json")
    require(sha(seal_path) == PLAN_SEAL_SHA, "Sealed diagnostic plan differs")
    seal = json.loads(seal_path.read_text())
    require(seal["manifest"]["sha256"] == PLAN_MANIFEST_SHA, "Plan manifest identity differs")
    manifest = read_json(seal["manifest"])
    require(manifest["root_diagnostic_execution_authorized"] is False, "Plan was changed into authorization")
    for name, row in manifest["files"].items():
        require(Path(name).name == name, "Plan file path differs")
        checked({"path": PLAN_ROOT + "/" + name, **row})
    plan = json.loads((PHASE / PLAN_ROOT / "PROTOCOL.json").read_text())
    require(plan["execution_authorized"] is False and plan["timing_provenance"]["preregistered_primary_endpoint"] is False, "Plan classification or disabled gate differs")
    source_ref = request["source_manifest"]
    source_path = checked(source_ref)
    require(source_path == HERE / "SOURCE_MANIFEST_v1.json", "Exact implementation manifest required")
    source = json.loads(source_path.read_text())
    require(source["schema"] == "gnnm-shared-message-decision-analysis-source-manifest-v1", "Implementation schema differs")
    names = set()
    for row in source["files"]:
        name = row["relative_path"]
        path = confined(HERE / name, HERE)
        require(name not in names and str(path.relative_to(HERE)) == name, "Duplicate/noncanonical source")
        names.add(name)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Implementation source differs")
    # No local source imports are needed. No bytecode, compiled modules, shadow
    # package directories or extra Python source can enter this packet.
    allowed_py = {name for name in names if name.endswith(".py")}
    found_py = set()
    for directory, dirs, files in os.walk(HERE, followlinks=False):
        parent = Path(directory)
        for name in dirs:
            require(not (parent / name).is_symlink() and name != "__pycache__", "Unsealed source/cache directory")
        for name in files:
            path = parent / name
            require(not path.is_symlink(), "Packet symlink forbidden")
            if path.suffix in {".py", ".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"}:
                relative = str(path.relative_to(HERE))
                require(relative in allowed_py, "Unsealed packet importable")
                found_py.add(relative)
    require(found_py == allowed_py, "Missing sealed Python source")
    for relative, wanted in source["dependency_sha256"].items():
        require(sha(PHASE / relative) == wanted, "Pinned source/metadata dependency differs")
    return plan, source


def verify_live(request, request_path, supervisor):
    route_path = confined(PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json")
    route = json.loads(route_path.read_text())
    require(route["ssh_destination"] == DESTINATION and Path.cwd().resolve() == Path(route["repository"]).resolve(), "Original authorized route/cwd required")
    supervisor = confined(supervisor)
    outer_path = confined(os.environ["GNNM_BOUND_START_JSON"])
    outer = json.loads(outer_path.read_text())
    command = json.loads((supervisor / "command.json").read_text())
    environment = json.loads((supervisor / "environment.json").read_text())
    interpreter = str(Path(route["repository"]) / ".venv/bin/python")
    argv = [interpreter, str(HERE / "decision_analysis.py"), "--request", str(request_path), "--supervisor", str(supervisor)]
    expected = [interpreter, str(PHASE / "protocols/run_authorized_v2.py"), "--output", str(supervisor), "--", *argv]
    require(outer_path.name == "START.json" and outer.get("schema") == "gnnm-whole-process-bound-start-v1", "Bounded outer START required")
    require(outer.get("root_request") == reference(request_path) and outer.get("root_request_child_flag") == "--request", "Outer request identity differs")
    require(outer.get("child_argv") == argv and command.get("argv") == argv and outer.get("argv") == expected, "Exact CPU child and supervisors required")
    require(outer.get("inner_supervisor_directory") == str(supervisor) and outer.get("outer_output") == str(outer_path.parent), "Supervisor/output identity differs")
    require(outer.get("whole_cap_seconds") == CAPS["whole_wall_seconds"] and 0 < outer.get("termination_grace_seconds", 0) < CAPS["whole_wall_seconds"], "Whole wall/termination bound differs")
    require(outer.get("normal_child_budget_seconds") == CAPS["whole_wall_seconds"] - outer["termination_grace_seconds"], "Termination grace must be inside wall cap")
    require(outer.get("shell") is False and command.get("shell") is False and outer.get("start_new_session") is True, "No-shell supervised session required")
    require(outer.get("cwd") == command.get("cwd") == environment.get("repository") == route["repository"], "Repository chain differs")
    require(outer.get("phase") == environment.get("phase") == os.environ.get("GNNM_PHASE_ROOT") == str(PHASE), "Phase chain differs")
    require(outer.get("ssh_destination") == os.environ.get("GNNM_SSH_DESTINATION") == DESTINATION, "Original route differs")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" and os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Original repository wrapper required")
    for key, path in (("source_sha256", PHASE / "protocols/bounded_run_v1.py"), ("authorization_source_sha256", PHASE / "protocols/run_authorized_v2.py"), ("logging_source_sha256", PHASE / "protocols/run_logged.py"), ("wrapper_sha256", PHASE / "protocols/repo_env.sh"), ("research_script_sha256", HERE / "decision_analysis.py")):
        require(outer.get(key) == sha(path), "Live supervisor source differs")
    require(outer["source_sha256"] == BOUND_SHA and environment.get("script_sha256") == sha(HERE / "decision_analysis.py"), "Bounded/research entry source differs")
    require(environment.get("supervisor_sha256") == sha(PHASE / "protocols/run_logged.py") and environment.get("wrapper_sha256") == sha(PHASE / "protocols/repo_env.sh"), "Environment source differs")
    for name, value in environment["cache_and_temp"].items():
        require(os.environ.get(name) == value and Path(value).resolve().is_relative_to(PHASE), "Cache/temp escaped phase")
    started = datetime.fromisoformat(outer["start_UTC"])
    require(started.tzinfo is not None, "Timezone-aware external START required")
    return {"outer_start": reference(outer_path), "supervisor_directory": str(supervisor.relative_to(PHASE)), "command": reference(supervisor / "command.json"), "environment": reference(supervisor / "environment.json"), "started": started.isoformat(), "normal_child_budget_seconds": outer["normal_child_budget_seconds"]}


def admission(path, supervisor):
    path = confined(path, HERE)
    request = json.loads(path.read_text())
    require(request.get("schema") == "gnnm-shared-message-decision-analysis-root-request-v1" and request.get("root_authorized") is True, "Separate approved diagnostic request required")
    require(request.get("runtime_policy") == RUNTIME and request.get("caps") == CAPS, "Exact bounded CPU runtime required")
    require(all(request.get(k) is False for k in ("training_or_inference_authorized", "test_target_access_authorized", "provider_access_authorized", "GPU_operations_authorized", "automatic_scientific_promotion_authorized")), "Forbidden diagnostic permission")
    require(request.get("plan_seal_sha256") == PLAN_SEAL_SHA, "Request plan seal differs")
    for name in ("source_manifest", "consumer_audit", "implementation_review", "resource_approval", "root_approval"):
        require(isinstance(request.get(name), dict), "Missing required admission receipt: " + name)
    plan, source = verify_plan_and_source(request)
    review = read_json(request["implementation_review"])
    require(review.get("schema") == "gnnm-shared-message-decision-analysis-independent-review-v1" and review.get("status") == "approved_for_bounded_execution", "Independent implementation approval required")
    require(review.get("independent_of_implementation_author") is True and same_ref(review.get("source_manifest"), request["source_manifest"]), "Independent review source identity differs")
    require(review.get("numerical_execution_performed") is False and review.get("unresolved_blocking_issues") == [], "Source-only independent review must resolve blockers")
    resource_receipt = read_json(request["resource_approval"])
    require(resource_receipt.get("schema") == "gnnm-shared-message-decision-analysis-root-resource-approval-v1" and resource_receipt.get("root_approved") is True, "Separate root resource reservation required")
    require(resource_receipt.get("reserved_caps") == CAPS and resource_receipt.get("frozen_training_allocation_extended") is False and resource_receipt.get("separate_wall_and_disk_charged") is True, "Resource accounting differs")
    approval = read_json(request["root_approval"])
    require(approval.get("schema") == "gnnm-shared-message-decision-analysis-root-approval-v1" and approval.get("root_authorized") is True, "Root approval missing")
    require(approval.get("approved_request_payload_sha256") == digest({k: v for k, v in request.items() if k != "root_approval"}), "Root approval payload differs")
    live = verify_live(request, path, supervisor)
    output = confined(PHASE / request["output_relative_to_phase"], HERE)
    require(output.parent == HERE / "runs" and not output.exists(), "Exclusive new local runs output required")
    return request, reference(path), plan, source, live, output


class Budget:
    def __init__(self, live, output):
        self.started = datetime.fromisoformat(live["started"])
        self.seconds = live["normal_child_budget_seconds"]
        self.output = output

    def check(self):
        elapsed = (datetime.now(timezone.utc) - self.started).total_seconds()
        require(0 <= elapsed < self.seconds, "Whole-process CPU wall cap reached")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = int(rss * (1024 if sys.platform.startswith("linux") else 1))
        require(rss <= CAPS["host_RSS_bytes"], "CPU RSS cap exceeded")
        if self.output.exists():
            require(sum(p.stat().st_size for p in self.output.iterdir() if p.is_file()) <= CAPS["new_output_disk_bytes"], "Diagnostic output disk cap exceeded")


def verify_consumer(request, plan, budget):
    """No binary input is opened until complete-family root qualification."""
    consumer = read_json(request["consumer_audit"], budget)
    require(consumer.get("schema") == "gnnm-shared-message-complete30-diagnostic-consumer-audit-v1" and consumer.get("status") == "qualified_complete30", "Actual full-family consumer audit required")
    require(consumer.get("root_audited") is True and consumer.get("all30_fits_and_replays_and_lock_qualified") is True and consumer.get("test_targets_loaded") is False, "Partial/unaudited/test family refused")
    require(isinstance(consumer.get("completed_family_root_audit"), dict), "Missing original complete-family audit")
    root_audit = read_json(consumer["completed_family_root_audit"], budget)
    require(root_audit.get("all30_fits_and_replays_qualified") is True and root_audit.get("test_targets_loaded") is False, "Original root full-family qualification missing")
    binding = read_json(consumer["binding"], budget)
    freeze = read_json(consumer["validation_freeze"], budget)
    family = read_json(consumer["family_lock"], budget)
    require(family.get("schema") == "gnnm-shared-message-complete30-validation-family-lock-v2" and family.get("all30_fits_and_replays_qualified") is True, "Full frozen family lock required")
    require(family.get("fixed_updates") == 9000 and family.get("test_target_export_or_evaluation_authorized") is False and family.get("locked_rows_sha256") == digest(family["rows"]), "Family identity/content differs")
    require(same_ref(family.get("binding"), consumer["binding"]) and same_ref(family.get("validation_freeze"), consumer["validation_freeze"]), "Family binding/freeze differs")
    cells = [{k: r[k] for k in ("task", "seed", "arm")} for r in plan["study_scope"]["cells"]]
    require(family.get("cells") == cells and binding.get("cells") == cells and len(family["rows"]) == 30, "Canonical complete30 family differs")
    require(binding.get("test_target_export_or_evaluation_authorized") is False, "Test authority present")
    science_root = confined(PHASE / binding["output_root"])
    rows = consumer.get("rows")
    require(isinstance(rows, list) and len(rows) == 30 and [r.get("cell") for r in rows] == cells, "All30 consumer rows/order required")
    require(set(consumer.get("validation_targets", {})) == set(plan["study_scope"]["tasks"]), "Both existing target bundles required")
    inputs = json.loads((PHASE / PLAN_ROOT / "INPUT_CONTRACT.json").read_text())
    payload_refs, metadata_refs = [], [request["consumer_audit"], consumer["completed_family_root_audit"], consumer["binding"], consumer["validation_freeze"], consumer["family_lock"]]
    for cell_row, locked, expected in zip(rows, family["rows"], plan["study_scope"]["cells"]):
        budget.check()
        cell = cell_row["cell"]
        require({k: locked[k] for k in ("task", "seed", "arm")} == cell, "Locked cell identity differs")
        fit_ref, replay_ref = cell_row["fit"], cell_row["replay"]
        require(same_ref(fit_ref, locked["fit"]) and same_ref(replay_ref, locked["replay"]), "Consumer receipts differ from family")
        require(checked(fit_ref, budget) == science_root / "fits" / expected["cell_id"] / "FIT_END.json" and checked(replay_ref, budget) == science_root / "replays" / expected["cell_id"] / "REPLAY_END.json", "Receipt route differs")
        fit, replay = read_json(fit_ref, budget), read_json(replay_ref, budget)
        require(fit.get("schema") == "gnnm-shared-message-complete300-fit-v2" and fit.get("status") == "completed" and fit.get("cell") == cell and fit.get("epochs") == 300 and fit.get("test_targets_loaded") is False, "Fit not qualified complete300")
        require(replay.get("schema") == "gnnm-shared-message-validation-replay-v2" and replay.get("status") == "passed" and replay.get("cell") == cell and replay.get("test_targets_loaded") is False, "Replay not qualified")
        require(replay.get("all_scalar_records_and_selection_rules_replayed") is True and replay.get("selected_and_final_separate_forwards_replayed") is True, "Both frozen replay roles required")
        require(all(same_ref(record.get("binding"), consumer["binding"]) and same_ref(record.get("validation_freeze"), consumer["validation_freeze"]) for record in (fit, replay)), "Fit/replay binding differs")
        require(same_ref(replay["fit"], fit_ref), "Replay fit identity differs")
        require(locked["selected_epoch"] == fit["selected_epoch"] == cell_row["selected_epoch"] and locked["selected_member"] == replay["selected_member"] == cell_row["selected_member"], "Frozen epoch/member differs")
        require(type(cell_row["selected_member"]) is int and 0 <= cell_row["selected_member"] < plan["study_scope"]["members"][cell["arm"]], "Invalid frozen member index")
        require(locked["final_validation"] == replay["final_primary_validation"] and locked["secondary_validation"] == replay["secondary_validation"], "Locked validation context differs")
        require(set(cell_row["logits"]) == {"final", "selected"}, "Both role logits required")
        require([x["role"] for x in replay["reports"]] == ["selected", "final"], "Replay roles/order differs")
        for report in replay["reports"]:
            role = report["role"]
            field = "final_checkpoint" if role == "final" else "selected_checkpoint"
            require(report["epoch"] == (300 if role == "final" else cell_row["selected_epoch"]), "Replay epoch differs")
            require(same_ref(report["checkpoint"], fit[field]) and same_ref(report["checkpoint"], locked[field]), "Replay checkpoint identity differs")
        for role in ("final", "selected"):
            logit_ref = cell_row["logits"][role]
            require(set(logit_ref) == {"path", "sha256", "bytes"} and same_ref(logit_ref, fit["validation_logits"][role]), "Exact saved fit logits required")
            # Size/path only here; full binary hash passes occur immediately
            # before loading and again after every diagnostic is complete.
            path = confined(PHASE / logit_ref["path"])
            require(path == science_root / "fits" / expected["cell_id"] / "validation_logits" / (role + ".pt") and path.stat().st_size == logit_ref["bytes"] <= MAX_LOGIT_FILE_BYTES, "Logit inventory path/size differs")
            payload_refs.append(logit_ref)
        metadata_refs.extend((fit_ref, replay_ref))
    for task, shape in inputs["task_shapes"].items():
        target_ref = consumer["validation_targets"][task]
        require(set(target_ref) == {"path", "sha256", "bytes"} and same_ref(target_ref, freeze["validation_exports"][task]["bundle"]), "Only existing frozen validation targets allowed")
        path = confined(PHASE / target_ref["path"])
        require(path == science_root / "targets" / shape["path_prefix"] / "validation_targets.pt" and path.stat().st_size == target_ref["bytes"] <= MAX_TARGET_FILE_BYTES, "Target inventory differs")
        payload_refs.append(target_ref)
        target_receipt_ref = freeze["validation_exports"][task]["receipt"]
        target_receipt = read_json(target_receipt_ref, budget)
        require(target_receipt.get("schema") == "gnnm-shared-message-validation-target-admission-v2" and target_receipt.get("task") == task and target_receipt.get("split") == "valid", "Existing authorized validation export receipt required")
        require(same_ref(target_receipt.get("bundle"), target_ref) and same_ref(target_receipt.get("binding"), consumer["binding"]), "Validation export binding differs")
        require(target_receipt.get("indexed_target_splits") == ["train", "valid"] and target_receipt.get("unrequested_target_values_indexed_or_summarized") is False, "Unrequested target indexing refused")
        metadata_refs.append(target_receipt_ref)
        entry = inputs["degree_inputs"][task]
        row = entry["record_copied_without_payload_access"]
        payload_refs.append({"path": entry["degree_file_relative_to_phase"], "sha256": row["sha256"], "bytes": row["bytes"]})
    require(len(payload_refs) == len({r["path"] for r in payload_refs}) == 64, "Exact64 unique binary inputs required")
    return consumer, family, inputs, payload_refs, metadata_refs


def runtime_receipt(torch):
    require(importlib.metadata.version("torch") == RUNTIME["torch_version"] and torch.__version__ == RUNTIME["torch_version"], "Pinned Torch version differs")
    distribution = importlib.metadata.distribution("torch")
    installed = Path(distribution.locate_file("torch/__init__.py")).resolve()
    require(Path(torch.__file__).resolve() == installed and installed.is_relative_to(Path(sys.prefix).resolve()), "Torch import origin differs")
    require(torch.get_num_threads() == 4, "Torch CPU thread count differs")
    phase_modules = []
    for name, module in list(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if raw and Path(raw).resolve().is_relative_to(PHASE):
            require(Path(raw).resolve() == HERE / "decision_analysis.py", "Unapproved phase source imported")
            phase_modules.append({"name": name, "path": str(Path(raw).resolve()), "sha256": sha(Path(raw).resolve())})
    require(not any(n == "torch_geometric" or n.startswith(("torch_geometric.", "ogb.")) or n == "ogb" for n in sys.modules), "Graph/provider package imported")
    return {"torch_version": torch.__version__, "torch_origin": str(installed), "CPU_threads": torch.get_num_threads(), "phase_modules": phase_modules, "sys_path": list(sys.path), "scientific_package_identity_scope": "Exact Torch distribution version/origin; transitive installed environment under repository venv, not a full third-party file hash seal."}


def load_targets(torch, ref, task, shape, budget):
    value = torch.load(checked(ref, budget, MAX_TARGET_FILE_BYTES), map_location="cpu", weights_only=True)
    require(set(value) == {"schema", "task", "split", "node_ids", "targets"} and value["schema"] == "gnnm-shared-message-official-validation-targets-v2" and value["task"] == task and value["split"] == "valid", "Frozen target bundle schema differs")
    ids, y = value["node_ids"], value["targets"]
    n = shape["validation_nodes"]
    require(ids.device.type == y.device.type == "cpu" and ids.dtype == y.dtype == torch.int64 and tuple(ids.shape) == tuple(y.shape) == (n,), "Target/ID shape/dtype differs")
    require(ids.unique().numel() == n and int(ids.min()) >= 0 and int(ids.max()) < shape["graph_nodes"], "Validation IDs invalid")
    require(int(y.min()) >= 0 and int(y.max()) < shape["classes"], "Validation class targets invalid")
    return ids, y


def load_degree(torch, entry, shape, ids, budget):
    row = entry["record_copied_without_payload_access"]
    ref = {"path": entry["degree_file_relative_to_phase"], "sha256": row["sha256"], "bytes": row["bytes"]}
    path = checked(ref, budget, row["bytes"])
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", "Degree array is not NPY")
        version = stream.read(2)
        require(version in (b"\x01\x00", b"\x02\x00"), "Unadmitted NPY format")
        size_bytes = 2 if version == b"\x01\x00" else 4
        header_bytes = struct.unpack("<H" if size_bytes == 2 else "<I", stream.read(size_bytes))[0]
        require(0 < header_bytes <= 4096, "NPY header outside bound")
        header = ast.literal_eval(stream.read(header_bytes).decode("latin1"))
        require(header == {"descr": "<i8", "fortran_order": False, "shape": (shape["graph_nodes"],)}, "Exact incoming degree array required")
        raw = stream.read(row["payload_bytes"] + 1)
        require(len(raw) == row["payload_bytes"] == shape["graph_nodes"] * 8, "Degree payload size differs")
    values = array.array("q")
    require(values.itemsize == 8, "Host int64 array required")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    tensor = torch.frombuffer(values, dtype=torch.int64)
    require(int(tensor.min()) >= 0, "Negative incoming degree")
    chosen = tensor.index_select(0, ids)
    budget.check()
    return chosen


def load_decisions(torch, ref, cell, role, epoch, ids, shape, m, budget):
    value = torch.load(checked(ref, budget, MAX_LOGIT_FILE_BYTES), map_location="cpu", weights_only=True)
    require(set(value) == {"schema", "cell", "node_ids", "member_logits", "pooled_logits", "epoch"} and value["schema"] == "gnnm-shared-message-validation-logits-v2" and value["cell"] == cell and value["epoch"] == epoch, "Saved logits schema/cell/epoch differs")
    require(value["node_ids"].dtype == torch.int64 and value["node_ids"].device.type == "cpu" and torch.equal(value["node_ids"], ids), "Logit node order differs")
    member, pooled = value["member_logits"], value["pooled_logits"]
    n, c = shape["validation_nodes"], shape["classes"]
    require(member.device.type == pooled.device.type == "cpu" and member.dtype == pooled.dtype == torch.float32 and tuple(member.shape) == (m, n, c) and tuple(pooled.shape) == (n, c), "CPU float32 complete logits required")
    require(bool(torch.isfinite(member).all()) and bool(torch.isfinite(pooled).all()), "Nonfinite logit value")
    # Casting each member separately preserves the frozen float32 values and
    # argmax tie order, with bounded float64 scratch. Use the saved pooled tensor.
    predictions = torch.stack([member[k].double().argmax(-1) for k in range(m)])
    pooled_prediction = pooled.double().argmax(-1)
    if m == 1:
        require(torch.equal(predictions[0], pooled_prediction), "Single pooled/member decisions differ")
    budget.check()
    return predictions, pooled_prediction


def fraction(count, denominator):
    return count / denominator if denominator else None


def partition(a, b, correct_a, correct_b, mask):
    n = int(mask.sum())
    both = correct_a & correct_b & mask
    lost = correct_a & ~correct_b & mask
    recovered = ~correct_a & correct_b & mask
    wrong = ~correct_a & ~correct_b & mask
    different = a != b
    counts = {"both_correct": int(both.sum()), "lost_correct": int(lost.sum()), "recovered_correct": int(recovered.sum()), "both_wrong_same_class": int((wrong & ~different).sum()), "both_wrong_different_classes": int((wrong & different).sum()), "decision_disagreement": int((mask & different).sum())}
    require(sum(counts[k] for k in counts if k != "decision_disagreement") == n, "Decision partition count identity fails")
    require(counts["decision_disagreement"] == counts["lost_correct"] + counts["recovered_correct"] + counts["both_wrong_different_classes"], "Disagreement identity fails")
    return n, counts


def strata(torch, degree, bins):
    masks = {"overall": torch.ones(degree.shape, dtype=torch.bool, device="cpu")}
    for row in bins:
        mask = degree >= row["lower_inclusive"]
        if row["upper_inclusive"] is not None:
            mask &= degree <= row["upper_inclusive"]
        masks[row["id"]] = mask
    require(sum(int(mask.sum()) for name, mask in masks.items() if name != "overall") == degree.numel(), "Degree bins do not cover validation IDs")
    return masks


def base_row(task, role, seed, stratum, n):
    return {"task": task, "role": role, "seed": seed, "stratum": stratum, "n_nodes": n, "small_stratum": 0 < n < 30, "undefined_reason": "empty_stratum" if n == 0 else None, "validation_only": True, "postlaunch_diagnostic": True}


def arm_tables(torch, task, role, seed, arm, epoch, chosen_member, predictions, pooled, y, masks):
    arms, groups, pairs = [], [], []
    m = predictions.shape[0]
    member_correct = predictions == y.unsqueeze(0)
    pooled_correct = pooled == y
    k_correct = member_correct.sum(0)
    for stratum, mask in masks.items():
        n = int(mask.sum())
        common = {**base_row(task, role, seed, stratum, n), "arm": arm, "epoch": epoch}
        pc = int((pooled_correct & mask).sum())
        zero = mask & (k_correct == 0)
        some = mask & (k_correct >= 1)
        recovery = int((zero & pooled_correct).sum())
        loss = int((some & ~pooled_correct).sum())
        zero_n, some_n = int(zero.sum()), int(some.sum())
        member_counts = [int((member_correct[k] & mask).sum()) for k in range(m)]
        pair_rows = []
        for a in range(m):
            for b in range(a + 1, m):
                _, counts = partition(predictions[a], predictions[b], member_correct[a], member_correct[b], mask)
                exactly_one = counts["lost_correct"] + counts["recovered_correct"]
                both_wrong = counts["both_wrong_same_class"] + counts["both_wrong_different_classes"]
                either_wrong = both_wrong + exactly_one
                pair_counts = {"both_correct": counts["both_correct"], "exactly_one_correct": exactly_one, "both_wrong_same_class": counts["both_wrong_same_class"], "both_wrong_different_classes": counts["both_wrong_different_classes"], "decision_disagreement": counts["decision_disagreement"], "both_wrong": both_wrong, "either_wrong": either_wrong}
                pair_rows.append({**common, "member_a": a, "member_b": b, "raw_counts": pair_counts, "denominators": {"unconditional": n, "wrong_disagreement_given_both_wrong": both_wrong, "error_Jaccard": either_wrong}, "decision_disagreement_fraction": fraction(counts["decision_disagreement"], n), "one_correct_disagreement_fraction": fraction(exactly_one, n), "wrong_disagreement_fraction": fraction(counts["both_wrong_different_classes"], n), "wrong_disagreement_given_both_wrong": fraction(counts["both_wrong_different_classes"], both_wrong), "error_Jaccard": fraction(both_wrong, either_wrong), "conditional_undefined_reasons": {"wrong_disagreement_given_both_wrong": "no_both_wrong_nodes" if both_wrong == 0 else None, "error_Jaccard": "no_either_wrong_nodes" if either_wrong == 0 else None}})
        pair_means = {}
        for key in ("decision_disagreement_fraction", "one_correct_disagreement_fraction", "wrong_disagreement_fraction"):
            pair_means[key] = statistics.mean(row[key] for row in pair_rows) if pair_rows and n else None
        arms.append({**common, "members": m, "raw_counts": {"pooled_correct": pc, "member_correct": member_counts, "pool_correct_all_members_wrong": recovery, "pool_wrong_some_member_correct": loss}, "denominators": {"unconditional": n, "zero_correct_members": zero_n, "some_correct_members": some_n}, "accuracy_percent_context": 100 * pc / n if n else None, "pool_recovery_fraction": fraction(recovery, n), "pool_recovery_given_no_correct_member": fraction(recovery, zero_n), "pool_loss_fraction": fraction(loss, n), "pool_loss_given_some_correct_member": fraction(loss, some_n), "conditional_undefined_reasons": {"pool_recovery_given_no_correct_member": "no_zero_correct_member_nodes" if zero_n == 0 else None, "pool_loss_given_some_correct_member": "no_some_correct_member_nodes" if some_n == 0 else None}, "frozen_selected_member": chosen_member, "frozen_selected_member_correct_count": member_counts[chosen_member], "member_pair_within_seed_means": pair_means, "member_pair_not_applicable_reason": "no_member_pairs" if m == 1 else None})
        local_groups = []
        for k in range(m + 1):
            group = mask & (k_correct == k)
            nk = int(group.sum())
            correct = int((group & pooled_correct).sum())
            local_groups.append({**common, "k_correct_members": k, "raw_counts": {"group_nodes": nk, "pooled_correct": correct}, "denominators": {"group_nodes": nk, "stratum_nodes": n}, "pooled_correct_fraction_given_k": fraction(correct, nk), "undefined_reason": "empty_member_correct_group" if nk == 0 else None})
        require(sum(r["raw_counts"]["group_nodes"] for r in local_groups) == n and sum(r["raw_counts"]["pooled_correct"] for r in local_groups) == pc, "Member-correct group identity fails")
        if m == 1:
            require(recovery == loss == 0 and pc == member_counts[0], "Single-model pooling identity fails")
        groups.extend(local_groups)
        pairs.extend(pair_rows)
    return arms, groups, pairs, pooled_correct


def descriptive_interval(values):
    require(len(values) == 3, "Three paired seeds required")
    if any(x is None for x in values):
        require(all(x is None for x in values), "Seed denominators unexpectedly differ")
        return {"values": values, "mean": None, "sample_sd": None, "two_sided_95_interval": None, "n": 0, "df": None, "undefined_reason": "empty_stratum"}
    require(all(math.isfinite(x) for x in values), "Nonfinite seed quantity")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = 4.302652729911275 * sd / math.sqrt(3)
    return {"values": values, "mean": mean, "sample_sd": sd, "two_sided_95_interval": [mean - half, mean + half], "n": 3, "df": 2, "undefined_reason": None}


def summarize_seeds(rows, plan):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("task", "role", "reference", "candidate", "stratum"))
        groups.setdefault(key, []).append(row)
    output = []
    for key, local in groups.items():
        local.sort(key=lambda r: plan["study_scope"]["seeds"].index(r["seed"]))
        require([r["seed"] for r in local] == plan["study_scope"]["seeds"] and len({r["n_nodes"] for r in local}) == 1, "Complete paired seed group required")
        output.append({**dict(zip(("task", "role", "reference", "candidate", "stratum"), key)), "n_nodes_per_seed": local[0]["n_nodes"], "seeds": [r["seed"] for r in local], "epochs_by_seed": [{"seed": r["seed"], "reference": r["reference_epoch"], "candidate": r["candidate_epoch"]} for r in local], "summaries": {quantity: descriptive_interval([r[quantity] for r in local]) for quantity in plan["seed_summary"]["cross_arm_quantities"]}, "interval_scope": "Descriptive optimizer-seed variation on one graph/split; not simultaneous intervals; approximate normality uncheckable with three seeds.", "validation_only": True, "postlaunch_diagnostic": True})
    return output


def svg_text(x, y, value, size=12, anchor="start", color="#17202a"):
    return '<text x="%s" y="%s" font-family="Arial,sans-serif" font-size="%s" text-anchor="%s" fill="%s">%s</text>' % (x, y, size, anchor, color, html.escape(str(value)))


def svg_circle(x, y, radius, fill, title):
    return '<circle cx="%s" cy="%s" r="%s" fill="%s" stroke="#17202a" stroke-width="0.7"><title>%s</title></circle>' % (x, y, radius, fill, html.escape(title))


def figure_panels(plan):
    return [(task, role) for task in plan["study_scope"]["tasks"] for role in plan["study_scope"]["checkpoint_roles"]]


def figures(tables, plan):
    """Three deterministic SVG figures from the complete aggregate tables only."""
    colors = {"lost": "#c44e52", "recovered": "#2878b5", "useful": "#2878b5", "wrong": "#c44e52"}
    seed_fills = ["#ffffff", "#aab2bd", "#17202a"]
    panel_order = figure_panels(plan)
    pairs = plan["cross_arm_comparisons"]["emphasized_pairs"]
    contrasts = tables["cross_arm_decisions"]
    max_count = max([r["raw_counts"][k] for r in contrasts if r["stratum"] == "overall" for k in ("lost_correct", "recovered_correct")] + [1])
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="650" viewBox="0 0 1240 650">', '<rect width="1240" height="650" fill="white"/>', svg_text(25, 30, "Recovered and lost correct validation decisions", 20), svg_text(25, 52, "Bars: seed means. Points: seeds 17 (white), 29 (grey), 43 (black). Signed count axis is shared across panels.", 12)]
    for index, (task, role) in enumerate(panel_order):
        x0, y0 = 25 + (index % 2) * 610, 85 + (index // 2) * 245
        local = [r for r in contrasts if r["task"] == task and r["role"] == role and r["stratum"] == "overall"]
        center, scale = x0 + 320, 245 / max_count
        out.append(svg_text(x0, y0, "%s — %s; validation nodes %d" % (task, role, local[0]["n_nodes"]), 15))
        out.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="#9ca3af"/>' % (center, y0 + 20, center, y0 + 175))
        for j, pair in enumerate(pairs):
            selected = sorted([r for r in local if r["reference"] == pair["reference"] and r["candidate"] == pair["candidate"]], key=lambda r: r["seed"])
            require(len(selected) == 3, "Figure lacks paired seed contrast")
            y = y0 + 55 + j * 80
            label = "shared versus exact" if j == 0 else "detached versus shared"
            out.append(svg_text(x0, y - 22, label, 13))
            for key, sign, color in (("lost_correct", -1, colors["lost"]), ("recovered_correct", 1, colors["recovered"])):
                vals = [r["raw_counts"][key] for r in selected]
                width = statistics.mean(vals) * scale
                out.append('<rect x="%s" y="%s" width="%s" height="16" fill="%s" opacity="0.75"/>' % (center - width if sign == -1 else center, y - 8, width, color))
                for i, value in enumerate(vals):
                    out.append(svg_circle(center + sign * value * scale, y - 4 + i * 4, 3, seed_fills[i], "%s seed%d: %d" % (key, selected[i]["seed"], value)))
                out.append(svg_text(center + sign * 255, y + 5, "%.1f" % statistics.mean(vals), 11, "end" if sign == -1 else "start", color))
        out.append(svg_text(center - 240, y0 + 200, "lost", 12, "start", colors["lost"]))
        out.append(svg_text(center + 240, y0 + 200, "recovered", 12, "end", colors["recovered"]))
    out.append(svg_text(25, 620, "Validation-only descriptive diagnostic planned after study launch; all ten comparisons are retained in the tables.", 12))
    out.append("</svg>")
    result = {"lost_recovered.svg": "\n".join(out)}
    arm_rows = tables["arm_summary"]
    metric_keys = ("one_correct_disagreement_fraction", "wrong_disagreement_fraction")
    max_frac = max([r["member_pair_within_seed_means"][key] for r in arm_rows if r["stratum"] == "overall" and r["members"] == 4 for key in metric_keys] + [0.01])
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="730" viewBox="0 0 1240 730">', '<rect width="1240" height="730" fill="white"/>', svg_text(25, 30, "Member disagreement separated by correctness", 20), svg_text(25, 52, "Blue: exactly one member correct. Red: both members wrong with different predictions. Six pairs averaged within each seed.", 12)]
    for index, (task, role) in enumerate(panel_order):
        x0, y0 = 25 + (index % 2) * 610, 85 + (index // 2) * 285
        out.append(svg_text(x0, y0, task + " — " + role, 15))
        for j, arm in enumerate(plan["study_scope"]["arms"][:3]):
            selected = sorted([r for r in arm_rows if r["task"] == task and r["role"] == role and r["arm"] == arm and r["stratum"] == "overall"], key=lambda r: r["seed"])
            y = y0 + 40 + j * 70
            out.append(svg_text(x0, y - 8, arm.replace("_mean_message", "").replace("_member", ""), 12))
            for k, (key, color) in enumerate(zip(metric_keys, (colors["useful"], colors["wrong"]))):
                vals = [r["member_pair_within_seed_means"][key] for r in selected]
                xx = x0 + 150
                yy = y + k * 22
                out.append('<rect x="%s" y="%s" width="%s" height="14" fill="%s" opacity="0.75"/>' % (xx, yy - 8, statistics.mean(vals) / max_frac * 365, color))
                for i, value in enumerate(vals):
                    out.append(svg_circle(xx + value / max_frac * 365, yy - 4 + i * 4, 3, seed_fills[i], "seed%d: %.6f" % (selected[i]["seed"], value)))
                out.append(svg_text(x0 + 570, yy + 4, "%.2f%%" % (statistics.mean(vals) * 100), 11, "end", color))
    out.append(svg_text(25, 705, "All points use optimizer seeds; nodes and member pairs are not independent repetitions. A correct member does not establish a pooling benefit.", 12))
    out.append("</svg>")
    result["member_disagreement_by_correctness.svg"] = "\n".join(out)
    summaries = tables["seed_summary"]
    all_values = [v for row in summaries for v in row["summaries"]["net_accuracy_change_pp"]["values"] if v is not None]
    limit = max([abs(v) for v in all_values] + [0.01])
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1480" height="1270" viewBox="0 0 1480 1270">', '<rect width="1480" height="1270" fill="white"/>', svg_text(20, 30, "Degree groups: candidate minus reference accuracy (percentage points)", 20), svg_text(20, 52, "Numbers: means over three seeds. Dots: individual seeds 17/29/43. Global symmetric color/point scale: ±%.3f pp." % limit, 12)]
    labels = {"exact_member": "exact", "shared_mean_message": "shared", "detached_mean_message": "detached", "single_base64": "single64", "single_parameters": "singleParam"}
    bins = plan["degree_analysis"]["bins"]
    bin_labels = ["0", "1–4", "5–16", "17–64", "65–256", "257+"]
    for index, (task, role) in enumerate(panel_order):
        x0, y0 = 20 + (index % 2) * 735, 90 + (index // 2) * 560
        out.append(svg_text(x0, y0, task + " — " + role, 16))
        for col, b in enumerate(bins):
            local = next(r for r in summaries if r["task"] == task and r["role"] == role and r["stratum"] == b["id"])
            n = local["n_nodes_per_seed"]
            out.append(svg_text(x0 + 180 + col * 88 + 42, y0 + 26, bin_labels[col], 12, "middle"))
            out.append(svg_text(x0 + 180 + col * 88 + 42, y0 + 42, "n=%d%s" % (n, " small" if 0 < n < 30 else ""), 9, "middle"))
        for row_index, pair in enumerate(plan["cross_arm_comparisons"]["all_pairs"]):
            y = y0 + 52 + row_index * 43
            out.append(svg_text(x0, y + 23, labels[pair["candidate"]] + " − " + labels[pair["reference"]], 12))
            for col, b in enumerate(bins):
                cell = next(r for r in summaries if r["task"] == task and r["role"] == role and r["reference"] == pair["reference"] and r["candidate"] == pair["candidate"] and r["stratum"] == b["id"])
                stat = cell["summaries"]["net_accuracy_change_pp"]
                mean = stat["mean"]
                x = x0 + 180 + col * 88
                if mean is None:
                    color = "#e5e7eb"
                else:
                    intensity = min(abs(mean) / limit, 1)
                    destination = (40, 120, 181) if mean >= 0 else (196, 78, 82)
                    rgb = tuple(round(255 + intensity * (v - 255)) for v in destination)
                    color = "#%02x%02x%02x" % rgb
                out.append('<rect x="%s" y="%s" width="84" height="39" fill="%s" stroke="#d1d5db"/>' % (x, y, color))
                out.append(svg_text(x + 42, y + 16, "NA" if mean is None else "%+.3f" % mean, 11, "middle"))
                if mean is not None:
                    for i, value in enumerate(stat["values"]):
                        out.append(svg_circle(x + 42 + value / limit * 30, y + 27 + (i - 1) * 3, 2.4, seed_fills[i], "seed%d: %+.6f pp" % (cell["seeds"][i], value)))
        out.append(svg_text(x0, y0 + 507, "Incoming raw edge-entry degree; duplicates/existing self-edges retained. Empty groups: NA.", 10))
    out.append(svg_text(20, 1230, "Degree associations are descriptive, not causal. Final and validation-selected checkpoint roles remain separate.", 12))
    out.append(svg_text(20, 1250, "The CSV/JSON atlas includes the overall split, all raw counts, seed values and non-simultaneous descriptive intervals.", 12))
    out.append("</svg>")
    result["degree_signed_map.svg"] = "\n".join(out)
    return result


def write_csv(path, rows):
    require(rows, "Empty table unexpectedly generated")
    keys = list(rows[0])
    require(all(set(r) == set(keys) for r in rows), "CSV row fields differ")
    with confined(path, HERE).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (list, dict)) else ("NA" if value is None else value) for key, value in row.items()})


def analyze(torch, consumer, family, inputs, plan, budget):
    tables = {key: [] for key in plan["output_contract"]["expected_rows"]}
    lookup = {(r["cell"]["task"], r["cell"]["seed"], r["cell"]["arm"]): r for r in consumer["rows"]}
    locked_lookup = {(r["task"], r["seed"], r["arm"]): r for r in family["rows"]}
    for task in plan["study_scope"]["tasks"]:
        shape = inputs["task_shapes"][task]
        ids, y = load_targets(torch, consumer["validation_targets"][task], task, shape, budget)
        degree = load_degree(torch, inputs["degree_inputs"][task], shape, ids, budget)
        masks = strata(torch, degree, plan["degree_analysis"]["bins"])
        for role in plan["study_scope"]["checkpoint_roles"]:
            for seed in plan["study_scope"]["seeds"]:
                block = {}
                for arm in plan["study_scope"]["arms"]:
                    row = lookup[(task, seed, arm)]
                    locked = locked_lookup[(task, seed, arm)]
                    epoch = 300 if role == "final" else row["selected_epoch"]
                    m = plan["study_scope"]["members"][arm]
                    member, pooled = load_decisions(torch, row["logits"][role], row["cell"], role, epoch, ids, shape, m, budget)
                    arm_rows, group_rows, pair_rows, correct = arm_tables(torch, task, role, seed, arm, epoch, row["selected_member"], member, pooled, y, masks)
                    context = locked["final_validation" if role == "final" else "secondary_validation"]
                    overall = arm_rows[0]
                    require(overall["stratum"] == "overall" and abs(overall["accuracy_percent_context"] - context["accuracy_percent"]) <= 1e-10, "Stored decision counts disagree with locked accuracy")
                    require(context["nodes"] == shape["validation_nodes"] and context["members"] == m, "Locked dimensions differ")
                    tables["arm_summary"].extend(arm_rows)
                    tables["correct_member_count"].extend(group_rows)
                    tables["member_pair"].extend(pair_rows)
                    block[arm] = {"pooled": pooled, "correct": correct, "epoch": epoch}
                    del member
                    budget.check()
                for pair in plan["cross_arm_comparisons"]["all_pairs"]:
                    a, b = block[pair["reference"]], block[pair["candidate"]]
                    for stratum, mask in masks.items():
                        n, counts = partition(a["pooled"], b["pooled"], a["correct"], b["correct"], mask)
                        difference = int((b["correct"] & mask).sum()) - int((a["correct"] & mask).sum())
                        require(difference == counts["recovered_correct"] - counts["lost_correct"], "Net correctness identity fails")
                        tables["cross_arm_decisions"].append({**base_row(task, role, seed, stratum, n), **pair, "reference_epoch": a["epoch"], "candidate_epoch": b["epoch"], "raw_counts": counts, "denominators": {key: n for key in counts}, "fractions": {key: fraction(value, n) for key, value in counts.items()}, "net_accuracy_change_pp": 100 * difference / n if n else None, "lost_correct_fraction": fraction(counts["lost_correct"], n), "recovered_correct_fraction": fraction(counts["recovered_correct"], n), "decision_disagreement_fraction": fraction(counts["decision_disagreement"], n)})
                del block
                budget.check()
        del ids, y, degree, masks
    tables["seed_summary"] = summarize_seeds(tables["cross_arm_decisions"], plan)
    require({key: len(rows) for key, rows in tables.items()} == plan["output_contract"]["expected_rows"], "Complete atlas row count differs")
    for task in plan["study_scope"]["tasks"]:
        for stratum in ["overall"] + [b["id"] for b in plan["degree_analysis"]["bins"]]:
            require(len({r["n_nodes"] for r in tables["arm_summary"] if r["task"] == task and r["stratum"] == stratum}) == 1, "Structural group changed across arms/seeds/roles")
    return tables


def run(request, request_ref, plan, source, live, output):
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(live, output)
    write_json(output / "START.json", {"schema": "gnnm-shared-message-decision-analysis-start-v1", "request": request_ref, "source_manifest": request["source_manifest"], "plan_seal_sha256": PLAN_SEAL_SHA, "live": live, "postlaunch_prospective_diagnostic_not_preregistered_primary": True, "test_targets_loaded": False, "training_or_inference_or_GPU_operations": False})
    result_names = {"AGGREGATES.json", "RESULT.json", "artifact_manifest.json", "lost_recovered.svg", "member_disagreement_by_correctness.svg", "degree_signed_map.svg"} | {name + ".csv" for name in plan["output_contract"]["expected_rows"]}
    try:
        budget.check()
        consumer, family, inputs, payload_refs, metadata_refs = verify_consumer(request, plan, budget)
        require(importlib.metadata.version("torch") == RUNTIME["torch_version"], "Pinned Torch distribution missing")
        # This is the only scientific import, strictly after every admission gate.
        import torch
        torch.set_num_threads(RUNTIME["CPU_threads"])
        before_runtime = runtime_receipt(torch)
        budget.check()
        with torch.inference_mode():
            tables = analyze(torch, consumer, family, inputs, plan, budget)
        # Rehash all64 inputs after consumption; no duplicate tensor loads.
        for ref in payload_refs:
            checked(ref, budget)
        for ref in metadata_refs + [request_ref, request["source_manifest"], request["implementation_review"], request["resource_approval"], request["root_approval"]]:
            checked(ref, budget)
        verify_plan_and_source(request)
        after_runtime = runtime_receipt(torch)
        require(before_runtime == after_runtime, "Runtime import identity changed")
        budget.check()
        rendered = figures(tables, plan)
        write_json(output / "AGGREGATES.json", {"schema": "gnnm-shared-message-validation-decision-atlas-v1", "tables": tables, "plan_seal_sha256": PLAN_SEAL_SHA, "postlaunch_diagnostic": True, "preregistered_primary_endpoint": False, "test_targets_loaded": False, "new_NLL_scoring_or_learning_or_inference": False, "causal_or_unseen_graph_generalization_established": False, "automatic_scientific_promotion_authorized": False})
        for name, rows in tables.items():
            write_csv(output / (name + ".csv"), rows)
            budget.check()
        for name, text in rendered.items():
            with confined(output / name, HERE).open("x") as stream:
                stream.write(text)
            budget.check()
        write_json(output / "RESULT.json", {"schema": "gnnm-shared-message-decision-analysis-end-v1", "status": "complete_pending_successful_outer_terminal_and_root_audit", "request": request_ref, "live": live, "consumer_audit": request["consumer_audit"], "family_lock": consumer["family_lock"], "plan_seal_sha256": PLAN_SEAL_SHA, "source_manifest": request["source_manifest"], "implementation_review": request["implementation_review"], "resource_approval": request["resource_approval"], "runtime": after_runtime, "row_counts": {key: len(rows) for key, rows in tables.items()}, "binary_inputs": payload_refs, "all_binary_inputs_hashed_before_loading_and_after_analysis": True, "postlaunch_diagnostic_not_primary_preregistration": True, "test_targets_loaded": False, "provider_or_checkpoint_or_training_target_accessed": False, "new_learning_or_inference_or_GPU_calls": False, "automatic_scientific_promotion_authorized": False, "qualification": "A normal zero exit, successful outer/inner terminals and root integrity audit remain required."})
        inventory = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p, budget)} for p in sorted(output.iterdir()) if p.is_file()}
        write_json(output / "artifact_manifest.json", {"schema": "gnnm-shared-message-decision-analysis-artifacts-v1", "files": inventory})
        budget.check()
        print(json.dumps({"status": "complete_pending_outer_and_root_audit", "row_counts": {k: len(v) for k, v in tables.items()}, "output": str(output.relative_to(PHASE))}), flush=True)
    except BaseException as error:
        # Remove only this process's named aggregate products on handled failure.
        # A forced kill may leave unqualified forensic files; no success receipt
        # or failed outer terminal is admissible as scientific completion.
        for name in result_names:
            path = output / name
            if path.is_file() and not path.is_symlink():
                path.unlink()
        if not (output / "FAILURE.json").exists():
            write_json(output / "FAILURE.json", {"schema": "gnnm-shared-message-decision-analysis-failure-v1", "exception": type(error).__name__, "reason": str(error)[:500], "request": request_ref, "partial_scientific_summary_qualified": False, "automatic_retry_authorized": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    args = parser.parse_args()
    request, request_ref, plan, source, live, output = admission(args.request, args.supervisor)
    run(request, request_ref, plan, source, live, output)


if __name__ == "__main__":
    main()
