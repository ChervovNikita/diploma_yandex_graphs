"""Stdlib-only fixed-family and actual source/input/runtime/resource admission."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
TASKS = ("Reddit2", "ogbn-products")
ARMS = ("exact_member", "shared_mean_message", "detached_mean_message", "single_base64", "single_parameters")
SEEDS = (17, 29, 43)
EPOCHS = 300
MODEL_ARMS = dict(exact_member="exact_member", shared_mean_message="shared_mean",
                  detached_mean_message="detached_shared_mean", single_base64="single_D64",
                  single_parameters="single_matched")
MODEL_MANIFEST_SHA = "607c487f83696a21d5792c3f22b408491f9fb55c8f7e0b1dd1527b8f68653dcc"
BOUNDED_SOURCE_SHA = "b096966c1570da0a998dbb9fb464a9ad71490b74d4ba3406b751390637d19a23"
DESIGN = {"path": "shared_message_resource_amendment_v3/PLAN_ROOT_before_validation_v3.json",
          "sha256": "66b242e582c61205e19264cdcaad2d17ad8e7611f39706b9b1ba648b68e0f198"}
ORDER_POLICY = {"path": "shared_message_study_design_v1/RNG_MEMBER_ORDER_ROOT_before_validation_v1.json",
                "sha256": "0aecc212def571291e4f2e842fbc50c31b14edb5dfb4ae0490ee9eceadf3dcc1"}
CAP_GPU = 21600
FORECAST_CAP = 18000
CAP_MEMORY = 60 * 2**30
CAP_DISK = 200 * 2**30
TIMING = {"warmups": 5, "samples": 10, "execution": "frozen_deployment", "order": "single_cell_isolated"}
_CACHE = {}
_HASH_STATS = {"files": 0, "bytes": 0, "unchanged_signature_reuses": 0, "wall_seconds": 0.0}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def within(path, root=PHASE):
    root = Path(root).resolve()
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(root):
        raise ValueError("Outside declared project phase")
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Evidence/output symlinks forbidden")
    return path.resolve()


def sha(path):
    path = within(path)
    before = path.stat()
    signature = tuple(getattr(before, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
    if str(path) in _CACHE:
        previous, result = _CACHE[str(path)]
        if signature != previous:
            raise ValueError("Previously verified file changed")
        _HASH_STATS["unchanged_signature_reuses"] += 1
        return result
    start = time.perf_counter()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(block)
    after = path.stat()
    if signature != tuple(getattr(after, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("File changed during hash")
    result = h.hexdigest()
    _CACHE[str(path)] = signature, result
    _HASH_STATS["files"] += 1
    _HASH_STATS["bytes"] += before.st_size
    _HASH_STATS["wall_seconds"] += time.perf_counter() - start
    return result


def hash_report():
    return {**_HASH_STATS, "cache_scope": "process; first full byte hash then unchanged stat signature", "adversarial_integrity_guarantee": False}


def checked(ref):
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Exact phase-relative path/SHA reference required")
    path = within(PHASE / ref["path"])
    if str(path.relative_to(PHASE)) != ref["path"] or not path.is_file() or sha(path) != ref["sha256"]:
        raise ValueError("Unqualified/noncanonical/changed reference")
    return path


def reference(path):
    path = within(path)
    return {"path": str(path.relative_to(PHASE)), "sha256": sha(path)}


def write(path, value):
    path = within(path, HERE)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def cells():
    return [{"task": t, "seed": s, "arm": a} for t in TASKS for s in SEEDS for a in ARMS]


def cell_id(cell):
    if cell not in cells():
        raise ValueError("Cell outside the new fixed30 family")
    return ("reddit2" if cell["task"] == "Reddit2" else "products") + "_seed%d_%s" % (cell["seed"], cell["arm"])


def prescribed(admitted, stage, cell=None, task=None):
    root = within(PHASE / admitted["output_root"], HERE)
    if stage in ("fits", "replays"):
        return within(root / stage / cell_id(cell), HERE)
    if stage == "targets" and task in TASKS:
        return within(root / "targets" / ("reddit2" if task == "Reddit2" else "products"), HERE)
    if stage == "family_lock":
        return within(root / stage, HERE)
    raise ValueError("Unprescribed output stage; no test stage exists")


def source_check():
    value = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    if value.get("schema") != "gnnm-shared-message-predictive-source-bindings-v2":
        raise ValueError("New source closure is not sealed")
    for path, wanted in value["sha256"].items():
        if sha(PHASE / path) != wanted:
            raise ValueError("Sealed source/dependency changed: " + path)
    for path in HERE.glob("*.py"):
        if str(path.relative_to(PHASE)) not in value["sha256"]:
            raise ValueError("Unsealed local Python import possibility")
    from .import_guard import verify_packet
    verify_packet(sha(HERE / "SOURCE_MANIFEST_v1.json"))
    return value


def runtime_import_receipt(*, require_generated=False):
    from .import_guard import (verify_packet, imported_modules,
        verify_imported_receipts, verify_generator_installation)
    policy, tree = verify_packet(sha(HERE / "SOURCE_MANIFEST_v1.json"))
    rows = imported_modules(policy)
    verify_imported_receipts(rows, policy, require_generated=require_generated)
    return dict(runtime_import_tree=tree, imported_phase_modules=rows,
        generated_module_policy_sha256=policy["generated_module_policy_sha256"],
        installed_generated_code_sources=verify_generator_installation(policy))


def audit_runtime_import_receipt(value, *, require_generated=False):
    from .import_guard import verify_packet, verify_imported_receipts, verify_generator_installation
    policy, tree = verify_packet(sha(HERE / "SOURCE_MANIFEST_v1.json"))
    if (value.get("runtime_import_tree") != tree
            or value.get("generated_module_policy_sha256") != policy["generated_module_policy_sha256"]
            or value.get("installed_generated_code_sources") != verify_generator_installation(policy)):
        raise ValueError("Exact scientific runtime import/source receipt changed")
    verify_imported_receipts(value["imported_phase_modules"], policy, require_generated=require_generated)


def completed(directory, outer_start_ref):
    """Exact corrected inner chain plus its successful bounded outer terminal."""
    from shared_message_quality_v3.contract import completed as audit_completed
    result = audit_completed(directory)
    start_path = checked(outer_start_ref)
    start = json.loads(start_path.read_text())
    terminal_path = start_path.parent / "TERMINAL.json"
    terminal = json.loads(terminal_path.read_text())
    verify_outer_start(start_path, checked(start["root_request"]), directory, result["argv"])
    names = {"authorization.json", "completion.json", "command.json", "environment.json", "stdout_stderr.log"}
    if (terminal.get("schema") != "gnnm-whole-process-bound-terminal-v1" or terminal.get("START_sha256") != outer_start_ref["sha256"]
            or terminal.get("complete") is not True or terminal.get("root_request_unchanged") is not True
            or terminal.get("child_exit_code") != 0 or terminal.get("timed_out") is not False
            or terminal.get("interruption") is not None or terminal.get("within_whole_cap") is not True
            or type(terminal.get("whole_supervised_seconds")) not in (int, float)
            or not math.isfinite(terminal["whole_supervised_seconds"])
            or not 0 <= terminal["whole_supervised_seconds"] <= start["whole_cap_seconds"]
            or terminal.get("stdout_stderr_sha256") != sha(start_path.parent / "stdout_stderr.log")
            or set(terminal.get("inner_evidence", {})) != names):
        raise ValueError("Unsuccessful/unrelated whole-process terminal")
    for name in names:
        file = within(Path(directory) / name)
        if terminal["inner_evidence"][name] != {"sha256": sha(file), "bytes": file.stat().st_size}:
            raise ValueError("Bounded outer/inner evidence differs")
    return {**result, "inner_seconds": result["seconds"], "seconds": terminal["whole_supervised_seconds"],
            "live": {**result["live"], "gpu_uuid": result["gpu_uuid"], "outer_start": outer_start_ref},
            "outer_start": outer_start_ref, "outer_terminal": reference(terminal_path)}


def verify_outer_start(path, request_path, supervisor, argv):
    start = json.loads(within(path).read_text())
    route = json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
    flag = "--request" if "--request" in argv else "--binding"
    expected = [argv[0], str(PHASE / "protocols/run_authorized_v2.py"), "--output", str(within(supervisor)), "--require-idle-gpu", "--", *argv]
    if (Path(path).name != "START.json" or start.get("schema") != "gnnm-whole-process-bound-start-v1"
            or start.get("root_request") != reference(request_path) or start.get("root_request_child_flag") != flag
            or argv.count(flag) != 1 or Path(argv[argv.index(flag) + 1]).resolve() != within(request_path)
            or argv.count("--supervisor") != 1 or Path(argv[argv.index("--supervisor") + 1]).resolve() != within(supervisor)
            or start.get("child_argv") != argv or start.get("argv") != expected
            or start.get("inner_supervisor_directory") != str(within(supervisor)) or start.get("outer_output") != str(Path(path).parent)
            or start.get("cwd") != route["repository"] or start.get("phase") != str(PHASE)
            or start.get("ssh_destination") != route["ssh_destination"]
            or start.get("shell") is not False or start.get("start_new_session") is not True
            or type(start.get("whole_cap_seconds")) not in (int, float) or not math.isfinite(start["whole_cap_seconds"])
            or not 0 < start.get("termination_grace_seconds", 0) < start["whole_cap_seconds"] <= CAP_GPU
            or start.get("normal_child_budget_seconds") != start["whole_cap_seconds"] - start["termination_grace_seconds"]
            or start.get("source_sha256") != BOUNDED_SOURCE_SHA
            or sha(PHASE / "protocols/bounded_run_v1.py") != BOUNDED_SOURCE_SHA):
        raise ValueError("Exact source-bound whole-process START required")
    for key, source_path in (("authorization_source_sha256", PHASE / "protocols/run_authorized_v2.py"),
                      ("logging_source_sha256", PHASE / "protocols/run_logged.py"),
                      ("wrapper_sha256", PHASE / "protocols/repo_env.sh"),
                      ("research_script_sha256", Path(argv[1]))):
        if start.get(key) != sha(source_path):
            raise ValueError("Bounded process source linkage differs")
    if datetime.fromisoformat(start["start_UTC"]).tzinfo is None:
        raise ValueError("Bound start needs timezone")
    return reference(path), start


def live(supervisor, entrypoint, request_path, admitted):
    from propagation_cost_engineering_v1.contract import live_context
    context = live_context(supervisor, entrypoint)
    argv = json.loads((Path(supervisor) / "command.json").read_text())["argv"]
    if argv[0] != str(Path(json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())["repository"]) / ".venv/bin/python"):
        raise ValueError("Exact repository interpreter required")
    outer_ref, outer = verify_outer_start(within(os.environ["GNNM_BOUND_START_JSON"]), request_path, supervisor, argv)
    if "per_child_wall_cap_seconds" in admitted:
        cap = admitted["per_child_wall_cap_seconds"]
    else:
        cap = min(admitted["whole_process_policy"]["per_child_wall_cap_seconds"], CAP_GPU - prior_seconds(admitted))
    if outer["whole_cap_seconds"] > cap:
        raise ValueError("Outer deadline exceeds approved child/remaining family wall budget")
    import subprocess
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    rows = [[part.strip() for part in line.split(",")] for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][0] != "0" or rows[0][1] != admitted["gpu_uuid"] or rows[0][2] != "NVIDIA A100-SXM4-80GB" or int(rows[0][3]) != 81920:
        raise ValueError("Wrong actual sole-GPU allocation")
    return {**context, "gpu_uuid": rows[0][1], "outer_start": outer_ref}


def runtime_before_import(admitted):
    policy = admitted["runtime_policy"]
    if (os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != policy["CUBLAS_WORKSPACE_CONFIG"]
            or policy["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8")):
        raise ValueError("Frozen CUDA workspace/cache policy required before scientific imports")


def quality_gate(row, admitted, name):
    """Resolve exact quality-v3 request/result/source/outer/artifact evidence."""
    from shared_message_quality_v3 import contract as quality
    from shared_message_quality_v3.counts import plan, variants, variant_id
    from shared_message_quality_v3.import_guard import (verify_packet,
        verify_imported_receipts, verify_generator_installation)
    if sha(PHASE / "shared_message_quality_v3/SOURCE_MANIFEST_v1.json") != MODEL_MANIFEST_SHA:
        raise ValueError("Exact numerically qualified model source generation required")
    policy, tree = verify_packet(MODEL_MANIFEST_SHA)
    common = dict(gpu_uuid=admitted["gpu_uuid"], runtime_policy=admitted["runtime_policy"],
        equivalence_tolerance=admitted["execution_equivalence_policy"], memory_cap_bytes=CAP_MEMORY)
    if name == "synthetic_numerical":
        quality.synthetic_chain(dict(common, synthetic_request=row["request"],
            synthetic_gate=row["report"], synthetic_supervisor_directory=row["supervisor_directory"],
            synthetic_outer_terminal=row["outer_terminal"]), MODEL_MANIFEST_SHA)
        return
    request_path = checked(row["request"])
    request, request_sha = quality.request(request_path, "real")
    report_path = checked(row["report"])
    output = quality.within(PHASE / request["output_relative_to_phase"], quality.HERE)
    supervisor = within(PHASE / row["supervisor_directory"])
    chain = quality.completed(supervisor)
    argv = quality.expected_cli(request_path, "real", supervisor)
    outer = quality.outer_completed(row["outer_terminal"], request_path, request, supervisor, argv)
    live = dict(chain["live"], outer_start=outer)
    report = json.loads(report_path.read_text())
    start = json.loads((output / "START.json").read_text())
    if (report_path != output / "RESULT.json" or chain["argv"] != argv
            or report.get("schema") != "gnnm-shared-message-complete-training-qualification-v3"
            or report.get("status") != "complete_engineering_paths"
            or report.get("source_manifest_sha256") != MODEL_MANIFEST_SHA
            or request.get("source_manifest_sha256") != MODEL_MANIFEST_SHA
            or report.get("request_sha256") != request_sha or report.get("live") != live
            or start.get("live") != live or start.get("operation") != "real"
            or start.get("request_sha256") != request_sha or start.get("source_manifest_sha256") != MODEL_MANIFEST_SHA
            or any(report.get(k) is not False for k in ("heldout_targets_loaded", "quality_scores_computed",
                "public_provider_storage_opened", "tuned_weights_or_predictions_saved", "scientific_training_admitted"))
            or report.get("gpu", {}).get("uuid") != admitted["gpu_uuid"]
            or chain["gpu_uuid"] != admitted["gpu_uuid"] or request.get("gpu_uuid") != admitted["gpu_uuid"]
            or report.get("runtime_policy") != admitted["runtime_policy"]
            or report.get("observed_runtime_policy") != admitted["runtime_policy"]
            or request.get("runtime_policy") != admitted["runtime_policy"]
            or report.get("equivalence_tolerance") != admitted["execution_equivalence_policy"]
            or request.get("equivalence_tolerance") != admitted["execution_equivalence_policy"]
            or report.get("synthetic_admission_binding") != request["_synthetic_chain"]
            or report.get("runtime_import_tree") != tree
            or report.get("generated_module_policy_sha256") != policy["generated_module_policy_sha256"]
            or report.get("installed_generated_code_sources") != verify_generator_installation(policy)):
        raise ValueError("Complete model qualification request/source/runtime/outer chain differs")
    verify_imported_receipts(report["imported_phase_modules"], policy)
    tasks = report["tasks"]
    if [r["task"] for r in tasks] != list(TASKS):
        raise ValueError("Both full graphs required in the model qualification result")
    expected_files = {"START.json", "RESULT.json"}
    for task_row in tasks:
        task = task_row["task"]
        rows = task_row["variants"]
        if (task_row.get("plan") != plan(task) or task_row.get("every_arm_has_qualified_path") is not True
                or task_row.get("complete_data_binding", {}).get("acquisition_sha256") != admitted["tasks"][task]["acquisition"]["sha256"]
                or request["engineering_phases"][task] != admitted["tasks"][task]["engineering_phase"]
                or len(rows) != len(variants()) or {r["id"] for r in rows} != {variant_id(v) for v in variants()}):
            raise ValueError("Full task inputs/parameter/variant family differs")
        for variant_row in rows:
            filename = ("reddit2" if task == "Reddit2" else "products") + "__" + variant_row["id"] + ".json"
            expected_files.add(filename)
            if json.loads((output / filename).read_text()) != variant_row:
                raise ValueError("Complete task variant evidence differs")
        for arm, policy_row in admitted["execution_policy"][task].items():
            for execution in {policy_row[k] for k in ("training", "validation", "deployment")}:
                matching = [r for r in rows if r["variant"] == dict(arm=MODEL_ARMS[arm],
                    execution=execution, checkpoint_blocks=policy_row["checkpoint_blocks"])]
                if (len(matching) != 1 or matching[0].get("status") != "qualified_bounded_training_path"
                        or matching[0].get("peak_memory", {}).get("peak_reserved_bytes", math.inf) > CAP_MEMORY
                        or matching[0].get("runtime_parameter_count") != plan(task)["parameter_counts"][MODEL_ARMS[arm]]):
                    raise ValueError("Scientific execution is absent from a qualified complete training path")
    artifact = json.loads((output / "artifact_manifest.json").read_text())
    if (artifact.get("schema") != "gnnm-shared-message-engineering-artifacts-v3"
            or set(artifact.get("files", {})) != expected_files
            or {p.name for p in output.iterdir()} != expected_files | {"artifact_manifest.json"}):
        raise ValueError("Exact complete qualification artifact file set required")
    for filename in expected_files:
        file = within(output / filename)
        if artifact["files"][filename] != dict(bytes=file.stat().st_size, sha256=sha(file)):
            raise ValueError("Complete qualification artifact changed")


def qualified_gates(value):
    expected = {"source_audit", "protocol_audit", "synthetic_numerical", "runtime_freeze",
                "execution_equivalence", "complete_training_Reddit2", "complete_training_ogbn-products",
                "complete30_resources", "validation_export_source_audit"}
    if set(value["qualified_gates"]) != expected:
        raise ValueError("Every new complete-study qualification gate required")
    supervised = {"synthetic_numerical", "execution_equivalence", "complete_training_Reddit2", "complete_training_ogbn-products"}
    for name, row in value["qualified_gates"].items():
        report = checked(row["report"])
        if row.get("qualified") is not True:
            raise ValueError("Unqualified gate: " + name)
        if name in supervised:
            quality_gate(row, value, name)
        if row.get("report_sha256") != sha(report):
            raise ValueError("Gate report binding differs")


def binding(path):
    value = json.loads(within(path, HERE).read_text())
    if (value.get("schema") != "gnnm-shared-message-fixed30-admission-v2"
            or value.get("scientific_training_authorized") is not True
            or value.get("validation_target_export_authorized") is not True
            or value.get("test_target_export_or_evaluation_authorized") is not False
            or value.get("cells") != cells() or value.get("model_arm_adapters") != MODEL_ARMS
            or value.get("design") != DESIGN or value.get("rng_member_order_policy") != ORDER_POLICY):
        raise ValueError("Exact new root-approved fixed30 admission required; old protocols cannot admit it")
    source_check()
    if (value.get("source_manifest_sha256") != sha(HERE / "SOURCE_MANIFEST_v1.json")
            or value["source_bindings_sha256"] != sha(HERE / "SOURCE_BINDINGS.json")):
        raise ValueError("Source closure is not root-bound")
    design = json.loads(checked(DESIGN).read_text())
    order = json.loads(checked(ORDER_POLICY).read_text())
    if value.get("fit_order") != order["fit_order"] or sorted(map(digest, order["fit_order"])) != sorted(map(digest, cells())):
        raise ValueError("Root prescribed all30 outcome-independent cells in an exact order")
    if (value["recipe"] != design["recipe"] or value["validation_policy"] != design["validation_policy"]
            or value["quality_screens"] != design["quality_screens"] or value["epochs_by_task"] != {t: EPOCHS for t in TASKS}):
        raise ValueError("Fixed 9,000 updates, primary final epoch and practical screens are immutable")
    approval = json.loads(checked(value["root_approval"]).read_text())
    if (approval.get("schema") != "gnnm-shared-message-fixed30-root-approval-v2"
            or approval.get("approved_payload_sha256") != digest({k: v for k, v in value.items() if k != "root_approval"})
            or approval.get("qualified") is not True or approval.get("all_complete_inputs_and_gates_qualified") is not True
            or approval.get("validation_export_authorized") is not True or approval.get("test_access_authorized") is not False):
        raise ValueError("Actual full inputs/runtime/execution/resources are not root-approved")
    qualified_gates(value)
    ledger = json.loads(checked(value["resource_ledger"]).read_text())
    required_costs = {"training_updates", "validation_forwards", "CPU_metrics", "device_transfers",
                      "checkpoint_writes", "logit_writes", "hashes", "selected_final_replays", "trained_timings",
                      "validation_export", "input_loading_and_operator_setup", "source_and_artifact_audits"}
    if (ledger.get("qualified") is not True or ledger.get("fit_count") != 30 or ledger.get("updates") != 9000
            or set(ledger.get("included_cost_components", [])) != required_costs
            or ledger.get("total_GPU_wall_seconds_forecast", math.inf) > FORECAST_CAP
            or ledger.get("peak_reserved_bytes", math.inf) > CAP_MEMORY
            or ledger.get("additional_phase_disk_bytes_forecast", math.inf) > CAP_DISK):
        raise ValueError("Expanded actual-measurement complete30 resource forecast is closed")
    if set(value["tasks"]) != set(TASKS) or set(value["execution_policy"]) != set(TASKS):
        raise ValueError("Both complete graphs required")
    for task in TASKS:
        row = value["tasks"][task]
        phase = json.loads(checked(row["engineering_phase"]).read_text())
        if row["acquisition"] != design["tasks"][task]["acquisition"] or phase["task"] != task:
            raise ValueError("Wrong full-graph acquisition")
        for name in ("acquisition", "prepared_manifest", "input_completed_audit", "parameter_control_report", "provider_storage"):
            checked(row[name])
        if set(value["execution_policy"][task]) != set(ARMS):
            raise ValueError("Every arm execution choice must be frozen")
        for arm, variant in value["execution_policy"][task].items():
            if set(variant) != {"training", "validation", "deployment", "checkpoint_blocks"} or type(variant["checkpoint_blocks"]) is not bool:
                raise ValueError("Exact execution schema required")
            if any(variant[k] not in ("packed", "serial") for k in ("training", "validation", "deployment")):
                raise ValueError("Unknown execution variant")
            if arm in ("shared_mean_message", "detached_mean_message") and any(variant[k] != "packed" for k in ("training", "validation", "deployment")):
                raise ValueError("Shared and detached means always require packed states")
            if arm.startswith("single_") and any(variant[k] != "packed" for k in ("training", "validation", "deployment")):
                raise ValueError("Single model interface has one declared packed trajectory")
    policy = value["runtime_policy"]
    if (policy.get("device") != "cuda:0" or policy.get("TF32") is not False or policy.get("AMP") is not False
            or type(policy.get("CPU_threads")) is not int or policy["CPU_threads"] < 1
            or type(policy.get("deterministic_algorithms")) is not bool
            or not {"torch", "numpy", "torch-geometric"}.issubset(policy.get("package_versions", {}))
            or value.get("timing_policy") != TIMING
            or value.get("execution_equivalence_policy") != {"atol": 1e-5, "rtol": 1e-5}):
        raise ValueError("Frozen actual runtime, precision, equivalence and trained timing required")
    remaining = value["remaining_resource_policy"]
    if set(remaining) != {"prior_GPU_wall_seconds", "prior_additional_disk_bytes"} or any(type(v) not in (int, float) or v < 0 for v in remaining.values()):
        raise ValueError("Actual prior phase resource charges required")
    prior = json.loads(checked(value["prior_accounting"]).read_text())
    if (prior.get("schema") != "gnnm-shared-message-prior-resource-accounting-root-v1"
            or remaining["prior_GPU_wall_seconds"] < prior["all_job_wall_seconds_conservatively_charged_to_allocation_and_CPU_wall"]
            or remaining["prior_additional_disk_bytes"] < prior["phase_disk_conservative_prior_bytes"]):
        raise ValueError("Completed failed/successful children and prior phase bytes cannot be omitted")
    if ledger["total_GPU_wall_seconds_forecast"] < remaining["prior_GPU_wall_seconds"]:
        raise ValueError("Resource forecast omitted actual prior charges")
    whole = value.get("whole_process_policy", {})
    if (set(whole) != {"bounded_source_sha256", "per_child_wall_cap_seconds"}
            or whole.get("bounded_source_sha256") != BOUNDED_SOURCE_SHA
            or type(whole.get("per_child_wall_cap_seconds")) not in (int, float)
            or not math.isfinite(whole["per_child_wall_cap_seconds"])
            or not 0 < whole["per_child_wall_cap_seconds"] <= CAP_GPU):
        raise ValueError("Prospective source-bound complete-process cap required")
    within(PHASE / value["output_root"], HERE)
    return value, reference(path)


def manifest(output):
    output = within(output, HERE)
    paths = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"]
    write(output / "ARTIFACT_MANIFEST.json", {"schema": "gnnm-shared-message-artifacts-v2", "files": [reference(p) for p in paths]})


def audit_manifest(output):
    output = within(output, HERE)
    if (output / "FAILURE.json").exists():
        raise ValueError("Failed partial evidence cannot qualify")
    value = json.loads((output / "ARTIFACT_MANIFEST.json").read_text())
    actual = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"]
    if value.get("schema") != "gnnm-shared-message-artifacts-v2" or value["files"] != [reference(p) for p in actual]:
        raise ValueError("Missing/extra/changed artifact bytes")


def prior_seconds(admitted):
    elapsed = admitted["remaining_resource_policy"]["prior_GPU_wall_seconds"]
    root = within(PHASE / admitted["output_root"], HERE)
    if root.exists():
        for start in root.rglob("START.json"):
            row = json.loads(start.read_text())
            elapsed += completed(PHASE / row["supervisor_directory"], row["live"]["outer_start"])["seconds"]
    return elapsed


class Budget:
    def __init__(self, admitted, context=None):
        self.start = time.monotonic()
        if context is not None:
            outer = json.loads(checked(context["outer_start"]).read_text())
            capture = datetime.fromisoformat(outer["start_UTC"])
            self.start -= max(0, (datetime.now(timezone.utc) - capture).total_seconds())
        self.prior = prior_seconds(admitted)
        self.root = within(PHASE / admitted["output_root"], HERE)
        self.base_disk = admitted["remaining_resource_policy"]["prior_additional_disk_bytes"]
        self.stored = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file()) if self.root.exists() else 0
        if self.prior >= CAP_GPU:
            raise ValueError("Actual phase resource budget already exhausted")

    def check(self, prospective_bytes=0):
        if self.prior + time.monotonic() - self.start >= CAP_GPU:
            raise RuntimeError("Six-hour scientific GPU wall cap exhausted; no partial promotion")
        if self.base_disk + self.stored + prospective_bytes + 512 * 2**20 > CAP_DISK:
            raise RuntimeError("Additional phase disk budget would be exceeded")

    def committed(self, path):
        self.stored += within(path, HERE).stat().st_size
