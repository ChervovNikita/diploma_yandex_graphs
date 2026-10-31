"""Stdlib source preparation and fail-closed future admission checks."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
DESTINATION = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
PLAN = PHASE / "objective_lookahead_molhiv_v1/INTEGRATION_PLAN_v1.json"
RECIPE = dict(hidden_dim=128, num_layers=3, hidden_dim_multiplier=2, members=4,
    output_dim=1, dropout=0.1, optimizer="AdamW", lr=0.001, weight_decay=0.0,
    betas=[0.9, 0.999], eps=1e-8, batch_size=128, epochs=100, dtype="float32",
    AMP=False, TF32=False, deterministic_algorithms=True, num_workers=0,
    checkpoint="max_pooled_validation_ROC_AUC_then_min_pooled_BCE_then_earliest_epoch",
    mask="original floor10percent largest-remainder stratification, class RNG101/102",
    order="inherited DataLoader generator19709+1000*seed+epoch",
    dropout_rng="continuous device RNG; seed optimizer_seed once; no batch reseeding",
    initialization="canonical fresh tied seed; private copies/zero-delta low-rank before fresh optimizer; never P25")
LAYOUTS = ("tied", "first_private", "middle_private", "last_private", "declared_low_rank")
OBJECTIVES = ("own", "pooled")
DEPS = ("new_tasks/molhiv_pilot.py", "new_tasks/gine_adapters.py",
    "new_tasks/shareable_guard.py", "source_snapshot/models.py",
    "closest_controls_v1/trainable_low_rank_gine_v2/adapter.py",
    "closest_controls_v1/trainable_low_rank_gine_v2/SOURCE_BINDINGS.json",
    "protocols/run_authorized_v2.py", "protocols/run_logged.py", "protocols/repo_env.sh",
    "objective_lookahead_molhiv_v1/INTEGRATION_PLAN_v1.json")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def inside(path, root=PHASE):
    p, root = Path(path).resolve(), Path(root).resolve()
    if not p.is_relative_to(root):
        raise ValueError("Path outside project phase")
    return p


def write_new(path, value):
    p = inside(path, HERE)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def sources():
    files = {str(p.relative_to(PHASE)): p for p in sorted(HERE.glob("*.py"))}
    files.update({name: PHASE/name for name in DEPS})
    return {name: sha(p) for name, p in files.items()}


def matrix():
    plan = json.loads(PLAN.read_text())
    cells = plan["stages"]["primary_complete_screen"]
    expected = {(s, o, l, 0.1) for s in range(3) for o in OBJECTIVES for l in LAYOUTS}
    actual = [(x["seed"], x["training_loss"], x["layout"], x["label_fraction"]) for x in cells]
    if len(actual) != 30 or set(actual) != expected:
        raise ValueError("Prescribed complete primary30 matrix changed")
    return cells


def checked(reference):
    if set(reference) != {"path", "sha256"}:
        raise ValueError("Reference needs exactly path and SHA256")
    p = inside(PHASE/reference["path"])
    if sha(p) != reference["sha256"]:
        raise ValueError("Reference hash mismatch: " + reference["path"])
    return p


def authorized_completed(directory):
    p = inside(directory)
    a, c, e, command = [json.loads((p/n).read_text()) for n in
                         ("authorization.json", "completion.json", "environment.json", "command.json")]
    if (a.get("schema") != "gnnm-authorized-allocation-supervision-v2"
        or a.get("ssh_destination") != DESTINATION or a.get("visible_gpu_count") != 1
        or a.get("gpu_name") != "NVIDIA A100-SXM4-80GB"
        or a.get("allocation_guard_sha256") != sha(PHASE/"protocols/run_authorized_v2.py")
        or a.get("child_exit_code") != 0 or c.get("exit_code") != 0
        or a.get("inner_completion_sha256") != sha(p/"completion.json")):
        raise ValueError("Unqualified or wrong-allocation completed supervisor")
    for field, name in (("environment_sha256", "environment.json"),
                        ("command_sha256", "command.json"), ("log_sha256", "stdout_stderr.log")):
        if c.get(field) != sha(p/name):
            raise ValueError("Supervisor completion chain changed")
    if e.get("supervisor_sha256") != sha(PHASE/"protocols/run_logged.py"):
        raise ValueError("Wrong inner supervisor source")
    route = json.loads((PHASE/"protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
    if a.get("route_record_sha256") != sha(PHASE/"protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json"):
        raise ValueError("Route reference changed")
    if e.get("repository") != route["repository"] or command.get("cwd") != route["repository"]:
        raise ValueError("Supervisor repository/cwd mismatch")
    return {"authorization_sha256": sha(p/"authorization.json"), "completion_sha256": sha(p/"completion.json")}


def audit_future_binding(value):
    """Checks format/receipts only; NEVER admits real data in this generation.

    A separately approved admission must additionally verify P25 record/replay,
    train/validation canonical tensor fingerprints, grouping/masks and global
    selector replay. These semantic requirements cannot be fabricated here.
    """
    required = {"schema", "source_sha256", "coordinator_approval", "semantic_admission_report", "six_P25_cells",
        "six_training_method_locks", "mechanism_lock", "qualified_gates",
        "train_export", "validation_export", "data_freeze", "runtime_freeze", "admission_entrypoint"}
    if not required <= set(value) or value["schema"] != "gnnm-primary-molhiv-future-admission-v1":
        raise ValueError("Incomplete future admission")
    if value["source_sha256"] != sources():
        raise ValueError("Prospective predictive sources differ")
    for name in ("coordinator_approval", "semantic_admission_report", "data_freeze", "runtime_freeze", "admission_entrypoint"):
        checked(value[name])
    expected = {(s, f) for s in range(3) for f in (0.1, 1.0)}
    for family in ("six_P25_cells", "six_training_method_locks"):
        rows = value[family]
        if len(rows) != 6 or {(r["seed"], r["label_fraction"]) for r in rows} != expected:
            raise ValueError("All six distinct training-only cells/locks required")
        for row in rows:
            for reference in row["artifacts"].values():
                checked(reference)
            authorized_completed(PHASE/row["supervisor_directory"])
            if family == "six_P25_cells" and row.get("epoch") != 25:
                raise ValueError("P25 qualification must be exact epoch25")
            if row.get("validation_or_test_used") is not False:
                raise ValueError("Method lock/source used predictive labels")
    mechanism = json.loads(checked(value["mechanism_lock"]).read_text())
    if (mechanism.get("schema")!="gnnm-molhiv-qualified-lookahead-mechanism-lock-v3"
        or mechanism.get("status")!="qualified_training_only_mechanism_screen"
        or mechanism.get("all_six_cells_used") is not True
        or mechanism.get("all_48_trajectories_replayed") is not True
        or mechanism.get("validation_or_test_used") is not False):
        raise ValueError("Require qualified complete v3 training-only mechanism lock")
    checked(mechanism["binding"])
    for name,expected_hash in mechanism["source_sha256"].items():
        if sha(inside(PHASE/name))!=expected_hash:raise ValueError("Locked mechanism source changed")
    for name,artifacts in (("execute",("summary","artifact_manifest")),("replay",("report",))):
        row=mechanism[name]
        for artifact in artifacts:checked(row[artifact])
        receipts=authorized_completed(PHASE/row["supervisor_directory"])
        if any(receipts[k]!=row[k] for k in receipts):raise ValueError("Mechanism execute/replay chain differs from qualified lock")
    # The finalizer cannot authorize itself while still running. Require its
    # later completed wrapper and tie the captured command to this exact lock.
    finalizer=mechanism["finalizer_live_supervisor"]
    if finalizer.get("authorization_status")!="pending until guarded child exits; qualification requires completed authorization":
        raise ValueError("Unexpected finalizer live-authorization declaration")
    finalizer_directory=inside(PHASE/finalizer["supervisor_directory"])
    authorized_completed(finalizer_directory)
    for field,name in (("command_sha256","command.json"),("environment_sha256","environment.json")):
        if finalizer.get(field)!=sha(finalizer_directory/name):raise ValueError("Finalizer command/environment changed after lock")
    command=json.loads((finalizer_directory/"command.json").read_text())
    environment=json.loads((finalizer_directory/"environment.json").read_text())
    argv=command.get("argv",[])
    if argv.count("--output")!=1 or argv.index("--output")+1>=len(argv):raise ValueError("Finalizer needs one explicit output target")
    remote_phase=Path(environment["phase"])
    route=json.loads((PHASE/"protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
    if remote_phase!=Path(route["repository"])/"experiments_iclr"/PHASE.name:raise ValueError("Finalizer ran in another project phase")
    raw_output=Path(argv[argv.index("--output")+1])
    actual_output=Path(os.path.normpath(str(raw_output if raw_output.is_absolute() else Path(command["cwd"])/raw_output)))
    relative_lock=checked(value["mechanism_lock"]).relative_to(PHASE)
    if actual_output!=remote_phase/relative_lock:raise ValueError("Completed finalizer did not write this referenced mechanism lock")
    locks=mechanism["loss_families"]
    if set(locks)!=set(OBJECTIVES):raise ValueError("Both global objective-family locks required")
    for objective,lock in locks.items():
        if lock.get("training_loss")!=objective or lock.get("all_six_cells_used") is not True or lock.get("validation_or_test_used") is not False:
            raise ValueError("Incomplete objective-family global lock")
        chosen=lock.get("private_indices")
        if chosen not in ([], [0], [1], [2]):
            raise ValueError("Reject multiple, invalid or implicit placement")
        candidate=None if not chosen else ("first","middle","last")[chosen[0]]
        if lock.get("candidate")!=candidate or lock.get("abstain")!=(not chosen):raise ValueError("Locked candidate/abstention identity differs")
    for row in value["qualified_gates"]:
        checked(row["report"])
        authorized_completed(PHASE/row["supervisor_directory"])
    for name in ("train_export", "validation_export"):
        checked(value[name]["bundle"])
        authorized_completed(PHASE/value[name]["supervisor_directory"])
    return {"format_and_receipts_passed": True, "real_admission": False}


def refuse_real():
    # Deliberately no mutable SOURCE_ONLY switch and no command-line override.
    raise ValueError("Predictive v1 is source-only. No binding, dataset, validation or checkpoint opened. Require root-supplied qualified P25/selector locks and a separately versioned, sealed, approved admission entry point.")


def live_job_context(supervisor):
    """For the future admitted child; authorization arrives only after exit."""
    p = inside(supervisor)
    e = json.loads((p/"environment.json").read_text())
    c = json.loads((p/"command.json").read_text())
    if (os.environ.get("GNNM_SSH_DESTINATION") != DESTINATION
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or e.get("supervisor_sha256") != sha(PHASE/"protocols/run_logged.py")
        or e.get("wrapper_sha256") != sha(PHASE/"protocols/repo_env.sh")):
        raise ValueError("Incorrect live route/supervisor/cache wrapper")
    route = json.loads((PHASE/"protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
    if c.get("cwd") != route["repository"] or str(Path.cwd().resolve()) != route["repository"]:
        raise ValueError("Incorrect live repository")
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],check=True,capture_output=True,text=True)
    rows = [r.strip().split(",") for r in query.stdout.splitlines() if r.strip()]
    if len(rows)!=1 or [x.strip() for x in rows[0]][::2] != ["0","NVIDIA A100-SXM4-80GB"] or int(rows[0][3])!=81920:
        raise ValueError("Only the authorized one-A10080GB allocation is allowed; seven-GPU account forbidden")
    return {"supervisor_directory": str(p.relative_to(PHASE)),
            "entrypoint_sha256": e["script_sha256"],
            "command_sha256": sha(p/"command.json"), "environment_sha256": sha(p/"environment.json"),
            "authorization_status": "pending until run_authorized_v2 exits; completed-chain audit mandatory before qualification"}
