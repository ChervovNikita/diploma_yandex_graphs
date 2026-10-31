"""Stdlib-only, fail-closed admission for the separately versioned GPU bridge."""
import ast
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PRIOR = PHASE / "objective_lookahead_molhiv_v1"
QUEUE = PHASE / "sharing_policy/gine_transfer_gpu_queue_v1"
PARENT = PHASE / "sharing_policy/gine_transfer_grouping_fix_v6"
DESTINATION = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
CELLS = tuple((s, f) for s in range(3) for f in (0.1, 1.0))
PLACEMENTS = ("tied", "first", "middle", "last")
LOSSES = ("own", "pooled")
OWN_FILES = ("admission.py", "admitted_core.py", "runner.py", "verify_admission.py", "recovery.py", "README.md")
PINNED = {
 'objective_lookahead_molhiv_admitted_v3/admission.py': '322930bc9d5ca5989433b8a544f913eb242af7c7d1c4708984c9a67b8a91c48e',
 'objective_lookahead_molhiv_admitted_v3/admitted_core.py': '164551f69f8f1bca746efbfbbf543e6725154bf365f9df8e1052505576110b97',
 'objective_lookahead_molhiv_admitted_v3/runner.py': '263c2c13912562fa54138289344bf9fdc4a6c03b12c15cb72b63053c8cedeb05',
 'objective_lookahead_molhiv_admitted_v3/verify_admission.py': 'ff7ed457934d2e7318099b3a864c7714a232cfc8c12b21001226b2bd4b64b9dc',
 'objective_lookahead_molhiv_admitted_v3/README.md': 'd692e6acbed8772e4206bcd240fdb44a638683e471e4ed5012921a243550e6b4',
 "objective_lookahead_molhiv_v1/INTEGRATION_PLAN_v1.json": "ccd683329b4f532f5d0686db37c536fc9cd833662a7ca13548b064ba40b47afb",
 "objective_lookahead_molhiv_v1/gpu_lookahead_core_v2.py": "8d16dbb981d7f1a4f69d337c14304e8557e1664b1a84320b94a21c47b96f930f",
 "objective_lookahead_molhiv_v1/lookahead_contract.py": "87273226752616712ecf40110d1ba66d7e425283d1c34036e46d7c36182ea83d",
 "objective_lookahead_molhiv_v1/lookahead_core.py": "8e3e4bd0352c60ebc237c6b90a5a933a4750471210c2605f9e2fabd1ee07a9b7",
 "objective_lookahead_molhiv_v1/verify_gpu_v2.py": "9c5636de72d695b3eda642d538e8b66182401d74f20114ebd2e148552d6905b3",
 "objective_lookahead_molhiv_v1/verify_cpu.py": "5c5603bedeae81b29b74806a745ab05d0b0ff42022a304c194997c0d3b65256b",
 "objective_lookahead_molhiv_v1/PROTOCOL.md": "0913540361db8cf09ffb593002034e2fee0f6cb2d940731f4629dbb984838ca8",
 "objective_lookahead_molhiv_v1/README.md": "c04af368e2046f1a769e64f552f6b79159f2eed9f8decbe19f9c57dd82e06358",
 "objective_lookahead_molhiv_v1/runner.py": "5590e02f464f24ebf3dbdd82f1e9b3588b35e6943f525f80390a8830cae7dd99",
 "new_tasks/gine_adapters.py": "7416ddddd5a4057ec80624c648f32a5c8369c8c2628e8b58c450c82fe22e3c86",
 "source_snapshot/models.py": "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(path, root=PHASE):
    p, root = Path(path).resolve(), Path(root).resolve()
    if p == root or root not in p.parents:
        raise ValueError("Require a file/directory strictly inside the authorized project phase")
    return p


def relative_path(value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("Only explicit confined phase-relative paths are allowed")
    return inside(PHASE / value)


def reference(path):
    p = inside(path)
    return {"path": str(p.relative_to(PHASE)), "sha256": sha(p)}


def checked(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("Reference must contain exactly path and sha256")
    p = relative_path(value["path"])
    if sha(p) != value["sha256"]:
        raise ValueError("Evidence bytes differ: " + value["path"])
    return p


def read_json(path):
    return json.loads(inside(path).read_text())


def write_new(path, value):
    p = inside(path, HERE)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def queue_modules():
    sys.path.insert(0, str(QUEUE)) if str(QUEUE) not in sys.path else None
    modules = []
    for name in ("queue_common", "qualify_inputs", "gpu_queue"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != QUEUE / (name + ".py"):
            raise ValueError("Conflicting queue module import")
        modules.append(module)
    return modules


def sources():
    result = {}
    for name, expected in PINNED.items():
        if sha(PHASE / name) != expected:
            raise ValueError("Pinned prior source differs: " + name)
        result[name] = expected
    qc, _, _ = queue_modules()
    result.update({str((QUEUE / name).relative_to(PHASE)): digest
                   for name, digest in qc.source_hashes().items()})
    result.update({str((HERE / name).relative_to(PHASE)): sha(HERE / name) for name in OWN_FILES})
    plan = qc.plan()
    protocol = read_json(PHASE / plan["new_authorized_protocol_phase_relative"])
    qc.check_parent_sources(protocol)
    result.update({str((PARENT / name).relative_to(PHASE)): digest
                   for name, digest in protocol["source_sha256"].items()})
    for key in ("allocation_guard_phase_relative", "inner_supervisor_phase_relative"):
        name = plan[key]
        result[name] = sha(PHASE / name)
    result["protocols/repo_env.sh"] = sha(PHASE / "protocols/repo_env.sh")
    return result


def prior_literal(name):
    tree = ast.parse((PRIOR / "lookahead_contract.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing sealed prior literal")


def manifest(path):
    p = inside(path)
    stored = read_json(p / "artifact_manifest.json")
    actual = {str(f.relative_to(p)) for f in p.rglob("*") if f.is_file() and f.name != "artifact_manifest.json"}
    if set(stored) != actual:
        raise ValueError("Artifact manifest does not cover the complete run")
    for name, digest in stored.items():
        target = inside(p / name, p)
        if sha(target) != digest:
            raise ValueError("Run artifact changed: " + name)
    return reference(p / "artifact_manifest.json")


def verify_gate(entry, generation, expected_uuid):
    qc, _, _ = queue_modules()
    report_path = checked(entry["report"])
    supervisor = relative_path(entry["supervisor_directory"])
    record = qc.authorization(supervisor, expected_uuid, "--output", report_path)
    command = read_json(supervisor / "command.json")["argv"]
    script = PRIOR / "verify_gpu_v2.py" if generation == 2 else HERE / "runner.py"
    if Path(command[1]).resolve() != script:
        raise ValueError("Gate used a different executable source")
    if generation == 4 and command[2] != "synthetic-gate":
        raise ValueError("Require separately executed v3 synthetic gate")
    report = read_json(report_path)
    schema = ("gnnm-GPU-lookahead-source-only-synthetic-gate-v2" if generation == 2
              else "gnnm-GPU-lookahead-admitted-sibling-synthetic-gate-v4")
    if (report.get("schema") != schema or report.get("passed") is not True
        or report.get("dataset_loaded") is not False or report.get("checkpoint_loaded") is not False
        or report.get("exact_GPU_paired_null") is not True
        or report.get("full_block_storage_optimizer_identity_checked") is not True):
        raise ValueError("Incomplete or failed synthetic actual-optimizer gate")
    for name, digest in report["sources"].items():
        if sha(relative_path(name)) != digest:
            raise ValueError("Gate scientific/runtime source changed")
    if generation == 4 and report["sources"] != sources():
        raise ValueError("V4 gate does not bind this exact bridge")
    return {"report": reference(report_path), "supervisor_directory": str(supervisor.relative_to(PHASE)), **record}


@dataclass(frozen=True)
class Admission:
    binding: dict
    evidence: dict
    cells: tuple
    qualified_gpu_uuid: str

    @property
    def descriptors(self):
        return self.cells


def validate_six_rows(rows, qualified):
    """Unchanged complete six-source contract/hash/replay audit, with one explicit recovery report path."""
    qc, qualify, _ = queue_modules()
    uuid = qualified["qualified_gpu_uuid"]
    if len(rows) != 6 or {(r["seed"], r["label_fraction"]) for r in rows} != set(CELLS):
        raise ValueError("Complete six-source audit required")
    protocol_path = relative_path(qualified["protocol_phase_relative"])
    bundle = relative_path(qualified["bundle_phase_relative"])
    protocol = read_json(protocol_path)
    descriptors = []
    for seed, fraction in CELLS:
        row = next(r for r in rows if (r["seed"], r["label_fraction"]) == (seed, fraction))
        run = qc.phase_path(row["parent_run_phase_relative"])
        if PARENT / "authorized_gpu_probes_v1" not in run.parents:
            raise ValueError("Reject an old or substituted probe binary path")
        probe_manifest = manifest(run)
        contract = read_json(run / "contract.json")
        budget = protocol["budgets"][str(fraction)]
        if (contract["seed"] != seed or contract["label_fraction"] != fraction
            or contract["device"] != "cuda:0" or contract["validation_or_test_accessed"] is not False
            or contract["source_sha256"] != protocol["source_sha256"]
            or contract["recipe"] != protocol["recipe"]
            or contract["private_parameter_budget"] != protocol["private_parameter_budget"]
            or contract["data_manifest_sha256"] != protocol["data_manifest_sha256"]
            or contract["train_bundle_sha256"] != sha(bundle)
            or contract["protocol_sha256"] != sha(protocol_path)
            or contract["folds"] != budget["folds"] or contract["mask"] != budget["label_mask"]
            or contract["warmup_counts_toward_adaptive_method_cost"] is not True):
            raise ValueError("P25 source fold/mask/source/recipe scope differs")
        receipts = {}
        default_replay = run.parent / "replays" / (run.name + ".json")
        replay_path = relative_path(row["full_replay_phase_relative"]) if "full_replay_phase_relative" in row else default_replay
        for stage in ("probe", "replay"):
            supplied = row["stages"][stage]
            directory = relative_path(supplied["supervisor_phase_relative"])
            target = run if stage == "probe" else replay_path
            receipt = qc.authorization(directory, uuid, "--output", target)
            if supplied != {"supervisor_phase_relative": str(directory.relative_to(PHASE)), **receipt}:
                raise ValueError("Queue stage receipt changed")
            argv = read_json(directory / "command.json")["argv"]
            if Path(argv[1]).resolve() != PARENT / "probe.py" or argv[2] != stage:
                raise ValueError("Source/replay used a different scientific executable")
            qualify.command_input_path(argv, "--train-bundle", bundle)
            if stage == "probe":
                qualify.command_input_path(argv, "--protocol", protocol_path)
                if int(qualify.command_value(argv, "--seed")) != seed or float(qualify.command_value(argv, "--label-fraction")) != fraction:
                    raise ValueError("Probe receipt belongs to another source cell")
            else:
                qualify.command_input_path(argv, "--run", run)
            receipts[stage] = {"supervisor_directory": str(directory.relative_to(PHASE)), **receipt}
        replay = read_json(replay_path)
        if (replay.get("status") != "passed" or replay.get("selector_lock_changed") is not False
            or replay.get("validation_or_test_loaded") is not False
            or replay.get("predictive_score") is not None
            or replay.get("artifact_manifest_sha256") != probe_manifest["sha256"]
            or sha(replay_path) != row["full_replay_sha256"]
            or sha(run / "selector_lock.json") != row["selector_lock_sha256"]
            or [r["epoch"] for r in replay["autograd"]] != [0, 10, 25]):
            raise ValueError("Full P25 checkpoint/autograd/decision replay is incomplete")
        record = read_json(run / "probe_epoch25.json")
        checkpoint = run / "probe_epoch25.pt"
        if (record["checkpoint_epoch"] != 25 or record["optimizer_seed"] != seed
            or record["probe_checkpoint_filename"] != checkpoint.name
            or record["probe_checkpoint_sha256"] != sha(checkpoint)
            or record["dataset_identity"]["label_fraction"] != fraction
            or record["dataset_identity"]["grouped_folds_sha256"]
               != hashlib.sha256(json.dumps(contract["folds"], sort_keys=True).encode()).hexdigest()
            or record["dataset_identity"]["contract_sha256"] != sha(run / "contract.json")
            or record["dataset_identity"]["train_bundle_sha256"] != sha(bundle)):
            raise ValueError("P25 physical checkpoint does not match its replayed record")
        descriptors.append({"seed": seed, "label_fraction": fraction, "epoch": 25,
            "run": str(run.relative_to(PHASE)), "checkpoint": reference(checkpoint),
            "contract": reference(run / "contract.json"), "record": reference(run / "probe_epoch25.json"),
            "artifact_manifest": probe_manifest, "full_replay": reference(replay_path), "stages": receipts})
    return tuple(descriptors)


def binding_format(value):
    fields = {"schema", "source_sha256", "coordinator_approval", "queue_qualification",
              "recovered_queue_result", "recovery_assembler_supervisor_directory", "v2_gpu_gate", "v4_gpu_gate"}
    if not isinstance(value, dict) or set(value) != fields or value.get("schema") != "gnnm-molhiv-lookahead-real-binding-v4":
        raise ValueError("Incomplete or unsupported technical-recovery real-input binding")
    return value


def admit(binding_path):
    """Qualify the explicit failed queue plus one final replay before any tensor import."""
    qc, qualifier, gpu_queue = queue_modules()
    value = binding_format(read_json(binding_path))
    if value["source_sha256"] != sources():
        raise ValueError("Real binding does not seal the exact v4 recovery bridge")
    expected = {"schema": "gnnm-molhiv-lookahead-coordinator-approval-v4", "approved_source_sha256": sources(),
                "recovered_queue_result_sha256": value["recovered_queue_result"]["sha256"],
                "queue_qualification_sha256": value["queue_qualification"]["sha256"],
                "v2_gpu_gate_sha256": value["v2_gpu_gate"]["report"]["sha256"],
                "v4_gpu_gate_sha256": value["v4_gpu_gate"]["report"]["sha256"],
                "stages": ["execute", "replay", "finalize"], "training_only": True,
                "maximum_trajectories": 48, "optimizer_steps_per_trajectory": 8,
                "single_final_replay_technical_recovery_allowed": True, "no_source_retraining": True,
                "automatic_retry": False, "previous_wrong_account_evidence_admissible": False}
    if read_json(checked(value["coordinator_approval"])) != expected:
        raise ValueError("Missing exact coordinator approval for this recovery generation")
    idle = qc.idle_guard()
    qualification_path = checked(value["queue_qualification"])
    qualified = read_json(qualification_path)
    gpu_queue.qualification_binding(qualified)
    fresh = qualifier.qualification_value(QUEUE / "QUALIFICATION_REQUEST_authorized1gpu_run01.json", idle)
    gpu_queue.compare_requalification(qualified, fresh)
    uuid = qualified["qualified_gpu_uuid"]
    recovered_path = checked(value["recovered_queue_result"])
    recovered = read_json(recovered_path)
    from recovery import assemble_value
    stable = assemble_value(checked(recovered["recovery_request"]), qualified=qualified)
    if {k: v for k, v in recovered.items() if k != "assembler_live_supervisor"} != stable:
        raise ValueError("Recovered result differs from fresh full semantic recovery re-audit")
    if recovered["queue_qualification"] != value["queue_qualification"]:
        raise ValueError("Recovery substitutes a different qualified input universe")
    assembler_dir = relative_path(value["recovery_assembler_supervisor_directory"])
    assembler = qc.authorization(assembler_dir, uuid, "--output", recovered_path)
    argv = read_json(assembler_dir / "command.json")["argv"]
    if Path(argv[1]).resolve() != HERE / "runner.py" or argv[2] != "recovery-assemble":
        raise ValueError("Recovery result did not originate from the exact separately supervised assembler")
    qualifier.command_input_path(argv, "--request", checked(recovered["recovery_request"]))
    live = recovered["assembler_live_supervisor"]
    if (live["supervisor_directory"] != str(assembler_dir.relative_to(PHASE))
        or live["command_sha256"] != assembler["command_sha256"] or live["environment_sha256"] != assembler["environment_sha256"]
        or live["authorization_status"] != "pending until guarded child exits; qualification requires completed authorization"):
        raise ValueError("Recovery assembler live record does not link to its completed chain")
    descriptors = validate_six_rows(recovered["results"], qualified)
    gates = {"v2": verify_gate(value["v2_gpu_gate"], 2, uuid), "v4": verify_gate(value["v4_gpu_gate"], 4, uuid)}
    technical = {"recovery_result": reference(recovered_path), "original_failed_queue_chain": recovered["original_failed_queue"],
        "one_final_replay_only": True, "no_retraining": True, "new_final_replay": recovered["new_final_replay"],
        "coordinator_preexecution_reason": recovered["coordinator_preexecution_reason"],
        "separate_prelaunch_final_manifest_digest_recorded": False, "provenance_limit": recovered["provenance_limit"]}
    evidence = {"binding": reference(binding_path), "source_sha256": sources(),
        "bundle": {"path": qualified["bundle_phase_relative"], "sha256": qualified["bundle_sha256"]},
        "protocol": {"path": qualified["protocol_phase_relative"], "sha256": qualified["protocol_sha256"]},
        "queue_qualification": reference(qualification_path), "queue_result": reference(recovered_path),
        "recovery_assembler_receipt": assembler, "gates": gates, "technical_recovery": technical,
        "all_six_cells_qualified_before_tensor_loading": True, "target_tensor_deserialized_by_admission": False,
        "wrong_allocation_evidence_accepted": False, "admission_generation": 4}
    return Admission(value, evidence, descriptors, uuid)


def live_supervisor(path, argv):
    qc, _, _ = queue_modules()
    plan = qc.runtime_scope()
    p = inside(path)
    environment, command = read_json(p / "environment.json"), read_json(p / "command.json")
    if (command["argv"][1:] != argv or command["cwd"] != str(qc.REPO)
        or environment["supervisor_sha256"] != plan["inner_supervisor_sha256"]
        or environment["script_sha256"] != sha(HERE / "runner.py")
        or environment["wrapper_sha256"] != sha(PHASE / "protocols/repo_env.sh")
        or environment["CUDA_VISIBLE_DEVICES"] != "0"
        or environment["python_bytecode_disabled"] is not True):
        raise ValueError("Live job does not bind the exact guarded invocation")
    if (p / "authorization.json").exists() or (p / "completion.json").exists():
        raise ValueError("Execution must use the currently running supervisor, never a prior receipt")
    return {"supervisor_directory": str(p.relative_to(PHASE)),
            "command_sha256": sha(p / "command.json"), "environment_sha256": sha(p / "environment.json"),
            "authorization_status": "pending until guarded child exits; qualification requires completed authorization"}


def choose_all_gains(gains):
    """Exact v1 selection arithmetic, retained here without importing torch."""
    rule = prior_literal("RULE")
    expected = {(s, f, loss, placement) for s, f in CELLS for loss in LOSSES for placement in PLACEMENTS}
    if set(gains) != expected or any(not math.isfinite(x) for x in gains.values()):
        raise ValueError("Complete finite 48-arm gain matrix required")
    if any(value != 0. for key, value in gains.items() if key[3] == "tied"):
        raise ValueError("Actual tied-after-K gains must be exactly zero")
    result = {}
    for loss in LOSSES:
        rows = {}
        for placement in PLACEMENTS[1:]:
            primary = [gains[s, .1, loss, placement] for s in range(3)]
            adverse = [gains[s, 1., loss, placement] for s in range(3)]
            eligible = (min(primary) >= rule["primary_each_seed_minimum_gain"] + rule["numerical_margin"]
                        and math.fsum(primary) / 3 >= rule["primary_mean_minimum_gain"] + rule["numerical_margin"]
                        and min(adverse) >= -rule["full_label_each_seed_maximum_harm"] + rule["numerical_margin"])
            rows[placement] = {"primary_seed_gains": primary, "full_label_seed_gains": adverse,
                               "mean_primary_gain": math.fsum(primary) / 3, "mechanism_screen_eligible": eligible}
        eligible = [p for p in PLACEMENTS[1:] if rows[p]["mechanism_screen_eligible"]]
        chosen = max(eligible, key=lambda p: rows[p]["mean_primary_gain"]) if eligible else None
        result[loss] = {"placements": rows, "candidate_for_separate_full_predictive_study": chosen, "abstain": chosen is None}
    return {"loss_families": result, "accuracy_promotion_allowed": False,
            "screening_is_not_independent_confirmation": True,
            "all_48_gains": {str(k): v for k, v in gains.items()}, "rule": rule}


def qualify_completed_lock(lock_path):
    """Return Admission after exact completed chains and selection replay.

    No torch, provider, graph or checkpoint tensor is imported/deserialized.
    The coordinator may use the returned TRAIN bundle in a separately sealed
    validation exporter only after this call succeeds on the authorized route.
    """
    lock_path = inside(lock_path, HERE)
    lock = read_json(lock_path)
    if (lock.get("schema") != "gnnm-molhiv-qualified-lookahead-mechanism-lock-v3"
        or lock.get("status") != "qualified_training_only_mechanism_screen"
        or lock.get("all_six_cells_used") is not True or lock.get("all_48_trajectories_replayed") is not True
        or lock.get("validation_or_test_used") is not False
        or lock.get("source_sha256") != sources()
        or lock.get("finalizer_completed_authorization_required_before_downstream_use") is not True):
        raise ValueError("Require the exact completed six-source mechanism lock")
    admitted = admit(checked(lock["binding"]))
    if lock.get("admission_generation") != 4 or lock.get("technical_recovery") != admitted.evidence["technical_recovery"]:
        raise ValueError("Completed lock omits or substitutes explicit v4 technical recovery lineage")
    qc, qualifier, _ = queue_modules()
    execute = lock["execute"]
    execute_summary_path = checked(execute["summary"])
    execute_dir = execute_summary_path.parent
    manifest_ref = manifest(execute_dir)
    execute_supervisor = relative_path(execute["supervisor_directory"])
    receipt = qc.authorization(execute_supervisor, admitted.qualified_gpu_uuid, "--output", execute_dir)
    if execute != {"summary": reference(execute_summary_path), "artifact_manifest": manifest_ref,
                   "supervisor_directory": str(execute_supervisor.relative_to(PHASE)), **receipt}:
        raise ValueError("Locked execution completed chain changed")
    argv = read_json(execute_supervisor / "command.json")["argv"]
    if Path(argv[1]).resolve() != HERE / "runner.py" or argv[2] != "execute":
        raise ValueError("Locked execution used another executable stage")
    qualifier.command_input_path(argv, "--binding", checked(lock["binding"]))
    summary = read_json(execute_summary_path)
    if (summary.get("status") != "all_48_completed_pending_authorization_and_independent_replay"
        or summary.get("source_sha256") != sources() or summary.get("binding") != admitted.evidence["binding"]
        or summary.get("all_six_cells_used") is not True or summary.get("trajectories") != 48
        or summary.get("optimizer_steps") != 384 or summary.get("full_snapshot_count") != 480
        or summary.get("validation_or_test_loaded") is not False
        or len(summary.get("cells", [])) != 6
        or {(r["seed"], r["label_fraction"]) for r in summary["cells"]} != set(CELLS)):
        raise ValueError("Incomplete locked execution48 matrix")
    run_lock = read_json(checked(summary["run_lock"]))
    if (run_lock.get("admission") != admitted.evidence or run_lock.get("source_sha256") != sources()
        or run_lock.get("cells") != list(admitted.cells) or run_lock.get("recipe") != prior_literal("RECIPE")
        or run_lock.get("rule") != prior_literal("RULE") or run_lock.get("trajectories") != 48
        or run_lock.get("validation_or_test_loaded") is not False
        or run_lock.get("automatic_retry") is not False or run_lock.get("outcome_scheduling") is not False):
        raise ValueError("Locked execution source/input/scientific run lock changed")
    replay = lock["replay"]
    replay_path = checked(replay["report"])
    replay_supervisor = relative_path(replay["supervisor_directory"])
    replay_receipt = qc.authorization(replay_supervisor, admitted.qualified_gpu_uuid, "--output", replay_path)
    if replay != {"report": reference(replay_path), "supervisor_directory": str(replay_supervisor.relative_to(PHASE)), **replay_receipt}:
        raise ValueError("Locked replay completed chain changed")
    argv = read_json(replay_supervisor / "command.json")["argv"]
    if Path(argv[1]).resolve() != HERE / "runner.py" or argv[2] != "replay":
        raise ValueError("Locked replay used another executable stage")
    for flag, path in (("--binding", checked(lock["binding"])), ("--run", execute_dir), ("--execute-supervisor", execute_supervisor)):
        qualifier.command_input_path(argv, flag, path)
    replay_report = read_json(replay_path)
    replay_cells = replay_report.get("cells", [])
    if (replay_report.get("status") != "exact_pass_pending_replay_authorization"
        or replay_report.get("binding") != admitted.evidence["binding"] or replay_report.get("execute") != execute
        or replay_report.get("source_sha256") != sources() or replay_report.get("all_six_cells_used") is not True
        or replay_report.get("all_48_trajectories_replayed") is not True
        or replay_report.get("snapshots_compared") != 480 or replay_report.get("optimizer_steps_replayed") != 384
        or replay_report.get("validation_or_test_loaded") is not False or len(replay_cells) != 6
        or {(r["seed"], r["label_fraction"]) for r in replay_cells} != set(CELLS)
        or any(r.get("snapshots") != 80 or r.get("status") != "exact_pass"
               or r.get("all_forward_losses_logits_gradients_states_optimizers_RNG_equal") is not True for r in replay_cells)):
        raise ValueError("Incomplete exact replay48 matrix")
    finalizer = lock["finalizer_live_supervisor"]
    directory = relative_path(finalizer["supervisor_directory"])
    final_receipt = qc.authorization(directory, admitted.qualified_gpu_uuid, "--output", lock_path)
    if (finalizer.get("command_sha256") != final_receipt["command_sha256"]
        or finalizer.get("environment_sha256") != final_receipt["environment_sha256"]
        or finalizer.get("authorization_status") != "pending until guarded child exits; qualification requires completed authorization"):
        raise ValueError("Finalizer live record does not link to its completed guarded chain")
    argv = read_json(directory / "command.json")["argv"]
    if Path(argv[1]).resolve() != HERE / "runner.py" or argv[2] != "finalize":
        raise ValueError("Mechanism lock did not originate from exact finalizer")
    for flag, path in (("--binding", checked(lock["binding"])), ("--run", execute_dir),
                       ("--execute-supervisor", execute_supervisor), ("--replay", replay_path),
                       ("--replay-supervisor", replay_supervisor)):
        qualifier.command_input_path(argv, flag, path)
    gains = {}
    for cell in summary["cells"]:
        report_path = checked(cell["report"])
        if report_path.parent.parent != execute_dir or cell.get("snapshot_count") != 80:
            raise ValueError("Substituted/missing cell artifact reference")
        cell_dir = report_path.parent
        stages = {"initial", "final", *("step%02d" % i for i in range(1, 9))}
        expected_names = {l + "_" + p for l in LOSSES for p in PLACEMENTS}
        if (checked(cell["input_lock"]).parent != cell_dir
            or {str(p.relative_to(cell_dir / "snapshots")) for p in (cell_dir / "snapshots").rglob("*.pt")}
               != {n + "/" + stage + ".pt" for n in expected_names for stage in stages}):
            raise ValueError("Locked full initial/K8/final artifacts are incomplete")
        report = read_json(report_path)
        if (report.get("schema") != "gnnm-molhiv-GPU-actual-optimizer-lookahead-cell-v3"
            or report.get("seed") != cell["seed"] or report.get("label_fraction") != cell["label_fraction"]
            or report.get("screen_scored_completely") is not True or report.get("source_and_inputs_preserved") is not True
            or report.get("validation_or_test_loaded") is not False
            or set(report.get("trajectories", {})) != {l + "_" + p for l in LOSSES for p in PLACEMENTS}):
            raise ValueError("Locked cell does not provide exact complete screening48 endpoints")
        for name, trajectory in report["trajectories"].items():
            loss, placement = name.split("_", 1)
            if trajectory["optimizer_step_counts"] != [8.] or len(trajectory["steps"]) != 8:
                raise ValueError("Locked trajectory does not contain the exact K8 update budget")
            gains[cell["seed"], cell["label_fraction"], loss, placement] = trajectory["paired_pooled_bce_gain_vs_actual_tied_after_K"]
    mechanism = choose_all_gains(gains)
    if lock.get("mechanism") != mechanism:
        raise ValueError("Locked decision fails exact all48 selection replay")
    families = {}
    for loss in LOSSES:
        chosen = mechanism["loss_families"][loss]["candidate_for_separate_full_predictive_study"]
        families[loss] = {"training_loss": loss, "candidate": chosen,
            "private_indices": [] if chosen is None else [PLACEMENTS.index(chosen) - 1], "abstain": chosen is None,
            "all_six_cells_used": True, "validation_or_test_used": False}
    if lock.get("loss_families") != families:
        raise ValueError("Locked placement aliases do not match the replayed training-only choice")
    return Admission(admitted.binding, {**admitted.evidence,
        "completed_mechanism_lock": reference(lock_path), "execute": execute, "replay": replay,
        "finalizer_completed_receipt": final_receipt, "loss_families": families,
        "all_48_gains_and_selection_replayed_without_tensor_loading": True}, admitted.cells, admitted.qualified_gpu_uuid)
