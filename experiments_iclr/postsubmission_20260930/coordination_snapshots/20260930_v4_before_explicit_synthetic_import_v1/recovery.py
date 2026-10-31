"""Exactly one pre-child-refused final replay recovery; no training or tensors."""
from datetime import datetime
from pathlib import Path

from admission import (HERE, PHASE, QUEUE, PARENT, CELLS, checked, inside, read_json,
    reference, relative_path, sha, sources, queue_modules)

FINAL = (2, 1.0)
FINAL_LABEL = "seed2_label100"
GUARD_ERROR = "Authorized GPU is occupied; no child started"
FAILURE_MESSAGE = "Supervised replay failed for seed2_label100; exact failure retained, no retry"


def utc(value):
    if not isinstance(value, str):
        raise ValueError("Explicit UTC timestamp required")
    time = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if time.tzinfo is None:
        raise ValueError("Require timezone-aware evidence timestamps")
    return time


def verify_technical_case(failure, events, reason, launcher):
    """Bounded causal gate on recorded queue facts, with no outcome inspection."""
    expected_order = [(s, f) for f in (.1, 1.) for s in range(3)]
    rows = failure.get("completed_cells", [])
    if (failure.get("type") != "RuntimeError" or failure.get("message") != FAILURE_MESSAGE
        or failure.get("remaining_cells_not_qualified") is not True
        or failure.get("retry_or_outcome_truncation") is not False
        or [(r["seed"], r["label_fraction"]) for r in rows] != expected_order[:-1]):
        raise ValueError("Recovery is limited to the explicit final replay pre-child refusal")
    expected = []
    for s, f in expected_order:
        label = "seed%d_label%d" % (s, int(100 * f))
        for stage in ("probe", "replay"):
            expected += [(label, stage + "_idle_guard"), (label, stage + "_started"), (label, stage + "_terminal")]
    if [(e.get("cell"), e.get("stage")) for e in events] != expected:
        raise ValueError("Queue event order does not prove six fits and only the final replay refusal")
    terminal = [e for e in events if e["stage"].endswith("_terminal")]
    if [e.get("exit_code") for e in terminal] != [0] * 11 + [1]:
        raise ValueError("Any source/earlier replay failure or different failure budget rejects this recovery")
    before = events[-3]["guard"]
    gpu = before["GPU"]
    if (before.get("status") != "idle_no_compute_context" or gpu.get("memory_MiB") != 0
        or gpu.get("utilization_percent") != 0 or gpu.get("compute_context_count") != 0):
        raise ValueError("Require the actual immediately preceding zero-context idle observation")
    if GUARD_ERROR not in launcher or "RuntimeError: " + GUARD_ERROR not in launcher:
        raise ValueError("Original final launcher does not record the exact guard refusal")
    if (reason.get("completed_probe_fits") != 6 or reason.get("completed_full_replays") != 5
        or reason.get("blocked_cell") != {"seed": 2, "label_fraction": 1.0, "stage": "replay"}
        or reason.get("guard_error") != GUARD_ERROR
        or reason.get("child_or_replay_supervisor_created_by_failed_attempt") is not False
        or reason.get("automatic_retry") is not False
        or reason.get("scientific_source_or_masks_or_selection_rule_changed") is not False
        or reason.get("predictive_validation_or_test_scored") is not False
        or reason.get("preceding_queue_idle_snapshot") != {
            "memory_MiB": 0, "utilization_percent": 0, "compute_context_count": 0}
        or utc(reason["UTC"]) <= utc(events[-1]["UTC"])):
        raise ValueError("Missing genuine pre-execution, training-only single-replay recovery reason")
    return {"exact_final_cell": {"seed": 2, "label_fraction": 1.0},
            "original_probe_fits": 6, "original_completed_replays": 5,
            "failure_before_replay_child_or_supervisor_creation": True,
            "failure_was_not_training_data_gradient_or_replay_error": True,
            "guard_snapshot_disagreement_cause": "unconfirmed; no other process inspected"}


def failed_queue_receipt(directory, expected_uuid, expected_output, qualification):
    """Accept exit1 only for the exact preserved original gpu_queue invocation."""
    qc, qualifier, _ = queue_modules()
    plan = qc.plan()
    p = inside(directory)
    a, c, e, command = [read_json(p / name) for name in
        ("authorization.json", "completion.json", "environment.json", "command.json")]
    if (a.get("schema") != "gnnm-authorized-allocation-supervision-v2"
        or a.get("ssh_destination") != plan["ssh_destination"]
        or a.get("route_record_sha256") != plan["route_sha256"]
        or a.get("allocation_guard_sha256") != plan["allocation_guard_sha256"]
        or a.get("visible_gpu_count") != 1 or a.get("gpu_uuid") != expected_uuid
        or a.get("gpu_name") != "NVIDIA A100-SXM4-80GB"
        or a.get("child_exit_code") != 1 or c.get("exit_code") != 1
        or a.get("inner_completion_sha256") != sha(p / "completion.json")
        or e.get("supervisor_sha256") != plan["inner_supervisor_sha256"]
        or e.get("script_sha256") != sha(QUEUE / "gpu_queue.py")
        or command.get("cwd") != str(qc.REPO)
        or Path(command["argv"][1]).resolve() != QUEUE / "gpu_queue.py"):
        raise ValueError("Only the exact completed failed original queue chain qualifies recovery")
    for field, name in (("environment_sha256", "environment.json"), ("command_sha256", "command.json"),
                         ("log_sha256", "stdout_stderr.log")):
        if c.get(field) != sha(p / name):
            raise ValueError("Original failed-supervisor artifact chain changed")
    qualifier.command_input_path(command["argv"], "--output", expected_output)
    qualifier.command_input_path(command["argv"], "--qualification", qualification)
    return {"supervisor_directory": str(p.relative_to(PHASE)), "exit_code": 1,
            "authorization_sha256": sha(p / "authorization.json"), "completion_sha256": sha(p / "completion.json"),
            "command_sha256": sha(p / "command.json"), "environment_sha256": sha(p / "environment.json"),
            "log_sha256": sha(p / "stdout_stderr.log"), "seconds": c["seconds"]}


def request_format(value):
    fields = {"schema", "source_sha256", "queue_qualification", "coordinator_reason", "original_queue_failure",
        "original_queue_events", "original_final_replay_launcher", "original_queue_supervisor_directory", "new_final_replay"}
    if (not isinstance(value, dict) or set(value) != fields
        or value.get("schema") != "gnnm-molhiv-single-replay-technical-recovery-request-v4"
        or value.get("source_sha256") != sources()):
        raise ValueError("Incomplete/unsealed narrow technical recovery request")
    return value


def assemble_value(request_path, *, qualified=None):
    """Re-audit complete six-cell metadata and one exact new replay, no tensors."""
    from admission import validate_six_rows
    qc, qualifier, gpu_queue = queue_modules()
    request = request_format(read_json(request_path))
    qualification_path = checked(request["queue_qualification"])
    stored_qualification = read_json(qualification_path)
    gpu_queue.qualification_binding(stored_qualification)
    if qualified is None:
        idle = qc.idle_guard()
        fresh = qualifier.qualification_value(QUEUE / "QUALIFICATION_REQUEST_authorized1gpu_run01.json", idle)
        gpu_queue.compare_requalification(stored_qualification, fresh)
        qualified = stored_qualification
    elif qualified != stored_qualification:
        raise ValueError("Recovery request substitutes another queue qualification")
    uuid = qualified["qualified_gpu_uuid"]
    failure_path = checked(request["original_queue_failure"])
    queue_dir = failure_path.parent
    if (failure_path.name != "failure.json" or queue_dir != QUEUE / "queues/all_six_authorized1gpu_run01"
        or (queue_dir / "queue_result.json").exists()):
        raise ValueError("Do not overwrite or reinterpret the failed original queue as a completed queue")
    events_path = checked(request["original_queue_events"])
    launcher_path = checked(request["original_final_replay_launcher"])
    if events_path != queue_dir / "events.jsonl" or launcher_path != queue_dir / (FINAL_LABEL + "_replay_launcher.log"):
        raise ValueError("Recovery must bind exact original events and guard-refusal launcher")
    events = [__import__("json").loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    failure = read_json(failure_path)
    reason_path = checked(request["coordinator_reason"])
    reason = read_json(reason_path)
    if reason.get("ssh_destination") != qc.plan()["ssh_destination"] or reason.get("original_queue") != str(queue_dir.relative_to(PHASE)):
        raise ValueError("Recovery reason does not name the exact corrected allocation and failed queue")
    technical = verify_technical_case(failure, events, reason, launcher_path.read_text())
    lock_path = queue_dir / "queue_lock.json"
    lock = read_json(lock_path)
    if (lock.get("source_sha256") != qc.source_hashes() or lock.get("qualification_sha256") != sha(qualification_path)
        or lock.get("bundle_sha256") != qualified["bundle_sha256"] or lock.get("protocol_sha256") != qualified["protocol_sha256"]
        or lock.get("GPU_uuid") != uuid or lock.get("outcome_scheduling") is not False
        or lock.get("automatic_retry") is not False or lock.get("maximum_parallel_cells") != 1):
        raise ValueError("Original queue scientific/input lock changed")
    failed = failed_queue_receipt(relative_path(request["original_queue_supervisor_directory"]), uuid, queue_dir, qualification_path)
    run_root = PARENT / "authorized_gpu_probes_v1" / queue_dir.name
    final_run = run_root / FINAL_LABEL
    original_missing_supervisor = queue_dir / "supervisors" / (FINAL_LABEL + "_replay")
    original_missing_report = run_root / "replays" / (FINAL_LABEL + ".json")
    if original_missing_supervisor.exists() or original_missing_report.exists():
        raise ValueError("Original pre-child-refused replay artifacts must remain absent")
    entry = request["new_final_replay"]
    if set(entry) != {"report", "supervisor_directory"}:
        raise ValueError("One explicit newly authorized replay required")
    new_report_path = checked(entry["report"])
    new_supervisor = qc.phase_path(entry["supervisor_directory"])
    if (new_report_path == original_missing_report or new_report_path.parent != run_root / "replays"
        or new_supervisor == original_missing_supervisor):
        raise ValueError("Recovery must use separate new outputs without replacing original failure evidence")
    new_receipt = qc.authorization(new_supervisor, uuid, "--output", new_report_path)
    argv = read_json(new_supervisor / "command.json")["argv"]
    if Path(argv[1]).resolve() != PARENT / "probe.py" or argv[2] != "replay":
        raise ValueError("Recovery accepts replay only; no new fit or modified scientific executable")
    qualifier.command_input_path(argv, "--run", final_run)
    qualifier.command_input_path(argv, "--train-bundle", relative_path(qualified["bundle_phase_relative"]))
    a = read_json(new_supervisor / "authorization.json")
    if utc(reason["UTC"]) >= utc(a["start_UTC"]):
        raise ValueError("The actual coordinator recovery reason must precede the single new replay")
    final_probe_supervisor = queue_dir / "supervisors" / (FINAL_LABEL + "_probe")
    probe_receipt = qc.authorization(final_probe_supervisor, uuid, "--output", final_run)
    final = {"seed": 2, "label_fraction": 1.0,
        "stages": {"probe": {"supervisor_phase_relative": str(final_probe_supervisor.relative_to(PHASE)), **probe_receipt},
                   "replay": {"supervisor_phase_relative": str(new_supervisor.relative_to(PHASE)), **new_receipt}},
        "selector_lock_sha256": sha(final_run / "selector_lock.json"), "full_replay_sha256": sha(new_report_path),
        "full_replay_phase_relative": str(new_report_path.relative_to(PHASE)),
        "parent_run_phase_relative": str(final_run.relative_to(PHASE))}
    rows = [*failure["completed_cells"], final]
    descriptors = validate_six_rows(rows, qualified)
    if len(descriptors) != 6:
        raise ValueError("Recovery cannot publish a partial source matrix")
    return {"schema": "gnnm-molhiv-single-replay-technical-recovery-result-v4",
        "status": "all_six_original_fits_and_complete_replays_qualified_after_one_technical_recovery",
        "source_sha256": sources(), "recovery_request": reference(request_path),
        "queue_qualification": reference(qualification_path), "results": rows,
        "original_failed_queue": {"failure": reference(failure_path), "events": reference(events_path),
            "queue_lock": reference(lock_path), "final_replay_launcher": reference(launcher_path),
            "failed_supervisor": failed},
        "coordinator_preexecution_reason": reference(reason_path),
        "new_final_replay": {"report": reference(new_report_path), "supervisor_directory": str(new_supervisor.relative_to(PHASE)), **new_receipt},
        "technical_case": technical, "all_six_cells_audited": True, "all_original_probe_fits_retained": True,
        "new_probe_fits": 0, "new_replay_jobs": 1, "automatic_retry": False, "outcome_dependent_retry_or_subset": False,
        "separate_prelaunch_final_manifest_digest_recorded": False,
        "current_original_manifest_sha256": next(d["artifact_manifest"]["sha256"] for d in descriptors if (d["seed"], d["label_fraction"]) == FINAL),
        "provenance_limit": "Pre-execution reason did not separately seal the final manifest digest. The authorized unchanged replay verifies the current complete original manifest, checkpoint/record identities and autograd. No separate prelaunch digest seal is claimed.",
        "original_missing_replay_report_and_supervisor_still_absent": True,
        "validation_or_test_loaded": False, "scientific_recipe_or_masks_or_selection_changed": False,
        "cost_accounting": {"failed_original_queue_seconds": failed["seconds"],
            "new_final_replay_seconds": new_receipt["seconds"],
            "original_successful_probe_seconds": sum(r["stages"]["probe"]["seconds"] for r in rows),
            "original_successful_replay_seconds": sum(r["stages"]["replay"]["seconds"] for r in rows[:-1]),
            "failure_and_recovery_overhead_must_be_retained": True}}
