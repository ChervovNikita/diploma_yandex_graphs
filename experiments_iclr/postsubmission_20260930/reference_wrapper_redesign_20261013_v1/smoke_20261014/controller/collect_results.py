"""Collect real-source engineering smoke receipts without importing model code.

The collector is read-only except for one fresh report directory. It does not
decode checkpoints, score TEST data, or qualify the full scientific campaign.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

CASE_SCHEMA = "real-source-short-smoke-v1"
PAIR_SCHEMA = "two-slot-real-source-pair-memory-v1"
DISPATCH_SCHEMA = "real-source-sequential-smoke-dispatch-v1"
EXCLUDED_DATASETS = {"roman-empire", "amazon-ratings", "minesweeper", "questions", "tolokers"}
SCOPE = (
    "Authentic-source short engineering tests only. Source-horizon and main "
    "tracks reuse this initial code path only where initial operator, optimizer "
    "and schedule semantics are identical. These receipts do not qualify full "
    "epochs, stopping, later phase transitions, convergence, distributed worker "
    "lifecycle, mature-training memory or all 2,480 planned fits. No full campaign "
    "fit or scientific accuracy result is implied."
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def receipt(path, raw=None):
    raw = path.read_bytes() if raw is None else raw
    return {"path": str(path), "bytes": len(raw), "sha256": digest(raw)}


def inside(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Deliberate path is outside the project: " + str(path))
    return resolved


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def finite_values(values, required_count):
    if any(not number(v) for v in values):
        return False
    return True if len(values) == required_count else None


def tri_checks(rows):
    values = [row.get("passed") if isinstance(row, dict) else None for row in rows]
    if False in values:
        return False
    return True if values and all(v is True for v in values) else None


def maxima(rows, name):
    values = [r[name] for r in rows if number(r.get(name))]
    return max(values) if values else None


def seed_for(case):
    recipe = case.get("recipe", {})
    return recipe.get("_seed", recipe.get("seed", case.get("seed")))


def utc_seconds(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.timestamp()
    except ValueError:
        return None


def is_warmup(path, explicit_roots):
    if any(path.is_relative_to(root) for root in explicit_roots):
        return True
    return any(any(word in part.lower() for word in ("warmup", "preflight", "unseeded"))
               for part in path.parts)


def work_counts(record):
    """Observer totals are exact; step-only sums are explicit lower bounds."""
    totals = record.get("actual_optimizer_step_calls_total")
    if isinstance(totals, dict):
        attempts, completed = totals.get("attempts"), totals.get("completed")
        if (isinstance(attempts, list) and isinstance(completed, list)
                and len(attempts) == len(completed)
                and all(type(v) is int and v >= 0 for v in attempts + completed)):
            return {"attempts_by_owner": attempts, "completed_by_owner": completed,
                    "attempts": sum(attempts), "completed": sum(completed),
                    "coverage": "exact_observer_total", "owner_count": len(attempts)}
    rows = [s.get("actual_optimizer_step_calls") for s in record.get("steps", [])]
    valid = [r for r in rows if isinstance(r, dict)
             and isinstance(r.get("attempts"), list) and isinstance(r.get("completed"), list)
             and len(r["attempts"]) == len(r["completed"])
             and all(type(v) is int and v >= 0 for v in r["attempts"] + r["completed"])]
    if valid:
        n = max(len(r["attempts"]) for r in valid)
        attempts = [sum(r["attempts"][i] for r in valid if len(r["attempts"]) > i) for i in range(n)]
        completed = [sum(r["completed"][i] for r in valid if len(r["completed"]) > i) for i in range(n)]
        return {"attempts_by_owner": attempts, "completed_by_owner": completed,
                "attempts": sum(attempts), "completed": sum(completed),
                "coverage": "recorded_steps_lower_bound", "owner_count": n}
    return {"attempts_by_owner": None, "completed_by_owner": None,
            "attempts": None, "completed": None, "coverage": "unknown", "owner_count": None}


def cost_summary(attempts, absent_terminal=0):
    counts = [a["work"] for a in attempts]
    walls = [a.get("wall_seconds") for a in attempts if number(a.get("wall_seconds"))]
    exact = sum(c["coverage"] == "exact_observer_total" for c in counts)
    lower = sum(c["coverage"] == "recorded_steps_lower_bound" for c in counts)
    unknown = sum(c["coverage"] == "unknown" for c in counts) + absent_terminal
    return {"unique_terminal_attempts": len(attempts),
            "absent_terminal_attempts": absent_terminal,
            "exact_observer_totals": exact, "step_only_lower_bounds": lower,
            "unknown_optimizer_work_attempts": unknown,
            "observed_optimizer_attempts": sum(c["attempts"] or 0 for c in counts),
            "observed_optimizer_completed": sum(c["completed"] or 0 for c in counts),
            "optimizer_sum_is_complete": unknown == 0 and lower == 0,
            "known_case_wall_seconds_sum": sum(walls),
            "wall_seconds_unknown_attempts": len(attempts) - len(walls) + absent_terminal,
            "wall_sum_is_not_elapsed_queue_time": True,
            "recorded_train_callback_rows": sum(a["train_callback_rows"] for a in attempts),
            "replayed_third_callback_rows": sum(a["replayed_callback_rows"] for a in attempts)}


def case_checks(record):
    steps = record.get("steps", [])
    validations = [record[k] for k in ("validation_after_second", "validation_uninterrupted_third",
                                      "validation_resumed_third") if isinstance(record.get(k), dict)]
    identities = [record.get("initial_parameter_identity"), record.get("reloaded_parameter_identity")]
    identities += [s.get("parameter_identity") for s in steps]
    replay = record.get("resume_checks", {})
    checkpoint = record.get("checkpoint", {})
    gradients = [s.get("gradients") for s in steps]
    failure_gradients = record.get("failure", {}).get("gradient_diagnostics") if isinstance(record.get("failure"), dict) else None
    gradient_status = tri_checks(gradients) if len(steps) == 4 else None
    if any(isinstance(g, dict) and g.get("passed") is False for g in gradients) or failure_gradients:
        gradient_status = False
    owners = record.get("initial_parameter_identity", {}).get("optimizer_owners")
    return {"finite_train_losses": finite_values([s.get("loss") for s in steps], 4),
            "finite_validation_losses": finite_values([v.get("loss") for v in validations], 3),
            "gradients": gradient_status,
            "parameter_identity": tri_checks(identities) if len(steps) == 4 else None,
            "checkpoint_receipt": bool(checkpoint.get("sha256") and number(checkpoint.get("bytes"))
                                       and checkpoint.get("full_source_state") is True) or None,
            "replay_outputs": replay.get("outputs", {}).get("passed"),
            "replay_model_optimizer_private_state": replay.get("model_optimizer_and_declared_private_state", {}).get("passed"),
            "replay_loss": replay.get("loss", {}).get("passed"),
            "replay_global_RNG": replay.get("global_RNG", {}).get("passed"),
            "immediate_loaded_full_state": tri_checks(list(record.get("immediate_loaded_state_checks", {}).values())),
            "independent4_aggregation": tri_checks([v.get("independent4_mean_aggregation") for v in validations])
                 if record.get("procedure") == "independent4" else "not_applicable",
            "source_data_provenance_supplied": bool(record.get("source_data_provenance")) and not
                 record.get("source_data_provenance", {}).get("additional_source_tensor_provenance_not_supplied", False),
            "optimizer_owners": owners}


def leading_output_count(validation):
    output = validation.get("outputs", {})
    tensors = output.get("tensors", [])
    if output.get("tensor_leaves") == 1 and tensors and tensors[0].get("shape"):
        return tensors[0]["shape"][0]
    return None


def valid_scope(scope, output_count):
    if not isinstance(scope, dict):
        return None
    if any(scope.get(k) is True for k in ("TEST_accessed", "TEST_opened", "TEST_scored", "test_scoring")):
        return False
    members = scope.get("member_validation_evidence")
    if isinstance(members, list):
        if len(members) != 4:
            return False
        statuses = [valid_scope(member, output_count) for member in members]
        return False if False in statuses else True if all(s is True for s in statuses) else None
    if scope.get("VALID_scope") == "complete_source_VALID":
        count = scope.get("VALID_nodes", scope.get("VALID_graphs"))
        if not number(count) or count <= 0 or output_count is None:
            return None
        return count == output_count
    pos, neg = scope.get("complete_VALID_positive_records"), scope.get("complete_VALID_negative_records")
    if number(pos) and number(neg) and pos > 0 and neg > 0:
        if scope.get("work", {}).get("all_official_VALID_records") is not True:
            return None
        return pos + neg == output_count
    return None


def complete_validation(record):
    rows = [record.get(k) for k in ("validation_after_second", "validation_uninterrupted_third",
                                   "validation_resumed_third")]
    metadata = record.get("source_metadata")
    legacy_count = metadata.get("valid_nodes") if isinstance(metadata, dict) else None
    if isinstance(metadata, dict) and isinstance(metadata.get("members"), list):
        counts = [m.get("valid_nodes") for m in metadata["members"] if isinstance(m, dict)]
        if len(counts) == 4 and len(set(counts)) == 1:
            legacy_count = counts[0]
    statuses = []
    bases = []
    for validation in rows:
        if not isinstance(validation, dict):
            statuses.append(None); bases.append("validation_summary_missing"); continue
        observed = leading_output_count(validation)
        status = valid_scope(validation.get("source_VALID_evidence"), observed)
        basis = "explicit_callback_scope_and_observed_output_count"
        if status is None and number(legacy_count) and legacy_count > 0:
            status = observed == legacy_count if observed is not None else None
            basis = "legacy_source_metadata_valid_nodes_and_observed_output_count"
        if status is None:
            basis = "complete_VALID_not_recorded; do_not_infer_from_pass_status"
        statuses.append(status); bases.append(basis)
    return {"passed": False if False in statuses else True if all(s is True for s in statuses) else None,
            "per_validation": statuses, "evidence_basis": bases}


def functional_completion(record, descriptor):
    """Separate executed save/reload work from strict floating replay outcome."""
    steps = record.get("steps", [])
    labels = [s.get("label") for s in steps]
    expected_labels = ["first_update", "second_update", "third_uninterrupted", "third_resumed"]
    owners = descriptor.get("expected_optimizer_owners", 4 if record.get("procedure") == "independent4" else 1)
    transitions = []
    for step in steps:
        calls = step.get("actual_optimizer_step_calls", {}).get("completed")
        transitions.append(isinstance(calls, list) and len(calls) == owners
                           and all(type(n) is int and n > 0 for n in calls))
    checks = case_checks(record)
    validity = complete_validation(record)
    loaded = record.get("immediate_loaded_state_checks")
    exact_supplied = isinstance(loaded, dict) and bool(loaded)
    exact_pass = tri_checks(list(loaded.values())) if exact_supplied else None
    replay = record.get("resume_checks", {})
    restored_global = loaded.get("global_RNG", {}).get("passed") if exact_supplied else None
    basic = {"all_four_callback_rows": labels == expected_labels,
             "real_owner_transitions_each_callback": len(transitions) == 4 and all(transitions),
             "finite_train_losses": checks["finite_train_losses"],
             "finite_validation_losses": checks["finite_validation_losses"],
             "finite_gradients": checks["gradients"],
             "complete_VALID": validity["passed"], "source_parameter_identity": checks["parameter_identity"],
             "checkpoint_receipt": checks["checkpoint_receipt"],
             "reloaded_parameter_identity": record.get("reloaded_parameter_identity", {}).get("passed"),
             "global_RNG_restore_evidence": restored_global if exact_supplied else replay.get("global_RNG", {}).get("passed")}
    if exact_supplied:
        basic["immediate_loaded_full_state"] = exact_pass
    status = False if False in basic.values() else True if all(v is True for v in basic.values()) else None
    strict = tri_checks([replay.get(k) for k in ("outputs", "model_optimizer_and_declared_private_state", "loss", "global_RNG")])
    return {"observed_functional_completion": status, "checks": basic,
            "complete_VALID_evidence": validity,
            "exact_restoration_qualified": exact_pass is True,
            "restoration_evidence_policy": "immediate_exact_full_saved_state_and_RNG" if exact_supplied
                 else "legacy_replay_evidence; not_exact_restoration_qualified",
            "strict_post_update_replay_passed": strict,
            "terminal_status_unchanged": record.get("status"),
            "functional_diagnostic_does_not_override_failure": True}


def eligibility(record, expected, path, pair_roots, warmup_roots):
    reasons = []
    cid = record.get("case_id")
    case = expected.get(cid)
    if case is None:
        reasons.append("unexpected_case_id")
    if any(path.is_relative_to(root) for root in pair_roots):
        reasons.append("pair_memory_child")
    if is_warmup(path, warmup_roots):
        reasons.append("warmup_or_preflight")
    if record.get("controller_seed", {}).get("bound_before_factory") is not True:
        reasons.append("global_seed_not_bound_before_factory")
    if record.get("status") not in ("passed", "failed"):
        reasons.append("not_terminal")
    if record.get("scientific_accuracy_result") is not False or record.get("dummy_data_fallback") is not False:
        reasons.append("incompatible_engineering_scope")
    if case is not None:
        if record.get("procedure") != case.get("procedure"):
            reasons.append("procedure_mismatch")
        if record.get("recipe") != case.get("recipe"):
            reasons.append("recipe_mismatch")
        if record.get("controller_seed", {}).get("value") != seed_for(case):
            reasons.append("seed_mismatch")
        if case.get("callback_sha256") and record.get("callback_source", {}).get("sha256") != case["callback_sha256"]:
            reasons.append("callback_hash_mismatch")
    return reasons


def write_csv(path, fields, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: canonical(row[name]) if isinstance(row.get(name), (dict, list))
                             else row.get(name) for name in fields})


def load_manifests(paths):
    expected = {}
    receipts = []
    for path in paths:
        raw = path.read_bytes()
        manifest = json.loads(raw)
        receipts.append(receipt(path, raw))
        if manifest.get("expected_case_count") is not None and manifest["expected_case_count"] != len(manifest["cases"]):
            raise ValueError("Manifest count mismatch: " + str(path))
        for case in manifest["cases"]:
            cid = case["id"]
            normalized_dataset = case.get("recipe", {}).get("dataset", "").lower().replace("_", "-").replace(" ", "-")
            if normalized_dataset in EXCLUDED_DATASETS:
                raise ValueError("Original-paper dataset is excluded: " + cid)
            if cid in expected and canonical(expected[cid]) != canonical(case):
                raise ValueError("Conflicting descriptors for case: " + cid)
            expected[cid] = case
    return expected, receipts


def queue_estimate(expected, selected, attempts, dispatches):
    """Observed short-case durations, with coverage and extrapolation explicit."""
    durations = {}
    all_completed = []
    for attempt in attempts:
        if not attempt["matrix_eligible"] or attempt["status"] != "passed" or not number(attempt.get("wall_seconds")):
            continue
        key = (attempt["callback_source"].get("sha256"), attempt["procedure"])
        durations.setdefault(key, []).append(attempt["wall_seconds"])
        all_completed.append(attempt["wall_seconds"])
    estimates = []
    uncovered = []
    groups = []
    for (sha, procedure), values in sorted(durations.items(), key=lambda row: str(row[0])):
        groups.append({"callback_sha256": sha, "procedure": procedure, "n_completed_passes": len(values),
                       "minimum_seconds": min(values), "median_seconds": statistics.median(values),
                       "maximum_seconds": max(values)})
    for cid, case in expected.items():
        if cid in selected:
            continue  # Failed terminal cases have ended; no automatic retry is implied.
        sha = case.get("callback_sha256")
        values = durations.get((sha, case["procedure"])) if sha else None
        if values is None:
            # Only a uniquely identified staged family hash can supply a source-family proxy.
            hashes = {a["callback_source"].get("sha256") for a in attempts
                      if a["matrix_eligible"] and a["callback_module"] == case.get("callback_module")}
            if len(hashes) == 1:
                values = durations.get((next(iter(hashes)), case["procedure"]))
        if values:
            estimates.append({"case_id": cid, "basis": "completed_same_callback_hash_and_procedure",
                              "n_observations": len(values), "median_proxy_seconds": statistics.median(values),
                              "observed_min_seconds": min(values), "observed_max_seconds": max(values)})
        else:
            uncovered.append(cid)
    queues = []
    by_case = {e["case_id"]: e for e in estimates}
    for dispatch in dispatches:
        rows = dispatch["data"].get("cases", [])
        pending = [r["case_id"] for r in rows if r.get("status") == "running"]
        # Unstarted rows are not present in a running sequential SUMMARY; the manifest
        # receipt may refer to a queue subset. Recover exact membership only if readable.
        queue_cases = dispatch.get("manifest_case_ids")
        if queue_cases:
            ended = {r["case_id"] for r in rows if r.get("status") in ("passed", "failed")}
            pending = [cid for cid in queue_cases if cid not in ended]
        known = [by_case[cid] for cid in pending if cid in by_case]
        queues.append({"summary_path": dispatch["receipt"]["path"],
                       "declared_queue_count": dispatch["data"].get("case_count"),
                       "terminal_rows": sum(r.get("status") in ("passed", "failed") for r in rows),
                       "remaining_exact_membership_known": queue_cases is not None,
                       "observed_running_case_ids": [r["case_id"] for r in rows if r.get("status") == "running"],
                       "pending_case_count_if_known": len(pending) if queue_cases is not None else None,
                       "covered_pending_case_count": len(known),
                       "sum_covered_short_case_median_proxies_seconds": sum(e["median_proxy_seconds"] for e in known),
                       "uncovered_pending_case_ids": [cid for cid in pending if cid not in by_case],
                       "actual_remaining_wall_ETA_seconds": None})
    return {"basis": "actual completed seeded passing short-case wall durations; family/procedure proxies only",
            "full_campaign_training_ETA": None, "actual_remaining_wall_ETA_seconds": None,
            "all_completed_short_cases_count": len(all_completed),
            "all_completed_short_cases_median_seconds": statistics.median(all_completed) if all_completed else None,
            "duration_groups": groups, "pending_case_proxies": estimates,
            "uncovered_pending_case_ids": uncovered, "queues": queues,
            "limitations": ["Dataset, dimensions and procedure work can change within one callback family.",
                            "Running-case elapsed work is not subtracted without a monotonic source clock.",
                            "Parallel sequential queues overlap; proxy sums are not elapsed wall time.",
                            "Failures, process startup and source preparation can change completion time."]}


def collect(root, manifests, results_roots, output, expected_count=167, warmup_roots=(), pair_extra=()):
    expected, manifest_receipts = load_manifests(manifests)
    if len(expected) != expected_count:
        raise ValueError(f"Expected {expected_count} unique cases, received {len(expected)}")
    if output.exists():
        raise ValueError("Preserve earlier reports; choose a fresh output directory")
    if any(output == r or r.is_relative_to(output) for r in results_roots):
        raise ValueError("Report output may not contain a supplied evidence root")
    paths = sorted({p.resolve() for base in results_roots for p in base.rglob("RESULT.json")})
    summaries = sorted({p.resolve() for base in results_roots for p in base.rglob("SUMMARY.json")})
    entries = []
    parse_errors = []
    for path in paths:
        inside(path, root)
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            entries.append({"path": path, "receipt": receipt(path, raw), "data": value,
                            "mtime_ns": path.stat().st_mtime_ns})
        except (OSError, ValueError, TypeError) as error:
            parse_errors.append({"path": str(path), "message": str(error)})
    pair_roots = {e["path"].parent for e in entries if e["data"].get("schema") == PAIR_SCHEMA}
    pair_roots.update(pair_extra)
    # An active pair has no terminal pair RESULT yet, but its barrier identifies children.
    pair_roots.update(p.parent for base in results_roots for p in base.rglob("barrier") if p.is_dir())
    dispatches = []
    timestamps = {}
    dispatch_only = []
    for path in summaries:
        inside(path, root)
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            if value.get("schema") != DISPATCH_SCHEMA:
                continue
            item = {"receipt": receipt(path, raw), "data": value, "manifest_case_ids": None}
            manifest_path = value.get("manifest", {}).get("path")
            if manifest_path:
                candidate = Path(manifest_path)
                if not candidate.is_absolute():
                    candidate = root / candidate
                if candidate.resolve().is_relative_to(root) and candidate.is_file():
                    queue_manifest = json.loads(candidate.read_text())
                    # A selected dispatcher subset has case_count smaller than declared.
                    if value.get("case_count") == len(queue_manifest.get("cases", [])):
                        item["manifest_case_ids"] = [c["id"] for c in queue_manifest["cases"]]
            dispatches.append(item)
            for row in value.get("cases", []):
                evidence = row.get("result", {}).get("sha256")
                start = utc_seconds(row.get("started_UTC"))
                if evidence and start is not None:
                    timestamps.setdefault(evidence, []).append((start, row["started_UTC"]))
                if not evidence and row.get("status") in ("passed", "failed"):
                    dispatch_only.append({"summary_receipt": item["receipt"], **row,
                                          "optimizer_cost": "unknown_without_terminal_observer_receipt"})
        except (OSError, ValueError, TypeError) as error:
            parse_errors.append({"path": str(path), "message": str(error)})
    unique = {}
    for entry in entries:
        unique.setdefault(entry["receipt"]["sha256"], []).append(entry)
    attempts = []
    pairs = []
    unknown_schemas = []
    step_rows = []
    for sha, copies in unique.items():
        entry = min(copies, key=lambda e: str(e["path"]))
        data = entry["data"]
        copy_paths = [str(c["path"]) for c in copies]
        if data.get("schema") == PAIR_SCHEMA:
            snapshots = [data.get("both_constructed", {})] + data.get("samples", [])
            pair = {"receipt": entry["receipt"], "copy_paths": copy_paths,
                    "case_ids": data.get("case_ids"), "status": data.get("status"),
                    "wall_seconds": data.get("wall_seconds"), "failure": data.get("failure"),
                    "two_jobs_same_gpu": data.get("two_jobs_same_gpu"),
                    "sampling_is_observation_not_complete_upper_bound": True,
                    "maximum_sampled_own_GPU_usage_mib": data.get("maximum_sampled_own_GPU_usage_mib",
                        maxima(snapshots, "own_GPU_usage_mib")),
                    "minimum_sampled_device_free_mib": data.get("minimum_sampled_device_free_mib"),
                    "device_before": data.get("before", {}).get("device"),
                    "both_constructed": data.get("both_constructed"), "workers": data.get("workers", []),
                    "sample_count": len(data.get("samples", []))}
            if pair["minimum_sampled_device_free_mib"] is None:
                free = [s.get("device", {}).get("free_mib") for s in snapshots]
                free = [v for v in free if number(v)]
                pair["minimum_sampled_device_free_mib"] = min(free) if free else None
            pairs.append(pair)
            continue
        if data.get("schema") != CASE_SCHEMA:
            unknown_schemas.append({"receipt": entry["receipt"], "schema": data.get("schema"), "copy_paths": copy_paths})
            continue
        # Identical copies are one attempt. Any copy rooted in the pair/warmup
        # evidence retains that classification even if a report mirror is supplied.
        reasons = set()
        for copied in copies:
            reasons.update(eligibility(data, expected, copied["path"], pair_roots, warmup_roots))
        if timestamps.get(sha):
            stamp, stamp_text = min(timestamps[sha])
            stamp_basis = "dispatcher_started_UTC"
        else:
            stamp = min(c["mtime_ns"] for c in copies) / 1_000_000_000
            stamp_text = dt.datetime.fromtimestamp(stamp, dt.timezone.utc).isoformat()
            stamp_basis = "earliest_available_copy_filesystem_mtime_fallback"
        resource = data.get("resource_observations", [])
        cid = data.get("case_id")
        case = expected.get(cid, {})
        work = work_counts(data)
        attempt = {"attempt_sha256": sha, "receipt": entry["receipt"], "copy_paths": copy_paths,
                   "case_id": cid, "procedure": data.get("procedure"), "status": data.get("status"),
                   "recipe_id": data.get("recipe", {}).get("id"), "dataset": data.get("recipe", {}).get("dataset"),
                   "recipe_sha256": digest(canonical(data.get("recipe")).encode()),
                   "controller_seed": data.get("controller_seed"),
                   "callback_source": data.get("callback_source", {}),
                   "callback_module": case.get("callback_module"),
                   "source_metadata": data.get("source_metadata"),
                   "source_data_provenance": data.get("source_data_provenance"),
                   "matrix_eligible": not reasons, "excluded_reasons": sorted(reasons),
                   "attempt_order_UTC": stamp_text, "attempt_order_basis": stamp_basis,
                   "_order": stamp, "checks": case_checks(data), "work": work,
                   "train_callback_rows": len(data.get("steps", [])),
                   "replayed_callback_rows": sum(s.get("label") == "third_resumed" for s in data.get("steps", [])),
                   "logical_trajectory_updates": data.get("logical_trajectory_updates"),
                   "final_optimizer_state_clocks": data.get("final_optimizer_state_clocks"),
                   "construction_seconds": data.get("construction_seconds"), "wall_seconds": data.get("wall_seconds"),
                   "cuda_peak_allocated_bytes": maxima(resource, "cuda_peak_allocated_bytes"),
                   "cuda_peak_reserved_bytes": maxima(resource, "cuda_peak_reserved_bytes"),
                   "peak_rss_bytes": maxima(resource, "peak_rss_bytes"),
                   "checkpoint": data.get("checkpoint"), "resume_checks": data.get("resume_checks"),
                   "controller_source": data.get("controller_source"),
                   "replay_criterion_version": data.get("replay_criterion_version", "legacy-coordinate-v1"),
                   "replay_criterion": data.get("replay_criterion"),
                   "immediate_loaded_state_checks": data.get("immediate_loaded_state_checks"),
                   "failure": data.get("failure"), "scientific_accuracy_result": False}
        attempt["functional_completion"] = functional_completion(data, case)
        attempt["source_operation_record_rows"] = sum(bool(s.get("source_operation_counts")) for s in data.get("steps", []))
        replay = data.get("resume_checks", {})
        attempt["failure_interpretation"] = (
            "legacy_coordinate_replay_check_definition_rejection; retained failed; needs newly executed amended check"
            if data.get("status") == "failed" and not data.get("replay_criterion_version")
            and replay.get("model_optimizer_and_declared_private_state", {}).get("passed") is False
            and all(replay.get(k, {}).get("passed") is True for k in ("outputs", "loss", "global_RNG"))
            else None)
        attempts.append(attempt)
        for step in data.get("steps", []):
            step_rows.append({"attempt_sha256": sha, "case_id": cid, "procedure": data.get("procedure"),
                              "attempt_status": data.get("status"), "label": step.get("label"),
                              "seconds": step.get("seconds"), "loss": step.get("loss"),
                              "actual_optimizer_step_calls": step.get("actual_optimizer_step_calls"),
                              "optimizer_state_clocks": step.get("optimizer_state_clocks"),
                              "source_operation_counts": step.get("source_operation_counts"),
                              "gradients": step.get("gradients"), "parameter_identity": step.get("parameter_identity")})
    attempts.sort(key=lambda a: (a["_order"], a["attempt_sha256"]))
    selected = {}
    for attempt in attempts:
        if attempt["matrix_eligible"]:
            selected[attempt["case_id"]] = attempt
    case_rows = []
    for cid, case in expected.items():
        chosen = selected.get(cid)
        if chosen is None:
            case_rows.append({"case_id": cid, "procedure": case["procedure"],
                              "recipe_id": case["recipe"].get("id"), "dataset": case["recipe"].get("dataset"),
                              "status": "missing", "terminal_matrix_receipt": False,
                              "all_terminal_attempts": sum(a["case_id"] == cid for a in attempts)})
            continue
        chosen["selected_matrix_attempt"] = True
        row = {k: chosen.get(k) for k in ("case_id", "procedure", "recipe_id", "dataset", "status",
                                         "attempt_sha256", "receipt", "attempt_order_UTC", "attempt_order_basis",
                                         "replay_criterion_version", "controller_source",
                                         "wall_seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "peak_rss_bytes")}
        row.update(terminal_matrix_receipt=True, all_terminal_attempts=sum(a["case_id"] == cid for a in attempts),
                   **chosen["checks"], actual_optimizer_attempts=chosen["work"]["attempts"],
                   actual_optimizer_completed=chosen["work"]["completed"],
                   optimizer_work_coverage=chosen["work"]["coverage"],
                   optimizer_owner_count=chosen["work"]["owner_count"],
                   checkpoint_sha256=(chosen.get("checkpoint") or {}).get("sha256"),
                   checkpoint_bytes=(chosen.get("checkpoint") or {}).get("bytes"),
                   final_optimizer_state_clocks=chosen.get("final_optimizer_state_clocks"),
                   failure_phase=(chosen.get("failure") or {}).get("phase"),
                   failure_message=(chosen.get("failure") or {}).get("message"))
        row.update(functional_completion=chosen["functional_completion"]["observed_functional_completion"],
                   functional_complete_VALID=chosen["functional_completion"]["complete_VALID_evidence"]["passed"],
                   exact_restoration_qualified=chosen["functional_completion"]["exact_restoration_qualified"],
                   functional_restoration_evidence_policy=chosen["functional_completion"]["restoration_evidence_policy"],
                   strict_post_update_replay_passed=chosen["functional_completion"]["strict_post_update_replay_passed"],
                   source_operation_record_rows=chosen["source_operation_record_rows"])
        case_rows.append(row)
    for attempt in attempts:
        attempt.setdefault("selected_matrix_attempt", False)
    warmups = [a for a in attempts if "warmup_or_preflight" in a["excluded_reasons"] or
               "global_seed_not_bound_before_factory" in a["excluded_reasons"]]
    pair_children = [a for a in attempts if "pair_memory_child" in a["excluded_reasons"]]
    excluded = [a for a in attempts if not a["matrix_eligible"]]
    known_receipts = {a["attempt_sha256"] for a in attempts}
    # A terminal dispatcher receipt may reference a missing/deleted RESULT file.
    for dispatcher in dispatches:
        for row in dispatcher["data"].get("cases", []):
            sha = row.get("result", {}).get("sha256")
            if sha and sha not in known_receipts and row.get("status") in ("passed", "failed"):
                dispatch_only.append({"summary_receipt": dispatcher["receipt"], **row,
                                      "optimizer_cost": "unknown_missing_terminal_receipt"})
    # Dispatcher copies are metadata duplicates, not additional launched work.
    dispatch_only = list({canonical({k: row.get(k) for k in ("case_id", "output", "started_UTC", "pid")}): row
                          for row in dispatch_only}.values())
    passed = sum(r["status"] == "passed" for r in case_rows)
    failed = sum(r["status"] == "failed" for r in case_rows)
    missing = [r["case_id"] for r in case_rows if r["status"] == "missing"]
    estimates = queue_estimate(expected, selected, attempts, dispatches)
    counts = {"expected_matrix_cases": len(expected), "selected_seeded_terminal_cases": len(selected),
              "passed": passed, "failed": failed, "missing": len(missing),
              "all_unique_terminal_case_attempts": len(attempts), "deduplicated_case_copy_paths":
              sum(len(a["copy_paths"]) - 1 for a in attempts),
              "warmup_or_unseeded_attempts": len(warmups), "pair_child_attempts": len(pair_children),
              "pair_memory_records": len(pairs), "dispatcher_terminal_without_receipt": len(dispatch_only)}
    policy_counts = {}
    for attempt in selected.values():
        version = attempt["replay_criterion_version"]
        policy_counts[version] = policy_counts.get(version, 0) + 1
    costs = {"all_unique_case_work": cost_summary(attempts, len(dispatch_only)),
             "selected_matrix": cost_summary(list(selected.values())),
             "excluded_attempts": cost_summary(excluded), "warmup_or_unseeded": cost_summary(warmups),
             "pair_children": cost_summary(pair_children),
             "categories_may_overlap": "warmup and pair categories are subsets of excluded; do not add these subtotals"}
    for attempt in attempts:
        attempt.pop("_order")
    report = {"schema": "real-source-short-smoke-collection-v1",
              "created_UTC": dt.datetime.now(dt.timezone.utc).isoformat(), "scope": SCOPE,
              "collector_source": receipt(Path(__file__).resolve()),
              "expected_manifest_receipts": manifest_receipts, "results_roots": [str(r) for r in results_roots],
              "selection_rule": "latest eligible globally seeded terminal attempt, including failures; never latest success",
              "deduplication_rule": "identical raw terminal RESULT bytes count once; all copy paths retained",
              "timestamp_fallback": "dispatcher start UTC; otherwise earliest available copy mtime; not an authoritative launch timestamp",
              "counts": counts, "missing_case_ids": missing, "case_matrix": case_rows,
              "selected_replay_criterion_versions": policy_counts,
              "functional_completion_counts": {
                  "selected_observed_functional_completion": sum(a["functional_completion"]["observed_functional_completion"] is True for a in selected.values()),
                  "selected_functional_completion_unknown": sum(a["functional_completion"]["observed_functional_completion"] is None for a in selected.values()),
                  "selected_exact_restoration_qualified": sum(a["functional_completion"]["exact_restoration_qualified"] for a in selected.values()),
                  "selected_strict_replay_failures": sum(a["functional_completion"]["strict_post_update_replay_passed"] is False for a in selected.values()),
                  "all_unique_recorded_callback_rows_without_source_operations": sum(a["train_callback_rows"] - a["source_operation_record_rows"] for a in attempts)},
              "attempts": attempts, "pair_memory": pairs, "actual_work_costs": costs,
              "queue_duration_observations": estimates,
              "dispatcher_receipts": [d["receipt"] for d in dispatches],
              "dispatcher_terminal_without_case_receipt": dispatch_only,
              "parse_errors": parse_errors, "unexpected_result_schemas": unknown_schemas,
              "matrix_complete_and_all_passed": passed == len(expected) and not parse_errors,
              "scientific_accuracy_claims": 0, "full_campaign_GO": False}
    output.mkdir(parents=True)
    (output / "RESULTS.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    case_fields = ["case_id", "procedure", "recipe_id", "dataset", "status", "terminal_matrix_receipt",
                   "attempt_sha256", "receipt", "all_terminal_attempts", "attempt_order_UTC", "attempt_order_basis",
                   "finite_train_losses", "finite_validation_losses", "gradients", "parameter_identity",
                   "checkpoint_receipt", "checkpoint_sha256", "checkpoint_bytes", "replay_outputs",
                   "replay_model_optimizer_private_state", "replay_loss", "replay_global_RNG",
                   "immediate_loaded_full_state", "replay_criterion_version", "controller_source",
                   "functional_completion", "functional_complete_VALID", "exact_restoration_qualified",
                   "functional_restoration_evidence_policy", "strict_post_update_replay_passed", "source_operation_record_rows",
                   "independent4_aggregation", "source_data_provenance_supplied", "optimizer_owner_count",
                   "actual_optimizer_attempts", "actual_optimizer_completed", "optimizer_work_coverage",
                   "final_optimizer_state_clocks", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
                   "peak_rss_bytes", "wall_seconds", "failure_phase", "failure_message"]
    write_csv(output / "CASE_CHECKS.csv", case_fields, case_rows)
    write_csv(output / "ATTEMPTS.csv", ["attempt_sha256", "case_id", "procedure", "recipe_id", "dataset", "status",
              "matrix_eligible", "selected_matrix_attempt", "excluded_reasons", "receipt", "copy_paths",
              "controller_seed", "callback_source", "attempt_order_UTC", "attempt_order_basis", "work",
              "replay_criterion_version", "controller_source", "failure_interpretation",
              "functional_completion", "source_operation_record_rows",
              "train_callback_rows", "replayed_callback_rows", "wall_seconds", "cuda_peak_allocated_bytes",
              "cuda_peak_reserved_bytes", "peak_rss_bytes", "failure"], attempts)
    write_csv(output / "STEP_WORK.csv", ["attempt_sha256", "case_id", "procedure", "attempt_status", "label",
              "seconds", "loss", "actual_optimizer_step_calls", "optimizer_state_clocks", "source_operation_counts",
              "gradients", "parameter_identity"], step_rows)
    write_csv(output / "PAIR_MEMORY.csv", ["receipt", "copy_paths", "case_ids", "status", "two_jobs_same_gpu",
              "sample_count", "maximum_sampled_own_GPU_usage_mib", "minimum_sampled_device_free_mib",
              "wall_seconds", "workers", "failure", "sampling_is_observation_not_complete_upper_bound"], pairs)
    lines = ["# Actual-source short smoke receipts", "", SCOPE, "", "## Matrix coverage", "",
             f"Expected {len(expected)}; selected seeded terminal cases {len(selected)}; passed {passed}; failed {failed}; missing {len(missing)}.",
             "", "Every declared case has a row in `CASE_CHECKS.csv`. All unique attempts, exclusions and failure records remain in `RESULTS.json` and `ATTEMPTS.csv`. Per-step source operations and optimizer clocks remain in `STEP_WORK.csv`.",
             "", "Selection uses the latest eligible terminal attempt, including failures. An initial unseeded pass or pair-memory child cannot satisfy a matrix case. Identical evidence copies are counted once.",
             "", "A separate functional-completion diagnostic reports actual four-callback optimizer work, finite losses/gradients, complete VALID, identity and save/reload evidence. Strict floating replay failures retain failed status. Legacy replay evidence is not exact-restoration-qualified; absent old VALID/source-operation receipts remain unknown.",
             "", "## Actual charged work", "",
             "| Category | Terminal attempts | Observed optimizer attempts | Completed calls | Unknown costs | Recorded replay callbacks |", "|---|---:|---:|---:|---:|---:|"]
    for category in ("all_unique_case_work", "selected_matrix", "excluded_attempts", "warmup_or_unseeded", "pair_children"):
        c = costs[category]
        lines.append(f"| {category} | {c['unique_terminal_attempts']} | {c['observed_optimizer_attempts']} | {c['observed_optimizer_completed']} | {c['unknown_optimizer_work_attempts']} | {c['replayed_third_callback_rows']} |")
    lines += ["", "Unknown work is not zero; step-only totals are lower bounds. Failed callbacks and four independent owners are charged by observed optimizer calls. Subcategories overlap and must not be added. Wall sums count worker time, not elapsed queue time.",
              "", "## Queue duration observations", "",
              f"Completed seeded passing duration observations: {estimates['all_completed_short_cases_count']}; median seconds: {estimates['all_completed_short_cases_median_seconds']}. Pending cases with same-source/procedure duration proxies: {len(estimates['pending_case_proxies'])}; pending cases without such coverage: {len(estimates['uncovered_pending_case_ids'])}.",
              "", "The report preserves actual completed durations and exact queue membership where available. It does not invent a completion ETA or infer a full-training ETA from four callbacks. Family-level proxies may span different dataset sizes and are explicitly labeled.",
              "", "## Pair-memory observations", "",
              f"Terminal pair records: {len(pairs)}. Separate sampled joint memory and child outcomes are in `PAIR_MEMORY.csv`. Samples are physical observations, not complete mature-training upper bounds.",
              "", "## Remaining evidence", "",
              f"Parse errors: {len(parse_errors)}; terminal dispatcher attempts without a case receipt: {len(dispatch_only)}; unexpected RESULT schemas: {len(unknown_schemas)}.", ""]
    if missing:
        lines += ["Missing eligible terminal case IDs:", ""] + ["- `" + cid + "`" for cid in missing] + [""]
    if failed:
        lines += ["Selected failed case IDs:", ""] + ["- `" + r["case_id"] + "`: " + str(r.get("failure_message"))
                  for r in case_rows if r["status"] == "failed"] + [""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifests", required=True, nargs="+")
    parser.add_argument("--results-roots", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-case-count", type=int, default=167)
    parser.add_argument("--warmup-roots", nargs="*", default=[])
    parser.add_argument("--pair-roots", nargs="*", default=[])
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    try:
        manifests = [inside(p, root) for p in args.manifests]
        results_roots = [inside(p, root) for p in args.results_roots]
        if not all(p.is_dir() for p in results_roots):
            raise ValueError("Every supplied results root must exist as a directory")
        output = inside(args.output, root)
        report = collect(root, manifests, results_roots, output, args.expected_case_count,
                         [inside(p, root) for p in args.warmup_roots],
                         [inside(p, root) for p in args.pair_roots])
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, "Collector failed: " + str(error) + "\n")
    print(json.dumps({"output": str(output), "counts": report["counts"],
                      "matrix_complete_and_all_passed": report["matrix_complete_and_all_passed"],
                      "scientific_accuracy_claims": 0}))
    if args.require_final and not report["matrix_complete_and_all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
