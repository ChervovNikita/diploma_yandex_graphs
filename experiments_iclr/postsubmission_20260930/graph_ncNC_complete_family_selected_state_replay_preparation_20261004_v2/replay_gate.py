"""Stdlib-only admission and immutable complete-family custody. No deserialization."""
from pathlib import Path
from hashlib import sha256
from math import isfinite
import json
import os

HERE = Path(__file__).resolve().parent
DRIVER_SHA = "a59c669356e1a1437ce90107a7c76eb8f0b5579dc48d67b4378a7e237e956b7e"
DESIGN_SHA = "e97aeb2655d54f261acca1874c8e03047608625aac1b5e1e980853f7d44e0004"
PROTOTYPE_SHA = "a99b0e3b0e8ec597b03e2c390ac176597b5b6422d365fc20624ab98a113d42f9"
RESOURCE_SHA = "031c6a1fc36514a7dd1b7b520e5ec251b2c3f385bcab917ca101f6184712790a"
DATA_SHA = "df6980064b3ee31377e6d96215146e1d954abf5464182464582801eddf17077d"
RUNTIME_SHA = "e63f602baa8a91e129fb4a0c0debd6ec282e7cd132be87eb66cd8c5cf0b7c882"
DATA_FILES = {"raw/node-feat.csv.gz", "raw/edge.csv.gz", "split/time/train.pt", "split/time/valid.pt"}
UNITS = ("native_bank4", "factor_private4", "factor_pooled4", "native70")
ARMS = ("native_single_64", "independent_native_4", "factorized_private_4", "factorized_pooled_after_clamp_4", "native_single_70")
EXPECTED_ARMS = {UNITS[0]: ARMS[:2], UNITS[1]: ARMS[2:3], UNITS[2]: ARMS[3:4], UNITS[3]: ARMS[4:]}
EXPECTED_UNITS = {(u, s) for u in UNITS for s in range(5)}
EXPECTED_CELLS = {(a, s) for a in ARMS for s in range(5)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and str(path) == str(path.resolve()), "Noncanonical/borrowed absolute path")
    return path


def bound(pin, *, root=None, name=None, decode=False):
    if root is None:
        path = canonical(pin["path"])
    else:
        require(Path(pin["path"]).name == pin["path"], "Unit artifact path escaped custody")
        path = root / pin["path"]
        require(path.resolve() == path and path.parent == root, "Unit artifact is borrowed/symlinked")
    require(name is None or path.name == name, "Bound artifact filename differs")
    require(type(pin["bytes"]) is int and pin["bytes"] >= 0 and path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"], "Bound artifact custody differs")
    return json.loads(path.read_text()) if decode else path


def verify_manifest(root, expected):
    root = canonical(str(root))
    require(file_sha(root / "MANIFEST.json") == expected, "Source manifest differs")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    rows = manifest["files"]
    require(rows and len({r["path"] for r in rows}) == len(rows), "Duplicate/empty source manifest")
    for row in rows:
        path = root / row["path"]
        require(path.resolve() == path and path.is_relative_to(root), "Source path escaped or borrowed")
        require(path.stat().st_size == row.get("bytes", row.get("size")) and file_sha(path) == row["sha256"], "Source payload differs")
    return manifest


def admission_context(release_path, output, *, expected_release_sha256=None):
    """Read-only metadata/source custody; output existence is checked separately."""
    release_path = canonical(str(Path(release_path).expanduser()))
    require(expected_release_sha256 is None or file_sha(release_path) == expected_release_sha256, "Admitted root release bytes changed")
    release = json.loads(release_path.read_text())
    require(release.get("schema") == "ncnc-selected-state-replay-root-release-v1" and release.get("execution_enabled") is True, "Replay root release is disabled/different")
    require(release.get("root_authorization_reference") and release.get("authorized_stages") == ["selected_state_replay"], "Replay lacks separate root authorization")
    sidecar_sha = file_sha(HERE / "MANIFEST.json")
    require(release["sidecar_manifest_sha256"] == sidecar_sha, "Release targets different sidecar")
    verify_manifest(HERE, sidecar_sha)
    source_bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    require(release["family_lock"] == source_bindings["original_lock_provenance"]["remote"], "Release targets another immutable lock descriptor")
    paths = {key: canonical(release[key]) for key in ("driver_root", "design_root", "prototype_root", "resource_root")}
    for key, expected in (("driver_root", DRIVER_SHA), ("design_root", DESIGN_SHA), ("prototype_root", PROTOTYPE_SHA), ("resource_root", RESOURCE_SHA)):
        require(release[key.replace("_root", "_manifest_sha256")] == expected, "Release source identity differs")
        verify_manifest(paths[key], expected)
    require(release["data_authority_sha256"] == DATA_SHA and release["runtime_authority_sha256"] == RUNTIME_SHA, "Original authority identity differs")
    authority = bound(release["data_authority"], decode=True)
    runtime = bound(release["runtime_authority"], decode=True)
    require(release["data_authority"]["sha256"] == DATA_SHA and release["runtime_authority"]["sha256"] == RUNTIME_SHA, "Authority binding differs")
    require(authority["schema"] == "ncnc-collab-TRAIN-VALID-data-authority-v1" and set(authority["files"]) == DATA_FILES and authority["test_file_opened"] is False, "Only original TRAIN/raw/VALID authority admitted")
    require(runtime["schema"] == "ncnc-collab-predictive-runtime-authority-v1" and runtime["ordinary_host_execution"] is True and runtime["TF32"] is False and runtime["mixed_precision"] is False and runtime["deterministic_algorithms"] is False, "Original ordinary runtime differs")
    output = canonical(str(Path(output).expanduser()))
    invocation = {"stage": "selected_state_replay", "output_directory": str(output), "cuda_visible_devices": release["cuda_visible_devices"]}
    require(release["authorized_invocations"] == [invocation], "Exactly one output/GPU invocation required")
    require(release["cuda_visible_devices"] in authority["physical_GPU_UUIDs"] and os.environ.get("CUDA_VISIBLE_DEVICES") == release["cuda_visible_devices"], "Exact admitted GPU UUID required")
    lock_root = canonical(release["family_lock_output_directory"])
    require(release["family_lock"]["path"] == str(lock_root / "FAMILY_LOCK.json"), "Borrowed family lock")
    identity = {"driver_manifest_sha256": DRIVER_SHA, "design_manifest_sha256": DESIGN_SHA, "prototype_manifest_sha256": PROTOTYPE_SHA, "resource_manifest_sha256": RESOURCE_SHA, "data_authority_sha256": DATA_SHA, "runtime_authority_sha256": RUNTIME_SHA, "family_id": release["family_id"], "family_lock_output_directory": str(lock_root)}
    require(type(identity["family_id"]) is str and bool(identity["family_id"]), "Missing original family identity")
    for key, schema in (("independent_source_review", "ncnc-selected-state-replay-source-review-v1"), ("ordinary_runtime_synthetic_qualification", "ncnc-selected-state-replay-synthetic-qualification-v1"), ("runtime_resource_admission", "ncnc-selected-state-replay-runtime-resource-admission-v1")):
        receipt = bound(release[key], decode=True)
        require(receipt["schema"] == schema and receipt["status"] == "PASS" and receipt["sidecar_manifest_sha256"] == sidecar_sha and receipt["identity"] == identity, "Separate sidecar review/qualification/admission is incomplete")
        if key == "ordinary_runtime_synthetic_qualification":
            expected_cases = {"baseline_all25", "custody_after_metadata", "selected_state", "raw_score", "selected_metric", "scoring", "accounting", "admitted_release_after_scoring"}
            require(receipt["source_entry"] == descriptor(HERE / "replay_synthetic.py") and receipt["runtime_authority"] == release["runtime_authority"] and receipt["fabricated_inputs_only"] is True and receipt["study_lock_data_outcome_checkpoint_accessed"] is False and receipt["TEST_opened"] is False and receipt["scientific_fit_updates"] == 0, "Qualification source/runtime/fabricated scope differs")
            require(receipt["case_count"] == len(receipt["cases"]) == 8 and {r["case"] for r in receipt["cases"]} == expected_cases and all(r["status"] == "PASS" for r in receipt["cases"]), "All reviewed qualification cases must pass")
            require(receipt["actual_work"] == {"actual_engineering_updates": 40, "reference_score_valid_calls": 44, "attempted_score_valid_calls": 284, "actual_original_score_valid_invocations": 283}, "Qualification work/routing evidence is incomplete")
        if key == "runtime_resource_admission":
            require(receipt["family_lock_sha256"] == release["family_lock"]["sha256"] and receipt["invocation"] == invocation and receipt["dispatch_recheck_required"] is True, "Runtime/resource admission binding differs")
    return {"release": release, "release_path": release_path, "release_sha256": file_sha(release_path), "sidecar_manifest_sha256": sidecar_sha, "paths": paths, "plan": json.loads((paths["design_root"] / "PILOT_PLAN.json").read_text()), "authority": authority, "runtime": runtime, "identity": identity, "output": output}


def fresh_output_gate(context):
    output = context["output"]
    roots = [HERE, *context["paths"].values()]
    if not context.get("synthetic_only"):
        roots.append(Path(context["identity"]["family_lock_output_directory"]))
    require(not output.exists() and all(not output.is_relative_to(r) for r in roots), "Output exists or belongs to immutable source/input")
    require(not any((p / "MANIFEST.json").exists() for p in (output, *output.parents)), "Output lies inside a sealed packet")


def preflight(release_path, output):
    context = admission_context(release_path, output)
    fresh_output_gate(context)
    return context


def runtime_and_data_custody(context):
    """Hash actual admitted data/runtime bytes, without importing numerical code."""
    import importlib.metadata
    import sys
    runtime, authority = context["runtime"], context["authority"]
    executable = Path(sys.executable).resolve()
    require(executable == Path(runtime["interpreter_path"]).resolve() and file_sha(executable) == runtime["interpreter_sha256"], "Admitted runtime interpreter changed")
    require({name: importlib.metadata.version(name) for name in runtime["distribution_versions"]} == runtime["distribution_versions"], "Admitted runtime distribution versions changed")
    for pin in [*runtime["runtime_source_pins"], *runtime["runtime_binary_files"], runtime["negative_sampler"], authority["ogb_evaluator"]]:
        path = canonical(pin["path"])
        require(("bytes" not in pin or path.stat().st_size == pin["bytes"]) and file_sha(path) == pin["sha256"], "Admitted runtime/evaluator/sampler bytes changed")
    if context.get("synthetic_only"):
        require(authority["files"] == {} and context["synthetic_tensor_custody"], "Synthetic custody must exclude study data and bind fabricated tensors")
        bound(context["synthetic_tensor_custody"])
    else:
        root = canonical(authority["dataset_root"])
        require(set(authority["files"]) == DATA_FILES, "Data authority file set changed")
        for relative, pin in authority["files"].items():
            path = root / relative
            require(path.resolve() == path and path.is_relative_to(root) and path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"], "Admitted actual TRAIN/raw/VALID bytes changed")


def admitted_input_custody(context):
    """Complete reusable final custody guard, with no output-freshness test."""
    if context.get("synthetic_only"):
        from replay_synthetic_admission import admitted_input_custody as synthetic_custody
        return synthetic_custody(context)
    repeated = admission_context(context["release_path"], context["output"], expected_release_sha256=context["release_sha256"])
    for key in ("release", "paths", "plan", "authority", "runtime", "identity", "sidecar_manifest_sha256"):
        require(repeated[key] == context[key], "Admitted metadata/source/receipt context changed")
    runtime_and_data_custody(context)
    return {"source_release_authority_qualification_actual_data_runtime_custody": "PASS", "output_freshness_rechecked": False}


def number(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def validate_complete(complete, identity, key):
    unit, seed = key
    count = 4 if unit == "native_bank4" else 1
    require(complete["schema"] == "ncnc-pilot-complete-unit-v1" and complete["identity"] == identity and (complete["unit"], complete["base_seed"]) == key, "Complete identity differs")
    require(complete["epochs"] == 100 and complete["unique_fits"] == count and complete["completed_optimizer_steps"] == 1700 * count, "Incomplete scientific fit budget")
    require(complete["scientific_initialization"] == "fresh" and complete["resource_state_donor"] is False and complete["test_file_opened"] is False and complete["all_official_VALID_rows_complete"] is True, "Donor/TEST/coverage contract differs")
    require(set(complete["selected_checkpoints"]) == set(complete["selection_artifacts"]) == set(EXPECTED_ARMS[unit]), "Missing/duplicate served arm")
    require(complete["independent_candidate_count"] == (101 if count == 4 else 0) and complete["candidate101_extra_VALID_evaluations"] == (4 if count == 4 else 0), "Candidate101 work omitted")
    require(len(complete["member_metadata"]) == count, "Fit member denominator differs")
    for member, record in enumerate(complete["member_metadata"]):
        require(record["member"] == member and record["fit_seed"] == (seed + 5 * member if count == 4 else seed), "Fit seed/member differs")
        require([e["epoch"] for e in record["epochs"]] == list(range(1, 101)), "Incomplete/duplicate/reordered epoch stream")
        for epoch in record["epochs"]:
            require(epoch["full_batches"] == epoch["optimizer_steps"] == 17 and epoch["positive_queries"] == 60084 and epoch["negative_queries"] == 100000, "TRAIN/VALID rows omitted")
            stream = epoch["stream"]
            require(stream["full_batches"] == 17 and stream["supervised_records"] == 1114112 and stream["dropped_tail_records"] == 64940 and stream["negative_rows_drawn"] >= 1179052, "Native epoch stream differs")


def validate_selector(candidate, arm):
    require(type(candidate["hits50"]) in (float, int) and isfinite(candidate["hits50"]) and 0 <= candidate["hits50"] <= 1, "Invalid original selected metric")
    order = candidate["order"]
    require(type(order) is int and 1 <= order <= (101 if arm == ARMS[1] else 100), "Selector outside original candidate order")
    expected = ("individual_validation_best_bank" if order == 101 else "same_epoch_" + str(order)) if arm == ARMS[1] else "epoch_" + str(order)
    require(candidate["candidate_id"] == expected, "Selector candidate identity differs")


def validate_pairing(units):
    for seed in range(5):
        p, q = (units[(u, seed)]["complete"] for u in UNITS[1:3])
        if p is None or q is None:
            continue
        p, q = p["member_metadata"][0], q["member_metadata"][0]
        require(p["initial_state_sha256"] == q["initial_state_sha256"] and p["initial_rng_sha256"] == q["initial_rng_sha256"], "F4 twin initial state/RNG differs")
        for pe, qe in zip(p["epochs"], q["epochs"]):
            require(pe["stream"] == qe["stream"] and pe["start_rng_sha256"] == qe["start_rng_sha256"] and pe["end_rng_sha256"] == qe["end_rng_sha256"], "F4 paired100epoch stream/RNG differs")


def validate_ledger(ledger, identity, key, disposition):
    require(ledger["schema"] == "ncnc-pilot-inclusive-attempts-v1" and ledger["identity"] == identity and ledger["attempts"], "Attempt ledger differs/empty")
    rows = ledger["attempts"]
    require([r["attempt"] for r in rows] == list(range(1, len(rows) + 1)), "Attempt ledger is reordered/duplicated")
    for row in rows:
        require(row["stage"] == "fit" and (row["unit"], row["base_seed"]) == key and row["status"] in ("COMPLETE", "FAILED", "INTERRUPTED_UNCLOSED"), "Pending/borrowed attempt")
        require(number(row["observed_wall_seconds"]), "Missing observed attempt cost")
        require((row.get("inclusive_wall_seconds") is None and row["status"] == "INTERRUPTED_UNCLOSED") or number(row.get("inclusive_wall_seconds")), "Unclosed cost is not preserved explicitly")
    require(rows[-1]["status"] == ("COMPLETE" if disposition == "COMPLETE" else "FAILED"), "Final attempt terminal state differs")
    unknown = [r for r in rows if r["inclusive_wall_seconds"] is None]
    return {"unit": key[0], "base_seed": key[1], "unique_fits_planned": 4 if key[0] == UNITS[0] else 1, "all_closed_attempt_inclusive_wall_seconds": sum(r["inclusive_wall_seconds"] for r in rows if r["inclusive_wall_seconds"] is not None), "interrupted_observed_wall_lower_bound_seconds": sum(r["observed_wall_seconds"] for r in unknown), "unknown_remainder_attempt_ids": [r["attempt"] for r in unknown], "total_cost_exact": not unknown, "terminal_write_tail_measured": False, "N64_reuses_I4_member0_fit": key[0] == UNITS[0]}


def family_gate(context):
    """All metadata and binary hashes first; no Torch, data access or pickle load."""
    release, identity = context["release"], context["identity"]
    lock = bound(release["family_lock"], name="FAMILY_LOCK.json", decode=True)
    require(lock["schema"] == "ncnc-pilot-full-family-lock-v1" and lock["identity"] == identity and lock["status"] in ("COMPLETE_FAMILY_LOCKED", "TERMINAL_FAILED_FAMILY_LOCKED"), "Family is not immutably complete-or-failed")
    require(release["family_lock"]["path"] == str(Path(identity["family_lock_output_directory"]) / "FAMILY_LOCK.json"), "Family lock path differs from admitted identity")
    require(lock["unique_fits_planned"] == 35 and lock["served_arm_seed_cells"] == 25 and lock["test_file_opened"] is False and lock["TEST_execution_authorized"] is False and lock["no_success_only_subset_summary"] is True, "Full denominator/TEST contract differs")
    inputs, pins = lock["inputs"], release["unit_custody"]
    require(len(inputs) == len(pins) == 20 and {(b["unit"], b["base_seed"]) for b in inputs} == {(b["unit"], b["base_seed"]) for b in pins} == EXPECTED_UNITS, "Exactly original20 units once required")
    require(len(lock["cells"]) == 25 and {(c["arm"], c["base_seed"]) for c in lock["cells"]} == EXPECTED_CELLS, "Missing/duplicate original25 cells")
    original = {(b["unit"], b["base_seed"]): b for b in inputs}
    lock_cells = {(c["arm"], c["base_seed"]): c for c in lock["cells"]}
    units, roots, costs = {}, set(), []
    for pin in pins:
        key = (pin["unit"], pin["base_seed"])
        b = original[key]
        root = canonical(pin["output_directory"])
        require(str(root) == b["output_directory"] and root not in roots, "Borrowed/reused unit output")
        require(not context["output"].is_relative_to(root) and not root.is_relative_to(context["output"]), "Replay output overlaps fit custody")
        roots.add(root)
        disposition = b.get("disposition", "COMPLETE")
        require(pin["disposition"] == disposition and disposition in ("COMPLETE", "TERMINAL_FAILED"), "Pending/unknown family unit")
        closure = bound(pin["closure"], root=root, name="FAMILY_CLOSURE.json", decode=True)
        require(closure == {"schema": "ncnc-pilot-immutable-family-closure-v1", "identity": identity, "family_lock_path": release["family_lock"]["path"], "family_lock_sha256": release["family_lock"]["sha256"], "status": lock["status"], "resume_or_seed_replacement_permitted": False}, "Missing/different immutable unit closure")
        complete = None
        terminal_name = "COMPLETE.json" if disposition == "COMPLETE" else "FAILED.json"
        terminal = bound(pin["terminal"], root=root, name=terminal_name, decode=True)
        require(pin["terminal"]["sha256"] == b["complete_sha256" if disposition == "COMPLETE" else "failure_sha256"], "Lock terminal custody differs")
        if disposition == "COMPLETE":
            validate_complete(terminal, identity, key)
            complete = terminal
            physical = bound(pin["physical_terminal"], decode=True)
            require((physical["unit"], physical["base_seed"]) == key and physical["output_directory"] == str(root) and type(physical["exit_code"]) is int and physical["exit_code"] == 0, "Complete unit lacks successful physical terminal")
        else:
            require(b["terminal_failure_immutable"] is True and b["terminal_failure_authorization_reference"], "Failed retirement is not explicit/immutable")
            require(terminal["schema"] == "ncnc-pilot-failed-unit-v1" and terminal["identity"] == identity and (terminal["unit"], terminal["base_seed"]) == key and terminal["stage"] == "fit" and terminal["status"] == "FAILED" and terminal["test_file_opened"] is False, "Failed terminal differs")
            require(pin["attempts"] == terminal["attempts_receipt"] and pin.get("journal") == terminal["journal_receipt"], "Failed custody differs")
        ledger = bound(pin["attempts"], root=root, name="ATTEMPTS.json", decode=True)
        cost = validate_ledger(ledger, identity, key, disposition)
        cost.update(unique_fits_completed=complete["unique_fits"] if complete is not None else 0,
                    completed_TRAIN_wall_seconds=sum(e["train_wall_seconds"] for m in complete["member_metadata"] for e in m["epochs"]) if complete is not None else None,
                    completed_epoch_VALID_wall_seconds=sum(e["valid_wall_seconds"] for m in complete["member_metadata"] for e in m["epochs"]) if complete is not None else None,
                    candidate101_extra_VALID_evaluations=complete["candidate101_extra_VALID_evaluations"] if complete is not None else None,
                    candidate101_extra_VALID_wall_seconds=None,
                    attempts_receipt={"output_directory": str(root), **pin["attempts"]},
                    cost_attribution="native_bank4_all_four_fits_and101_candidates_once;N64_exact_member0_fit_reuse" if key[0] == UNITS[0] else "one_unique_fit")
        costs.append(cost)
        journal = None
        if pin.get("journal") is not None:
            journal = bound(pin["journal"], root=root, name="JOURNAL.json", decode=True)
            require(journal["schema"] == "ncnc-pilot-private-epoch-journal-v1" and journal["identity"] == identity and (journal["unit"], journal["seed"]) == key and type(journal["revision"]) is int and journal["revision"] >= 0, "Own journal identity differs")
            require(journal["state_file"]["path"] == "STATE_SLOT_" + str(journal["revision"] % 2) + ".pt", "Journal is not its own alternating slot")
            bound(journal["state_file"], root=root)
        require(complete is None or (journal is not None and journal["epoch"] == 100), "Complete own journal missing/unfinished")
        if complete is not None:
            for arm in EXPECTED_ARMS[key[0]]:
                bound(complete["selected_checkpoints"][arm], root=root, name="SELECTED_" + arm + ".pt")
                bound(complete["selection_artifacts"][arm], root=root, name="PRIVATE_SELECTION_" + arm + ".json")
        units[key] = {"root": root, "pin": pin, "complete": complete, "journal": journal}
    validate_pairing(units)
    completed_fits = sum(v["complete"]["unique_fits"] for v in units.values() if v["complete"] is not None)
    completed_cells = sum(len(EXPECTED_ARMS[k[0]]) for k, v in units.items() if v["complete"] is not None)
    require(lock["unique_fits_completed"] == completed_fits and lock["complete_served_cells"] == completed_cells, "Locked denominator disagrees with authenticated units")
    full = lock["status"] == "COMPLETE_FAMILY_LOCKED"
    require(not full or (completed_fits == 35 and completed_cells == 25 and not lock["terminal_failures"]), "Full35/25 success required before predictive replay")
    require(full or (completed_cells < 25 and lock["terminal_failures"] and lock["arm_summaries"] is None and all(lock["primary_development_pilot"].get(k) is None for k in ("paired_VALID_hits50_differences", "mean_difference", "sample_sd", "range", "sign_count"))), "Failed family must have null contrasts")
    cells = []
    # Every20 terminal/journal/selected-byte receipt is checked before disclosure.
    for key, unit in sorted(units.items()):
        root, complete = unit["root"], unit["complete"]
        for arm in EXPECTED_ARMS[key[0]]:
            locked = lock_cells[(arm, key[1])]
            cell = {"unit": key[0], "base_seed": key[1], "arm": arm, "status": "NOT_REPLAYED" if complete is not None else "MISSING_TERMINAL_FAILED"}
            if complete is None:
                require(locked["status"] == "MISSING_TERMINAL_FAILED" and locked["served_VALID_hits50"] is None and locked["checkpoint"] is None, "Failed slot differs")
            else:
                selected = bound(complete["selection_artifacts"][arm], root=root, decode=True)
                require(selected["schema"] == "ncnc-pilot-validation-selected-checkpoint-v1" and selected["identity"] == identity and selected["arm"] == arm and selected["seed"] == key[1] and selected["checkpoint"] == complete["selected_checkpoints"][arm], "Selector/checkpoint identity differs")
                validate_selector(selected["selection"], arm)
                require(locked["status"] == "COMPLETE" and locked["selection"] == selected["selection"] and locked["served_VALID_hits50"] == selected["selection"]["hits50"] and locked["checkpoint"] == {"output_directory": str(root), **selected["checkpoint"]}, "Lock cell/selector/checkpoint differs")
                cell.update(selection=selected["selection"], checkpoint=selected["checkpoint"], root=root)
            cells.append(cell)
    return {"lock": lock, "units": units, "cells": cells, "costs": costs, "full_success": full, "unique_fits_completed": completed_fits, "complete_served_cells": completed_cells, "synthetic_only": bool(context.get("synthetic_only"))}
