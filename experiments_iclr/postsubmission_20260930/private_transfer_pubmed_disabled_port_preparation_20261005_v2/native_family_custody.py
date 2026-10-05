"""Metadata/hash-only complete-four preflight; no numerical imports or quality reads."""
import json
from pathlib import Path
import custody


def record(path):
    path = Path(path)
    return {"path": str(path.relative_to(custody.PHASE)), "bytes": path.stat().st_size, "sha256": custody.sha(path)}


def authenticate(row):
    if set(row) != {"path", "bytes", "sha256"}: raise ValueError("Exact artifact record required")
    path = custody.phase_file(row["path"])
    if record(path) != row: raise ValueError("Owned artifact bytes differ")
    return path


def complete_four(job, members):
    """Authenticate every owned member before opening any selected freeze/payload.

    The two metadata schemas below are prospective contracts. No supervisor or
    complete-family evidence is created here or assumed to exist.
    """
    custody.require_evidence(job, "owned_complete_native_fit_family")
    family_record = job["owned_complete_native_fit_family"].get("family_inventory")
    family_path = authenticate(family_record)
    if not any(row["path"] == family_record["path"] and row["sha256"] == family_record["sha256"]
        for row in job["owned_complete_native_fit_family"]["evidence"]):
        raise ValueError("Exact family inventory is absent from root-approved evidence")
    available = custody.phase_file(job["available_manifest_relative"])
    if custody.sha(available) != job["available_manifest_sha256"]: raise ValueError("Available manifest changed")
    inputs = json.loads(available.read_text())
    if set(inputs["files"]) != custody.AVAILABLE or inputs.get("TEST_available_to_loader") is not False:
        raise ValueError("Exactly four available metadata roles required")
    identities = {key: {"bytes": row["bytes"], "sha256": row["sha256"]} for key, row in inputs["files"].items()}
    common = {"source_manifest_sha256": job["source_manifest_sha256"], "input_files": identities,
        "runtime": job["runtime_versions"], "base_seed": job["base_seed"]}
    family = json.loads(family_path.read_text())
    if set(family) != {"schema", "status", "members", *common} or family.get("schema") != "owned_native_independent_four_family_v1" or family.get("status") != "COMPLETE":
        raise ValueError("Complete metadata-only native family contract differs")
    if any(family[key] != value for key, value in common.items()): raise ValueError("Family source/input/runtime/base identity differs")
    if len(family["members"]) != 4 or [row["member"] for row in family["members"]] != [0, 1, 2, 3]:
        raise ValueError("Complete-four owned member coverage differs")
    paths, outputs, jobs = [], set(), set()
    for supplied, owned in zip(members, family["members"]):
        if set(owned) != {"member", "seed", "fit_id", "model", "output_directory", "job", "terminal", "final_custody", "artifacts"}:
            raise ValueError("Metadata-only member fields differ")
        member = supplied["member"]
        identity = {**common, "member": member, "seed": job["base_seed"]+5*member,
            "fit_id": "pubmed_native_s"+str(job["base_seed"])+"_m"+str(member), "model": "source_native_NCNC"}
        if any(owned[key] != identity[key] for key in ("member", "seed", "fit_id", "model")):
            raise ValueError("Owned native member identity differs")
        if set(owned["artifacts"]) != {"freeze", "state", "scores"} or owned["artifacts"] != {role: supplied[role] for role in ("freeze", "state", "scores")}:
            raise ValueError("Supplied selected artifacts differ from the complete-family inventory")
        selected_paths = {role: authenticate(row) for role, row in owned["artifacts"].items()}
        directory = selected_paths["freeze"].parent
        if str(directory) != owned["output_directory"] or directory in outputs or any(p.parent != directory for p in selected_paths.values()):
            raise ValueError("Distinct exact owned member output required")
        outputs.add(directory)
        if {role: p.name for role, p in selected_paths.items()} != {"freeze": "FIT_FREEZE.json", "state": "selected_state.pt", "scores": "selected_VALID_scores.pt"}:
            raise ValueError("Native selected artifact roles differ")
        job_path = authenticate(owned["job"])
        if job_path in jobs: raise ValueError("Independent member jobs must be distinct")
        jobs.add(job_path)
        fitted_job = json.loads(job_path.read_text())
        if any(fitted_job.get(key) != identity[key] for key in ("source_manifest_sha256", "base_seed", "member", "seed", "fit_id")):
            raise ValueError("Owned fit job identity differs")
        if fitted_job.get("purpose") != "TRAIN_VALID_native_ncnc_member_fit" or fitted_job.get("fits_authorized") is not True or fitted_job.get("VALID_values_access") is not True or fitted_job.get("TEST_access") is not False:
            raise ValueError("Owned native fit job scope differs")
        if fitted_job.get("source_review_approved") is not True or fitted_job.get("external_hard_bound_confirmed") is not True or fitted_job.get("retry") is not False or type(fitted_job.get("soft_seconds")) is not int or fitted_job["soft_seconds"] < 1:
            raise ValueError("Owned fit reviewed source/bound/no-retry contract differs")
        if fitted_job.get("runtime_versions") != common["runtime"] or fitted_job.get("output_directory") != str(directory) or fitted_job.get("program_sha256") != custody.sha(custody.ROOT/"native_ncnc.py"):
            raise ValueError("Owned fit source/runtime/output differs")
        if fitted_job.get("available_manifest_sha256") != job["available_manifest_sha256"] or fitted_job.get("available_manifest_relative") != job["available_manifest_relative"]:
            raise ValueError("Owned fit available input manifest differs")
        final_path = authenticate(owned["final_custody"])
        if final_path.parent != directory or final_path.name != "FIT_ARTIFACT_CUSTODY.json": raise ValueError("Owned final custody path differs")
        final = json.loads(final_path.read_text())
        if set(final) != {"schema", "status", "output_directory", "job_sha256", "files", "artifacts", *identity} or final.get("schema") != "owned_native_member_artifact_custody_v1" or final.get("status") != "COMPLETE":
            raise ValueError("Complete metadata-only member custody differs")
        if any(final[key] != value for key, value in identity.items()) or final["output_directory"] != str(directory) or final["job_sha256"] != owned["job"]["sha256"] or final["artifacts"] != owned["artifacts"]:
            raise ValueError("Native final artifact/source/input/runtime/job relationships differ")
        inventory = [authenticate(row) for row in final["files"]]
        expected_names = {"START.json", "CONFIG.json", "HISTORY.jsonl", "FIT_FREEZE.json", "selected_state.pt", "selected_VALID_scores.pt"}
        if len(inventory) != len(expected_names) or {p.name for p in inventory} != expected_names or any(p.parent != directory for p in inventory):
            raise ValueError("Exact complete owned native fit artifact inventory differs")
        if any(owned["artifacts"][role] not in final["files"] for role in ("freeze", "state", "scores")):
            raise ValueError("Selected roles are absent from owned fit custody")
        terminal = json.loads(authenticate(owned["terminal"]).read_text())
        required = {"schema", "status", "exit_code", "timed_out", "fit_id", "output_directory", "job_sha256", "result", "final_custody"}
        if set(terminal) != required or terminal.get("schema") != "owned_native_member_terminal_v1" or terminal.get("status") != "COMPLETE" or type(terminal.get("exit_code")) is not int or terminal["exit_code"] != 0 or terminal.get("timed_out") is not False:
            raise ValueError("Successful metadata-only owned supervisor terminal absent")
        if terminal["fit_id"] != identity["fit_id"] or terminal["output_directory"] != str(directory) or terminal["job_sha256"] != owned["job"]["sha256"] or terminal["result"] != owned["artifacts"]["freeze"] or terminal["final_custody"] != owned["final_custody"]:
            raise ValueError("Owned terminal/job/result/final-custody relationships differ")
        paths.append(selected_paths)
    # Only after all four metadata inventories, artifacts and successful owned
    # terminals passed may any selected freeze (which contains quality) open.
    return paths, identities
