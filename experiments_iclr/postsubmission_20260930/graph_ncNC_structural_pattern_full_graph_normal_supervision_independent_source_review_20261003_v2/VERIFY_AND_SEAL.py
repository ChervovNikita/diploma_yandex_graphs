"""Independent minimal successor source review; standard library only."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

OUT = Path(__file__).resolve().parent
PHASE = OUT.parent
OLD = PHASE / "graph_ncNC_structural_pattern_full_graph_normal_supervision_preparation_20261003_v1"
NEW = PHASE / "graph_ncNC_structural_pattern_full_graph_normal_supervision_preparation_20261003_v2"
DRIVER = PHASE / "graph_ncNC_structural_pattern_pilot_preparation_20261003_v3"
UTC = datetime.now(timezone.utc).isoformat()
TEXT = {".py", ".json", ".md", ".sh", ".patch", ".diff", ".txt", ".log", ".csv"}


def desc(path):
    assert path.suffix in TEXT and path.resolve().is_relative_to(PHASE)
    raw = path.read_bytes()
    raw.decode("utf8")
    return {"path": str(path.relative_to(PHASE)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def verify(path, row):
    got = desc(path)
    assert got["sha256"] == row["sha256"] and got["bytes"] == row.get("bytes", row.get("size"))
    if path.suffix == ".json":
        json.loads(path.read_text())
    elif path.suffix == ".py":
        ast.parse(path.read_text())
    return got


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


assert desc(NEW / "MANIFEST.json")["sha256"] == "92a48396b6c635d04574ebd347e0a75b6bb7d7eb28377204c94473a15c164fd1"
assert desc(NEW / "MANIFEST.json")["bytes"] == 2561
assert desc(NEW / "SEAL.json")["sha256"] == "57766acc5dacd85b60fa8ef8aab1258b44a73729a11712a599e9227fc973ec8a"
assert json.loads((NEW / "SEAL.json").read_text())["manifest"] == desc(NEW / "MANIFEST.json")
rows = json.loads((NEW / "MANIFEST.json").read_text())["files"]
assert len(rows) == 13
payloads = [verify(NEW / r["path"], r) for r in rows]
assert {p.name for p in NEW.iterdir() if p.is_file()} == {r["path"] for r in rows} | {"MANIFEST.json", "SEAL.json"}
assert desc(DRIVER / "MANIFEST.json")["sha256"] == "fa7b2a7a2c6ec83362f3c820fb4f7ad5288e5cc9fb0ee5139614d6690f3f7f89"
driver_rows = json.loads((DRIVER / "MANIFEST.json").read_text())["files"]
assert len(driver_rows) == 49
driver_payloads = [verify(DRIVER / r["path"], r) for r in driver_rows]
oldroot = "graph_ncNC_structural_pattern_full_graph_execution_root_20261003_v1"
newroot = "graph_ncNC_structural_pattern_full_graph_execution_root_20261003_v2"
for name in ["full_graph_child.py", "supervise_full_graph.py", "DETACHED_COMMAND.sh"]:
    expected = (OLD / name).read_text().replace(oldroot, newroot)
    if name == "supervise_full_graph.py":
        expected = expected.replace("'cuda_peak_allocated_bytes':40*1024**3,'cuda_peak_reserved_bytes':48*1024**3", "'cuda_peak_allocated_bytes':70*1024**3,'cuda_peak_reserved_bytes':75*1024**3")
    if name == "DETACHED_COMMAND.sh":
        expected = expected.replace(OLD.name, NEW.name)
    assert expected == (NEW / name).read_text()
assert (NEW / "FULL_GRAPH_WORK_PLAN.json").read_bytes() == (OLD / "FULL_GRAPH_WORK_PLAN.json").read_bytes()
caps = dict(wall_seconds=1800, host_RSS_bytes=32 * 2**30, cuda_peak_allocated_bytes=70 * 2**30, cuda_peak_reserved_bytes=75 * 2**30)
assert json.loads((NEW / "CAPS.json").read_text()) == caps
template = json.loads((NEW / "ROOT_RELEASE_FULL_GRAPH_TEMPLATE.json").read_text())
assert template["full_graph_caps"] == caps
assert template["authorized_stages"] == [] and template["authorized_invocations"] == [] and template["root_authorization_reference"] is None
work = json.loads((NEW / "FULL_GRAPH_WORK_PLAN.json").read_text())
assert work["complete_optimizer_steps"] == 34 and work["additional_own_serialization_replay_steps"] == 6
assert work["complete_five_route_VALID_traversals"] == 2 and work["VALID_query_pools_each_arm"] == [60084, 100000]
custody = json.loads((NEW / "PREDECESSOR_FAILURE_CUSTODY.json").read_text())
failure_payloads = [verify(PHASE / r["path"], r) for r in custody["evidence"]]
terminal = json.loads((PHASE / custody["evidence"][0]["path"]).read_text())
assert terminal["status"] == "FAILED" and terminal["child_exit_code"] == 88 and terminal["qualification_receipt"] is None
assert terminal["supervisor_inclusive_wall_seconds"] == 26.04586512222886
assert terminal["CUDA_peak_observation"]["cuda_peak_allocated_bytes"] == 46954556416
assert terminal["CUDA_peak_observation"]["cuda_peak_reserved_bytes"] == 59624128512
attempts = json.loads((PHASE / custody["evidence"][2]["path"]).read_text())
assert attempts["attempts"][0]["work"]["arm"] == "J" and attempts["attempts"][0]["work"]["completed_batches"] == 0
resource = json.loads((NEW / "RESOURCE_PLANNING_EVIDENCE.json").read_text())
receipt = verify(PHASE / resource["current_resource_receipt"]["path"], resource["current_resource_receipt"])
observation = json.loads(json.loads((PHASE / receipt["path"]).read_text())["stdout"])
gpu1 = next(r for r in observation["GPUs"] if r[0] == "1")
assert gpu1[1] == "GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced" and int(gpu1[3]) == 899
assert resource["required_minimum_selected_GPU_free_MiB"] == 78 * 1024
assert resource["required_minimum_host_available_bytes"] == 40 * 2**30
assert int(gpu1[2]) - int(gpu1[3]) >= resource["required_minimum_selected_GPU_free_MiB"]
assert observation["host_available_bytes"] >= resource["required_minimum_host_available_bytes"]
checks = {
    "schema": "independent-ncnc-cap-successor-source-check-v1", "UTC": UTC,
    "status": "PASS_SOURCE_ONLY", "support_payloads": payloads, "driver_payloads": driver_payloads,
    "source_diff_exactly_fresh_root_and_caps": True, "work_plan_byte_identical": True,
    "disabled_template": True, "failure_metadata_verified": failure_payloads,
    "resource_observation_verified": receipt, "observation_not_current_admission": True,
    "wall_annotation_discrepancy": {"annotation_seconds": custody["paid_wall_seconds"], "authoritative_terminal_seconds": terminal["supervisor_inclusive_wall_seconds"], "annotation_minus_terminal_seconds": custody["paid_wall_seconds"] - terminal["supervisor_inclusive_wall_seconds"], "terminal_is_cost_authority": True},
    "numerical_or_project_imports": False, "arrays_checkpoints_binaries_GPU_SSH_upload_or_launch": False,
    "execution_authorized": False,
}
save("STATIC_VERIFICATION.json", checks)
save("READ_SCOPES.json", {
    "schema": "independent-ncnc-cap-successor-read-scopes-v1", "UTC": UTC,
    "scope": "Minimal successor check against completed V1 source review; source/text/AST/JSON/hash only.",
    "source_inspected": [{"path": str((NEW / "supervise_full_graph.py").relative_to(PHASE)), "lines": "1-188", "scope": "complete parent, caps and launch/resource/cleanup gates"}, {"path": str((NEW / "full_graph_child.py").relative_to(PHASE)), "lines": "1-170", "scope": "peak/capture/work gates; only root changed"}],
    "changed_diff_and_work_caps_template_resource_failure_text": True,
    "primary_literature_reads": 0, "whole_paper_certifications": 0, "new_scientific_execution": False,
    "old_driver_source_hashes_verified": 49, "old_driver_scientific_review_inherited": True,
})
save("REVIEW.json", {
    "schema": "independent-ncnc-cap-successor-source-review-v1", "UTC": UTC,
    "status": "PASS_SOURCE_ONLY", "blocking_findings": [],
    "source": {"manifest": desc(NEW / "MANIFEST.json"), "seal": desc(NEW / "SEAL.json")},
    "predecessor_review": desc(PHASE / "graph_ncNC_structural_pattern_full_graph_normal_supervision_independent_source_review_20261003_v1/REVIEW.json"),
    "driver_manifest": desc(DRIVER / "MANIFEST.json"), "support_payload_count": len(payloads), "support_payload_bytes": sum(r["bytes"] for r in payloads),
    "driver_payload_count": len(driver_payloads), "driver_payload_bytes": sum(r["bytes"] for r in driver_payloads),
    "scientific_changes": False, "work_shrunk": False,
    "checks": [
        {"id": "exact_minimal_successor", "status": "passed", "finding": "Exact 13 support payloads and 49 unchanged V3 scientific-driver payloads verify. Parent and child source differ only by fresh V2 external root and parent CUDA caps, and detached command selects that support/root. Work plan is byte-identical. Caps are 1800 seconds, 32 GiB host RSS, 70 GiB CUDA allocated and 75 GiB reserved."},
        {"id": "complete_work_and_normal_custody", "status": "passed_source_only", "finding": "Unchanged V1 review covers 34 real native updates plus 6 own replay updates and two complete five-route VALID traversals. Exact numerical prerequisite, peak/reset custody, ordinary child session, final qualification capture and owned cleanup remain. Root template is disabled; one fresh full_graph/pair/seed0 invocation only."},
        {"id": "prior_physical_failure", "status": "passed", "finding": "Pinned failure terminal is FAILED, child exit 88, allocated 46954556416 bytes and reserved 59624128512 bytes, before first completed J batch. Incomplete driver IN_PROGRESS record is a killed-child receipt, not continued work. Paid supervisor-inclusive wall is 26.04586512222886 seconds; the successor's annotation 26.045865029096603 differs by about 9.3e-8 seconds. Preserve both; terminal remains cost authority."},
        {"id": "prospective_resource_admission", "status": "pending_root_gate", "finding": "The 19:21:39 UTC observation shows GPU1 UUID GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced at 899/81920 MiB and host available 119102222336 bytes. It satisfied 78 GiB selected-GPU free and 40 GiB host available then. Supervisor records GPU profile/occupancy but does not enforce these minimum-free values automatically. Root must refresh and enforce them immediately before launch, select the available GPU1 UUID rather than the disabled template's occupied GPU0 example, and prevent conflicting work from being scheduled there."},
        {"id": "bounded_feasibility_answer", "status": "conditional_source_assessment", "finding": "A single successful invocation with unchanged complete work, physical COMPLETE/exit0, no cap violation and exact final peak/qualification custody can establish bounded feasibility for that invocation on one 80 GiB device under the proposed caps. Source evidence and a pre-forward failure do not establish backward or both-arm feasibility. Completion does not establish predictive results, native Adam parity or a need for more GPUs. No automatic retry, shrinking or unrelated process signal is introduced."},
    ],
    "limitations_inherited": ["250 ms session RSS samples may miss short combined peaks", "Exceptional cleanup lacks normal wait4 detail", "Terminal serialization tails are unmeasured", "Torch allocator peaks do not include every driver/context allocation"],
    "nonblocking_annotations": ["Paid-wall annotation differs insignificantly from the exact pinned physical terminal; use terminal for cost carry", "78 GiB/40 GiB free-space requirements remain root admission gates, not automatic checks in this unchanged supervisor"],
    "execution_authorized": False, "scientific_fit_authorized": False, "feasibility_established": False,
    "actual_current_resource_admission_established": False, "canonical_edits": False,
    "static_verification": desc(OUT / "STATIC_VERIFICATION.json"), "read_scopes": desc(OUT / "READ_SCOPES.json"),
    "read_accounting": {"source_JSON_AST_text_only": True, "project_or_numerical_modules_imported": False, "array_checkpoint_runtime_binary_reads_or_hashes": False, "GPU_SSH_upload_launch_or_other_jobs_mutation": False, "subagents": False, "new_output_directory_only": OUT.name},
})
(OUT / "REVIEW.md").write_text("""# NCNC V2 cap successor independent source review

**PASS_SOURCE_ONLY; no blocking source findings.** The exact 13 support payloads and 49 unchanged V3 driver payloads verify. Only fresh external execution paths and 70 GiB allocated / 75 GiB reserved caps change. The full 34 + 6 optimizer updates and two complete five-route VALID traversals remain.

One physical successful invocation can establish bounded feasibility for this work and device under its caps. Backward and both-arm feasibility are still unmeasured. The prior exit-88 failure before the first J update remains paid: authoritative wall 26.04586512222886 seconds. Its successor annotation differs by about 9.3e-8 seconds; use the pinned terminal for costs.

Before launch, root must refresh and enforce 78 GiB free on the selected GPU and 40 GiB host available, select the available GPU1 UUID and prevent conflicting scheduling. These are root gates; the unchanged supervisor records occupancy but does not automatically enforce the free thresholds. The disabled template's GPU0 is an example. The 19:21:39 observation is not current admission.

No execution, scientific-fit, predictive or acceptance authority is granted. No runtime/numerical code, arrays, checkpoints, GPU, SSH or uploads were used by this review. Earlier supervision measurement limits remain.
""")
sealed_payloads = [desc(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"MANIFEST.json", "SEAL.json"}]
save("MANIFEST.json", {"schema": "independent-source-review-manifest-v1", "UTC": UTC, "payload": sealed_payloads, "execution_authorized": False})
save("SEAL.json", {"schema": "independent-source-review-seal-v1", "UTC": UTC, "manifest": desc(OUT / "MANIFEST.json"), "review": desc(OUT / "REVIEW.json"), "execution_authorized": False})
for row in sealed_payloads:
    assert desc(PHASE / row["path"]) == row
for path in OUT.iterdir():
    if path.is_file():
        path.chmod(0o444)
print(json.dumps({"status": "PASS_SOURCE_ONLY", "review": desc(OUT / "REVIEW.json"), "manifest": desc(OUT / "MANIFEST.json"), "seal": desc(OUT / "SEAL.json")}))
