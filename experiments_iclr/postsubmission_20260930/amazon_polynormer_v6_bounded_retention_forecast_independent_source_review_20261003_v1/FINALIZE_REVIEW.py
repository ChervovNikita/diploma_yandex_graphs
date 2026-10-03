"""Seal the completed independent source review using standard-library metadata only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

OUT = Path(__file__).resolve().parent
PHASE = OUT.parent
V6 = PHASE / "amazon_polynormer_paired_family_source_preparation_20261003_v6"
CANDIDATE = PHASE / "amazon_polynormer_paired_family_execution_root_20261003_v3/fit_schedule_resource_candidate_v2_v6_bounded_retention"
UTC = datetime.now(timezone.utc).isoformat()


def descriptor(path):
    assert path.suffix in {".py", ".json", ".md"}
    raw = path.read_bytes()
    raw.decode("utf8")
    return {"path": str(path.relative_to(PHASE)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


check = json.loads((OUT / "STATIC_VERIFICATION.json").read_text())
assert check["status"] == "PASS_SOURCE_ONLY"
assert len(check["V6_payloads"]) == 36 and len(check["candidate_payloads"]) == 25
assert len(check["verified_text_metadata_descriptors"]) == 345
assert len(check["opaque_descriptors_carried_unopened"]) == 21
save("VERIFIER_ATTEMPT02_CORRECTION.json", {
    "schema": "reviewer-verifier-failure-correction-v1", "UTC": UTC,
    "preserved_record": descriptor(OUT / "VERIFIER_ATTEMPT02_FAILURE.json"),
    "correction": "The extra descriptor was the old remote interpreter binary, not a literature binary. The original failure wording was an incorrect provisional classification and is preserved unchanged.",
    "actual_extra_descriptor": next(r for r in check["opaque_descriptors_carried_unopened"] if r["path"].endswith("/.venv/bin/python")),
    "resolution": "Allow the exact old interpreter path as an opaque carried descriptor. Do not open, hash, execute, or remotely access it.",
    "source_or_candidate_failure": False, "scientific_execution": False,
})
scopes = {
    "schema": "independent-v6-read-scopes-v1", "UTC": UTC,
    "review_scope": "Source, JSON, text, AST, descriptor integrity and forecast arithmetic only; source inspection completed before this seal.",
    "source_scopes": [
        {"path": str((V6 / "retention.py").relative_to(PHASE)), "lines": "1-105", "scope": "complete retention implementation"},
        {"path": str((V6 / "study.py").relative_to(PHASE)), "lines": "11-84,98-203", "scope": "inheritance/admission and full fit/selector/retirement/replay path"},
        {"path": str((V6 / "qualify.py").relative_to(PHASE)), "lines": "293-327", "scope": "added retirement/full-state isolation proof"},
        {"path": str((V6 / "evaluate.py").relative_to(PHASE)), "lines": "1-57", "scope": "changed trace validation; unchanged evaluator functions compared by AST"},
        {"path": str((V6 / "worker.py").relative_to(PHASE)), "lines": "10,70-117", "scope": "second-registration rejection and closure/all inheritance"},
        {"path": str((V6 / "retention_fixture.py").relative_to(PHASE)), "lines": "1-105", "scope": "fabricated metadata fixture source inspected, not executed"},
        {"path": str((CANDIDATE / "prepare_v6_metadata_candidate.py").relative_to(PHASE)), "lines": "1-272", "scope": "forecast60-96,storage130-180,disabled releases199-255"},
    ],
    "source_diff_evidence": descriptor(V6 / "SOURCE_DIFF_EVIDENCE.json"),
    "author_fixture_receipt": descriptor(V6 / "RETENTION_FIXTURE_RESULT.json"),
    "hash_integrity_scope": "36 V6,29 V5,25 forecast payloads;345 unique recursively enumerated text/source metadata descriptors. Complete arrays, checkpoints and old interpreter remain unopened.",
    "primary_literature_reads": 0, "whole_paper_certifications": 0,
}
save("READ_SCOPES.json", scopes)
accounting = {
    "schema": "independent-v6-read-accounting-v1", "UTC": UTC,
    "V6_payloads_verified": 36, "V6_payload_bytes": 269343,
    "V5_payloads_verified": 29, "V5_payload_bytes": 227899,
    "forecast_payloads_verified": 25, "forecast_payload_bytes": 1173156,
    "unique_text_metadata_descriptors_verified": 345,
    "opaque_descriptors_carried_unopened": 21, "arrays_and_checkpoints": 20, "old_interpreter_binary": 1,
    "author_count_relationship": "The reviewer recursively includes complete V6 source metadata and manifest-local rows; its345 descriptors are a broader enumeration than the author's224, not a disagreement.",
    "checker_failures_preserved": [descriptor(OUT / "VERIFIER_ATTEMPT01_FAILURE.json"), descriptor(OUT / "VERIFIER_ATTEMPT02_FAILURE.json")],
    "checker_failure_correction": descriptor(OUT / "VERIFIER_ATTEMPT02_CORRECTION.json"),
    "failed_checker_scope": "Only standard-library JSON/hash/AST/arithmetic. Attempt01 used the phase root for a packet-local path. Attempt02 rejected the unopened old interpreter descriptor. No project modules or scientific fixtures were executed.",
    "project_or_numerical_imports": False, "Torch_or_NumPy_imports": False,
    "array_checkpoint_runtime_binary_reads_or_hashes": False,
    "GPU_SSH_upload_network_or_runtime_changes": False,
    "scientific_execution_or_scoring": False, "subagents": False,
    "canonical_or_existing_source_registry_claim_output_changes": False,
    "only_new_output_directory": str(OUT.relative_to(PHASE)),
}
save("READ_ACCOUNTING.json", accounting)
review = {
    "schema": "independent-amazon-v6-retention-forecast-source-review-v1", "UTC": UTC,
    "status": "PASS_SOURCE_ONLY", "review_type": "Independent changed-scope V6 bounded-retention and disabled forecast review",
    "source": {"manifest": descriptor(V6 / "MANIFEST.json"), "seal": descriptor(V6 / "SEAL.json")},
    "forecast": {"manifest": descriptor(CANDIDATE / "MANIFEST.json"), "seal": descriptor(CANDIDATE / "SEAL.json")},
    "predecessor_review": descriptor(PHASE / "amazon_polynormer_v5_adam_image_ownership_source_review_20261003_v1/REVIEW.json"),
    "blocking_findings": [], "execution_authorized": False, "training_authorized": False,
    "resource_admission_established": False, "numerical_qualification_established": False,
    "checks": [
        {"id": "exact_integrity", "status": "passed", "finding": "The exact requested V6 and forecast manifests/seals verify, with36 and25 payloads respectively. V5 prerequisite29payloads verify.345 recursively collected text/source metadata descriptors verify;20 arrays/checkpoints and one old interpreter descriptor are carried unopened."},
        {"id": "scientific_recipe_and_cohort", "status": "passed", "finding": "native_training.py,common.py,supervise.py,prepare_release.py,neural/reference bodies,DESIGN.json and PRESERVED_FAILURES.json are byte-identical to V5. Original15physical fits,nine families,seeds,widths,claims and paths remain. FIT/Adam/native transition/RNG,200local+2500global real updates,all2700 complete VAL scores,strict improvement and earliest ties remain."},
        {"id": "prospective_audit_scope", "status": "passed_source_reasoning", "finding": "Root explicitly authorized prospective removal of permanent reopening and next-step replay guarantees for superseded candidates before predictive fits. V6 declares both removals. Each strict improvement still creates the fresh full portable state with hash/size and snapshot/write cost; the score/decision trace is fsynced before retiring its predecessor."},
        {"id": "retention_custody", "status": "passed_source_reasoning", "finding": "Current best and selected-local transition images are protected, aliased once, with at most two persistent images plus one new temporary. Only a new owned.pt file under this exact fresh attempt/checkpoints directory may retire. Exact descriptor verification,fsynced intent,unlink,directory fsync,and fsynced completion/cost precede success. Validator checks pairing/order/bindings,later strict improvement,protected images,absence and original selection/write-cost metadata."},
        {"id": "retained_replay_and_qualification", "status": "passed_source_reasoning", "finding": "Retained local scratch replay jointly deep-copies(model,opt) and restores canonical caller RNG in finally. Its AST matches qualified V5 replay_scratch apart from function name. Retained local transition and final selected replay add four audit optimizer updates per fit. New five-form qualification retirement proof compares exact model/Adam/grad/modes/stage/RNG and protects local/global probes. Fresh exact V6 qualification remains mandatory; V5 does not substitute."},
        {"id": "evaluation_and_registration", "status": "passed", "finding": "Evaluation metrics/cohort/run function ASTs remain V5-identical. Trace validation accepts retirement and requires retained local/final replay/transition. worker.register rejects a second registration immediately. Registry retains its V5 source descriptor with explicit V6 inheritance; current release/admission source selects V6."},
        {"id": "metadata_fixture", "status": "inspected_not_rerun", "finding": "Author's source-only fabricated2700-record fixture covers global final,local final,first-update-only and every-update-improvement histories plus rejection of altered guarantee/receipt fields. Receipt and source are inspected and integrity-bound; the independent reviewer did not rerun it or execute runtime code."},
        {"id": "forecast_arithmetic", "status": "passed_arithmetic_only", "finding": "Full15+closure forecast is53.369053359398hours and80.053580039097hours with50%margin. Retained checkpoint bound is4405455504bytes and incremental space18391721268bytes against proposed32GiB. Scientific updates40500/member trajectories64800; audit60/96; totals40560/64896. Single aliases add zero fits. All40500 possible snapshot/hash/write events remain charged; retirement N-1 is conservative. Old full snapshot volume5947364930400bytes remains preserved."},
        {"id": "forecast_limits", "status": "not_resource_admission", "finding": "Rates derive from one bounded GNNM plus four native V5 forms. Trace fsync0.1s/update and retirement0.25s/image are unmeasured assumptions. Safe verify/load and train+VAL intervals are hash/inference charge proxies,not independent measurements. Actual V6 retirement timing,same-filesystem free-space/quota,runtime/consumer and fresh five-form qualification,allocation lifetime or user time budget remain required. Top-level forecast gates do not certify deep nested measurement completeness."},
        {"id": "disabled_candidates_and_history", "status": "passed", "finding": "All17fit/close/all release bodies keep execution/review false and source_review,runtime_receipt,consumer_release,qualification_freeze null. All15registered paths match originals. Old failure/source/history and denied full-retention candidate remain untouched. Review is engineering source evidence and grants no predictive progress,scientific acceptance or execution authority."},
    ],
    "pending_before_predictive_release": [
        "Exact V6 runtime receipt and consumer release",
        "Fresh exact five-form V6 qualification including retirement/full-state/RNG isolation",
        "Physical retirement/fsync/storage and fit memory/time resource measurements",
        "Same-filesystem free space/quota and explicit storage budget",
        "Allocation lifetime or user time budget and root adoption/release",
    ],
    "static_verification": descriptor(OUT / "STATIC_VERIFICATION.json"),
    "read_scopes": descriptor(OUT / "READ_SCOPES.json"),
    "read_accounting": descriptor(OUT / "READ_ACCOUNTING.json"),
    "nonblocking_annotations": ["Two reviewer metadata-checker failures are preserved; attempt02 literature-binary wording is corrected by a separate immutable annotation."],
}
save("REVIEW.json", review)
(OUT / "REVIEW.md").write_text("""# Independent V6 bounded-retention and forecast source review

**PASS_SOURCE_ONLY; no blocking source findings.** This review binds the exact V6 source and disabled resource candidate. It grants no execution, predictive-training or resource-admission authority.

The original15-fit/nine-family cohort, native scientific recipe,2700score/strict-selector history and earliest ties remain. V6 explicitly adopts the root-authorized prospective audit change: superseded image bodies may retire and no longer receive next-step replay. Every strict improvement still writes an exact portable full state and retains its descriptor and snapshot/write cost. Retirement is restricted to this fresh attempt's owned images, with trace fsync, exact hash/size, fsynced intent/completion and directory fsync. Current best and selected-local images remain protected. Retained local/final replay adds four audit optimizer updates per fit; joint scratch model/Adam ownership and caller RNG restoration match V5's qualified replay body.

The source/metadata integrity and forecast arithmetic pass. Full15+closure is53.369053359398hours,80.053580039097hours with50%margin; retained images4405455504bytes; incremental free space18391721268bytes. These are forecasts.0.1s trace-fsync and0.25s retirement allowances remain unmeasured; V5 short-sample rates and hash/inference proxies do not establish actual V6 feasibility. All17release candidates remain disabled with fresh inputs null.

Fresh exact V6 runtime/consumer, five-form qualification, retirement/resource/storage measurements, free-space/quota and allocation/time budget must be resolved by root before predictive release. Historical V5 qualification is not a substitute. No arrays, checkpoints or interpreter binary were opened; no numerical/project imports, GPU, SSH, uploads or scientific execution occurred. Two reviewer stdlib-checker failures are preserved, with a separate correction identifying the extra unopened descriptor as the old interpreter binary.

Detailed evidence is in REVIEW.json, READ_SCOPES.json, READ_ACCOUNTING.json and STATIC_VERIFICATION.json.
""")
payload = [descriptor(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"MANIFEST.json", "SEAL.json"}]
save("MANIFEST.json", {"schema": "independent-source-review-manifest-v1", "UTC": UTC, "payload": payload, "execution_authorized": False})
save("SEAL.json", {"schema": "independent-source-review-seal-v1", "UTC": UTC, "manifest": descriptor(OUT / "MANIFEST.json"), "review": descriptor(OUT / "REVIEW.json"), "execution_authorized": False})
for row in payload:
    assert descriptor(PHASE / row["path"]) == row
for p in OUT.iterdir():
    if p.is_file():
        p.chmod(0o444)
print(json.dumps({"status": "PASS_SOURCE_ONLY", "review": descriptor(OUT / "REVIEW.json"), "manifest": descriptor(OUT / "MANIFEST.json"), "seal": descriptor(OUT / "SEAL.json"), "payload_count": len(payload)}))
