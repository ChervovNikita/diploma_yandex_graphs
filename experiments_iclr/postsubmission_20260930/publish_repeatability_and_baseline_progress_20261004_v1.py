"""Preserve current status and build a compact, explicit progress inventory."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
PUB = P / "publication/repeatability_and_baseline_progress_20261004_v1"
SNAP = P / "coordination_snapshots/20261004_repeatability_and_baseline_before_state_v1"
PUB.mkdir()
SNAP.mkdir()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p, obj):
    with p.open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
for name in ("PUBLIC_STATUS.md", "RESEARCH_STATE.md", "research_ledger.json"):
    shutil.copy2(P / name, SNAP / name)
now = datetime.now(timezone.utc).isoformat()
adopt = json.loads((P / "pubmed_shared4_zero_update_first_batch_repeat_root_adoption_20261004_v1/ROOT_ADOPTION.json").read_text())
monitor = json.loads((P / "amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1/MONITOR_0026_RESULT.json").read_text())
progress = [{"fit_id": x["fit_id"], "updates": x["trace_progress"]["actual_update"], "terminal": (x.get("physical_terminal") or {}).get("status")} for x in monitor["registered_fit_progress"] if x.get("trace_progress")]
assert sum(x["terminal"] == "success" for x in progress) == 5
assert next(x for x in progress if x["fit_id"] == "split1_gnnm_boundary_4_seed29")["updates"] == 390
prior = json.loads((P / "publication/native_gradient_startup_progress_20261004_v1/PUSH_RECEIPT.json").read_text())
assert prior["verified"] and prior["remote_commit"] == "9d8efcece0251e1e3114cfef6158a4835f726113"
event = {
    "UTC": now,
    "status": "active_incomplete_predictive_extension_unconfirmed",
    "Pubmed": {"result": "Four controlled zero-update comparisons match the original rule; complete-epoch failure is preserved and remains unresolved.", "adoption": "pubmed_shared4_zero_update_first_batch_repeat_root_adoption_20261004_v1/ROOT_ADOPTION.json", "source_manifest_sha256": adopt["source_manifest_sha256"], "predictive_success": False},
    "Amazon": {"observation_UTC": monitor["UTC"], "progress": progress, "failures": monitor["failures"], "quality_or_TEST_scoring": False},
    "support_bucket": {"source_review": "PASS for bounded equivalence QA eligibility only", "source_manifest_sha256": "47d8fc112f82d05bfce84548f86538e2b259b67efaf3635c5d2de42c567a434b", "numeric_execution": False},
    "PENCIL": {"sealed_source_plan": "pencil_collab_representative_baseline_source_plan_20261004_v1", "manifest_sha256": "2c2ecd493f754f6901ca168005febbda65d65be0e4570e9fce326df188ff497b", "one_GPU_route": True, "representative_resource_execution": False, "paper_identities_added": 0},
    "new_predictive_advantage": False,
    "fresh_manuscript_acceptance": False,
    "original_paper_scores_unchanged": True,
    "next_actions": ["Finish minimal bucket equivalence and native resource check before paired auxiliary fits.", "Resolve complete-epoch Pubmed repeat blocker without changing its tolerance.", "Observe PENCIL dependencies and prepare a concrete representative resource qualifier.", "Continue original Amazon queue; preserve all paired outcomes."],
}
put(PUB / "RESEARCH_EVENT.json", event)
ledger = json.loads((P / "research_ledger.json").read_text())
key = "repeatability_and_baseline_progress_20261004_v1"
assert key not in ledger
ledger[key] = event
ledger["updated_UTC"] = now
ledger["updated_utc"] = now
ledger["publication_history"].append({"event": "native_gradient_startup_progress_20261004_v1", "commit": prior["remote_commit"], "receipt": "publication/native_gradient_startup_progress_20261004_v1/PUSH_RECEIPT.json", "exact_ref_verified": True})
ledger["latest_verified_publication"] = {"commit": prior["remote_commit"], "branch": prior["branch"], "receipt": "publication/native_gradient_startup_progress_20261004_v1/PUSH_RECEIPT.json", "receipt_sha256": sha(P / "publication/native_gradient_startup_progress_20261004_v1/PUSH_RECEIPT.json"), "GitHub_exact_ref_verified": True, "before_current_publication": True}
ledger["publication_state"] = {"branch": prior["branch"], "latest_pushed_commit": prior["remote_commit"], "pending_changes": "Pubmed actual zero-update result, bucket source and review, Amazon26 metadata, and PENCIL representative baseline plan await this explicit publication."}
(P / "research_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n")
status = (P / "PUBLIC_STATUS.md").read_text()
status = status.replace("Updated: 2026-10-04T10:05:25.364024+00:00", "Updated: " + now)
status = status.replace("At 2026-10-04T09:47:14.436380+00:00, Amazon training was **5/15 complete fits**; the sixth, `split1_gnnm_boundary_4_seed29`, had **146/2700 updates**.", "At " + monitor["UTC"] + ", Amazon training was **5/15 complete fits**; the sixth, `split1_gnnm_boundary_4_seed29`, had **390/2700 updates**.")
old = "Fresh review blocked the new zero-update comparison source because it omitted storage alias relationships between distinct tensor views; no runtime alias defect is inferred. V2's narrow repair is sealed and under a different fresh source review; it has not executed."
new = "Fresh review blocked v1's incomplete storage-alias comparison. A different reviewer passed the narrow v2 repair, and the actual zero-update comparison then completed: all four controlled first-batch comparisons passed the original rule for outputs, logits, loss and gradients. No optimizer updates or VALID/TEST access occurred. This does not replace the earlier failed complete-epoch repeat or identify its cause. [Bounded result](pubmed_shared4_zero_update_first_batch_repeat_root_adoption_20261004_v1/RESULTS_SUMMARY.md)."
assert old in status
status = status.replace(old, new)
status = status.replace("An exact-law support-bucket optimization is being inspected before broader training.", "The exact-law support-bucket candidate passed independent static review for equivalence-QA eligibility; actual numerical and native resource checks remain pending.")
status = status.replace("Feature-enabled PENCIL is one source-pinned future comparison; its resource cost is unknown.", "A concrete feature-enabled PENCIL source plan is sealed: direct one-process launch on GPU0, author hidden512/eight-layer/20epoch recipe, and actual152-node/306-column query shapes. Its package availability and complete-epoch resource cost remain unobserved; no baseline score or GNNM novelty follows from this plan.")
status = status.replace("Latest verified GitHub head before this update:31f3f581aed94b6a445aaaecd16f713c0e9ccb79.", "Latest verified GitHub head before this update:" + prior["remote_commit"] + ".")
(P / "PUBLIC_STATUS.md").write_text(status)
state = (P / "RESEARCH_STATE.md").read_text().replace("Updated: 2026-10-04T10:05:25.364024+00:00", "Updated: " + now)
state = state.replace("sixth at146/2700", "sixth at390/2700")
state = state.replace("Obtain fresh independent exact review of sealed Pubmedv2 storage-alias comparison repair, then execute only the fixed zero-update comparison. Do not change tolerance or infer a causal runtime alias defect.", "Use Pubmedv2's completed matched zero-update comparisons to narrow the remaining complete-epoch repeat discrepancy. Resolve the concrete blocker minimally, preserve the prior failure, and resume predictive study only after the original qualification. Do not change tolerance or infer a causal runtime alias defect.")
state = state.replace("Prepare the missing PENCIL launcher/data/query source scopes; do not infer its speed or superiority from NCNC measurements.", "Use the completed PENCIL launcher/data/query source plan; observe missing dependencies and prepare the smallest representative complete-epoch/full-VALID resource observation. Do not infer speed or superiority from NCNC measurements.")
(P / "RESEARCH_STATE.md").write_text(state)
roots = [
    "pubmed_shared4_zero_update_first_batch_repeat_root_adoption_20261004_v1",
    "pubmed_shared4_zero_update_first_batch_repeat_independent_source_review_20261004_v2",
    "pubmed_shared4_zero_update_first_batch_repeat_execution_root_20261004_v1",
    "exact_cb_support_bucket_implementation_hypothesis_preparation_20261004_v2",
    "exact_cb_support_bucket_implementation_hypothesis_independent_source_review_20261004_v2",
    "pencil_collab_representative_baseline_source_plan_20261004_v1",
    str(SNAP.relative_to(P)),
]
paths = [P / name for name in ("PUBLIC_STATUS.md", "RESEARCH_STATE.md", "research_ledger.json", "adopt_pubmed_zero_update_repeat_20261004_v1.py", "publish_repeatability_and_baseline_progress_20261004_v1.py", "prepare_pubmed_zero_update_execution_client_20261004_v1.py")]
for root in roots:
    paths.extend(p for p in (P / root).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
amazon = P / "amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1"
paths.extend(p for p in amazon.rglob("*") if p.is_file() and (p.name.startswith("MONITOR_0026") or "MONITOR_0026_files" in p.parts))
paths += [PUB / "RESEARCH_EVENT.json", P / "publication/native_gradient_startup_progress_20261004_v1/PUSH_RECEIPT.json", P / "publication/native_gradient_startup_progress_20261004_v1/ACKNOWLEDGEMENT.json"]
allowed = {".py", ".json", ".jsonl", ".md", ".txt", ".html", ".diff", ".patch", ".log", ".raw", ".sha256", ".csv", ".xml", ".sh", ".yml", ".yaml"}
rows = []
for p in sorted(set(paths)):
    assert p.is_file() and not p.is_symlink() and p.stat().st_size < 2_000_000
    assert p.suffix in allowed or p.name in {"SHA256SUMS", "NOTE_SHA256SUMS", ".gitignore"}
    source = str(p.relative_to(P))
    rows.append({"source": source, "target": "experiments_iclr/postsubmission_20260930/" + source, "bytes": p.stat().st_size, "sha256": sha(p)})
put(PUB / "INVENTORY.json", {"UTC": now, "branch": prior["branch"], "expected_head": prior["remote_commit"], "files": rows, "remove": [], "message": "Record bounded Pubmed repeat and representative baseline progress"})
print(json.dumps({"inventory": str((PUB / 'INVENTORY.json').relative_to(P)), "files": len(rows), "bytes": sum(r["bytes"] for r in rows), "no_predictive_score_changes": True}))
