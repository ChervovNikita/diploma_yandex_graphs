"""Record a completed bounded repeatability check without promoting it to training."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parent
OBSERVED = PHASE / "pubmed_shared4_zero_update_first_batch_repeat_execution_root_20261004_v1" / "owned_monitor01"
DEST = PHASE / "pubmed_shared4_zero_update_first_batch_repeat_root_adoption_20261004_v1"

def read(name):
    return json.loads((OBSERVED / name).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, value):
    with (DEST / name).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

observation = read("OBSERVATION.json")
result = observation["diagnostic_projection"]
terminal = read("supervision/first_batch_repeat/run01/TERMINAL.json")
custody = read("supervision/first_batch_repeat/run01/SUPERVISOR_CUSTODY.json")
final = read("first_batch_repeat/run01/FINAL_CUSTODY.json")
assert observation["supervisor_identity"] is None
assert terminal["status"] == "COMPLETE_ZERO_UPDATE_DIAGNOSTIC_ONLY"
assert terminal["physical_exit_code"] == 0
assert terminal["physical_session_closed"] and terminal["direct_child_reaped"]
assert not terminal["errors"] and not terminal["unresolved_cleanup"] and terminal["stop"] is None
assert custody["terminal_sha256"] == sha(OBSERVED / "supervision/first_batch_repeat/run01/TERMINAL.json")
assert terminal["linked"]["final_custody_sha256"] == sha(OBSERVED / "first_batch_repeat/run01/FINAL_CUSTODY.json")
assert final["completed"] and result["status"] == "COMPLETE"
assert result["source_manifest_sha256"] == "7219d7d7863b431d4d4c2b8775810211ce3ebdd426ec05de8c58e970efb1a36f"
assert result["raw_descriptor"]["sha256"] == terminal["linked"]["result_sha256"]
for row in custody["child_output_files"]:
    path = OBSERVED / "first_batch_repeat/run01" / row["path"]
    if path.exists():
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
    else:
        retained = next(r for r in observation["large_server_retained_metadata"] if r["path"] == "first_batch_repeat/run01/" + row["path"])
        assert retained["sha256"] == row["sha256"] and retained["bytes"] == row["bytes"]
assert result["all_four_state_matching_valid"] and result["saved_tree_unchanged"]
assert all(result["initial_restores_exact"].values())
assert all(not result[key] for key in ("engineering_qualification_PASS", "scientific_fit_admitted", "state_donor_allowed", "VALID_files_opened", "TEST_files_opened", "score_files_opened", "numeric_rule_changed", "kernel_causality_established"))
assert result["progress"]["Adam_completed"] == result["progress"]["Adam_started"] == 0
assert result["progress"]["prefixes_completed"] == result["progress"]["gradient_backward_calls"] == 4
comparisons = []
for comp in result["comparisons"]:
    assert comp["state_matching_valid"] and comp["numerical_comparison_collected"]
    assert comp["status"] == "MATCHED_COMPARISON"
    assert all(v["passed_original_fixed_predicate"] for v in comp["original_fixed_comparator_verdicts"])
    comparisons.append({key: comp[key] for key in ("label", "candidate", "reference", "status", "state_matching_valid", "native_streams_exact", "initial_restores_exact", "saved_and_idle_exact", "saved_and_cross_unit_step_alias_isolation_exact", "original_fixed_comparator_verdicts")})
assert {c["label"] for c in comparisons} == {"within_A", "within_B", "between_first", "between_repeat"}
DEST.mkdir()
adoption = {
    "UTC": datetime.now(timezone.utc).isoformat(),
    "status": "ADOPTED_BOUNDED_ZERO_UPDATE_REPEATABILITY_RESULT_ONLY",
    "source_manifest_sha256": result["source_manifest_sha256"],
    "root_release_sha256": result["root_release_sha256"],
    "observation_sha256": sha(OBSERVED / "OBSERVATION.json"),
    "raw_server_descriptor": result["raw_descriptor"],
    "physical_terminal_sha256": terminal["physical_terminal_sha256"],
    "supervisor_terminal_sha256": custody["terminal_sha256"],
    "comparisons": comparisons,
    "progress": result["progress"],
    "child_seconds": result["inclusive_child_wall_seconds"],
    "physical_session_closed": True,
    "scientific_fit_admitted": False,
    "earlier_full_epoch_failure_preserved": True,
    "original_paper_scores_unchanged": True,
    "new_predictive_success": False,
    "interpretation": "Four first-batch prefixes agree under the original rule from controlled repeated and reconstructed states. This narrows the previous discrepancy but does not reproduce a complete training epoch or identify its cause.",
    "next_action": "Resolve the concrete remaining full-epoch repeat blocker with a minimal paired qualification, then return to representative predictive experiments. Do not broaden to unrelated engineering checks or relax tolerances.",
}
save("ROOT_ADOPTION.json", adoption)
(DEST / "RESULTS_SUMMARY.md").write_text("""# Pubmed first-batch repeatability result

The fixed zero-update diagnostic completed on the authorized 18.77 GPU1. The physical child exited successfully and its session closed. Four first-batch forward/backward prefixes were evaluated in two independently reconstructed units from the exact owned engineering state.

All four comparisons passed the original rule for encoder outputs, positive and negative logits, loss, and gradients before Adam. Their complete pre-forward states matched, including tensor-storage alias relationships. There were no optimizer updates or validation/test reads.

This is a bounded repeatability result. It narrows the earlier discrepancy, but does not explain it or replace the failed full-epoch comparison. No Pubmed GNNM predictive result is available yet. The next task is the minimum complete-epoch qualification needed to resume the paired predictive study; the failed result and tolerance stay unchanged.

The large raw diagnostic and replay metadata remain on the server. Their hashes are recorded in ROOT_ADOPTION.json and the original monitor metadata. No methodological advantage or accuracy gain follows from this check.
""")
save("SEAL.json", {"files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(DEST.iterdir()) if p.is_file()]})
print(json.dumps({"adoption": DEST.name, "all_four_comparisons_matched": True, "predictive_success": False}))
