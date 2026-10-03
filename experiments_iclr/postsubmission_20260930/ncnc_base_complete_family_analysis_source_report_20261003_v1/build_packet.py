"""Seal source-only analysis-path findings; no model/data/outcome imports."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
DRIVER = PHASE / "graph_ncNC_collab_predictive_driver_preparation_20261003_v1"
DESIGN = PHASE / "graph_ncNC_collab_predictive_pilot_design_20261003_v1"
STAMP = datetime.now(timezone.utc).isoformat()


def descriptor(path, base=PHASE):
    data = path.read_bytes()
    return {"path": str(path.relative_to(base)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


verified_packets = []
for packet, expected in [(DESIGN, "e97aeb2655d54f261acca1874c8e03047608625aac1b5e1e980853f7d44e0004"),
                         (DRIVER, "a59c669356e1a1437ce90107a7c76eb8f0b5579dc48d67b4378a7e237e956b7e")]:
    assert descriptor(packet / "MANIFEST.json")["sha256"] == expected
    manifest = json.loads((packet / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        path = packet / item["path"]
        actual = descriptor(path)
        assert actual["bytes"] == item.get("bytes", item.get("size")) and actual["sha256"] == item["sha256"]
    verified_packets.append({"manifest": descriptor(packet / "MANIFEST.json"), "verified_source_payloads": len(manifest["files"])})

reads = {
    "pilot_lock.py": [[1, 152]],
    "pilot_common.py": [[1, 172]],
    "pilot_run.py": [[1, 112]],
    "pilot_fit.py": [[1, 151]],
    "pilot_accounting.py": [[1, 70]],
    "pilot_evaluate.py": [[1, 81]],
    "pilot_state.py": [[80, 171]],
    "static_check.py": [[1, 102]],
}
write("INPUT_BINDINGS.json", {"schema": "ncnc_base_family_analysis_source_inputs_v1", "UTC": STAMP,
    "verified_packets": verified_packets,
    "sources": [descriptor(DRIVER / name) for name in reads] + [descriptor(DRIVER / name) for name in ["ROOT_RELEASE_SCHEMA.md", "README.md", "STDLIB_PREPARATION_CHECK.json", "SOURCE_BINDINGS.json"]] + [descriptor(DESIGN / "PILOT_PLAN.json")],
    "source_review": descriptor(PHASE / "graph_ncNC_predictive_qualification_execution_root_20261003_v1/ROOT_SOURCE_INSPECTION.json"),
    "review_scope": "Root source inspection supports numerical engineering; it is not a separate selected-checkpoint replay review or fit/outcome certificate.",
    "existing_runtime_release_state_inspected": False,
    "fitted_outcomes_or_terminal_receipts_read": False,
    "actual_checkpoints_arrays_labels_runtime_binaries_read": False})
write("READ_SCOPES.json", {"schema": "ncnc_base_family_analysis_source_read_scopes_v1", "UTC": STAMP,
    "line_indexing": "one-based inclusive",
    "source_ranges": [{"source": descriptor(DRIVER / name), "ranges": ranges} for name, ranges in reads.items()],
    "design": "Selected frozen endpoint/selection/family/work/control fields in PILOT_PLAN.json. Combined display truncated in trailing root-context metadata; no certification of every trailing field.",
    "navigation": "Directory/file inventories and bounded source-pattern searches used to locate NCNC closure and distinguish unrelated Mixed40/calibration evaluators. Retrieved paths are not outcome reads.",
    "existing_review": "ROOT_SOURCE_INSPECTION.json fully displayed. Adjacent resource-v3 serialization source review displayed as dependency context; it does not certify the predictive evaluator.",
    "primary_literature_reads": 0, "scientific_execution": False,
    "source_checker_fixture_execution": False, "actual_OUTCOME_CHECKPOINT_TEST_access": False})

plan = {
    "schema": "minimal_ncnc_selected_state_replay_sidecar_source_plan_v1", "UTC": STAMP,
    "status": "PROPOSAL_ONLY_MISSING_REPLAY_PASS_NOT_IMPLEMENTED_OR_RELEASED",
    "existing_analysis_path": "Reuse sealed driver family_lock, pilot_evaluate and selected-state semantics; do not rewrite fitting, selectors or family closure.",
    "necessity_scope": "Only the independent selected-checkpoint numerical replay pass is missing. Existing family_lock already supplies the original full-family saved-VALID summary.",
    "release": {
        "separate_root_release_required": True,
        "bind": ["exact sidecar manifest/source review and qualification receipt", "driver/design/prototype/resource manifests", "TRAIN/raw/VALID authority and qualified runtime", "immutable FAMILY_LOCK.json", "all20 original unit COMPLETE/terminal-failure receipts", "all25 selected checkpoint/selection bindings", "each unit's closed attempt ledger and journal custody", "exact output path and one authorized replay invocation"],
        "forbidden": ["TEST authority or data accessor", "fit/resume/retrain", "new checkpoint selection", "threshold/margin/tolerance change after outcomes", "calibration or retuning", "successful subset summary"],
    },
    "gate_before_numerical_or_predictive_access": [
        "Stdlib verify exact source/release/data/runtime metadata and source custody.",
        "Require original20(unit,seed) slots once,35planned unique fits and25original arm/seed slots. Reject any incomplete/pending unit, duplicate identity or borrowed output directory.",
        "Accept only COMPLETE_FAMILY_LOCKED35complete fits/25complete cells for predictive replay. A bound TERMINAL_FAILED_FAMILY_LOCKED produces metadata-only failure/cost closure with null contrasts and no partial replay.",
        "Recheck100ordered epochs,1700optimizer steps per unique fit, all60084VALID positives/100000shared negatives,17full TRAIN batches and required101st I4 candidate/four extra VALID passes.",
        "Recheck five F4 twin initial-state/RNG identities and every one of100epoch sampler/permutation/start/end RNG identities.",
        "Bind all selected states/selection identities and candidate orders before loading the first checkpoint. Bind frozen family/closure markers and prohibit subsequent fit/resume/seed replacement.",
    ],
    "replay_contract": [
        "Load only authenticated selected checkpoints from this driver; require original schema, family/unit/seed/arm, one snapshot for N64/N70/F4 or four for I4, original selector identity, pool and TRAIN-only VALID graph.",
        "Construct exact pinned native/factor model, strict-load its complete saved model state and check flags/tree/state digest; no optimizer update. Use admitted float32/no-autocast/no-TF32 runtime.",
        "For N64 verify selected snapshot/selector equals native-bank member0's individually selected best in the bound own journal. Do not require equality to member0 in the I4 served checkpoint, whose selection can be a different bank/epoch.",
        "Call existing score_valid/evaluator/hits50; include every official row in canonical order and eval tails, complete TRAIN graph, exact private versus pooled-after-clamp mode, equal raw-logit mean before official Hits50, strict ties fail.",
        "Require each replayed selected Hits50 to equal its original saved selected Hits50 exactly. Retain mismatches and suppress aggregate conclusions; no new tolerance or score-threshold policy. Stored per-model score digests can be checked against the bound journal where available; no original pooled I4 score digest was explicitly exported.",
        "Keep intermediate metrics private until all25replays pass. A numerical failure/mismatch retains all25slots, all failed work and null family contrasts. No success-only subset.",
        "The source uses deterministic_algorithms=False; numerical replay success and bitwise raw-score reproduction remain unqualified runtime facts. A mismatch is not silently repaired or retrained.",
    ],
    "original_endpoints": {
        "primary": "Five paired separately VALID-selected factorized_private_4 minus factorized_pooled_after_clamp_4 official Hits50 differences; mean/sampleSD/range/sign counts and selection identities.",
        "baseline_checks": "Allfive N64/I4/N70 means/sampleSD and paired candidate-minus-control descriptive differences remain exploratory. N70 is required for the parameter-capacity interpretation; no new arm, margin, continuation or novelty threshold.",
        "inference": "Training randomness conditional on one fixed graph/time split. VALID selection optimism remains; no graph-row bootstrap or confirmatory TEST claim. Frozen two-sided exact-sign-flip minimum p with five is0.0625; no added significance rule.",
    },
    "costs": {
        "training_denominator": "35unique fits;20native64 member fits+10factorized fits+5native70 fits;100epochs/fit;59500optimizer steps. N64 is an alias of I4 member0 fitting, not five extra fits.",
        "replay_nominal_calls": "40top-level score_valid calls:5N64+20I4 member calls+5privateF4+5pooledF4+5N70; eachF4 call serves its four members. No training-compute equality inferred.",
        "measured": ["original all closed attempt inclusive wall", "observed interrupted-attempt lower bounds and unknown remainder separately", "full TRAIN/epochVALID and candidate101 work", "checkpoint custody/load/transfer and replay encoder/decoder/scoring wall", "fresh replay peak allocated/reserved memory", "finalization with disclosed unmeasured terminal-write tail"],
        "alias_policy": "Charge native_bank4 fit/101candidate work once. Disclose donor reuse; do not assign a fictitious standalone N64 setup time or call its deployment inference free.",
    },
    "required_source_qualification_before_any_run": [
        "Independent review of this sidecar and exact source seals; no inherited automatic release from fit engineering review.",
        "Fabricated complete35/25, missing/failed/pending, duplicate-alias, custody/selector/model-state/RNG mismatch and no-subset fixtures.",
        "Admitted ordinary-runtime synthetic replay of trusted fabricated saved states with exact Hits50/routing/mean/tie/coverage checks and mismatch suppression.",
        "Separate root admission of runtime/resource and complete selected-state replay; this plan contains no runtime command or release.",
    ],
    "scientific_execution": False, "outcome_access": False, "TEST_opened": False,
    "existing_evaluator_sources_changed": False, "canonical_edits": False,
}
write("MINIMAL_REPLAY_PLAN.json", plan)

report = """# NCNC base family: source-only analysis/evaluation path

**Reuse the existing `family_lock` analysis.** Sealed driver v1 already implements a separately root-released full-family saved-VALID evaluator/summary. Its `family_lock` stage is metadata-only and computes the frozen endpoint after complete-family admission. **A separate selected-checkpoint numerical replay evaluator is missing.** This report preserves the existing closure and proposes only that small replay sidecar; no source is rewritten or executed.

## Existing release and denominator gates

The source manifest is `a59c669356e1a1437ce90107a7c76eb8f0b5579dc48d67b4378a7e237e956b7e`; design manifest is `e97aeb2655d54f261acca1874c8e03047608625aac1b5e1e980853f7d44e0004`. All21 driver and9 design source payloads match their manifests. Existing root inspection supports numerical engineering, and retained preparation metadata declares fabricated complete35/25 and terminal-failure fixtures. Neither certifies selected-checkpoint numerical replay. Actual release/terminal/outcome state was not inspected.

`pilot_common.preflight/admit_invocation` requires the exact manifest, root stage authorization and exact unit/seed/absolute-output invocation. `family_id` and lock destination must match all fits; source/design/prototype/resource, TRAIN/raw/VALID and runtime authority are bound. Fit authorization does not automatically release `family_lock`. Closure needs a fresh external lock output and exactly20distinct unit/seed inputs: five seeds0–4 for each native_bank4/privateF4/pooledF4/native70 unit.

Before private selection disclosure, `pilot_lock` checks every COMPLETE or explicitly root-retired terminal failure, identity/path custody,100epochs,1700steps per unique fit, full TRAIN/VALID coverage, native seeds and all selected artifact/checkpoint hashes. Full success requires35unique fits and all25served cells. Failed closure retains missing arm/seed cells and costs, gives null arm/primary summaries, and authorizes no TEST. Closure markers prevent subsequent fit/resume/seed replacement. Pending inputs cannot be treated as terminal failures.

## Frozen scientific analysis and controls

Primary analysis is five paired differences of separately VALID-selected privateF4 minus pooled-after-clampF4 official Hits@50, with mean, sampleSD, range and signs; cells retain candidate/epoch identities. F4 initial model/emptyAdam/flags/RNG hashes and every100epoch sampler/permutation/start/end RNG stream must match. Pairing refers to training randomness on one graph/time split. VALID was used for selection, so this is an optimistic development pilot, not confirmatory held-out evidence. There is no quality margin or continuation threshold; five blocks cannot give a two-sided exact-sign-flip p below0.0625. No query-row graph bootstrap is justified.

| Control | Complete required scope and interpretation |
|---|---|
| N64 | Exact native-bank member0 fit, independently selected by its own VALID criterion. An alias, not five additional fits. |
| I4 | Four independently learned encoders/decoders/optimizers/RNGs per seed, seeds`s+5m`;100same-epoch banks followed by candidate101assembled from individual best states. Four additional full VALID passes are charged. |
| N70 | Required native width70 same-recipe100epoch single for parameter capacity;44663/39622 total/active parameters versus F4's43790/38793. It is not a compute match. |

Allfive baseline cells and failures remain required. Baseline contrasts are exploratory; existing closure emits arm means/SD and the primary paired contrast, not explicit paired baseline-difference arrays. Any later descriptive candidate-minus-N64/I4/N70 differences must use allfive locked cells and preserve this status. The family supplies these admitted controls; it does not establish a broader best-tuned structural single or SOTA claim.

N64's selected member0 state can differ from member0 in the served I4 checkpoint, because the I4 bank has its own selector. Donor verification must target member0's individual best, not demand equality to the served bank's first member. `pilot_fit` exports N64 directly from that individual best snapshot, but `family_lock` verifies custody/metadata rather than deserializing this alias equality.

## Missing replay pass: minimal source-only proposal

The current lock reads saved selection Hits50 values. It does not deserialize selected states, reconstruct predictions or verify their numeric replay. The existing `pilot_evaluate` does supply exact full60084positive/100000shared-negative scoring on the complete TRAIN graph, canonical order/tails, float32 equal raw-logit pooling and source-bound OGB strict Hits50 with an explicit arithmetic cross-check. Reuse those functions.

`MINIMAL_REPLAY_PLAN.json` specifies a separate root-released read-only sidecar: bind the immutable complete35/25lock and every original terminal/selected/journal/attempt receipt before Torch/data/outcome access; keep terminal-failed closure metadata-only; strict-load only the driver's authenticated selected snapshots and verify arm/seed/route/selector/state plus N64 individual-best alias; replay every25cell with original graph/pool/metric; require exact saved Hits50 agreement and retain any mismatch/failure without subset summaries or a new tolerance. Deterministic algorithms are disabled in the frozen runtime, so successful numeric replay remains an unqualified execution fact. Original individual-model score digests can be checked where the journal retained them; no explicit pooled I4 score digest was exported. Independent sidecar review and fabricated ordinary-runtime replay qualification would precede a separate release. This report contains no evaluator implementation or launch command.

## Cost accounting

Denominator:20native64 member fits+10factorized fits+5native70 fits=35unique fits,3500fit epochs and59500optimizer transitions. Each F4 fit has four served members; these counts do not claim equal work to a single. Charge the native bank's fit/101candidate work once, disclose N64 donor reuse, and do not call N64 inference free or invent standalone setup timing.

Existing lock reports all closed-attempt inclusive wall, TRAIN/epochVALID work, candidate101passes, failures and exactness flags; the original ledgers/`Attempts.receipt` retain observed interrupted-wall lower bounds and unknown remainders. The lock's compact unit-cost rows do not separately expose those lower-bound seconds, so preserve the ledgers in the analysis. Inclusive costs cover setup/custody/transfer/training/VALID/selection/journals/finalization, with a disclosed unmeasured terminal-write tail. A replay sidecar adds nominally40top-level full-VALID `score_valid` calls (5N64+20I4member+5privateF4+5pooledF4+5N70); measure its custody/load/transfer/scoring and peak memory separately. Do not double-count donor fitting or omit failed/replay work.

No fitted outcome, checkpoint/array/label/runtime binary, score subset or TEST file was read. No numerical library or project source was imported/executed; source fixtures were inspected, not run. No threshold, fit, evaluator source, canonical status/ledger or publication was changed.
"""
(ROOT / "REPORT.md").write_text(report)
for path in ROOT.glob("*.json"):
    json.loads(path.read_text())
builder = (ROOT / "build_packet.py").read_text()
ast.parse(builder); compile(builder, "build_packet.py", "exec")
write("VERIFICATION.json", {"schema": "ncnc_base_analysis_source_verification_v1", "UTC": STAMP,
    "status": "PASS_SOURCE_CUSTODY_JSON_AST_ONLY", "verified_existing_source_payloads": 30,
    "existing_driver_and_design_unchanged": True,
    "builder_AST_compile": True, "project_source_import_or_execution": False,
    "outcome_checkpoint_array_label_TEST_access": False,
    "canonical_or_live_execution_changes": False})
files = [p for p in sorted(ROOT.iterdir()) if p.is_file() and p.name not in {"MANIFEST.json", "SEAL.json"}]
write("MANIFEST.json", {"schema": "ncnc_base_analysis_source_packet_manifest_v1", "UTC": STAMP,
    "status": "SEALED_SOURCE_ONLY_EXISTING_ANALYSIS_REUSED_REPLAY_PROPOSAL_UNRELEASED",
    "payload": [descriptor(p, ROOT) for p in files], "payload_count": len(files),
    "payload_bytes": sum(p.stat().st_size for p in files),
    "execution_authorized": False, "outcome_access": False, "canonical_edits": False})
write("SEAL.json", {"schema": "ncnc_base_analysis_source_packet_seal_v1", "UTC": STAMP,
    "manifest": descriptor(ROOT / "MANIFEST.json", ROOT), "report": descriptor(ROOT / "REPORT.md", ROOT),
    "replay_plan": descriptor(ROOT / "MINIMAL_REPLAY_PLAN.json", ROOT),
    "decision": "REUSE_EXISTING_FULL_FAMILY_SAVED_VALID_ANALYSIS; MISSING_SELECTED_CHECKPOINT_REPLAY_PASS_PROPOSAL_ONLY",
    "execution_authorized": False, "outcome_access": False, "canonical_edits": False})
for p in ROOT.iterdir():
    if p.is_file(): p.chmod(0o444)
ROOT.chmod(0o555)
print(json.dumps({"manifest": descriptor(ROOT / "MANIFEST.json"), "seal": descriptor(ROOT / "SEAL.json"),
    "report": descriptor(ROOT / "REPORT.md"), "replay_plan": descriptor(ROOT / "MINIMAL_REPLAY_PLAN.json"),
    "payload_count": len(files), "payload_bytes": sum(p.stat().st_size for p in files)}, indent=2))
