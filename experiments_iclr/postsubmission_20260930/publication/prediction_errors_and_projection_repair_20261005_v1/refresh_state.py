"""Preserve predecessor state and record the completed diagnosis and fixed screen."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

PHASE = Path(__file__).resolve().parents[2]
utc = datetime.now(timezone.utc).isoformat()
snapshot = PHASE / 'coordination_snapshots/20261005_complete_prediction_errors_and_negative_fusion_before_state_v1'
snapshot.mkdir()
before = []
for name in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json'):
    path = PHASE / name
    shutil.copy2(path, snapshot / name)
    before.append(dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(snapshot / 'SNAPSHOT.json').write_text(json.dumps(dict(UTC=utc, files=before), indent=2) + '\n')

status = f'''# Current GNNM research status

Updated: {utc}. Goal active and unmet. Original manuscript scores remain unchanged. No new method has established novelty and confirmed predictive superiority; no fresh manuscript acceptance verdict exists.

## Prediction errors now guide the next method

The complete Amazon/Polynormer comparison finished fifteen fits and exact selected-checkpoint logit replay. Reserved TRAIN-control accuracy is 52.7203% native member0, 52.9243% shared four and 54.0533% independent four. Shared four trails independent four by 1.1290 accuracy points and 2.2709 NLL nats on average. TEST remains unopened. [Original complete outcome](amazon_polynormer_complete15_original_comparison_root_adoption_20261005_v1/REPORT.md).

The completed VALID diagnosis covers all three splits, 6,123 nodes per predictor per split and all 44,514 retained metric rows. Mean member accuracy is 52.2402% shared versus 52.5859% independent; averaging adds 0.1633 versus 0.5907 accuracy points. Their native pooled accuracy gap is -0.7730 points: -0.3457 from mean member quality and -0.4274 from smaller pooling benefit. Mean pairwise error correlation is 0.9292 versus 0.7304. Wrong pooled answers average 92.60% versus 81.09% confidence. All four members are wrong on 93.60% of shared wrong answers versus 74.86% of independent wrong answers.

This identifies repeated mistakes, slightly weaker members and very confident errors. It does not prove a causal parameter-sharing defect. Selecting one member cannot repair an all-member error; a probability combination or nonlinear correction can sometimes recover a class none selects. Class3 is a retained shared advantage; the high-degree deficit has limited support and remains exploratory. The independent interpretation audit reconciled every retained shard, 92 summary values, 32,504 stored ratios, confusion counts and exact decompositions. [Diagnosis](amazon_polynormer_valid_error_analysis_root_interpretation_20261005_v1/REPORT.md); [independent review](amazon_polynormer_valid_error_analysis_independent_interpretation_review_20261005_v1/REPORT.md).

The next training hypothesis must improve useful complementarity while retaining individual competence. The saved assessment derives the known NCL/GNCL own/pool loss equivalence. That regularizer is an attributed control, not methodological novelty. A graph-specific extension needs a concrete graph-conditioned learning dependency and competent single/untied controls. [Assessment](amazon_error_driven_training_control_assessment_20261005_v1/REPORT.md).

## Completed aggregation screen: no confirmation admitted

The fixed V2 VALID development protocol completed all nine split/family groups, all 99 configurations and 198 raw/corrected metric panels. It used 90 small MLP fits, 36 calibration fits and 18,900 updates, with complete processed independent and comparable single references. Whole-fold exclusions and all prespecified controls remain intact. Base VALID selection and processor tuning still make it retrospective development.

The prescribed Brier-selected shared processor scores 52.5450% accuracy, 0.637247 Brier and 1.833855 NLL. Processed independent four scores 53.2528%, 0.642339 and 1.784916; processed single scores 53.1276%, 0.613544 and 1.492955. Shared four fails the fixed full quality screen against both references. Its Brier advantage over independent cannot replace its lower accuracy, worse NLL or loss to the single. The capable score-context MLP wins every ensemble split; local/full error moments are unsupported against global/diagonal/posterior controls. No final refit, confirmation or extra tuning grid is admitted. [Complete outcome and failed gates](amazon_polynormer_logits_graph_moment_complete_outcome_root_20261005_v2/REPORT.md).

The first real execution failed after 6.1128 seconds before any scientific fitting or scoring. The actual node709 witness exposed equality conditioning/roundoff in the old projection solve; full-face rank loss is a separate issue. Preserved source V3 repairs probability-space projection arithmetic only, with an explicitly approximate numerical rank cutoff. It passed independent source review and five affected checks, including agreement with an 80-digit witness reference. The completed screen took 165.664 numerical seconds, 169.628 physical child seconds and 719,511,552 bytes peak child RSS. Actual counts are 144 batched graph propagations, 2,880 sparse steps and 684 logical field propagations. All attempts and historical base costs remain retained.

## Other studies and authorized execution

At 17:47 UTC the allocation private-transfer block had completed 10/10 with BLOCK_FREEZE; its owned queue and final child were gone. Its matched-single companion block is 3/3. No scientific scores were opened. Full30 quality/full39 mechanism requirements remain. The other two pilot blocks had previously begun on77; current status is unknown under withdrawn access, and six companions were never released. A complete26-cell allocation replication amendment is being prepared prospectively, preserving old attempts and costs. It is not a setup fallback or an activated new cohort. Methods cannot be selected from alternative attempts after outcomes.

PENCIL remains 2/3, its final seed at epoch213/300 and 856 updates at 17:47 UTC. The complete collector remains disabled until custody closes. Citeseer-HeaRT and saved Borda comparisons remain small/mixed with no new superiority. Historical Collab TEST is consumed and cannot confirm later choices.

Only anogena-2.ai0001053-01174 at port2222 is accessed, host anogena-2-0/UUID GPU-44039938-fd82-41d2-fefd-de71514e2fac. 18.77/MacLink remains withdrawn; wrong-allocation evidence remains excluded. Amazon TEST and hidden states stay unopened. TRAIN-control was consumed by the original comparison and cannot become fresh fusion confirmation or training.

## Literature and publication

Index72 remains 260 scoped records/207 paper identities/two software identities, not full-paper reading counts. Consult [saved supplements V3](literature_memory/ACTIVE_SUPPLEMENTS_20261005_v3.json) before repeating retrievals. Ordinary learned fusion, hidden/depth combination and error-coupled losses have direct priors. Shared channel indices are an operational convention, not proved semantic alignment. Five new legitimate GENNN locators yielded no primary method; an OA metadata flag does not clear novelty.

The separately acknowledged pushed head before this packet is e6fbb8b3a3e4de904994a9fbd3a168902c2a6d7a. This update, the complete outcome, diagnosis, audits, failures, repair and saved conclusions are being included through an explicit inspected publication inventory. Exact commit/push receipts are kept in publication/prediction_errors_and_projection_repair_20261005_v1. Earlier canonical state is preserved in [the snapshot](coordination_snapshots/20261005_complete_prediction_errors_and_negative_fusion_before_state_v1) and the append-only ledger. A fresh skill-based paper review follows an evidence-supported manuscript revision, with immutable evidence, no author history and no requested verdict.
'''
(PHASE / 'PUBLIC_STATUS.md').write_text(status)
state = f'''# Current state: graph-ensemble predictive quality

Updated: {utc}. Goal active and unmet; original paper scores unchanged. [Evidence and limits](PUBLIC_STATUS.md).

1. Only the authorized one-GPU allocation is accessed. 18.77/MacLink withdrawn; wrong-allocation evidence excluded.
2. Complete Amazon original15 comparison and exact replay retained. TEST unopened; TRAIN-control consumed by that comparison.
3. Completed and independently checked VALID error diagnosis: shared member accuracy 52.2402% versus 52.5859%; pooling gain 0.1633 versus 0.5907 points; error correlation 0.9292 versus 0.7304. All 44,514 rows retained. This is descriptive, not a causal sharing diagnosis.
4. Fixed aggregation screen complete: nine groups/99 configurations/90 MLP+36 calibration fits. Shared processor accuracy52.5450% versus53.2528% processed independent and53.1276% single. Both full gates fail; moment contribution unsupported. No final refit, confirmation or extra grid.
5. First CPU failure, actual witness, V3 arithmetic-only repair, source reviews and affected qualification preserved. Completed physical child169.628s/719,511,552B peak RSS. No new base inference or TEST access.
6. At17:47UTC private allocation block10/10+freeze, matched single3/3; full30/full39 incomplete. Prior77 blocks started, current status unknown; six companions unreleased. Prospective complete26-cell allocation replication amendment in preparation, without outcome selection or setup-fallback relabeling.
7. PENCIL2/3, final epoch213/300 and856updates; complete collector not released. Existing link/rank outcomes remain mixed.
8. Next method follows common-error/member-strength diagnosis. NCL/GNCL coupling is a known control, not novelty; graph-conditioned private learning still needs competent controls and complete outcomes.
9. Index72 unchanged260scoped records/207paper identities/two software identities. Saved supplementsV3 prevent repeated retrieval; GENNN primary method unresolved.
10. This packet is being explicitly inventoried, inspected and published. Last separately acknowledged push before it:e6fbb8b3a3e4de904994a9fbd3a168902c2a6d7a. No fresh manuscript acceptance verdict.

Prior state is preserved in coordination_snapshots/20261005_complete_prediction_errors_and_negative_fusion_before_state_v1 and ledger/Git.
'''
(PHASE / 'RESEARCH_STATE.md').write_text(state)
ledger_path = PHASE / 'research_ledger.json'
ledger = json.loads(ledger_path.read_text())
key = 'Complete_prediction_error_diagnosis_and_negative_fixed_fusion_20261005_v1'
assert key not in ledger
ledger[key] = dict(UTC=utc, snapshot=snapshot.name,
                   error_diagnosis='amazon_polynormer_valid_error_analysis_root_interpretation_20261005_v1',
                   independent_error_review='amazon_polynormer_valid_error_analysis_independent_interpretation_review_20261005_v1',
                   full_screen_outcome='amazon_polynormer_logits_graph_moment_complete_outcome_root_20261005_v2',
                   original_failed_attempt_preserved=True, all_99_configurations_preserved=True,
                   full_quality_screen='NO_GO', final_refit_or_confirmation_admitted=False,
                   training_control='Known NCL/GNCL ancestry; explicit own/pool equivalence, not novelty',
                   private_allocation_block_completed=10, private_matched_single_completed=3,
                   private_full30_and39_requirements_retained=True,
                   prior_77_blocks_already_started=True, current_77_status='unknown; access withdrawn',
                   allocation_replication26='prospective preparation only; preserve both attempt histories',
                   pencil_complete=2, pencil_total=3, latest_metadata_UTC='2026-10-05T17:47:20.837975+00:00',
                   manuscript_scores_changed=False, new_novelty_or_superiority=False,
                   new_paper_acceptance_verdict=False,
                   publication_packet='publication/prediction_errors_and_projection_repair_20261005_v1')
ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + '\n')
print(json.dumps(dict(UTC=utc, snapshot=str(snapshot.relative_to(PHASE)), ledger_entry=key)))
