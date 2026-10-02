# GNNM post-submission research state

Updated: 2026-10-02T02:02:10.843993+00:00. Goal incomplete.

PUBLIC_STATUS.md gives the current status. research_ledger.json retains prior study records. The prior coordination state is preserved in coordination_snapshots/20261002_publication_before_state_v1/.

## Completed representative comparison

All 54 fits completed normally: nine arms, Amazon Photo and Coauthor CS, three paired model seeds on core0. All 54 passed the fixed fit, provenance and competence assessment. The prospectively fixed decision is **STAGE1_NO_GO**. No 108-fit continuation is admitted. Test labels remain sealed and the original five-dataset scores remain unchanged.

Added factors do not establish benefit beyond permutation-only: relative mean validation NLL gain is +0.101% on Photo and -0.034% on CS. Combined factors and permutations beat factors alone by 1.285% on Photo (only 1/3 seed wins) and 13.487% on CS, but lose to Photo's strongest byte-matched untied control by 6.996%. This is exploratory three-seed, one-partition evidence. Byte matching does not match compute or learned capacity. Complete unsuccessful results are retained.

## Selected-state integrity replay

The 54-case replay terminated normally under supervision in 20.116 seconds: 15 cases verified; 39 exceeded the frozen selected-forward logit tolerance. Differences were 2.86e-6 to 8.58e-6. These cases remain inconclusive; no tolerance was widened and no primary score was replaced. Checkpoint, deployment, provenance, validation ordering and independent saved-logit metric checks precede the forward comparison. No training or test-label reading occurred.

## Current method work

The next pilot tests whether same-member covariance across neighboring nodes can make graph prediction sets smaller while retaining independently calibrated coverage. It includes full-marginal, alignment-shuffled, pooled, HeAD and CF-GNN controls. This is a hypothesis, not an established contribution.

The first Squirrel/Photo acquisition attempt failed during label-blind role preparation, before model training. The published Squirrel validation masks contain 718, 700 and 726 nodes; the frozen derived source reservoir allowed 667. The separate derived-role v2 amendment uniformly moves surplus published validation nodes into the final pool, preserves every original test node and retains the fixed total label budget. The final pool is mixed; results will not be called official-split accuracy. New acquisition completed in 20.634 supervised seconds. An independent payload/partition audit verified all six source partitions. No model fitting or held-out scoring took place. The failed v1 attempt and an initially unsupported Fortran-mask parser attempt are retained.

Recent backbone sources are being assessed before new primary fits. The released PolyFormer filtered-Squirrel schedule and Polynormer-r Photo schedule are concrete candidates; their reported paper scores do not transfer to this pilot's derived supervision roles. The user permits modest, fair validation tuning. Published ingredients can support a useful extension; the extension still needs a precise distinction and representative evidence.

A second source packet batches the four GNNM hidden-state paths for evaluation, with equally optimized untied/head controls. It has only syntax/source checks. No numerical equivalence, speedup or reduced memory has been measured. Graph-aware initialization and useful predictive diversity remain under literature assessment. Neural ensemble assimilation remains a source-only backup with unresolved native solver dependencies.

Paper conclusions are indexed in `literature_memory/`. Consult saved conclusions before rereading; revisit only a new version or an explicit unresolved passage. No new methodological novelty, manuscript improvement or acceptance recommendation is established. No remote scientific job is live at the latest recorded terminal observation.

## Scope and resources

Only the anogena-2 port 2222 route and GPU UUID 44039938-fd82-41d2-fefd-de71514e2fac are authorized. Never connect to the seven-GPU account. 18.77 remains unresolved. Work stays in this local phase and the remote project repository. No sudo, PDF compilation, GENLINK, Desktop/unrelated files or original-score recalculation.

Conservative reservations after acquisition admission: 133201.601915 / 133340 phase seconds, 100315.729657 / 100320 diagnostic seconds and 134810842784 / 214748364800 disk bytes. These are reservations, not actual GPU time. Earlier failed reservations remain charged. The acquisition failed after 9.1514 supervised seconds. Fetches 118–121 are complete; next unused fetch is 122.

## Evidence

- `coordinate_ensemble_execution_root_v1/stage1_assessment_run01/ASSESSMENT.json`
- `coordinate_ensemble_execution_root_v1/stage1_assessment_run01/ALL54_CELLS.csv`
- `coordinate_ensemble_execution_root_v1/TENSOR_REPLAY_EVIDENCE_INDEX_v1.json`
- `coordinate_ensemble_execution_root_v1/tensor_replay_selected_v1_run01/TERMINAL.json`
- `coordinate_ensemble_execution_root_v1/STAGE1_ASSESSMENT_AND_REPLAY_TERMINAL_v1.json`
- `coordinate_conformal_execution_root_v1/acquisition_run01/FAILED_ATTEMPT.json`
- `coordinate_conformal_execution_root_v1/acquisition_run01/Squirrel/PREPARATION.log`
- `protocols/PUBLICATION_METADATA_v1.json`
- `continuous_graph_efficiency_gap_v1/gnnm_vectorized_eval_source_v1/REPORT.md`
- `literature_memory/index_v2/LITERATURE_INDEX.json`

- `coordinate_conformal_execution_root_v1/ACQUISITION_ASSESSMENT_v3.json`

## Verified source publication

Commit `1751bc993446d95f4172def1bee35c39a47060e5` was pushed and GitHub advertised the same branch revision. Source and result summaries are published; binary evidence remains in the project archive. The research goal remains incomplete. See `publication/PUSH_VERIFIED_20261002_v1.json`.
