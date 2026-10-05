# Current GNNM research status

Updated: 2026-10-05T01:16:47.635160+00:00. Goal active and incomplete. Original paper scores remain unchanged.

## Accuracy evidence

No method has yet shown confirmed superiority over both competent singles and ordinary independent ensembles. No new manuscript acceptance exists.

The completed five-seed ogbl-collab TEST family reports Hits@50: single64 **66.4426%**, independent4 **67.6298%**, private completion4 **67.2909%**, pooled completion4 **67.0673%**, capacity70 **66.7642%**. Private completion improves over single but trails independent4. Its frozen private–pooled contrast is inconclusive: +0.2236 percentage points, paired descriptive interval [-0.7775,+1.2247], sign-flip p=.6875. These are seed contrasts on one split. TEST has been consumed. [Full results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## Current method and literature synthesis

The next candidate puts a compact learned private frame before the link head multiplies the two node embeddings coordinate by coordinate. Each route can then expose different mixed-coordinate endpoint interactions while retaining the common encoder. This is a directly served operation, rather than a claim based on embedding distances or auxiliary fit. A shared encoder and context paths can compensate for the original interface; no whole-model capacity separation is claimed. [Method and falsification](quality_method_synthesis_20261005_v1/REPORT.md).

BatchEnsemble/TabM sharing, graph-endpoint reflections in HousE/GoldE, function-preserving adapter tangents in LoRA-Ensemble/ETHER, and reflection sandwiches in HTA are prior. The possible incremental contribution is the placement and learning recipe, if it improves over same-operation singles and independent ensembles. No novelty clearance or predictive improvement follows from the construction. [Closest-prior assessment](endpoint_frame_function_preserving_prior_20261005_v1/REPORT.md).

The private-frame CPU fixture passed in 3.470 seconds with zero fits/updates. The stronger v2 controls also passed: shared-frame F4, one single receiving all four products, and ordinary independent single with the identical frame operation. This second check used zero fits and exactly one native Adam update, taking 4.101 seconds inclusive. All four single-frame tangent paths are accessible on the fixed fixture. This establishes implementation behavior, not baseline competence or accuracy. V1 controls remain preserved, superseded and unexecuted. [Actual v2 receipt](endpoint_frame_control_execution_20261005_v2/RECEIPT.json).

Literature memory v60 has **230 scoped conclusion records, 178 paper groups and two software groups**. These are not full-paper reading totals. ETHER and HTA are two added primary method scopes; LoRA initializer, HeaRT and PENCIL code reads are retained-identity upgrades. Earlier records/history are preserved. [Adoption](literature_memory/index_v60/ROOT_ADOPTION_NOTES.md).

## Representative new benchmark

The full released Citeseer-HeaRT inputs were acquired on the authorized one-GPU server. The official 880,240,878-byte archive matched both its MD5 and retained SHA256. Only six Citeseer files were extracted; TEST inputs remain separately withheld. All large files stay on the server. [Acquisition](citeseer_heart_acquisition_execution_20261005_v1/OBSERVATION_001.json).

CPU input qualification passed: supplied float32 features [3327,3703] equal every row/column of independently pinned raw Planetoid Citeseer features. Native TRAIN has 3,870 nonself positives; VALID has 227 positives and 500 fixed hard negatives each. Negative endpoint ranges and query-row association were verified without changing native duplicates/collision semantics. No withheld TEST input or prediction outcome was read. The first check's unsupported PyG keyword failure is preserved; the corrected check took 11.351 seconds inclusive with zero fits/updates. [Qualification](citeseer_feature_qualification_20261005_v2/RESULT.json).

The source-native NCN runner is implemented and sealed. It must use Citeseer's puregcn/JK, width256, batch1024, complete VALID-MRR selection and full native early stopping; Collab's width/encoder/schedule cannot be transplanted. NCN and an ordinary independent four-model bank are required quality references. The native-single CUDA probe passed with one masked TRAIN batch backward and all113,500 VALID negative forwards, zero optimizer updates and no ranking metric. Peak allocated memory was896,887,808bytes; runtime and sparse operations are qualified. The missing sparse dependency and pip-bootstrap failures are preserved. A private in-repository runtime/overlay leaves the existing training environment untouched. [Runtime evidence](citeseer_ncn_native_runtime_qualification_20261005_v2/RESULT.json).

One full native NCN single seed0 was launched at01:26UTC and physically observed running at01:27UTC, epoch88/264updates, worker404460. It uses the complete native schedule and all fixed VALID negatives. Its selected quality remains unopened. A prospective three-block36-fit frame screen fixes all arms/seeds/member0 reuse before this first fit; only this baseline has been admitted, with the remaining dispatch pending exact adapter runtime checks and full-fit resource assessment. [Frozen development plan](citeseer_endpoint_frame_paired_development_20261005_v1/PLAN.json), [actual baseline observation](citeseer_endpoint_frame_paired_development_20261005_v1/MONITOR_001.json). [Exact recipe](small_representative_link_benchmark_scout_20261005_v1/REPORT.md).

## Existing frozen cohorts

Amazon observation56 at **2026-10-05 01:09 UTC** verifies 10/15 complete, current `split2_gnnm_boundary_4_seed43` update1189/2700, worker403194 live, no failures. [Receipt](amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1/MONITOR_0056_RESULT.json). Next sequence57 only when useful.

18.77 remains unobservable after SSH connection timeouts. Its last successful compactv3 observation at00:01 UTC has conditional Collab5/9 and DDI3/12 complete. A lost route does not establish job failure; no jobs were signalled or restarted. [Last successful metadata](compact_owned_queue_monitor_20261005_v3/OBSERVATION.json). Comparative outcomes remain unopened until complete frozen cohorts.

The earlier graph-curvature selector is closed for absent intended intervention in a required block. Matching, graph-view and conditional-pattern proposals retain their exact prior/serving limits; no failed version is renamed into a success. Their detailed records remain in the ledger and [preserved previous status](coordination_snapshots/20261005_candidate_controls_and_citeseer_inputs_v1/PUBLIC_STATUS.md).

## Publication and boundaries

Latest verified GitHub ref is **542db059cbf94b70edc294d9a8ac9b1c53b8a89d**, branch `codex/postsubmission-research-20260930`. Current controls, v60, qualified inputs, runner and actual baseline launch await the next reviewed publication. [Verified prior push](publication/endpoint_frame_synthesis_20261005_v1/PUSH_RECEIPT.json). No18.77 Git synchronization is claimed.

Fresh manuscript reviews use the supplied skill, immutable evidence and no requested verdict. Deliberate operations stay in the authorized repositories; seven-GPU access is forwarding only. No sudo, PDF compilation, GENLINK, Desktop writes, host changes or unrelated-data access.
