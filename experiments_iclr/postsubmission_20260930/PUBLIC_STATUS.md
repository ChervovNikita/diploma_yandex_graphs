# Current GNNM research status

Updated: 2026-10-04T08:57:08.128168+00:00. Goal active and incomplete. Original paper scores unchanged.

## Predictive evidence

The five-seed official ogbl-collab TEST comparison measured GNNM private completion at **67.2909% Hits@50**, native single64 at **66.4426%** and an independent four-model ensemble at **67.6298%**. The exploratory **+0.8483-point** single-model gain is positive in all five seeds. The independent ensemble remains higher.

The frozen private-minus-pooled primary contrast is **+0.2236 points**, with a paired descriptive 95% seed interval **[-0.7775, +1.2247]** and exact sign-flip p=0.6875. It is inconclusive. TEST is consumed for this family. [Complete results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

**No confirmed new methodological advantage or fresh manuscript acceptance exists.**

## Current hypothesis

The new auxiliary trains which candidate neighbours belong together after fixing their observed TRAIN counts. The main comparison links one member responsibility across the two ends of a target edge versus mixing the ends independently. Native inference still uses its existing count-free ranking model.

Classical conditional Bernoulli, mixtures and cardinality inference are prior. New author-code reading confirms that MaskGAE already combines topology/degree supervision with count-free edge scoring, and GRAN already shares mixture components across generated graph blocks. The surviving claim requires useful transfer on native residual supports beyond the separate-side control. Its predictive value is untested. The ordered structured-single control is not guaranteed equivariant to node relabeling; its failure cannot establish an equivariant bank advantage. [Prior assessment](graph_count_conditioned_auxiliary_graph_distinction_prior_assessment_20261004_v1/REPORT.md).

The complete exact core-v3 fabricated CPU qualification passed **22,855 comparison reports**, including both-sided genuine subset patterns, structured-single equivalence and complete gradient cases. Separate core and supervisor reviews passed before execution. The bounded child completed in 37.33 seconds with 539.7 MB peak RSS; physical collection passed. This establishes neither native full-batch feasibility nor predictive benefit. [Actual QA](graph_count_conditioned_pattern_cpu_qualification_root_adoption_20261004_v3/RESULTS_SUMMARY.md).

The complete TRAIN census found both sides variable in **5.8242% of positive queries**, with genuine subset choices on both sides in **1.3599%**. Sampled negatives supply essentially no cross-side pattern signal. The next test measures same-state target and joint/separate auxiliary gradients on one fixed full native TRAIN batch, with zero optimizer updates. [Support and limits](graph_count_conditioned_train_support_census_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## Runs

Amazon training remains **4/15 complete fits**. At 2026-10-04T08:51:24 UTC the fifth fit had reached **1,701/2,700 updates**, without recorded failure or restart. No partial quality or TEST decisions were made.

Six native Pubmed baselines are complete; no GNNM predictive result exists. The earlier shared4 continuation diagnostic still fails its unchanged numerical rule. A new native-only control completed **252 updates** with exact repeat agreement and all ten comparator predicates passing. Its independent compact audit passed 20 checks. Different warm-state and validation history prevent a causal conclusion about shared4. No scientific state or fit is admitted by this control. [Native result](pubmed_native_only_continuation_control_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## History and boundaries

Literature memory index_v45 retains 198 scoped conclusions across 147 paper identities and two software identities, not full-paper read totals. New MaskGAE/GRAN author-code scopes are integrated; they add no new paper identities or full-paper reads. Inactive literature assets were offloaded with hash verification; conclusions and history remain.

Latest verified GitHub head before this update: c4366a4301392a04ad8281d87362dd8620922327. New evidence awaits publication. Science uses the authorized anogena-2 and 18.77 repositories only. The seven-GPU route is forwarding only. No sudo, PDF compilation, GENLINK, server configuration or unrelated-job changes.
