# Current GNNM research status

Updated: 2026-10-04T10:05:25.364024+00:00. Goal active and incomplete. Original paper scores unchanged.

## Predictive evidence

The strongest verified result remains the five-seed official ogbl-collab TEST comparison: GNNM private completion **67.2909% Hits@50**, native single64 **66.4426%**, independent ensemble4 **67.6298%**. The exploratory single-model gain is **+0.8483 points**, positive in all five seeds; the independent ensemble remains higher.

The frozen private-minus-pooled contrast is **+0.2236 points**, with paired descriptive95% seed interval **[-0.7775,+1.2247]** and exact sign-flip p=.6875. It is inconclusive. TEST is consumed. [Complete results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

**No confirmed new methodological advantage or fresh manuscript acceptance exists.**

## Current hypothesis and actual execution

The auxiliary supervises which candidate neighbours belong together after fixing their observed TRAIN counts. J_K shares one member responsibility across both ends of an edge; J_K_sep mixes the ends independently. Serving retains the existing count-free ranker. Conditional Bernoulli likelihoods, mixtures/cardinality inference, MaskGAE degree supervision and GRAN shared components are prior. Predictive transfer beyond the separate-side control is the unresolved question.

The fixed full native TRAIN-batch gradient diagnostic completed on18.77 GPU1: **childexit0, physical sessionclosed and custody matched**. One retained forward graph supplied three reverse evaluations, with **zero optimizer updates and no VALID/TEST access**. The auxiliary reaches encoder/member parameters. Joint gradient norm is2.785% of target norm; joint-minus-separated norm is0.0561% of target norm and2.014% of joint norm. These are a small derivative opportunity at one initialization, not predictive improvement. Per-slot differences remain within the unchanged tolerance, whose absolute scale exceeds mean-reduced derivatives.

The child took87.30s, with10.69s forward and30.13/31.14s auxiliary reverse passes; peak allocator memory was28.63GB allocated/42.42GB reserved. Dispatching1,505 genuine groups and132,447 slot loops is expensive. An exact-law support-bucket optimization is being inspected before broader training. [Actual result and limits](graph_count_conditioned_pattern_native_gradient_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

The first startup attempt failed before data/model loading because memory-stat reset preceded CUDA initialization; failure receipts are preserved. V4 moved that reset after authenticated initialization, with separate source review and a fresh execution directory.

Complete fabricated core-v3 CPU QA passed22,855 comparison reports. That establishes law/gradient implementation consistency, not predictive usefulness. The complete TRAIN census found both sides variable on5.8242% and both genuine on1.3599% of positive queries; negatives supply essentially no cross-side pattern signal. [Prospective decision](graph_count_conditioned_pattern_predictive_decision_root_20261004_v1/DECISION.md).

## Predictive queue and baselines

At 2026-10-04T09:47:14.436380+00:00, Amazon training was **5/15 complete fits**; the sixth, `split1_gnnm_boundary_4_seed29`, had **146/2700 updates**. No failures, restarts or partial quality/TEST decisions were recorded.

A completed source assessment confirms native NCNC64 matches the pinned official Collab core width/depth/100epoch recipe. Five rather than ten runs reduces replication, not per-fit capacity. Published numerical reproduction and broad recent-method competitiveness remain unverified. Feature-enabled PENCIL is one source-pinned future comparison; its resource cost is unknown. [Assessment](ncnc_collab_baseline_competitiveness_assessment_20261004_v1/REPORT.md).

Six native Pubmed baselines are complete; no GNNM predictive result exists. The earlier shared4 repeat remains failed under its unchanged rule. The native-only252-update repeat passed in a different warm-state context. Fresh review blocked the new zero-update comparison source because it omitted storage alias relationships between distinct tensor views; no runtime alias defect is inferred. V2's narrow repair is sealed and under a different fresh source review; it has not executed.

## History and boundaries

Literature index_v45 retains198 scoped conclusions across147 paper identities and two software identities; these are not full-paper read totals. The new NCNC assessment reuses existing source/literature scopes and adds no primary reads or identities. Failed experiments, original scores, decisions and reviews remain preserved.

Latest verified GitHub head before this update:31f3f581aed94b6a445aaaecd16f713c0e9ccb79. Current changes await explicit publication. Science uses authorized anogena-2 and18.77 project repositories only; the seven-GPU route is forwarding only. No sudo, PDF compilation, GENLINK, server configuration or unrelated-job changes.
