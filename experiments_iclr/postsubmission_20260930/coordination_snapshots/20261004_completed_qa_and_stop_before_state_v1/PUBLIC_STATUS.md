# Current GNNM research status

Updated: 2026-10-04T10:26:49.929057+00:00. Goal active and incomplete. Original paper scores unchanged.

## Predictive evidence

The strongest verified result remains the five-seed official ogbl-collab TEST comparison: GNNM private completion **67.2909% Hits@50**, native single64 **66.4426%**, independent ensemble4 **67.6298%**. The exploratory single-model gain is **+0.8483 points**, positive in all five seeds; the independent ensemble remains higher.

The frozen private-minus-pooled contrast is **+0.2236 points**, with paired descriptive95% seed interval **[-0.7775,+1.2247]** and exact sign-flip p=.6875. It is inconclusive. TEST is consumed. [Complete results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

**No confirmed new methodological advantage or fresh manuscript acceptance exists.**

## Current hypothesis and actual execution

The auxiliary supervises which candidate neighbours belong together after fixing their observed TRAIN counts. J_K shares one member responsibility across both ends of an edge; J_K_sep mixes the ends independently. Serving retains the existing count-free ranker. Conditional Bernoulli likelihoods, mixtures/cardinality inference, MaskGAE degree supervision and GRAN shared components are prior. Predictive transfer beyond the separate-side control is the unresolved question.

The fixed full native TRAIN-batch gradient diagnostic completed on18.77 GPU1: **childexit0, physical sessionclosed and custody matched**. One retained forward graph supplied three reverse evaluations, with **zero optimizer updates and no VALID/TEST access**. The auxiliary reaches encoder/member parameters. Joint gradient norm is2.785% of target norm; joint-minus-separated norm is0.0561% of target norm and2.014% of joint norm. These are a small derivative opportunity at one initialization, not predictive improvement. Per-slot differences remain within the unchanged tolerance, whose absolute scale exceeds mean-reduced derivatives.

The child took87.30s, with10.69s forward and30.13/31.14s auxiliary reverse passes; peak allocator memory was28.63GB allocated/42.42GB reserved. Dispatching1,505 genuine groups and132,447 slot loops is expensive. The exact-law support-bucket candidate passed independent static review for equivalence-QA eligibility; actual numerical and native resource checks remain pending. [Actual result and limits](graph_count_conditioned_pattern_native_gradient_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

The first startup attempt failed before data/model loading because memory-stat reset preceded CUDA initialization; failure receipts are preserved. V4 moved that reset after authenticated initialization, with separate source review and a fresh execution directory.

Complete fabricated core-v3 CPU QA passed22,855 comparison reports. That establishes law/gradient implementation consistency, not predictive usefulness. The complete TRAIN census found both sides variable on5.8242% and both genuine on1.3599% of positive queries; negatives supply essentially no cross-side pattern signal. [Prospective decision](graph_count_conditioned_pattern_predictive_decision_root_20261004_v1/DECISION.md).

## Predictive queue and baselines

At 2026-10-04T10:21:25.913053+00:00, Amazon training was **5/15 complete fits**; the sixth, `split1_gnnm_boundary_4_seed29`, had **390/2700 updates**. No failures, restarts or partial quality/TEST decisions were recorded.

A completed source assessment confirms native NCNC64 matches the pinned official Collab core width/depth/100epoch recipe. Five rather than ten runs reduces replication, not per-fit capacity. Published numerical reproduction and broad recent-method competitiveness remain unverified. A concrete feature-enabled PENCIL source plan is sealed: direct one-process launch on GPU0, author hidden512/eight-layer/20epoch recipe, and actual152-node/306-column query shapes. Its package availability and complete-epoch resource cost remain unobserved; no baseline score or GNNM novelty follows from this plan. [Assessment](ncnc_collab_baseline_competitiveness_assessment_20261004_v1/REPORT.md).

Six native Pubmed baselines are complete; no GNNM predictive result exists. The earlier shared4 repeat remains failed under its unchanged rule. The native-only252-update repeat passed in a different warm-state context. Fresh review blocked v1's incomplete storage-alias comparison. A different reviewer passed the narrow v2 repair, and the actual zero-update comparison then completed: all four controlled first-batch comparisons passed the original rule for outputs, logits, loss and gradients. No optimizer updates or VALID/TEST access occurred. This does not replace the earlier failed complete-epoch repeat or identify its cause. [Bounded result](pubmed_shared4_zero_update_first_batch_repeat_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## History and boundaries

Literature index_v45 retains198 scoped conclusions across147 paper identities and two software identities; these are not full-paper read totals. The new NCNC assessment reuses existing source/literature scopes and adds no primary reads or identities. Failed experiments, original scores, decisions and reviews remain preserved.

Latest verified GitHub head before this update:9d8efcece0251e1e3114cfef6158a4835f726113. Current changes await explicit publication. Science uses authorized anogena-2 and18.77 project repositories only; the seven-GPU route is forwarding only. No sudo, PDF compilation, GENLINK, server configuration or unrelated-job changes.
