# Current GNNM research status

Updated: 2026-10-04T12:57:54.633724+00:00. Goal active and incomplete. Original paper scores unchanged.

## Predictive evidence

The strongest verified new result remains five-seed official ogbl-collab TEST: GNNM private completion 67.2909% Hits@50, native single64 66.4426%, independent ensemble4 67.6298%. The exploratory single-model gain is +0.8483 percentage points, positive in all five seeds. The independent ensemble remains higher by 0.3389 points.

The frozen private-minus-pooled contrast is +0.2236 points, paired descriptive 95% seed interval [-0.7775,+1.2247], exact sign-flip p=.6875. It is inconclusive. Collab TEST is consumed; successor development must disclose this history. [Full results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

**No confirmed methodological advantage or fresh manuscript acceptance exists.**

## Current methodological test

The TRAIN-only pattern objective asks whether one member can explain residual-neighbour subsets at both endpoints of an edge. The served link ranker stays count-free. A fixed comparison of target-only, shared-responsibility and independently mixed endpoints uses three paired seeds and nine fresh full 100-epoch fits. Fresh independent technical source review passed. The fixed nine-fit queue launched at 2026-10-04T12:28:04UTC on authorized 18.77 GPU1. At 12:45:39UTC the target-only seed0 fit was in epoch49/100, with no queue failure. No predictive result is available yet. Planned serial runtime is 24–30 GPU-hours, rather than a measured duration.

The joint-versus-separate contrast measures endpoint responsibility overlap. Identical members give a zero contrast, so it does not guarantee diversity or prevent collapse. [Algebra and limits](pattern_responsibility_overlap_analysis_20261004_v1/ANALYSIS.md). Conditional Bernoulli laws, mixtures/cardinality inference, MaskGAE, GRAN and neighbourhood reconstruction are prior. [New scoped comparisons](conditional_neighbourhood_generation_prior_check_20261004_v1/CONCLUSIONS.md).

The actual native direct-versus-bucket comparison passed all 96 numerical reports plus 16 both-unused records under the original tolerances. Loss forward took 8.7041s versus .8715s; joint reverse took 29.0650s versus 4.5597s. This one fixed-order diagnostic establishes implementation agreement and practical feasibility, with zero updates or heldout reads. It does not establish end-to-end speed, fit memory, novelty or prediction gains. [Native result](exact_cb_support_bucket_native_equivalence_resource_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

The earlier native gradient signal is small: joint norm 2.785% of target and joint-minus-separate 0.0561% of target, with both-genuine supports on 1.3599% of positive queries. Preserve that concern; no numerical non-equivalence or predictive transfer is claimed.

## Active comparisons and failures

Amazon's original queue at 2026-10-04T12:45:38.910055+00:00 had five of fifteen fits complete. The sixth, `split1_gnnm_boundary_4_seed29`, had 1444/2700 updates. No failure, restart or partial quality/TEST selection was observed.

PENCIL v3's feature-enabled author architecture completed one full native training epoch and all160,084 VALID queries on authorized18.77GPU0. Its793TRAIN batches made100updates; TRAIN took337.68s and VALID39.76s. CUDA allocated/reserved peaks were32.80/50.34GB. Physical exit and custody passed. The disclosed pin_memory=False harness repair and one-rank/runtime adaptations remain explicit; the v2 failure is preserved. No quality scores, checkpoints, TEST reads or state donors were produced. Full20-epoch scientific baseline source for three fresh seeds is now being prepared. [Feasibility and limits](pencil_collab_resource_qualifier_root_adoption_20261004_v3/RESULTS_SUMMARY.md).

DDI's official TRAIN acquisition and exact native43-mask geometry census are complete:1,067,911 unique undirected edges,4,267nodes and1,056,768queries per population. Both endpoints have genuine subset choices on90.2831% of positives and19.9379% of sampled negatives. Collab's descriptive fractions were1.3599%/0%, under a different graph/mask recipe. This motivates an independent DDI test, not a prediction claim. The census usedGPU0, correcting the earlier CPU-only planning wording; it constructed no model or heldout scores. A separate agent is preparing the pinned HL-GNN full author recipe with fresh-seed initialization and VALID-only selection. [Geometry and limits](ddi_native_train_support_census_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

Six native Pubmed baselines are complete; no GNNM predictive result exists. The 72-update continuation diagnostic failed the original final parity rule, and the ad hoc engineering branch is closed. [Preserved result](pubmed_shared4_update_inclusive_continuation_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## History and boundaries

Literature index_v47 contains 204 scoped conclusions across 153 paper identities and two software identities; these are not full-paper read totals. Failed experiments, original scores, all reviews and decisions remain preserved. Latest verified GitHub head before this update:7258b45798af150089f8aacf7d78dfcf562299d0; current feasibility and monitoring evidence awaits publication.

Science uses authorized anogena-2 and 18.77 project repositories only; the seven-GPU route is forwarding only. No sudo, PDF compilation, GENLINK, Desktop writes, server configuration or unrelated-job changes.

The recent DDI scout saves four additional scoped primary method reads and three public author-recipe assessments. Index_v47 additive integration is complete and preserves prior records; these are not full-paper read certifications.
