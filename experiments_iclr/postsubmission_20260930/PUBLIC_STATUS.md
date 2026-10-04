# Current GNNM research status

Updated: 2026-10-04T12:11:40.870508+00:00. Goal active and incomplete. Original paper scores unchanged.

## Predictive evidence

The strongest verified new result remains five-seed official ogbl-collab TEST: GNNM private completion 67.2909% Hits@50, native single64 66.4426%, independent ensemble4 67.6298%. The exploratory single-model gain is +0.8483 percentage points, positive in all five seeds. The independent ensemble remains higher by 0.3389 points.

The frozen private-minus-pooled contrast is +0.2236 points, paired descriptive 95% seed interval [-0.7775,+1.2247], exact sign-flip p=.6875. It is inconclusive. Collab TEST is consumed; successor development must disclose this history. [Full results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

**No confirmed methodological advantage or fresh manuscript acceptance exists.**

## Current methodological test

The TRAIN-only pattern objective asks whether one member can explain residual-neighbour subsets at both endpoints of an edge. The served link ranker stays count-free. A fixed comparison of target-only, shared-responsibility and independently mixed endpoints uses three paired seeds and nine fresh full 100-epoch fits. Source preparation is complete; fresh technical review and root execution preparation are in progress. No fit has launched in this event. Planned serial runtime is 24–30 GPU-hours, rather than a measured duration.

The joint-versus-separate contrast measures endpoint responsibility overlap. Identical members give a zero contrast, so it does not guarantee diversity or prevent collapse. [Algebra and limits](pattern_responsibility_overlap_analysis_20261004_v1/ANALYSIS.md). Conditional Bernoulli laws, mixtures/cardinality inference, MaskGAE, GRAN and neighbourhood reconstruction are prior. [New scoped comparisons](conditional_neighbourhood_generation_prior_check_20261004_v1/CONCLUSIONS.md).

The actual native direct-versus-bucket comparison passed all 96 numerical reports plus 16 both-unused records under the original tolerances. Loss forward took 8.7041s versus .8715s; joint reverse took 29.0650s versus 4.5597s. This one fixed-order diagnostic establishes implementation agreement and practical feasibility, with zero updates or heldout reads. It does not establish end-to-end speed, fit memory, novelty or prediction gains. [Native result](exact_cb_support_bucket_native_equivalence_resource_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

The earlier native gradient signal is small: joint norm 2.785% of target and joint-minus-separate 0.0561% of target, with both-genuine supports on 1.3599% of positive queries. Preserve that concern; no numerical non-equivalence or predictive transfer is claimed.

## Active comparisons and failures

Amazon's original queue at 2026-10-04T12:00:46.275370+00:00 had five of fifteen fits complete. The sixth, `split1_gnnm_boundary_4_seed29`, had 1115/2700 updates. No failure, restart or partial quality/TEST selection was observed.

PENCIL's pinned feature-enabled author model was actually attempted on authorized 18.77 GPU0. It stopped after seven TRAIN batches, before the first optimizer update or VALID traversal, with a host-memory pinning error. The owned processes exited cleanly, and all failure evidence remains preserved. No baseline quality or full-epoch resource result exists. A disclosed transfer-only repair is the next action; no server configuration or unrelated job was changed.

DDI's official TRAIN acquisition succeeded: 1,067,911 unique undirected edges, 4,267 nodes, no self-loops. Only TRAIN and the node-count member were decoded; opaque archive bytes include heldout payloads. The HTTPS failure is preserved, and acquisition succeeded using the exact official HTTP URL. No support census, fit or DDI score exists yet.

Six native Pubmed baselines are complete; no GNNM predictive result exists. The 72-update continuation diagnostic failed the original final parity rule, and the ad hoc engineering branch is closed. [Preserved result](pubmed_shared4_update_inclusive_continuation_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## History and boundaries

Literature index_v46 contains 200 scoped conclusions across 149 paper identities and two software identities; these are not full-paper read totals. Failed experiments, original scores, all reviews and decisions remain preserved. Latest verified GitHub head before this update: 81241158e31bf2ee087fca25974468cd5e421c13; this event awaits publication.

Science uses authorized anogena-2 and 18.77 project repositories only; the seven-GPU route is forwarding only. No sudo, PDF compilation, GENLINK, Desktop writes, server configuration or unrelated-job changes.
