# Current GNNM research status

Updated: 2026-10-04T14:12:37+00:00. Goal active and incomplete. Original paper scores unchanged.

## Predictive evidence

The strongest verified new result remains five-seed official ogbl-collab TEST: GNNM private completion 67.2909% Hits@50, native single64 66.4426%, independent ensemble4 67.6298%. The exploratory single-model gain is +0.8483 percentage points, positive in all five seeds. The independent ensemble remains higher by 0.3389 points.

The frozen private-minus-pooled contrast is +0.2236 points, paired descriptive 95% seed interval [-0.7775,+1.2247], exact sign-flip p=.6875. It is inconclusive. Collab TEST is consumed; successor development must disclose this history. [Full results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

**No confirmed methodological advantage or fresh manuscript acceptance exists.**

## Current methodological test

The TRAIN-only pattern objective asks whether one member can explain residual-neighbour subsets at both endpoints of an edge. The served link ranker stays count-free. A fixed comparison of target-only, shared-responsibility and independently mixed endpoints uses three paired seeds and nine fresh full 100-epoch fits. Fresh independent technical source review passed. The fixed nine-fit queue launched at 2026-10-04T12:28:04UTC on authorized 18.77 GPU1. Observation 6 at 14:12:37 UTC shows target-only seed 0 complete, 100 epochs, exit 0, 2150.55 seconds; joint seed 0 is live at epoch 35/100. No queue failure or private quality reading occurred. No predictive result is available yet. Planned serial runtime is24–30GPU-hours; complete auxiliary-fit runtime is not yet measured.

The joint-versus-separate contrast measures endpoint responsibility overlap. Identical members give a zero contrast, so it does not guarantee diversity or prevent collapse. [Algebra and limits](pattern_responsibility_overlap_analysis_20261004_v1/ANALYSIS.md). Conditional Bernoulli laws, mixtures/cardinality inference, MaskGAE, GRAN and neighbourhood reconstruction are prior. [New scoped comparisons](conditional_neighbourhood_generation_prior_check_20261004_v1/CONCLUSIONS.md).

The actual native direct-versus-bucket comparison passed all 96 numerical reports plus 16 both-unused records under the original tolerances. Loss forward took 8.7041s versus .8715s; joint reverse took 29.0650s versus 4.5597s. This one fixed-order diagnostic establishes implementation agreement and practical feasibility, with zero updates or heldout reads. It does not establish end-to-end speed, fit memory, novelty or prediction gains. [Native result](exact_cb_support_bucket_native_equivalence_resource_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

The earlier native gradient signal is small: joint norm 2.785% of target and joint-minus-separate 0.0561% of target, with both-genuine supports on 1.3599% of positive queries. Preserve that concern; no numerical non-equivalence or predictive transfer is claimed.

## Active comparisons and failures

Amazon's original queue observation 32 at 2026-10-04T14:12:37UTC had five of fifteen fits complete. The sixth, `split1_gnnm_boundary_4_seed29`, had 2081/2700 updates. No failure, restart or partial quality/TEST selection was observed.

PENCIL's feature-enabled author architecture completed one full training epoch and all 160,084 VALID queries in the resource check. Its 793 TRAIN batches made 100 updates; TRAIN took 337.68 s and VALID 39.76 s. CUDA allocated/reserved peaks were 32.80/50.34 GB. The pin_memory=False harness repair, one-rank launch and runtime adaptations remain explicit; the earlier failure is preserved. No quality scores or donor states were produced by that resource check. [Feasibility and limits](pencil_collab_resource_qualifier_root_adoption_20261004_v3/RESULTS_SUMMARY.md).

The fresh non-author technical review of the full scientific source passed with no blockers, and three fresh fits launched at 13:45:31 UTC on authorized 18.77 GPU0. **The scientific v1 attempt failed**: the worker's CUDA reservation crossed the declared 75 GiB cap during epoch 2, after 106 optimizer updates and one full VALID pass. No OOM or scoring failure is identified by the available diagnostics. The worker's watchdog exited with code 2; supervisor shutdown also encountered live-session and missing-RSS exit races. The original supervisor ultimately closed/reaped its owned session, and the queue stopped without attempting seeds 1/2. Selected quality values have not been read or adopted. All failed source and metadata are preserved.

A surgical v2 source is being prepared with the same scientific recipe, fresh scratch initialization, a disclosed expandable allocator policy, an 80 GiB reserved-memory ceiling (70 GiB allocated cap unchanged), and exit-aware owned-process RSS sampling. It remains unreviewed and unlaunched. No checkpoint continuation or donor is allowed. The earlier 6.5–9-hour forecast does not establish feasibility of the full family. [Preserved source review](pencil_collab_paired_predictive_independent_source_review_20261004_v1/REVIEW.md).

DDI's official TRAIN acquisition and exact native43-mask geometry census are complete:1,067,911 unique undirected edges,4,267nodes and1,056,768queries per population. Both endpoints have genuine subset choices on90.2831% of positives and19.9379% of sampled negatives. Collab's descriptive fractions were1.3599%/0%, under a different graph/mask recipe. This motivates an independent DDI test, not a prediction claim. The census usedGPU0, correcting the earlier CPU-only planning wording; it constructed no model or heldout scores. The pinned HL-GNN fresh-seed author recipe and strict TRAIN+VALID artifact source are prepared. The CPU artifact and shared-propagation/private-hop output/gradient checks are complete; a separate agent is preparing the fixed DDI auxiliary integration. Training/runtime budgets and predictive utility remain unqualified. [Geometry and limits](ddi_native_train_support_census_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

DDI's strict TRAIN+VALID artifact completed on CPU in 4.91 seconds with 654 MB peak RSS and physical exit 0. It preserves all 2,135,822 directed graph entries, 1,067,911 TRAIN edges, 133,489 VALID positives and 101,882 fixed global negatives. Graph/TRAIN identity, candidate exclusion, native ordering and serialization checks passed. Weights are absent, so the actual HL-GNN branch is native AUC. TEST remains opaque; the 55 MB artifact stays on the server. No model or quality score was produced.

The shared-propagation/private-hop implementation passed one CPU check: 17 outputs and 92 full input/parameter gradients, maximum absolute error 2.44e-15, exact native M=1 reduction. This establishes the specified fixed-operator implementation only. A full F4 target-only DDI source is sealed; a prospective joint-versus-separate auxiliary integration is being prepared. No measured speed, novelty or predictive gain is inferred. [Numerical result and limits](hlgnn_member_filter_numerical_execution_20261004_v1/REPORT.md).

Six native Pubmed baselines are complete; no GNNM predictive result exists. The 72-update continuation diagnostic failed the original final parity rule, and the ad hoc engineering branch is closed. [Preserved result](pubmed_shared4_update_inclusive_continuation_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## History and boundaries

Literature index_v47 contains 204 scoped conclusions across 153 paper identities and two software identities; these are not full-paper read totals. Failed experiments, original scores, all reviews and decisions remain preserved. Latest verified GitHub head:392eb51b457e3522aa620c2d82c2895477d29118. The feasibility/geometry/monitoring inventory is published and the exact branch ref was verified.

Science uses authorized anogena-2 and 18.77 project repositories only; the seven-GPU route is forwarding only. No sudo, PDF compilation, GENLINK, Desktop writes, server configuration or unrelated-job changes.

The recent DDI scout saves four additional scoped primary method reads and three public author-recipe assessments. Index_v47 additive integration is complete and preserves prior records; these are not full-paper read certifications.

A further scoped primary read of Karrer/Newman's degree-corrected stochastic blockmodel is saved with exact pages and conclusion digest in `degree_corrected_latent_prior_check_20261004_v1`. It is not yet in index_v47. Fixed-count conditioning cancels common logit offsets but does not remove candidate-specific degree bias. DDI degree-only references and responsibility diagnostics are specified before DDI predictive scores; the Collab protocol is unchanged.

Two further scoped primary reads, Newman–Leicht and Amini et al., identify latent neighbourhood preferences, responsibilities and conditional-degree pseudolikelihood as prior. The exact queried-edge/two-subset/filter composition remains unresolved, with no novelty clearance. Saved counterexamples show collapsed optima and improved joint likelihood without improved served edge scores. Predictive confirmation is therefore essential. [Closest prior and counterexamples](shared_endpoint_latent_mixture_delta_check_20261004_v1/CONCLUSIONS.md).
