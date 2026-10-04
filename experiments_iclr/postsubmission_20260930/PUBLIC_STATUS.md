# Current GNNM research status

Updated: 2026-10-04T11:16:58.092848+00:00. Goal active and incomplete. Original paper scores unchanged.

## Predictive evidence

The strongest verified new result remains five-seed official ogbl-collab TEST: GNNM private completion 67.2909% Hits@50, native single64 66.4426%, independent ensemble4 67.6298%. The exploratory single-model gain is +0.8483 percentage points, positive in all five seeds. The independent ensemble remains higher by 0.3389 points.

The frozen private-minus-pooled contrast is +0.2236 points, paired descriptive95% seed interval [-0.7775,+1.2247], exact sign-flip p=.6875. It is inconclusive. Collab TEST is consumed; successor development must disclose this history. [Full results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

**No confirmed methodological advantage or fresh manuscript acceptance exists.**

## Active research

The TRAIN-only pattern objective tests whether one latent member can explain residual-neighbour patterns at both endpoints of an edge. Joint and independently mixed endpoint responsibilities are distinct hypotheses. The serving ranker stays count-free. Conditional Bernoulli laws, mixture/cardinality inference, MaskGAE topology/degree supervision and GRAN shared components are prior; generic novelty is not cleared.

The actual fullTRAIN native gradient check connected this objective to shared/member parameters, with zero optimizer updates and no VALID/TEST. Joint gradient norm was2.785% of target; joint-minus-separate was0.0561% of target. Both-genuine supports occur on1.3599% of positive queries, a sparse-transfer concern. The small initial derivatives are not predictive evidence. [Native diagnostic](graph_count_conditioned_pattern_native_gradient_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

The exact bucket implementation now passed all119 declared CPU cases and3,288 comparisons against independent oracles under unchanged tolerances. Its full native GPU equivalence/resource source passed independent review; actual GPU result and speedup are not yet measured. [Bounded CPU result](exact_cb_support_bucket_cpu_qa_root_adoption_20261004_v1/RESULTS_SUMMARY.md). A three-block paired target/joint/separate predictive decision is recorded prospectively; complete paired fits and an independent benchmark remain required.

DDI was selected for a prospective TRAIN-support census from official documentation and pinned native source, before own data acquisition/scoring. No DDI dataset, support census or fit exists yet. Published whole-graph density does not establish actual TRAIN support.

## Predictive queue and baselines

At 2026-10-04T11:08:10.625832+00:00, Amazon's original paired queue had5/15 completed fits; the sixth, `split1_gnnm_boundary_4_seed29`, had730/2700 updates. The monitor recorded no failures, new launches, restarts or partial quality/TEST selection.

PENCIL's stronger feature-enabled link-prediction baseline has a pinned author recipe and complete-epoch/fullVALID resource source. Its16 missing packages are now installed inside the18.77 project overlay; core versions are unchanged. No numerical resource run or baseline score exists. Independent source review raised a concrete torchrun-session ownership concern; the original packet stays immutable while a minimal direct-rank repair is prepared.

Six native Pubmed baselines are complete; no GNNM predictive result exists. The latest two36-update continuations completed physically, but final loss, encoder, predictor, Adam and gradient parity fail the original rule. Initial isolation and selected first-two-update checks pass without establishing a causal repair. Adopted decision: STOP_ENGINEERING_BRANCH_NO_DEMONSTRATED_REPAIR. No further ad hoc repeats, relaxed tolerance or automatic continuation/donor admission. [Result](pubmed_shared4_update_inclusive_continuation_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## History and boundaries

Literature index_v45 retains198 scoped conclusions across147 paper identities and two software identities; these are not full-paper read totals. New software/documentation scopes are not counted as new full papers. Failed experiments, all reviews, original scores and prior decisions remain preserved.

Latest verified GitHub head before this update: c03ae8df191b5a06816ee928ab9b06db51b4d40e. Current changes await explicit publication. Science uses authorized anogena-2 and18.77 project repositories only; the seven-GPU route is forwarding only. No sudo, PDF compilation, GENLINK, Desktop writes, server configuration or unrelated-job changes.
