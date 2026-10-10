# Live-route TRAIN-label retrieval: independent source assessment

10 October 2026. One source/theory assessment of the proposal communicated by
the new-method researcher and root. No method implementation, numerical import,
data/checkpoint/prediction access, outcome read, fit, launch, or public retrieval.
Saved UniMP, label-only, two-hop and MPNP scopes were read before this assessment.

## Conclusion

The proposed operation is a Matching Networks type decoder on live graph
embeddings, trained alongside the native classifier, then combined across shared
routes. It supplies an intelligible member-learning hypothesis: masked label
retrieval gives both query and support embeddings a supervised similarity signal
that ordinary native CE does not explicitly impose. The retrieval primitive and
the ensemble composition alone establish neither a new method nor useful shared
learning. A source-qualified same-information comparison can test the hypothesis.

## Exact equivalence and the graph distinction

Let S be the visible TRAIN support, Q the common disjoint TRAIN query set, and
u_m(i) the unit-normalized, label-input-free native hidden state. At tau=.1,

    K_m(i,a) = exp(u_m(i)^T u_m(a) / tau)
    r_m(c|i,S) = sum_{a in S:y_a=c} K_m(i,a) / sum_{a in S} K_m(i,a).

This is exactly a positive learned-kernel class vote. Equivalently, its class
logits are `g_m,c(i)=logsumexp_{a in S:y_a=c}(u_m(i)^T u_m(a)/tau)` and
`r_m=softmax(g_m)`. Every support exemplar is a prototype with equal anchor prior.
It is generally **not** reducible to cosine similarity with one centroid per
class: the exponential class sum retains the within-class exemplar distribution.
Unequal class support sizes affect the class mass. Four routes learn four such
kernels; averaging their served probabilities is an ordinary mixture.

The exact operational change from the saved local label-only routes is access to
**every permitted anchor label**, regardless of graph distance or component. For
a native encoder with a finite local receptive field this can include anchors
beyond that field; a global native encoder need not have that restriction. The
kernel uses route-specific graph states, and retrieval CE sends live gradients
through both query and support states into their native graph computations.
This is graph-dependent representation learning
with nonlocal exemplar retrieval. Retrieval itself uses no edges and remains
defined with a feature-only encoder. Its advantage therefore need not arise from
a uniquely graph-specific propagation rule or from parameter sharing.

## Information and masking

Whole-Q removal from every route's support before retrieval blocks literal target
label input in the stated design. Since labels never enter H, retrieval outputs
never feed any node/support state, and there is no propagated label cache,
UniMP/two-hop label-return walks do not arise. Query nodes may influence support
H through ordinary feature message passing; this carries factual graph/features,
not an exposed query label. At fixed parameters, fixed Q/S and fixed graph inputs,
changing y_Q must leave H, keys, similarities, values and predictions unchanged;
only the supervised losses may change.

There are three narrower qualifications:

- Native CE still supervises **all TRAIN nodes**, including Q. Their H are
  in-sample representations shaped by their labels through previous optimization.
  This is ordinary transductive fitting with masked retrieval episodes, not
  withheld-label representation learning, pipeline cross-fitting, or Bayesian
  conditioning. Native all-TRAIN CE must also be retained in the controls.
- Class-stratified Q/S construction uses TRAIN labels to choose the episode.
  This is a declared training sampler, but it is not a target-label-independent
  mask. Known per-class quotas/totals can constrain the withheld class counts.
  Literal input exclusion is assessed conditional on the fixed episode; do not
  claim unconditional counterfactual label hiding or inherit the saved uniform
  half-mask theorem. VALID/unrevealed TEST truth cannot choose supports or masks.
- Softmax normalization changes when supports change. The saved inverse-inclusion
  identity for a fixed linear label operator does not apply. Half-support training
  and all-TRAIN serving are different kernel estimates, with no unbiasedness or
  competence guarantee. This familiar support-size transfer needs equal handling
  across controls. A live differentiable H is required; detached/cached H would
  define a different learning operation.

## Closest saved prior and decisive controls

The saved UniMP source already jointly learns masked-TRAIN prediction after
labels are mixed into graph states, with complementary query/support roles and
all-TRAIN serving. MPNP already conditions graph predictions on observed label
contexts; its message-passing encoder, global latent and ELBO differ here. The
saved label-only and two-hop recipes use detached native H and locally propagated
linear label values. Global support retrieval and native retrieval gradients are
the present operational differences. These scopes support attribution, not an
absence or novelty certificate. The new-method agent reports reading Matching
Networks 1606.04080v2 §2 and TPN 1805.10002v3 §3; their papers were not reread here.

The decisive controls are competent native M1 and genuine ordinary I4 fitted
with this **same** retrieval decoder, Q/S draws, native and retrieval CE weights,
tau, normalization, all-TRAIN serving, .5/.5 native/retrieval probability mixture,
optimization opportunity and complete-task selection. Plain M1/I4 establish the
total intervention's utility; retrieval-equipped M1/I4 separate extra label
information and similarity supervision from shared-route utility. A shared bank
with identical forward retrieval but stopped similarity gradients isolates the
added representation-learning signal while retaining native CE. Do not interpret
that ablation as isolating global label access, since it retains that access.

**One refutation:** if a competent same-information single or ordinary I4 explains
the complete-task improvement and the shared candidate has no protected pooled
and member-quality advantage, the claim that this mechanism solves a shared-route
learning deficiency fails. Generic kernel-decoder utility may remain. Hidden
separation, masked TRAIN retrieval success, or improvement over a plain predictor
with fewer inputs cannot rescue that attribution.

## Saved source scopes read

- `unimp_author_code_reference_scope_20261008_v1/REPORT.md`: original arxiv mask,
  label embedding and evolving Q/K/V states; stronger-v2 and reproduction limits.
- `label_only_staged_posterior_full_WikiCS_source_20261008_v1/README.md`: fixed
  detached H, common-Q exclusion, local label-only values and capable joint single.
- `label_only_common_mask_correction_covariance_20261008_root_v1/NOTE.md`: fixed
  linear operator's uniform-mask first moment and its limitations.
- `label_only_protected_backbone_closest_prior_gap_scout_20261008_v1/REPORT.md`:
  Echoless-LP partition-wide exclusion and learned combined encoder.
- `label_two_hop_shared_kernel_prior_design_20261009_v1/REPORT.md`: source masking,
  rebuilt intermediate fields, exact-two-edge support and prototype restriction.
- `new_learning_gap_independent_v1/ASSESSMENT.md` and `PASSAGES.md`,
  2009.13895v1 PDF pp.3–4: MPNP label concatenation/unknown zeros, global latent,
  message-passing decoder and Eq.5 ELBO.

Only these saved textual scopes and the communicated proposal were used. No
empirical claims or published metric values were adopted, and no execution is
authorized by this report.
