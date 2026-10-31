# Prospective paired comparison and replay requirements

All items are future qualification requirements, not executed tests or an experiment freeze.

Verify the sealed dependency fingerprints in `contracts.py`/the manifest before future imports. The ordinary package imports do not enforce those hashes automatically. Package-root/backend setup remains an admission gate.

## Attribution pairs

1. Compare the earlier private-factored recurrent decoder with the new **factored-head** control using the same source encoder recipe, graph information, input/GRU/head convention, hidden and embedding dimensions, M, fixed pi, ordering policy, target law, mixture objective, graph splits, admitted draw counts and outcome reporting. The controlled difference is private transition/initial-state parameters versus fully shared transition/h0. Width, parameter and cost matching are separate prospective designs; none are established by equal API names.
2. Include **ordinary private heads** to avoid attributing an output-factor parameterization effect to private recurrent transitions. Its raw parameter count differs. In the binary domain, logit differences and parameter redundancy must be considered when interpreting head expressiveness.
3. Include the same full-matrix mean-branch-KL objective control and disclose its natural cheaper one-head-per-replay cost. Keep the complete mixture density in the mixture reward; do not replace it with a generating or leave-one-out approximation.
4. Retain single/untied and appropriate structured/variational controls. Exact density reuse is an implementation property; it does not demonstrate better target-law fidelity, feasible conditional inference or a new method.

Each independently trained arm has its own charged encoder forward. Sharing one actual H across arm comparisons is only legitimate if the scientific protocol explicitly ties those encoder parameters; identical initialization alone does not imply identical learned H.

## Paired initialization and common-law boundary

No initializer is supplied or invoked here. A future paired-law check can set every earlier private transition factor to the neutral value, copy the new shared W_x/W_h into the old shared matrices, repeat common b_x/b_h/h0 across old branches, and match the factored output matrix/factors/biases. With identical H, graph/order and supplied y, all branch state sequences and complete densities should then agree. This is a prospective boundary check, not an initialization freeze or fitted result.

For the ordinary-head boundary, derive each private dense output map from the effective factored map `diag(s_o_m) W_o diag(r_o_m)` with the same bias. Verify numerical equality of binary probabilities and full densities; do not infer that all arbitrary ordinary-head tensors share that parameter representation without checking the relevant law. Tensor storage must be independently allocated where private parameters are intended, and shared exactly where the architecture requires it. Stride/aliasing, dtype, initialization scale and optimizer treatment require audit.

## Exact replay and gradient gates

- Against a deliberately unshared reference replay of this **same** shared-transition model, verify all-member full densities on every feasible assignment of tiny complete graphs. Replaying all branches independently is a qualification reference, not the efficient production path.
- Verify one actual recurrence/mask trajectory per supplied assignment and all M heads per step; compare intermediate prefix states. No head/index/logit/probability or pi may enter the transition. Same-prefix reuse must stop being claimed if a future edit makes h0, normalization, inputs or transition member-dependent.
- Verify generating-score gradients through shared recurrence/encoder and generating private head against the naive reference, with other heads forward only. Compare exact enumerated expected score gradients and objective gradients under nonuniform fixed pi. The full reward remains detached; its omitted explicit derivative has zero expectation by normalized common support.
- Test force-zero support and zero direct score derivative while preserving later state effects. Check genuine normalization, complete-density/sampling agreement, ordering/remapping and numerical saturation failures. Do not silently retry, filter or clip failures into a different law.
- Audit actual current-parameter sampling, unchanged H/graph/order/parameters during score construction, declared iid or fixed stratification and independent baseline provenance. Existing generated-record IDs and certifications do not establish these facts.
- Verify symbolic counters against actual routing. A retained score graph is not a backward completion. Cost comparisons must include generation, all density/head work, source encoding, baselines/RNG/reference/task work and adverse/failed runs, with measured peak memory and wall time under unchanged caps.

Python/package/backend compatibility and all numerical checks remain unperformed. Full prospective task/outcome/config/control/resource freeze and representative resource qualification must precede fits; no source-only seal admits training.
