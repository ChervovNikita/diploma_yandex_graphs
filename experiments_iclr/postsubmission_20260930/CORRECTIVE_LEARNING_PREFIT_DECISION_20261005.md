# Decision before corrective-learning acquisition

5 October 2026. This is a prospective design decision. No candidate fit or predictive outcome has been obtained. Sealed protocol V1 remains unchanged.

## Error analysis and the intervention

The completed Amazon validation analysis separates weaker individual predictions from reduced benefit of averaging. Shared members have error correlation 0.9292, compared with 0.7304 for independent members. Every member is wrong on 93.60% of shared pooled errors, compared with 74.86% for independent members. These are descriptive measurements on three already selected validation panels, not a causal explanation or an untouched test result.

This rules out choosing one member's predicted class as a complete repair for that population. It does not rule out probability combinations, changes to training, or recovering a correct class that no member currently ranks first. The completed fixed aggregation comparison failed against equally processed single and independent references. Its unsuccessful result remains part of the evidence.

The next candidate measures which member learns a true-class versus competitor distinction after a small private update. It allocates extra supervision using that response, while retaining every member's own classification loss. The served prediction remains the arithmetic probability pool. Member competence and pooled quality are measured together. Neither additional disagreement nor a smaller training loss counts as predictive success.

## Adopt a prospective W/S/R split

The source-only independent design review recommends separating acquisition labels from the labels that first drive the corrective response. Root adopts that recommendation before data acquisition or outcome access.

Keep V1's held A IDs and its B complement. Sort B once by the label-independent key `sha256(UTF8(amazon-response-G0|split=0|seed=17|WSR|<node_id>))`, breaking ties by node ID. Set W to the first floor(|B|/2), S to the next floor(|B|/4), and R to the remainder. Expected counts are W=4898, S=2449, R=2450 and A=2449; these are protocol-derived expectations, not observed counts.

Use W alone in the fixed 200 local plus 200 global acquisition updates. S and R first supply supervision during the subsequent correction episodes. They are initially label-unseen, not permanently held out: later episodes repeatedly train on both. All public graph nodes and features remain available throughout. A stays excluded from acquisition, fitting, selection and response estimates until every fixed endpoint is immutable.

Retain G0, H16, the six declared arms, full native context and the existing utility/member-competence thresholds. Preserve a missing-class or native-derivative failure; do not repair it with a new seed, changed graph, smaller model or post-outcome coefficients. V2 source and its exact budget must be frozen and reviewed before execution.

## Interpret response scale explicitly

At the common acquisition endpoint, record per-pair centered response RMS, its scale relative to epsilon=0.001, assignment movement, entropy, positivity and marginal residuals. A response RMS below epsilon means epsilon contributes more than half of the squared normalization scale. This is an analytic description, not an empirically chosen admission threshold. Uniform assignments alone do not prove that their derivative is zero.

Reuse the initial live computation if a numerically qualified tensor-only auxiliary path permits it. Otherwise freeze an explicit additional read-only response call before execution and charge its 16 full-member forwards and eight private gradient constructions. Discard its returned states. Do not call post-core recomputation a warm-state diagnostic or hide an additional call in the declared budget.

## What is still required

The native/sparse source has passed independent static review. Full native output, gradient, higher-order, recompute/commit, state-restoration and resource checks remain pending. A representative predictive comparison has not started. A useful first result would still require subsequent paired comparisons with competent single models, independent ensembles and a same-capacity multibranch model trained directly on ordinary own/pool losses. Close prior methods remain acknowledged; novelty and acceptance are unestablished.

References: `finite_corrective_learning_prefit_design_review_20261005_v1/REPORT.md`, `ERROR_ANALYSIS_TO_METHODS_20261005.md`, `amazon_learnability_responsibility_train_response_protocol_20261005_v1/PROTOCOL.md`, and `learnability_responsibility_native_amazon_sparse_port_independent_review_20261005_v1/REPORT.md`.
