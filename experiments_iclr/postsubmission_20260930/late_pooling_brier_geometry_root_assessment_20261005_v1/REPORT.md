# What graph-informed late pooling can add

5 October 2026. Analytical assessment of the existing aggregation shortlist. No new source retrieval, target-data access, numerical experiment, training or manuscript edit.

## Practical interpretation

Four completed routes provide four probability vectors for a node. A dense late aggregator can choose how much to trust each vector. It can also use private hidden features to make a prediction outside those vectors' convex hull. Both operations have published ancestry. Their scientific value depends on the information they use and the quality they achieve, rather than on naming a gate differently.

The immediate logits-only hypothesis remains graph transport of member-error moments. This assessment identifies its precise limit: neighboring labels and neighboring predictions jointly affect its reliability estimate. It can therefore fail when neighboring predictions differ substantially, even if labels are locally smooth. This is a reason to measure that behavior and use strong controls, not a reason to reject the hypothesis without a representative test.

## Conditional Brier geometry

Fix the visible graph, the fitted bank and the information available at a target node. Let P be the M-by-C matrix whose rows are its member probabilities, and q the unknown conditional class probabilities at that node. For a one-hot label Y, the conditional uncentered error second moment is

`R = P P^T - (P q) 1^T - 1 (P q)^T + 1 1^T`.

This follows by expanding `(P - 1 Y^T)(P - 1 Y^T)^T`, using `E[Y]=q` and `Y^T Y=1`. For any served simplex weights w, measurable under the same conditioning information as P,

`w^T R w = ||P^T w - q||² + 1 - ||q||²`.

Consequently, exact unregularized moment weighting projects the conditional class distribution into the convex hull of the member probabilities. A fixed positive lower bound on weights restricts that hull; ridge adds a preference for less concentrated weights. These statements concern arithmetic mixtures of member probabilities before any Correct-and-Smooth operation. Averaging raw logits, class-specific scaling and unrestricted nonlinear readouts have different output sets; in a multiclass problem, softmax of mean logits can lie outside the member-probability hull. None of these identities is a new ensemble theorem. They do not supply q or guarantee improved accuracy, finite-sample Brier loss or generalization.

This also clarifies why a reliability matrix does not contain independent new labeled information once local member competence and pairwise disagreement are supplied. A competent stacker with the same inputs is a necessary reference. If it matches the proposed estimator, report the benefit of the information or regularization; do not claim an exclusive aggregation capability.

## Transported error moments are a different estimator

The current shortlist transports anchor matrices `G_l = (P_l - 1 Y_l^T)(P_l - 1 Y_l^T)^T` with fixed nonnegative normalized graph weights a_l. These matrices use the probabilities at the anchors, P_l. They do not directly estimate the target conditional risk using fixed P_v.

For comparison, replacing every P_l with P_v in the same average gives a counterfactual matrix whose simplex objective is projection toward `q_hat_v = sum_l a_l Y_l`. Without shrinkage or ridge, that is ordinary label diffusion followed by convex-hull projection. With a density bound, use the restricted hull. A ridge penalty changes the projection objective; global shrinkage retains the label-diffusion interpretation only if its global term is also built using target P_v and the global anchor labels, rather than original anchor P_l. It provides an intelligible cheap reference, not a novel method. Its labels must use exactly the same permitted anchors, with the entire evaluation fold excluded everywhere.

The two matrices differ because the frozen predictor bank changes across the neighborhood. An elementary spectral operator-norm bound makes the condition explicit. All matrix norms in this display and proof are spectral norms:

`||R_transport - R_counterfactual|| <= 2 sqrt(2M) sum_l a_l ||P_l - P_v||`.

Proof: for each anchor define A=P_l-1Y_l^T and B=P_v-1Y_l^T. Then `||AA^T-BB^T|| <= (||A||+||B||)||A-B||`. A probability row minus a one-hot vector has Euclidean norm at most sqrt(2), hence both matrix norms are at most sqrt(2M). Average the bound with nonnegative normalized a_l. This is a deterministic estimator comparison under the stated simplex/normalization conditions, not a generalization guarantee or evidence of smoothness on the actual graph. Even zero predictor drift leaves finite-anchor label estimation error and possibly biased graph context. This bound does not address either error. Global shrinkage and ridge must be handled identically if compared.

Anchor predictor variation can encode real local expertise, or can introduce harmful estimator drift. The representative test must distinguish those possibilities. Label homophily alone is insufficient evidence that member-error moments are smooth.

## Consequence for a small experiment

Preserve native probability averaging. Compare graph-local full moments against global moments, local diagonal competence and a same-information contextual stacker on both the shared and independent banks. A common graph corrector must receive the same permitted labels. A label-diffusion projection is worth retaining only if its small incremental cost helps distinguish label information from anchor-specific competence; it must not expand into a tuning grid.

Use prediction variation or rescued common errors as descriptive diagnostics fixed before outcomes. They cannot rescue a failed served-quality comparison. Three splits of one graph still represent one graph, and existing VALID checkpoint selection makes aggregator-only VALID evaluation retrospective development. Untouched confirmation remains necessary under the unchanged contracts.

A hidden readout is the subsequent candidate if the score bank is insufficient. It may recover information discarded by class heads, but cannot reconstruct information absent from the joint retained hidden bank and supplied context. Joint-only information can exist even when each member state alone is uninformative. The nonlinear score/context readout and competent single-feature readout remain decisive controls. Extraction has a real server cost; it should follow evidence rather than be treated as free.

## Source and attribution

This analysis derives elementary identities from the definitions in the saved [member-error scout](../dense_graph_ensemble_error_correction_scout_20261005_v1/REPORT.md). Its prior-work records cover ensemble error algebra, stacking and Correct-and-Smooth. [Hidden-fusion ancestry](../cross_member_hidden_graph_error_aggregation_prior_scout_20261005_v1/REPORT.md) includes PCL and FFL. The [latest closest-source follow-up](../dense_graph_ensemble_unread_primary_closure_20261005_v1/REPORT.md) shows graph-conditioned class scaling in GETS's pinned code; it is a relevant accuracy comparator, although its native reproduction is unqualified. The unresolved Graph ensemble neural network manuscript supplies no novelty clearance.

Zero new paper-reading credit. No predictive result or methodological novelty claim is established by this assessment.

## Independent mathematical review response

The [sealed independent review](../late_pooling_brier_geometry_independent_math_review_20261005_v1/REPORT.md) found the expansion, unregularized projection and bound sound. Its four scope clarifications are incorporated above: probability versus logit pooling, counterfactual global regularization, estimator versus risk error, and joint-bank information. The exact reviewed version is preserved as REPORT_BEFORE_MATH_REVIEW.md. The review does not establish novelty or predictive performance.
