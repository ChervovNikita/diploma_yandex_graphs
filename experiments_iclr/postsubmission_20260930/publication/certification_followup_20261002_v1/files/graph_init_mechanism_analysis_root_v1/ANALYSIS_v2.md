# What centered route initialization can change

Research analysis, revision 2, 2 October 2026. This is a conditional derivation, not an empirical result or a claim of a new theorem. It does not change the frozen experiment or manuscript. Prior ingredients are recorded in the graph tangent literature review: BatchEnsemble/TabM private factors, graph residual propagation, gradient diversity, and warm ensemble copying all have published ancestry.

## Mechanism in plain language

The proposed initializer starts from one trained predictor. It applies the same small descent update to every route, then adds four different changes derived from graph-filtered training errors. The extra changes sum to zero and are orthogonal to the training-loss gradient. This construction makes all routes improve at the same first-order rate on training loss. It does **not** give the pooled prediction an additional first-order improvement over the common update.

Its possible benefit therefore comes from nonlinear effects and later training. The decisive comparison is graph-informed initialization against the common-descent control, random directions, shuffled topology, and unchanged warm copies. Their safeguarded line searches can accept different step lengths; these are comparisons of complete initialization operations. Increased disagreement at initialization is insufficient evidence. Independent dropout already lets unchanged warm copies diverge during continuation.

## Local descent statement

Fix the shared model, dropout state and all parameters outside the active private slice. Let its training cross-entropy be L(theta), with gradient g at the warm state theta0. In exact real arithmetic, the installed direction of route m is d_m = -g - t_m, where mean_m t_m = 0 and g^T t_m = 0. The FP32 implementation instead admits these identities within its fixed algebra tolerance 2e-5; an exact theorem must not be described as an exact machine certificate. With e_m = g^T t_m, the actual directional derivative is -||g||^2 - e_m.

If L is differentiable at theta0, every route has directional derivative g^T d_m = -||g||^2. If L has a locally beta-Lipschitz gradient, then

L(theta0 + alpha d_m) <= L(theta0) - alpha ||g||^2 + (beta alpha^2/2)||d_m||^2.

For nonzero g and a finite set of directions, sufficiently small positive alpha therefore decreases every route's training loss. This is a local statement. The executable bounded line search can still fail to find that step within its fixed attempts; its common/unchanged fallback must remain in the results. At g = 0, no strict descent conclusion follows.

## Why mean logits cancel the extra first-order effect

Let z_i(theta) be the class-logit vector of node i. If it is differentiable at theta0, mean_m z_i(theta0 + alpha d_m) equals z_i(theta0) - alpha J_i g + o(alpha), because the centered t_m cancel. A hypothetical common-only update with that same alpha has that same first-order term. The executable common_only control chooses alpha separately, from its own direction norm and bounded backtracking. It therefore need not have the same first-order displacement as the graph arm actually tested.

A stronger bound is available if z_i has a Hessian bounded in operator norm by B_i throughout the relevant neighborhood. Here the vector-valued Hessian bound means ||D^2 z_i(theta)[u,v]|| <= B_i||u||||v|| for all u,v in Euclidean norms. With theta_c = theta0 - alpha g, Taylor expansion around theta_c and mean_m t_m = 0 gives

||mean_m z_i(theta_c - alpha t_m) - z_i(theta_c)|| <= (B_i alpha^2/2) mean_m ||t_m||^2.

Thus the additional pooled-logit displacement relative to the same-alpha common step is at most second order under those assumptions and exact centering. With a nonzero mean tangent, add alpha B_J ||mean_m t_m|| to the bound, where B_J bounds the logit Jacobian locally. The recorded tolerance and accepted alpha must accompany a numerical interpretation. The actual graph model can cross activation boundaries; the smooth-neighborhood condition must be stated, not presumed from numerical automatic differentiation. First-order function checks detect null/private directions but do not establish this smoothness assumption or heldout utility.

## Individual loss and pooled loss serve different purposes

Cross-entropy is convex as a function of class logits, so at each labeled node

CE(mean_m z_m, y) <= mean_m CE(z_m, y).

This inequality compares one ensemble with its own members. It does not prove superiority to a single predictor, common-only initialization or another ensemble. It applies to the frozen mean-raw-logit primary pool. Mean-probability pooling is a different operation and cannot be exchanged in this argument.

For small centered logit perturbations delta_m with mean zero, the leading Jensen gap is one half of mean_m delta_m^T [diag(p)-pp^T] delta_m, where p is the softmax at their common center. The matrix is positive semidefinite. Consequently, optimizing mean member cross-entropy locally charges a nonnegative cost for predictive spread at a fixed pooled center. More spread alone is not a favorable optimization argument. An embedding repulsion objective could increase spread without helping this task, and representation rotations can produce different embeddings with identical decisions.

## Reading the already frozen diagnostics

The initializer already records its actual accepted branch, alpha, gradient/band-sum checks, tangent norm, centered-logit pair distances, finite loss tests and fallback attempts. Read these with the later member quality, pooled NLL/accuracy and decision-change records. A common fallback is a valid outcome of the graph arm, not an omitted cell.

Useful evidence would show that the graph-informed safeguarded operation changes later decisions beneficially beyond norm-matched random and topology-permuted controls, without member deterioration accounting for the apparent disagreement. Norm matching is over the full tangent Frobenius norm. It does not equate maximum per-route direction norms, accepted alpha or fallback branches across arms. Any comparison therefore concerns the complete operation and cannot isolate graph direction independently of the resulting step length. If those contrasts fail, the graph initializer has no demonstrated practical contribution. Warm-copy success alone establishes the value of warming/continuation, not of the graph operation.

No new inference gate or contrastive loss is introduced by this analysis. No numerical dataset/model/checkpoint was read or executed. No original score was recalculated. A future paper can use a short local-descent lemma if the method works; the first-order cancellation must accompany it so the theory is not presented as an ensemble-performance guarantee.
