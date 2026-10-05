# Independent narrow review of late-pooling Brier geometry

5 October 2026. Analytical review of the root assessment and its necessary operator definitions only. No paper review, empirical verdict, experiments, primary retrieval, remote contact or payload access. The root note was not edited.

## Identities that are correct

Let the conditioning information be a fixed sigma-field F. Require P to be F-measurable, each row to lie in the class-probability simplex, Y to be one-hot, and q=E[Y|F]. Then the displayed expansion is exactly

`R=PPᵀ−(Pq)1ᵀ−1(Pq)ᵀ+11ᵀ`.

For affine weights with `1ᵀw=1`, its quadratic form is `||Pᵀw−q||²+1−||q||²`. Nonnegativity supplies the convex-hull interpretation. To interpret this as the conditional risk of the served rule, w must also be fixed/measurable given the same F; a weight chosen using the unobserved target Y cannot be taken outside the conditional expectation. No iid-label or independent-member assumption is needed for the algebra.

Exact unregularized minimization therefore gives the Euclidean projection of q into the feasible prediction set. With `w_m≥ε/M`, `0≤ε≤1`, that set is precisely `ε p_bar+(1−ε)conv{p_m}`. Positive ridge adds `λ||w||²`; it favors less concentrated coefficients but is no longer ordinary projection in probability space. The unregularized projected probability is unique, although its representing weights need not be.

The norm inequality is valid with every matrix norm interpreted as the spectral/operator norm. For each anchor, `A=P_l−1Y_lᵀ`, `B=P_v−1Y_lᵀ`, and

`AAᵀ−BBᵀ=(A−B)Aᵀ+B(A−B)ᵀ`.

Submultiplicativity gives `(||A||₂+||B||₂)||P_l−P_v||₂`. Each row has Euclidean norm at most sqrt(2), hence `||A||₂,||B||₂≤||·||_F≤sqrt(2M)`. Nonnegative normalized averaging proves the stated bound. Explicit norm subscripts would remove ambiguity; using a Frobenius norm on the right also gives a valid looser bound.

## Material limits and concrete clarifications

**1. Probability geometry is not raw-logit geometry.** The projection statement applies to `Pᵀw` before a graph corrector. It does not characterize softmax of weighted raw logits, class-specific rescaling, an unrestricted score decoder, or the post-C&S prediction. For example, two member probabilities `(0.6,0.3,0.1)` and `(0.1,0.3,0.6)` have second coordinate 0.3 everywhere in their hull. Averaging their log-probability logits and applying softmax yields the normalized vector `(sqrt(0.06),0.3,sqrt(0.06))`, whose second coordinate exceeds 0.3. Thus leaving the probability hull does not require hidden features. Keep native raw-logit averaging and probability averaging separately named; failure of a constrained probability pool alone does not establish that the score bank lacks useful information.

**2. The counterfactual projection requires a specified regularization recipe.** For the bare matrix with P_v replacing every anchor P_l, the objective is exactly the empirical label-distribution risk for `q_hat=sum_l a_lY_l`. The original operator additionally uses `(1−η)R_local+ηR_0+λI`. If both arms share the original anchor-based R_0 and λI, their matrix difference is `(1−η)` times the local difference, but the counterfactual arm is not merely projection toward q_hat. If its global term is also counterfactualized using P_v, the label target becomes `(1−η)q_hat_local+ηq_hat_global`; ridge still makes this regularized coefficient fitting. State which reference is intended. For zero local anchor mass, specify the global fallback rather than implying that normalized local a_l exist.

**3. The bound does not compare either estimator with the target conditional risk.** Predictor variation can be informative local competence or estimator drift; the sign is not identified by the bound. Even zero drift does not remove label-estimation error. Take constant member predictions P_l=P_v=I₂, independent anchor/target labels with common conditional class probabilities `(1/2,1/2)`, and one observed anchor label `(1,0)`. Transported and counterfactual matrices coincide, so their norm difference is zero. The unregularized estimator selects `(1,0)`, with target conditional Brier risk 1; the true optimum `(1/2,1/2)` has risk 1/2. The labels have the same generating distribution, yet finite-anchor noise remains. Target-risk claims require additional control of `q_hat−q` or of transported-moment bias/variance, beyond predictor drift. A small matrix discrepancy also does not by itself certify stable argmin weights.

**4. Hidden information must be defined jointly.** Replace “information absent from every retained hidden state” with “information absent from the joint retained hidden bank and supplied context.” Two independent fair bits H₁,H₂ can each have zero marginal information about `Y=H₁ XOR H₂`, while their joint bank determines Y. A nonlinear readout can recover such joint-only information. Information absent from the entire available input cannot be recovered by a deterministic decoder; absence from each feature individually is insufficient.

The root note already limits its claims to analytical identities and an unconfirmed utility hypothesis. The clarifications above preserve that scope. They establish no predictive gain, estimator consistency, novel operator, protocol admission or manuscript conclusion.

Only this fresh review subtree was written. Bindings identify the exact root note and the saved operator-definition document; no model, score, dataset, checkpoint, source-code or primary-paper payload was opened or hashed.
