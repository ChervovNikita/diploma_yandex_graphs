# When common structure plus private learning can help

This is a plausibility analysis, not a new theorem or a guarantee for the implemented GNNM. A capable single predictor with the same structural bias can match the construction below. Beating that single still requires experiments.

## The error condition that any ensemble must satisfy

For squared predictive error, let `e_m(x)` be the error of member m relative to the true conditional mean. For the uniform average, the risk is exactly

\[
R_{\rm pool}=M^{-2}\sum_{m,k}\mathbb E[e_m e_k].
\]

With exchangeable members of bias b, centered error variance v and centered pairwise covariance c, this becomes `b² + v/M + (M−1)c/M`. Sharing wins only if its reduction of bias/member variance outweighs any increase of correlated error. Parameter tying alone changes none of these terms in a known favourable direction. Models trained on the same finite graph share sampling error even when they use independent weights and optimizer seeds.

This identity is for squared error. It is not an identity for accuracy, MRR, Hits@K or NLL. Spread of embeddings or parameters is not one of its terms.

## A finite-sample example with exact assumptions

Consider K predeclared data regimes with conditional means `μ + δ_j`, where the average of the fixed offsets δ is zero. Regimes can illustrate different relation or neighbourhood types; the example does not assert that a real graph provides independent regimes. Each regime supplies an unbiased TRAIN estimate `z_j = μ + δ_j + ε_j`. Assume finite second moments, `Var(ε_j)=v=σ²/n`, and equal cross-regime covariance `κv`, with `0≤κ<1`. These are explicit assumptions about the sampling errors, not consequences of endpoint exclusion. No Gaussian assumption is needed.

A fully shared constant predictor uses `z̄`. A group-capable unregularized learner uses `z_j`. Independently initializing several copies of the same full least-squares learner on these same data does not remove the error ε_j: their served average still equals z_j. This is a stylized ordinary ensemble, not a model of every neural ensemble.

Starting from the common predictor, one private gradient step on the regime squared loss gives the correction `a_j=λ(z_j−z̄)`, for a fixed step coefficient `0≤λ≤1`. The predictor is

\[
\hat f_\lambda(j)=\bar z+\lambda(z_j-\bar z).
\]

It can be represented by a uniform private-head ensemble: member m predicts `z̄ + K a_m 1{j=m}` and their mean equals the expression above. The known group indicator is a toy input feature. It is not a new learned router, and these members are not guaranteed individually competent. The same expression is equally representable by one hierarchical linear model.

Let `A=K⁻¹Σ_j δ_j²` be between-regime variation, `v_c=v[1+(K−1)κ]/K` the common estimation variance, and `B=v(1−κ)(1−1/K)` the residual estimation variance. Averaging squared error over regimes yields

\[
R(\lambda)=(1-\lambda)^2A+v_c+\lambda^2B.
\]

**Proof.** The deterministic bias is `−(1−λ)δ_j`. Noise is `ε̄+λ(ε_j−ε̄)`. Under the stated equal-covariance model the two noise terms are uncorrelated, with variances v_c and B. Squaring, taking expectation and averaging regimes gives the expression. This is standard bias–variance/shrinkage algebra.

**Conditional proposition.** If `A>0` and `B>0`, the fixed oracle choice `λ*=A/(A+B)` gives

\[
R(\lambda^*)=v_c+\frac{AB}{A+B}<\min(v_c+A,\ v_c+B).
\]

Thus it improves on both the fully shared constant predictor (`λ=0`) and the unregularized group learner/ensemble (`λ=1`) under these assumptions. The inequality follows because `AB/(A+B)` is smaller than both A and B. The oracle values are unknown in a real task. Choosing λ from heldout outcomes after seeing results would not be this proposition’s experimental validation.

## What this does and does not imply for GNNM

The useful regime has real private structure (`A>0`), compatible common structure and appreciable residual sampling noise (`B>0`). Partial private learning trades estimation error against the bias from forcing all regimes to agree. When private differences are very large, the useful coefficient approaches one and the gain from sharing becomes small. When graph dependence approaches perfect correlation (`κ→1`), B approaches zero and this mechanism offers no strict improvement. If A is zero, private corrections are unnecessary. A wrong shared representation introduces misspecification not covered by the example.

The sampling units are regimes and their observations, not optimizer seeds. Four members on one graph do not create four independent datasets. The toy’s covariance and linear update assumptions have not been measured for the current pilot. Its predictor classes are also restricted: a competent single with learned regime-dependent shrinkage can reproduce the same solution, and a regularized independent ensemble could erase the advantage. The example therefore motivates that comparator instead of proving ensemble necessity.

For the current private-learning rule, a small hypothetical SGD inner step has the familiar expansion `F_out(φ−α∇L_in)=F_out(φ)−α〈∇F_out,∇L_in〉+O(α²)` under smoothness. This explains training for transfer/gradient compatibility and is already part of the retained MLDG/ANIL ancestry. The actual pilot uses stateful native Adam; its displacement and derivative are not this SGD formula. No native-Adam convergence or ranking guarantee follows.

## The fixed outer objective is not repulsion

For one outer query with common label y, let `ℓ_y(z)=log(1+exp(z))−yz` be BCE in the raw logit and let z̄ be the member mean. Convexity gives `J=mean_m ℓ_y(z_m)−ℓ_y(z̄)≥0`. The fixed half aggregate/half member loss is exactly `ℓ_y(z̄)+J/2`. Thus the competence term adds a Jensen-gap penalty to pure aggregate loss; it can discourage logit dispersion and does not enforce error diversity. This is an objective identity, not a claim of actual member collapse or a quality failure. The live private-update derivative, rather than a repulsion term, is the specific training intervention under test.

## Decisive empirical unknowns

1. Does the real common representation reduce estimation error without suppressing useful graph-specific evidence? Existing descriptive Citeseer gains do not settle this.
2. Does the proposed private learning help the served ranking, beyond ordinary same-budget training and beyond generic meta-regularization?
3. Is endpoint eligibility necessary after matching target masking, exposure and support, rather than a generic random-separation effect?
4. Does a capable single with the same training rule match the gain? Does untied four with the same rule match or exceed it?
5. Does any development lead persist on new graphs/splits with a locked method and competent contemporary baselines?

The frozen pilot already addresses several of these questions. This note adds no fit, coefficient, tuning rule, trace requirement or promotion criterion. It does not turn a failed pilot into a theorem-based success.
