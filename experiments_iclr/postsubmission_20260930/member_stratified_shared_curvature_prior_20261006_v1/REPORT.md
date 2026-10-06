# Member-stratified curvature on shared ensemble weights

6 October 2026. Static follow-on assessment, distinct from the closed AdaTask proposal. The loss, four predictions, backbone and private factors are preserved. No model implementation, fits, data/held-label/tensor reads, remote allocation contacts, source qualification or manuscript edits were performed. Elementary scalar formulas were evaluated only to check the derivation.

## Decision

**No defensible new estimator or fast-factor geometry is established; close the generic curvature novelty path.** Aggregating the current joint gradient once and applying one damped SPD inverse avoids the previous memberwise-Adam ascent mechanism. That is the standard descent property of SPD preconditioning, not an ensemble-specific optimization principle.

The proposed member-conditioned \(M^{-1}\sum_m A_m\otimes G_m\) is a legitimate conditional KFAC approximation to the **member-loss component**. It differs from KFAC after pooling the factor statistics. It is also exactly obtained by applying ordinary KFAC to untied member layers and pulling the resulting curvature back through the shared-weight/fast-factor tying map. The private diagonal factors change the activations and cotangents in the usual Jacobian geometry. No additional geometric mechanism is required.

The inspected sharing paper does not literally publish this conditional per-member bank with the current composite loss, so this is not a claim of verbatim algorithm identity to its KFAC-expand/reduce equations. Nevertheless, its exact tied Jacobians and general GGN already contain the required loss and sharing structure. EKFAC and multi-Kronecker Fisher approximations further establish that retaining information beyond one product of pooled factors is prior. A specific numerical choice of conditioning is an attributed approximation variant, without novelty or quality clearance.

The pooled probability likelihood has **real cross-route curvature**. The member-only sum cannot stand in for the full objective curvature. Dropping those terms, confusing the observed Hessian with a PSD Fisher/GGN, or using the outer product of the total supervised gradient would define different estimators. These distinctions are derived below.

## 1. Saved memory and primary boundary

Index_v72 and bounded saved report searches were checked first. No saved KFAC-expand/reduce, EKFAC or KPSVD method scope was located. Saved Bayesian LoRA conclusions already attribute adapter-only Laplace/KFAC curvature, and saved BatchEnsemble conclusions establish the diagonal input/output fast-factor architecture. Those primary passages were not reread.

Three new bounded primary method scopes were assessed, zero full papers and zero author-code reads:

| Source | Inspected scope | Consequence |
|---|---|---|
| Eschenhagen, Immer, Turner, Schneider and Hennig, **Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures**, arXiv:2311.00636v2; primary metadata identifies NeurIPS 2023 | PDF pp. 4–7: §2.3–2.4, §3.1–3.3, Eqs. 4–10; expand/reduce definitions, exact shared Jacobians, dropped cross-position terms, approximation and cost statements. Proposition 1–2 statements read, not Appendix B proofs. Graph-layer example on p. 4 visible. | General weight-sharing GGN and both separate-loss and pooled-output settings are direct prior. Their deep-linear/Gaussian exactness claims do not apply automatically to nonlinear fast-factor graph routes or pooled probability CE. |
| George, Laurent, Bouthillier, Ballas and Vincent, **Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis**, arXiv:1806.03884v2; metadata identifies NeurIPS 2018 | PDF p. 3 regular-layer factorization and §3.1; p. 4 §3.2 and footnote 3. Appendix proofs and later experiments unread. | EKFAC replaces products of factor eigenvalues with directly estimated score second moments in a Kronecker eigenbasis. Its explicit caveat says smaller Frobenius curvature error does not ensure a better inverse direction. |
| Koroko, Anciaux-Sedrakian, Ben Gharbia, Garès, Haddou and Tran, **Efficient Approximations of the Fisher Matrix in Neural Networks using Kronecker Product Singular Value Decomposition**, arXiv:2201.10285v6 | PDF p. 4 factor independence, moving estimates and damping; p. 6 model-label Monte Carlo paragraph, §3.2 rank-two objective/deflation; p. 7 Lanczos, KFAC-corrected and §4 structured sum inversion. Proposition 3.2 statement visible, proof unread. §3.1 derivation and experiments unread. | Sums of Kronecker Fisher approximations and structured solves are established. These SVD/correction estimators are not exactly the proposed member-stratified factor choice. |

Method equations were rendered and visually checked: sharing pp. 6–7, EKFAC p. 4, KPSVD pp. 6–7. Some figures/other page text were incidentally exposed; no empirical paper outcomes were audited or adopted. Mechanical PDF extraction of all pages was for locating bounded scopes and is not full-paper reading. EKFAC's p. 3 extraction contains extraneous embedded figure/poster text; the assessed regular-layer paragraph and §3.1 were isolated in the retained scoped text.

Five bounded OpenAlex metadata queries returned at most five rows each, including two exact-title locators. Searches for natural-gradient/KFAC BatchEnsemble produced no verified direct primary method lead. This is a locator limitation, not evidence of absence. The KPSVD publisher PDF route returned 403; a version-pinned author arXiv PDF was read instead, without certifying equivalence to the published version. No broad novelty/absence or global reading count is claimed.

## 2. Exact supervised gradient and pooled Hessian

For one input \(x\), let \(z_m(W,\phi_m)\) be member logits, \(p_m=\operatorname{softmax}(z_m)\), and

\[
q_c=\frac1M\sum_m p_{m,c},\qquad M=4.
\]

For observed class \(y\), the specified loss is

\[
L_y=-\frac{1}{2M}\sum_m\log p_{m,y}
     -\frac12\log q_y.
\]

Define shared-coordinate scores and pooled responsibilities

\[
h_m^y=\nabla_W\log p_{m,y},\qquad
\rho_m^y=\frac{p_{m,y}}{\sum_n p_{n,y}}
         =\frac{p_{m,y}}{M q_y}.
\]

Then \(\sum_m\rho_m^y=1\), the pooled score is
\(s_q^y=\nabla_W\log q_y=\sum_m\rho_m^y h_m^y\), and the actual current shared gradient is

\[
g_W=\nabla_W L_y
=-\frac{1}{2M}\sum_m h_m^y-\frac12 s_q^y.
\]

This gradient is aggregated **once**, including the pooled cotangent. Preconditioning is applied to this \(g_W\), without replacing it by historical momentum or individually preconditioned route contributions. Private parameters retain their ordinary gradients/updates; this does not assert that their simultaneous Adam steps are descent directions.

Let \(H_m^y=-\nabla_W^2\log p_{m,y}\). Differentiating the responsibilities gives

\[
\nabla_W\rho_m^y=\rho_m^y(h_m^y-s_q^y).
\]

Therefore the **observed pooled Hessian** is exactly

\[
H_{q,y}=-\nabla_W^2\log q_y
=\sum_m\rho_m^y H_m^y
-\left[\sum_m\rho_m^y h_m^y(h_m^y)^\top
        -s_q^y(s_q^y)^\top\right].
\]

The bracket is the responsibility-weighted score covariance. Even if each \(H_m^y\) is PSD, subtraction can make \(H_{q,y}\) indefinite. The full observed loss Hessian is

\[
H_{L,y}=\frac1{2M}\sum_m H_m^y+\frac12 H_{q,y}.
\]

On replicated private/shared variables, the pooled Hessian has mixed-route blocks proportional to \(\rho_m^y\rho_n^y h_m^y(h_n^y)^\top\) for \(m\ne n\), with within-route terms as above. They do not disappear after tying. For nonlinear networks, \(H_m^y\) also contains network second-derivative terms; a layer's conditional linearity alone does not remove nonlinear downstream curvature.

## 3. A well-defined PSD target for the actual composite loss

For each member's categorical distribution define

\[
F_m=\mathbb E_x\sum_c p_{m,c} h_m^c(h_m^c)^\top
=\mathbb E_x J_{W}z_m^\top
[\operatorname{diag}(p_m)-p_mp_m^\top]J_{W}z_m.
\]

For the **pooled predictive distribution**, its true Fisher is

\[
F_q=\mathbb E_x\sum_c q_c s_q^c(s_q^c)^\top
=\mathbb E_x J_W q^\top\operatorname{diag}(1/q)J_W q.
\]

The tangent columns of \(J_Wq\) sum to zero, so the expression is the categorical Fisher on the probability simplex. The objective-aligned termwise model-Fisher/GGN target is

\[
C_L=\frac1{2M}\sum_m F_m+\frac12 F_q\succeq0.
\]

This is the composite GGN when the member natural outputs are \(z_m\) and the pooled natural output is \(\log q\): CE on \(\log q\) equals \(-\log q_y\) because \(\sum_c q_c=1\). Its pooled GGN is precisely \(F_q\). Equivalently, this is the Fisher of a fixed head-index augmented model that chooses the pooled head with probability one half and each member head with probability \(1/(2M)\), then samples a class from the chosen head. This proper augmented interpretation does not assert that the unnormalized product of powered member/pooled likelihoods is itself the current categorical likelihood.

If the probability vector \(q\) is instead declared the network output for GGN construction, the **observed-label probability-output GGN** is

\[
Q_{q,y}=J_Wq^\top\frac{e_y e_y^\top}{q_y^2}J_Wq
=s_q^y(s_q^y)^\top\succeq0,
\]

and \(H_{q,y}=Q_{q,y}-q_y^{-1}\nabla_W^2q_y\). Taking labels from the pooled model turns \(\mathbb E_{y\sim q}Q_{q,y}\) into \(F_q\). These are different observed estimators/parameterizations; the target must be declared.

The outer product of the **total observed supervised gradient**, \(g_Wg_W^\top\), is not \(C_L\). It squares the one-half coefficients and adds member/member and member/pool products different from the composite GGN. Observed-label score second moments, true predictive Fisher, natural-output GGN and observed Hessian must not be interchanged under a generic “Fisher” label. Model-label Monte Carlo uses training inputs and predictive-label draws, not held labels; none was executed here.

### Genuine pooled cross-route terms

Expanding the pooled Fisher yields

\[
F_q=\mathbb E_x\sum_c\sum_{m,n}
q_c\rho_m^c\rho_n^c h_m^c(h_n^c)^\top.
\]

The terms with \(m\ne n\) are genuine covariance from differentiation of the **same pooled distribution**. Individual cross terms need not be PSD and their aggregate can be negative, but their complete sum with the diagonal terms is PSD. They can be decisive even when all members currently predict the same probabilities.

For one use of a shared layer, where \(h_m^c=\alpha_m\otimes b_m^c\), this becomes

\[
F_q=\mathbb E_{x,c\sim q}\sum_{m,n}
\rho_m^c\rho_n^c
(\alpha_m\alpha_n^\top)\otimes(b_m^c(b_n^c)^\top).
\]

Responsibilities depend on current outputs, classes and activations. A product of unweighted factor means does not reproduce these terms. Merely summing the four individual member Fishers does not estimate \(F_q\).

One conservative operator option is to preserve \(F_q\)'s exact matrix-free action, or a declared PSD sum of sampled pooled score outer products, while approximating the member term. That is a standard Fisher/GGN operator composition; it is not a new estimator principle established by this assessment.

## 4. Rank-one fast factors and conditional KFAC

Let the shared linear matrix have shape \(o\times i\). With fixed current private diagonal matrices \(R_m\) and \(S_m\), a member use is

\[
u_m=S_m W R_m a_m^0=S_mW\alpha_m,
\qquad \alpha_m=R_m a_m^0.
\]

For its log-probability score, let \(\delta_m^c=\nabla_{u_m}\log p_{m,c}\) and \(b_m^c=S_m^\top\delta_m^c\). Then

\[
\nabla_W\log p_{m,c}=b_m^c\alpha_m^\top,
\qquad h_m^c=\operatorname{vec}(b_m^c\alpha_m^\top)
=\alpha_m\otimes b_m^c
\]

using column-major vectorization. For this single-use layer, the exact member block is

\[
F_{m,W}=\mathbb E_{x,c\sim p_m}
[(\alpha_m\alpha_m^\top)\otimes(b_m^c(b_m^c)^\top)].
\]

Conditional KFAC makes the within-member activation/cotangent factorization

\[
F_{m,W}\approx A_m\otimes G_m,\quad
A_m=\mathbb E_x\alpha_m\alpha_m^\top,\quad
G_m=\mathbb E_{x,c\sim p_m}b_m^c(b_m^c)^\top.
\]

Consequently the member component is approximated by

\[
\widehat C_{\rm member}
=\frac1{2M}\sum_m A_m\otimes G_m.
\]

The half is required for the specified objective. This is not an exact Fisher unless the conditional factorization is justified. Covariances based on actual supervised label gradients instead define an empirical approximation, not the model-Fisher target above.

### Difference from pooling factors

Let \(\bar A=M^{-1}\sum_m A_m\), \(\bar G=M^{-1}\sum_m G_m\). A single pooled-factor product is

\[
\bar A\otimes\bar G=\frac1{M^2}\sum_{m,n} A_m\otimes G_n.
\]

Relative to the conditional mixture, it introduces artificial pairings of one member's activation covariance with another member's derivative covariance. Those pairings are distinct from the genuine \(\alpha_m\alpha_n^\top\), \(b_m(b_n)^\top\), responsibility-weighted terms in \(F_q\).

Exactly,

\[
\frac1M\sum_m A_m\otimes G_m-\bar A\otimes\bar G
=\frac1M\sum_m(A_m-\bar A)\otimes(G_m-\bar G).
\]

This difference has no universal PSD ordering. In scalar two-stratum examples, \(A=(1,4),G=(1,4)\) gives 8.5 versus 6.25; switching \(G\) to \((4,1)\) gives 4 versus 6.25. Nor is there universal curvature-error dominance. For the latter means, take equally likely nonnegative squared activation/derivative pairs \((X,Y)=(0,1.75),(2,6.25)\) in stratum one, and \((1.75,0),(6.25,2)\) in stratum two. Both exact \(\mathbb E[XY]\)'s are 6.25, so the pooled-factor product equals the target while the conditional mixture equals 4. Taking constant \((X,Y)=(1,4)\) and \((4,1)\) instead gives the same means but exact target 4, where the mixture is exact. These are moment-distribution examples, not claims about the native model. Retaining member association can reduce a particular approximation bias, but the mixture still drops conditional activation/derivative dependence. Neither estimator is universally closer to the full Fisher or better for predictive quality.

### Exact tying-map equivalence

Introduce untied effective matrices \(U_m\) and apply ordinary member KFAC there. The actual parameter map is

\[
\operatorname{vec}(U_m)=D_m\operatorname{vec}(W),
\qquad D_m=R_m^\top\otimes S_m.
\]

If the untied layer approximation is \(B_m=A_m^0\otimes G_m^0\), its pullback is

\[
D_m^\top B_mD_m
=(R_mA_m^0R_m^\top)\otimes(S_m^\top G_m^0S_m)
=A_m\otimes G_m.
\]

Stacking the tying maps into \(T\) gives
\(T^\top\operatorname{diag}(B_1,\ldots,B_M)T
=\sum_mD_m^\top B_mD_m\).
Objective coefficients simply weight these blocks. Thus the stratified shared curvature is an exact pullback of a conventional untied KFAC approximation, with ordinary diagonal-factor Jacobians. Preserving the ordinary private optimizer ignores shared/private curvature blocks. It does not confer full-model natural-gradient invariance to arbitrary fast-factor reparameterization; block approximations and isotropic damping limit such claims.

## 5. Multiple layer uses: members are not the only sharing axis

In a graph layer, the same \(W\) may be used at many nodes, edges or query positions \(t\). The score is then

\[
h_m^c=\sum_t \alpha_{m,t}\otimes b_{m,t}^c,
\]

so even the individual member Fisher includes true \(t\ne u\) terms. A covariance over flattened member/node positions can discard these before member stratification. The proposed \(A_m\otimes G_m\) therefore also needs a declared within-route expand/reduce or other sharing approximation; it cannot be called complete-member exact curvature merely because all four routes are retained.

Eschenhagen et al. Eq. 6 explicitly starts from these Jacobian sums and then drops unequal-position terms for expand. Eq. 7 collapses factor statistics into one product. Eqs. 9–10 replace the reduced-output Jacobian sum by a product of sums before forming one KFAC block. For the present loss, member CE is an expand component across members; pooled NLL is a reduce component across members. Nodes/edges introduce another sharing axis whose treatment must be specified separately. The composite curvature can be formed by summing the approximated components, as ordinary loss/GGN additivity permits.

The proposed conditional member sum is therefore **not literally equal** to pooled KFAC-expand Eq. 7 in general, and a correct pooled Fisher operator is **not literally equal** to the Jacobian approximation in KFAC-reduce Eqs. 9–10. General tied-weight curvature, conditional factorization, additive loss metrics and the tying pullback are still established algebra. The inspected source's exactness propositions require deep linear networks, Gaussian likelihoods and, for reduce, scaled-sum aggregation. Those assumptions do not cover nonlinear softmax members and a probability-mixture NLL.

If cross-route factorization is used for the pool, independent pairwise estimates need not preserve PSD. A sufficient PSD construction is to estimate consistent joint activation banks and consistent joint responsibility-weighted derivative banks, then factor between the two banks: the resulting cross-pair sum is the expectation of an outer product of \(\sum_m\alpha_m\otimes\widetilde b_m\) under independent bank draws. That is a conventional enlarged/block covariance factorization. It adds another approximation and up to quadratic-in-member factor storage; it is not promoted here as a novel or ready estimator.

## 6. What the SPD/current-gradient claim does guarantee

For any symmetric PSD approximation \(\widehat C\), choose \(\lambda>0\), set \(H=\widehat C+\lambda I\succ0\), and solve

\[
Hp=g_W,\qquad \Delta W=-\eta p.
\]

If \(g_W\ne0\), then

\[
g_W^\top\Delta W=-\eta g_W^\top H^{-1}g_W<0.
\]

This holds with stale **curvature** if it remains SPD and the gradient is current. It does not hold automatically when current \(g_W\) is replaced by stale gradients or momentum. Damping does not guarantee a finite-size loss decrease by itself. If the shared-block loss has Lipschitz gradient constant \(L_W\), a sufficient bound is

\[
0<\eta<\frac{2g_W^\top p}{L_W\|p\|^2};
\]

\(\eta<2\lambda/L_W\) is a conservative sufficient bound for an exact solve. A qualified line search is another standard way to choose a decreasing finite step. None was implemented.

For an approximate solve with residual \(r=g_W-Hp\),
\(g_W^\top p=p^\top Hp+r^\top p\). A positive dot product or an appropriate residual bound supplies the descent condition. In exact arithmetic, CG started at zero has the Galerkin orthogonality giving \(g_W^\top p=p^\top Hp>0\) at a nonzero iterate; finite-precision/numerical support still needs qualification. This is standard SPD linear algebra, not a new CG rule.

The guarantee is for the shared block with private parameters fixed. Simultaneous private Adam updates with history may oppose the full loss gradient, so no full-update descent or improved member competence is established. Smaller curvature Frobenius error does not imply a better inverse direction; EKFAC explicitly makes this qualification. Neither descent nor better curvature approximation establishes VALID ranking, generalization or ensemble predictive gains.

The inverse of a sum of curvature terms is not generally the sum or mean of their inverses. The intended step uses the joint gradient with the inverse of the declared total approximation; averaging member inverses silently defines a different SPD approximation even if applied to that same common gradient.

## 7. Explicit costs

For a layer \(W\in\mathbb R^{o\times i}\), one dense KFAC factor pair stores \(i^2+o^2\) scalars. Four separate member pairs store \(4(i^2+o^2)\). Against one pair, the extra FP32 factor bank alone is

\[
12(i^2+o^2)\text{ bytes}.
\]

At \(i=o=512\), four pairs require 8 MiB and the additional three pairs require 6 MiB. This excludes gradient tensors, factor update workspace, spectra/inverses, saved activations/cotangents, CG vectors, checkpoints and the pool term. Relative to the original shared Adam's \(2oi\) moment scalars, the optimizer-state comparison is \(M(i^2+o^2)\) factors versus \(2oi\) moments, not a claim that a KFAC pair is always smaller or cheaper. Replacing shared Adam can remove its moments, while ordinary private optimizer state remains.

The member operator action is

\[
\widehat C_{\rm member}\operatorname{vec}(V)
=\frac1{2M}\sum_m\operatorname{vec}(G_mVA_m^\top).
\]

For dense factors this costs \(O(M[o^2i+oi^2])\) per CG matrix-vector product. \(k\) iterations multiply that cost by \(k\), in addition to estimating/updating factors. Full factorizations/eigendecompositions can cost \(O(M[i^3+o^3])\) when used and cannot be advertised as four ordinary cheap Adam steps. Structured storage or low-rank factors would be extra approximations requiring their own qualification.

Consistent cross-route factor banks for the pool can have dimensions \(Mi\times Mi\) and \(Mo\times Mo\), hence \(O(M^2[i^2+o^2])\) storage before symmetry compression and \(O(M^2[o^2i+oi^2])\) pairwise operator work. Actual shared diagonal blocks may be reused, but the accounting depends on the declared estimator and label distribution.

An exact matrix-free \(F_qv=J_q^\top\operatorname{diag}(1/q)J_qv\) requires differentiation through all four member routes, typically a JVP and VJP per operator application. Multiple CG iterations repeatedly incur that work. A pooled-score Monte Carlo bank instead needs additional model-label backpropagation and score storage or repeated passes. Sparse graph JVP/VJP support, memory peaks and numerical behavior have not been qualified here. No source capability, wall-clock advantage, GPU feasibility, inference speed gain or equal-compute claim follows from the formulas.

## 8. Scalar checks and scientific disposition

`evidence/CURVATURE_SANITY.json` reuses logits \((-2W,-2W,-2W,8W)\) and target one, with the specified half member/half mean-probability loss. At \(W=0\), all probabilities are one half, but the pooled Fisher is 0.0625. Retaining only diagonal-route terms gives 1.1875; the real cross-route sum is \(-1.125\). Thus equal member probabilities do not imply absent pooled cross curvature.

At \(W=1\), the pooled observed Hessian is approximately \(-0.47758\) and the full observed loss Hessian is \(-0.07862\), while the termwise model Fisher/GGN is \(+0.21502\). Elementary finite differences agree with the observed-Hessian formula within \(10^{-6}\) at both points. These are mathematical checks, not data outcomes or evidence that these values occur in the native graph model.

A conditional utility rationale remains easy to state: member stratification may reduce bias caused by association between each member's activation and derivative covariance, and retaining real pool terms may approximate the served loss metric better. The moment-distribution examples above refute universal curvature-error dominance, and curvature error does not control inverse-direction or predictive quality. A specific prospective utility claim would require a declared full-objective curvature target and a matched approximation comparison; no new scientific cell, fit, threshold or acceptance gate is created here.

There is no genuinely distinct estimator/fast-factor ingredient surviving this bounded screen. The recommendation is to treat the construction, if retained, as an attributed conditional/shared-weight KFAC variant with a correctly declared pool metric. Current-gradient SPD descent is valid under its stated limits; architecture relabeling, preserved four predictions, more covariance state, and ordinary diagonal-factor pullbacks do not establish novelty or quality. No implementation or fit is justified by the present assessment.

Canonical indices and sealed scientific packets were left unchanged. No exhaustive search, complete publication-version equivalence, full-paper/proof audit, predictive benefit or execution authorization is claimed.
