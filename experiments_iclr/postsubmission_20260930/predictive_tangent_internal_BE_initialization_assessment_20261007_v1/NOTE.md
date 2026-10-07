# Prediction-tangent initialization of internal BE factors

7 October 2026. Research assessment only. Saved BE/TabM, NTK, local perturbation, GGN-head and adapter-Laplace scopes were checked first. One new bounded primary method scope, **Local Ensembles `1910.09573v2` §§2–3**, resolves the flat-versus-sensitive direction tradeoff. No model/data/current outcome/TEST access, scientific code, run, server contact, new grid or canonical-state edit.

## Disposition: one conditional hypothesis, not a new principle

**Worth retaining as a narrow orientation question:** at the same initial task-relevant predictive disturbance and comparable loss damage, do non-null prediction-guided directions in existing internal factors lead to more useful final pooled predictions than random internal directions from the same TRAIN-only warm model?

The identifiable mechanism is avoiding ineffective factor directions and placing the initial asymmetry in the real prediction function before joint continuation. This is more concrete than hidden-vector separation. It is not evidence that top Fisher directions are optimal, that the asymmetry survives, or that it uses different graph evidence. Jacobian construction, covariance-aware initialization, centered perturbations and full-model continuation already have close prior. The remaining operational distinction is **existing internal BE coordinates that can affect graph computations, followed by a live common core**, rather than only new terminal heads or frozen posterior functions.

Do not promote “leading-Fisher graph initialization” as a discovered method or add a contrastive/routing pipeline. If a future comparison cannot match predictive disturbance without unacceptable competence damage, or has no usable non-null span, this hypothesis has no demonstrated applicability. No desired positive verdict is presumed.

## 1. Define the tangent being measured

At one TRAIN-only warm predictor `(θ0,φ0,ψ0)`, hold shared θ and prediction boundaries ψ fixed for the initializer calculation. Let φ contain only the existing eligible internal factors. For one training object i,

`z_i(φ0+δ)=z_i(φ0)+J_i δ+O(||δ||²)`.

Use the actual prediction map and the declared TRAIN graph/support. A class-common logit shift is invisible to softmax; class-centered logits or probabilities/log-probabilities remove that particular nuisance. These are different metrics, so a claimed “predictive tangent” must name its output.

Two commonly conflated constructions are:

```
Observed-label gradient Gram:
  g_i = ∇_φ log p_i(y_i),  F_obs = mean_i g_i g_iᵀ

Model Fisher / classification GGN:
  F_model = mean_i J_iᵀ [diag(p_i)−p_i p_iᵀ] J_i
          = mean_i sum_c p_i(c) ∇_φ log p_i(c)∇_φ log p_i(c)ᵀ.
```

F_obs is not generally the model Fisher, the exact Hessian or a posterior covariance. When TRAIN labels are fitted confidently, `∇log p(y)` can become small even while logits or alternative probability directions remain sensitive. F_obs directions guarantee a local change in the selected true-class log-probability only when their corresponding gradients are nonzero. They can emphasize hard/noisy targets or confidence variation rather than different graph evidence.

F_model measures local expected predictive KL: for small δ, `mean KL(p0||p_δ)=δᵀF_modelδ/2+o(||δ||²)`. With finite probabilities, a null direction gives no first-order probability change on the measured TRAIN objects. It may still affect other objects or finite nonlinear behavior. Neither geometry includes uncertainty over the fitted core or the graph population. Its eigenvectors also depend on the factor coordinates; multiplicative gauges and rescaling can change a Euclidean “leading direction.” Match predictive effects rather than relying only on factor norm.

### Prediction visibility is weaker than decision visibility

Wiki softmax is invariant to a per-example all-class logit offset; class-centering/Fisher removes that direction. Confidence or positive temperature changes can still leave every member argmax unchanged. Such calibration can affect a mean-probability ensemble, but it does not establish different graph evidence or individual error repair.

For MolHIV, a common additive score shift across all molecules changes sigmoid probabilities/BCE while preserving score ordering and ROC AUC. Pointwise model-Fisher visibility therefore includes some ranking-invariant directions. Relative positive/negative score gradients can identify that distinction.

**For the current ogbl-collab contract**, root identifies 60,084 positive scores evaluated against **one global 100,000-negative pool/threshold**. A global common shift of every positive and negative score preserves Hits@50. A per-query common offset is **not generally invariant** because it can change ordering against that global threshold. Per-query cancellation applies to query-specific candidate evaluations such as Citation2/HeaRT, and must not be imported here.

A relative-score tangent uses differences `∇φ z_i^+−∇φ z_j^-` under the actual TRAIN comparator contract; it removes a global common shift. The common-negative-pool identity matters for Collab. No pairing rule, new objective or initializer is adopted here. If a later protocol retains pointwise probability/Fisher geometry, it must state that this is not an AUC/Hits geometry and require actual official-metric and task-margin gains before calling its directions useful. This is one conditional initialization question, not a grid of different tangent metrics.

## 2. The sensitive-versus-flat tradeoff

**Leading eigen-directions:** at the same raw parameter radius, these maximize local predictive KL or measured gradient response. That supplies visible initial asymmetry, but also the largest local disturbance. At a stationary warm point with a positive local risk Hessian, symmetric offsets increase average own risk to second order. Fisher/GGN is an approximation to that risk curvature, not a bound on finite CE changes. A short nonstationary warm point is not automatically a MAP point.

In a simple locally quadratic own-loss continuation, private differences evolve approximately as `d_{t+1}=(I−ηH)d_t`; highly constrained positive-curvature directions can be damped fastest. Thus “most visible now” need not mean “most persistent or useful after training.” This is a limiting model, not a theorem about native Adam, nonlinear gates or the current block-dependent loss.

**Inverse-curvature/flat directions:** these prioritize weakly constrained variation and can preserve TRAIN loss under a valid local approximation. In a model-Fisher nullspace they have no first-order TRAIN predictive spread. Large inverse-eigenvalue offsets can also leave the local regime. Out-of-TRAIN prediction changes may be useful uncertainty or arbitrary extrapolation; flatness does not imply complementary correct decisions. Full-Hessian flatness and Fisher nullness are not interchangeable.

No spectral-band, inverse-power or perturbation-strength sweep follows. A future protocol would freeze one stated direction rule and one displacement budget before outcomes. Initial predictive magnitude/Gram and finite loss damage must be reported so that greater spread alone cannot explain a gain.

### Symmetry and normalization limits

Four zero-sum offsets cancel the **first-order** mean prediction change at a common deterministic state. Internal factors enter nonlinear/multiplicative computation, so the mean function can move at second order and individual competence can be damaged. Exact mean-logit preservation of a linear last layer cannot be transferred automatically. Four members also provide at most three centered empirical function directions; two antithetic pairs use at most two. This is not a full posterior.

Norm placement can make some directions ineffective. Positive per-feature scaling immediately before TRAIN BatchNorm largely cancels if each member uses its own batch statistics, apart from epsilon and intervening operations. Shared/frozen statistics, branch addition or nonlinear placement change that conclusion. LayerNorm normalizes across channels: general channelwise scaling changes their ratios, while a uniform positive amplitude can still largely cancel. Root reports that the current public GINE uses stateless LayerNorm, unlike the typical BatchNorm GINE recipe; no runtime or source equivalence was checked here.

The actual predictor Jacobian exposes these conditional null directions, unlike hidden separation alone. But its mode and normalization state must be specified: a TRAIN-BN Jacobian and a serving-map Jacobian can differ. Near-unit tangent behavior also does not cover finite Rademacher sign changes that cross nonlinear activation regimes. This is established normalization/FiLM/BE parameterization reasoning, not a new norm-aware method or test.

## 3. Closest priors and exact limits

| Prior scope reused or newly read | Direct overlap | Remaining distinction / limit |
|---|---|---|
| BatchEnsemble `2002.06715v2`; TabM `2410.24210v3` | Shared matrices/private multiplicative factors; first-adapter randomization, later identity factors and own-member supervision. | Existing factor perturbations are prior. Warm prediction-guided internal orientation is the conditional choice under discussion; TabM already shows useful nonlinear paths, not a guarantee of strong standalone members. |
| **NTKGP `2007.05864v2`**, saved §3/Algorithm1/§§3.4–3.5/F–H scopes | Random ensemble functions `δ(x)=J(θ0,x)θ*`, fixed-primal JVP and matching output second moments. | Uses independent full learners plus untrainable prior functions; exact posterior statement is infinite-width/squared-loss. It does not qualify finite shared-factor CE initialization and joint continuation. |
| **How to Train a Shallow Ensemble `2602.15747v1`**, saved II.3/IV/S10 | One shared representation; centered GGN/Laplace last-layer samples; head-only or whole-model continuation; covariance orientation versus magnitude. | Very close initialization/continuation ancestry. Atomistic Gaussian-NLL/UQ and terminal heads differ from internal graph factors; reported UQ improvements do not establish classification gains. |
| **Laplace-LoRA `2308.13111v5`**, saved §§3–4/appendix scopes | Frozen-base adapter curvature, low-rank KFAC and linearized output covariance. Rank1BNN `2005.07186v2` adds factor distributions with point-estimated common W. | Conditional posterior/sampling differs from a one-time initialization followed by a live core. These do not license a posterior claim at an arbitrary warm point. |
| **Fort et al. `1912.02757v2`**, saved §§4–5/AppendixB | Random/local Gaussian perturbations around a trained solution; prediction distance versus parameter distance. | Already motivates predictive-geometry-matched controls. Its validation-filtered sample rejection cannot select this TRAIN-only initializer. |
| **Local Ensembles `1910.09573v2`**, new §§2–3 | Prediction-gradient projection into a low-Hessian-curvature subspace; first-order variance of a local family with similar training loss; implicit top-eigenvector calculation to define the excluded complement. | Audits a pretrained model for underspecification. It excludes leading constrained directions and does not initialize/train four fast-factor paths with a live shared core. Its Hessian is not automatically F_obs/F_model. |
| Existing graph-covariance/cache packets | Graph-error VJP spans, centered perturbations, matched prediction Gram/Jensen damage, curvature-aware head controls. | Those are project history. This question must not recreate their one-step selector, graph-band search or matched-geometry algebra as a new mechanism. |

**New primary citation:** David Madras, James Atwood and Alex D'Amour, [Detecting Underspecification with Local Ensembles, `1910.09573v2`](https://arxiv.org/abs/1910.09573v2), revised 7 December 2021. Version1's title was *Detecting Extrapolation with Local Ensembles*; title/version aliases are one paper. The relevant method is [§§2–3](https://arxiv.org/html/1910.09573v2#S2): `||U_flatᵀ ∇_θ prediction(x)||`, with the flat subspace formed as the complement of the leading Hessian eigenvectors. No author code, full proof/appendix, result verification or accepted-version equivalence was read.

No direct original SWAG method scope was located in the checked saved records. The saved Fort local-Gaussian controls already cover the needed local-sampling alternative; this review does not claim to have reverified SWAG's complete procedure or count it as another read. No extra scope was forced to meet a paper quota.

## 4. One falsifiable fallback hypothesis and strong controls

**Hypothesis:** after the same TRAIN-only warm acquisition, placing a matched initial task-relevant predictive disturbance in non-null internal-factor directions improves final fixed-pool quality and net common-error repair beyond equally disturbed random internal directions, without merely weakening members. A graph mechanism further requires persistent differences in neighbor/evidence class-margin responses beyond scalar confidence changes. A score gain without that evidence supports initialization utility for a GNN, not a graph-specific diversity principle.

Use one already declared representative complete graph/task and paired seed blocks under a separate prospective freeze. No new dataset or active-family edit is proposed. Keep the warm tensors, eligible factor/boundary partition, norm state, continuation objective, optimizer/state policy, views, full horizon, actual pool and checkpoint selector identical across initialization controls. Do not choose eigen-directions, radii, warm horizon or seeds from current/held outcomes.

The indispensable controls are:

- **Warm identity-copy continuation** and a capable warm single: distinguish additional training/warm competence from useful initialized paths.
- **Warm random internal directions with matched predictive disturbance/Gram and comparable finite loss damage:** distinguishes useful orientation from simply larger function perturbations; factor-norm matching alone is inadequate.
- **A centered classification GGN/head-tangent initializer with the same live continuation:** addresses the closest shallow-ensemble prior. A nonstationary warm point must be called GGN-inspired, not a fitted Laplace posterior.
- **Ordinary native/TabM-style shared BE, a genuinely independently acquired four-model bank and a packed untied reference**, at their declared complete budgets. These are strong quality controls, not four teachers used to construct the method. All comparable warm acquisition, continuation and selection costs remain visible.

These are fixed attribution/reference controls, not a grid of Fisher variants or strengths. If random matched orientation or head initialization explains the gain, the claimed internal predictive-orientation advantage is unestablished. If initial differences disappear, member competence degrades, only calibration improves, or repaired common errors are canceled by new full-population mistakes, narrow the result accordingly. A capable untied bank can remain better; sharing then needs an honest measured cost/quality case.

## 5. Costs and graph inference boundary

One warm shared GNN is allowed by the stated teacher-free premise: its own parameters become the live common core. It is neither four independently trained teacher donors nor compression into a student. It is still paid acquisition. Charge warm fitting, all gradient/JVP/VJP or curvature-vector work, eigenspace storage/approximations, initialization construction, failures, every continuation forward/reverse, selection and serving. Internal fast factors alter graph/attention states, so shared W does not make four member graph trajectories free. No complexity or speed gain is measured here.

The TRAIN geometry is conditional on one fitted model, graph/support, features and supervised objects. Correlated nodes/queries and overlapping neighborhoods do not invalidate the deterministic Gram algebra; they do invalidate treating its rows as independent graph-population replicates or automatic calibrated posterior evidence. Four directions do not restore omitted backbone uncertainty. Optimizer seeds on the same graph are not new graph draws.

## Scope and conclusion

**One new scoped primary method, zero full-paper/proof/code/performance credits.** All other listed primary methods were reused with original limitations. The first recalled title query missed; exact-title resolution found the versioned Local Ensembles identity. Only selected HTML/text extracts are retained; no PDF or source archive was downloaded. No scientific source edit, computation, current outcome, server or TEST access occurred.

The defensible outcome is a conditional, attributed predictive-orientation hypothesis. The broad operation is already established in Jacobian/curvature ensembles. This scope supplies neither an exact complete-equivalence certificate nor novelty clearance, and no new contrastive objective or adaptive strength method is justified.
