# Local prediction objective of the registered graph initializer

Source-bound mathematical interpretation, 2 October 2026. No empirical result, novelty claim, new arm, active-source edit, or manuscript text is supplied here. The derivation concerns `graph_init_cfg0_outcome_aware_precision_v2`, anchored at `graph_init_precision_execution_root_v2/study_v2`, and the exact registered round17 v3 sources. All outcome-bearing initialization, fit, comparison and report files remain unread. No labels, arrays, checkpoints, model imports, scientific runtime, SSH or GPU work were used.

## 1. The defensible interpretation

At a fixed warm predictor, the graph branch constructs four **detached linear functionals of training logits** from graph-filtered cross-entropy cotangents. It takes their steepest jointly bounded parameter step after removing the common loss-gradient component and the route mean. The resulting extra prediction changes are realizable through the admitted private factor slice and are first-order neutral to the original training CE. A common descent step is added to every route. A finite training-only safeguard decides whether the constructed operation is installed.

This is an elementary projected linear optimization characterization. It is useful for naming what the code does, not a major theoretical contribution. It provides no guarantee of complementary classification errors, held-out benefit, calibration, a Bayesian posterior, or low cost. The graph-specific construction selects a supervised orientation; generic Jacobian construction and output-scale matching are already prior.

The exact result below describes the nondegenerate graph candidate in ideal arithmetic. The installed arm may instead take common descent or retain identical warm copies. All three outcomes belong to the complete safeguarded operation.

## 2. Bound source and actual interface

The registered cohort is PolyFormer-Mono/Squirrel and Polynormer-r/Photo, configuration 0, seeds 17/29/43 and source splits 0/1/2. The registry points to round17 v3 `SOURCE_BINDINGS.json` and `PROTOCOL.json`; their local SHA256 values match those registered descriptors. The active slice is respectively `(stem.S, head.R)` with 512 coordinates and `(stem.S, global_head.R)` with 1024 coordinates. All multiplicative factors start at identity after a warm native fit. Common weights, other factors, copied biases and buffers are frozen in the initializer closure; dropout is off. The predictive logits are those of the whole model, with the final global branch used for Photo. [S1, S2, S3]

The boundary map is `linear(x * R, W) * S + B`; the method changes only the two admitted scale vectors. It is not an additive permanently frozen JVP head. During continuation, shared weights and private trainable parameters change under mean member CE, independent per-member dropout and the registered Adam transport. The initial orthogonality and centering constraints are not enforced throughout training. [S7, S8]

## 3. Notation and exact graph masking

Let `theta0 in R^d` be the admitted identity factor vector, and let `z(theta) in R^(n C)` flatten TRAIN logits of the deterministic fixed common model. All inner products and norms below are ordinary Euclidean ones on these flattened objects. Let `J = Dz(theta0)` and

`r_i = (softmax(z_i(theta0)) - onehot(y_i))/n`,

so `g = J^T r` is the training-CE gradient on that slice. Both `r` and every filtered cotangent below have zero class sum per row. Write `Gamma = I_n tensor (I_C - 11^T/C)` and `Jbar = Gamma J` for the class-centered prediction Jacobian. Then `Jbar^T r = g`.

Let `E` inject the `n` TRAIN rows into all `N` graph nodes. The released topology is an unweighted simple undirected graph without added self loops. Its normalized adjacency `S = D^(-1/2) A D^(-1/2)` has zero rows and columns for isolates. The four fixed degree-three Bernstein polynomials are

`H_b = choose(3,b) ((I+S)/2)^(3-b) ((I-S)/2)^b`, `b = 0,1,2,3`.

They are evaluated with three sparse products, without an eigendecomposition. For this symmetric normalized `S`, its spectrum is contained in `[-1,1]`; therefore `H_b` are positive semidefinite contractions and `sum_b H_b = I`. They overlap spectrally and are not disjoint spectral projectors. Define `K_b = E^T H_b E`, implicitly tensoring with the class identity when acting on `r`. Then `sum_b K_b = I_(nC)`.

The code first propagates `E r` through the whole graph and then masks the resulting cotangent **back to TRAIN before the VJP**. Thus

`h_b = Jbar^T K_b r`, `sum_b h_b = g`.

Unlabeled nodes can occur along paths in the polynomial graph filter, but no unlabeled logit receives a cotangent in this VJP. This is the masked operator `E^T H_b E`, not whole-node pseudo-label fitting. The graph filter and predictive encoder need not commute. Calling route outputs four isolated graph-frequency components would be incorrect. [S4, S5]

All of `r`, `K_b r` and `h_b` are formed at the warm state and detached. Differentiating an objective that re-evaluates `r(theta)` would add terms that this code does not use.

## 4. The constrained objective and its solution

Assume `g != 0`. Let

`P_g = I - gg^T/||g||^2`,

`q_b = (K_b - I/4) r`,

`a_b = P_g Jbar^T q_b = P_g h_b`,

`A_* = (sum_b ||a_b||^2)^(1/2)`, `rho = 0.5 ||g||`.

Here `sum_b q_b = 0`, `sum_b a_b = 0` and `g^T a_b = 0`. In the nondegenerate case `A_* > 0`, consider the first-order prediction objective

`min_(u_0,...,u_3) Phi(u) = sum_b <q_b, Jbar u_b>`

subject to

`sum_b u_b = 0`, `g^T u_b = 0 for every b`, `sum_b ||u_b||^2 <= rho^2`.   (1)

This objective rewards route-specific changes against the **contrasts of detached graph-filtered errors**. It is linear in prediction changes; its joint radius is in parameter space. It is not an objective maximizing prediction disagreement or minimizing squared residual reconstruction.

For any feasible `u`, projection and centering leave its inner product unchanged:

`Phi(u) = sum_b <a_b,u_b> >= -A_* (sum_b ||u_b||^2)^(1/2) >= -rho A_*`.

Equality is attained by

`u_b^* = -(rho/A_*) a_b`.

This is exactly the ideal graph candidate, with the code's tangent `t_b = -u_b^*`, scalar `lambda = rho/A_*` and direction `d_b = -g + u_b^* = -g - t_b`. The repeated centering/projection in FP64 implements this projected gradient before casting to model dtype. [S6]

An important detail: `lambda = CAP*||g||/||a||_F` is **not** `min(1, CAP*||g||/||a||_F)`. For a nonzero sufficiently resolved projected gradient, it rescales to the radius, including amplification when the unscaled gradient is small. The exact characterization assumes the denominator floor `1e-20` is inactive. It is not a claim that arbitrarily tiny graph signals yield stable orientations.

The full `d_b` also solve the corresponding first-order graph-functional problem with the common component fixed to `-g`: the common contribution `sum_b <q_b,-Jbar g>` is zero. Using `K_b r` instead of `q_b` adds the fixed scalar `-||g||^2` to the full-direction objective, and does not change its optimizer.

### The prediction-space version keeps the induced metric

Let `Delta_b = Jbar u_b`. The same optimization can be written over realizable class-centered prediction changes as

`min sum_b <q_b,Delta_b>`

subject to

`Delta_b in range(Jbar)`, `sum_b Delta_b = 0`, `r^T Delta_b = 0`,

`sum_b ||Jbar^+ Delta_b||^2 <= rho^2`.   (2)

`Jbar^+` is the Euclidean Moore–Penrose inverse. For any feasible parameter vectors, their minimum-norm representatives have no larger norm; conversely, choosing `u_b = Jbar^+ Delta_b` satisfies all parameter constraints. The optimizer in (1) lies in `range(Jbar^T)` and wastes no radius on `ker(Jbar)`.

This is a constrained prediction objective with the metric induced by the private slice. It is not a prediction-space Euclidean-radius optimum. In particular, matching parameter Frobenius norms does not match output gains.

The centered logit response is

`Delta_b^* = -lambda B q_b`, `B = Jbar P_g Jbar^T >= 0`.

`B r = 0`; hence `r^T Delta_b^* = 0` exactly. Also `q_b^T Delta_b^* = -lambda ||P_g Jbar^T q_b||^2 <= 0`. A route reduces its own detached contrast functional to first order, but this is not a statement that its classification error, or even every individual TRAIN-node CE, improves.

If `A_* = 0`, the constrained objective is flat on the feasible directions. The source tests for null/duplicate **function** tangents and uses common/unchanged fallbacks; it does not force a new diversity direction. If the squared common gradient is at most `1e-20`, it returns unchanged warm factors. [S6]

## 5. Approximation statements with assumptions and remainders

These are local statements. Numerical AD qualification does not establish smoothness on a whole neighborhood. The retained bodies contain ReLUs; activation boundaries can invalidate a Hessian-bound argument. Differentiability alone gives an `o(alpha)` first-order remainder, without a usable finite-step bound. Where bounds are claimed below, assume the entire relevant set of line segments lies in a neighborhood with the stated bounded derivatives and frozen dropout/shared state.

### Loss descent

If the slice training loss `L` has a locally `beta`-Lipschitz gradient, then

`L(theta0 + alpha d_b) <= L(theta0) - alpha ||g||^2 + (beta alpha^2/2)||d_b||^2`.   (3)

The first-order derivative is identical across routes because `g^T u_b = 0`. With Armijo coefficient `c=1e-4`, every member meets its ideal bound for

`0 < alpha <= 2(1-c)||g||^2 / (beta max_b ||d_b||^2)`.

This proves existence of a sufficiently small ideal step if `g != 0`; it does not prove that one of the six executable trials finds it. Finite arithmetic and sufficiently small improvements can also defeat acceptance. By convexity of CE in raw logits, accepted member CE bounds imply the same bound for their mean-logit pool; the source nevertheless checks the pool explicitly.

The actual diagnostics permit relative orthogonality and route-mean errors up to `2e-5`. With an actual `u_b`, replace the first-order term in (3) by `-alpha||g||^2 + alpha g^T u_b`. A machine certificate is tolerance-bounded, not an exact real-arithmetic identity.

### Individual and pooled logits

Suppose the vector-valued logit Hessian obeys `||D^2 z(theta)[v,w]|| <= B_z ||v||||w||` throughout those segments. Then

`z(theta0+alpha d_b) = z0 + alpha J d_b + R_b`,

`||R_b|| <= (B_z alpha^2/2)||d_b||^2`.   (4)

In exact centering, the pooled logits have first-order term `z0 - alpha Jg`. For the hypothetical common point `theta_c = theta0 - alpha g`, expanding each route about `theta_c` gives the stronger comparison

`||mean_b z(theta_c+alpha u_b) - z(theta_c)|| <= (B_z alpha^2/2) mean_b ||u_b||^2`.   (5)

With route-mean defect `ubar`, add `alpha ||J(theta_c)|| ||ubar||` to the right side. The accepted graph step therefore has no extra **first-order** pooled-logit improvement over the hypothetical common update at that same alpha.

The active `common_only` arm chooses its own initial alpha from its maximum direction norm and backtracks independently. Its endpoint is generally not `theta_c` at the graph alpha. Comparing the actual arms includes the first-order difference `-(alpha_graph-alpha_common)Jg`, nonlinear terms and fallback differences. A same-alpha interpretation must not be attached to that actual contrast. [S2, S6]

### The detached graph objective at the finite candidate

Define the signed finite contrast functional, using the **fixed warm** `q_b`, by

`Psi(alpha) = sum_b <q_b, Gamma[z(theta0+alpha d_b)-z0]>`.

By (4) and `sum_b q_b=0`,

`Psi(alpha) = -alpha rho A_* + E_Psi`,

`|E_Psi| <= (B_z alpha^2/2) sum_b ||q_b|| ||d_b||^2`.   (6)

This names the conditional finite prediction effect the construction targets. No `B_z` is estimated or certified here, so a negative first-order coefficient alone is not proof that the accepted finite `Psi` is negative. The line search evaluates member/pool CE and nonzero pair differences, not `Psi`.

For small route-centered logit deviations at a fixed pooled center, convex CE charges a nonnegative leading spread cost: `(1/2) mean_b delta_b^T [diag(p)-pp^T] delta_b`. With bounded third derivative its remainder is bounded by a constant times `mean_b ||delta_b||^3`. This reused Jensen/Taylor observation explains why disagreement alone is insufficient; it does not establish an ensemble advantage. [S9]

## 6. What is graph-induced, and what already has prior

The precise graph-induced construction is the warm, supervised cotangent orientation

`q_b = (E^T H_b E - I/4) r`

pulled through the finite predictive Jacobian, projected orthogonally to the CE gradient, jointly normalized and installed into private multiplicative factors before ordinary continuation. The graph enters the cotangents; the predictive Jacobian also depends on the fixed graph-conditioned encoder. The projected kernel response mixes both. Node permutation changes filter-to-label/feature alignment while retaining graph spectrum, edge inventory and degree multiset; it does not merely relabel four bands. [S4–S6]

The retained initializer/cache review already establishes shared/private factor algebra (BatchEnsemble/TabM), graph error propagation (Correct & Smooth), Bernstein spectral filters (BernNet), warm graph transfer (PreGS), and a generic frozen-Jacobian ensemble/output-second-moment prior (He, Lakshminarayanan & Teh, 2007.05864v2). The latter adds a fixed random JVP function to independent learners under a different NTK/squared-loss setting. The present construction instead uses detached training errors and a constrained finite private-factor displacement under CE. That is a source-level difference; it does not establish global novelty or import a posterior theorem. Fort et al. 1912.02757v2 already separate weight distance from prediction distance. [S10]

The active random control matches only the joint parameter tangent norm, not its training centered-logit Gram matrix. The existing follow-up's prospective same-alpha Gram-matched control remains a separate unrun design. Nothing in this derivation amends the active five arms or turns those prior Jacobian/output-scale ideas into a new learner claim.

## 7. Feasible evidence after full cohort closure

`DIAGNOSTIC_PLAN.md` specifies descriptive checks using already preserved arrays/scalars, without a model rerun. The strongest feasible signed check reconstructs warm graph cotangents from saved warm logits and the frozen source TRAIN pack/topology, then measures (6) on saved initialized predictions. Six stored JVP pair RMS values also recover the route Gram matrix and allow comparison with the finite initialized Gram. JVP vectors themselves are discarded. Later deterministic logits at the frozen native midpoint and selected endpoint permit measurement of retained graph orientation; per-update traces contain losses, not full per-update predictions.

These checks can refute a claim that the finite installed perturbation actually retains the intended graph-error orientation. Successful realization is only a necessary mechanism check. Whether the complete operation has useful task effects requires the already registered full-cohort contrasts, including fallback cells and actual costs. The cohort/idea choice is explicitly outcome-aware and previously exposed; a favorable descriptive result would remain exploratory.

The source alone does not establish a useful contribution, and no cell outcome has been inspected in this packet. There was no need for a new primary read: existing conclusions plus the actual source determine the bounded interpretation.

## Source keys

Paths below are relative to `postsubmission_research_20260930`; exact bytes/hashes and line scopes are in `INPUT_BINDINGS.json` and `SOURCE_MAP.json`.

- **S1:** `graph_init_precision_execution_root_v2/study_v2/GRAPH_INIT_ATTEMPT_REGISTRY.json`, registered study/source/context fields only; wrapper `SEAL.json` and registered round17 `SOURCE_BINDINGS.json`.
- **S2:** `continuous_method_gap_search_v1/round17_graph_init_driver_integration_v3_precision/PROTOCOL.json`, active slices, pooling, five arms, continuation and complete-operation interpretation.
- **S3:** same packet `prototype/graph_band_route_initializer.py`, lines 28–111: slice and frozen predictive closure.
- **S4:** same initializer, lines 114–176: graph normalization, bands and alignment-null permutation.
- **S5:** same initializer, lines 212–253: detached CE residual, full-node propagation, TRAIN remasking and VJPs.
- **S6:** same initializer, lines 254–350: projection, common rescaling, JVP pair summary, bounded line search and fallbacks.
- **S7:** round17 `prototype/graph_init_driver.py`, lines 635–807: actual-warm qualification, graph arms, installed slices/logits, scalar diagnostics and warm RNG restoration.
- **S8:** round17 `prototype/graph_init_training_adapter.py`, lines 368–499; modern v3 `prototype/backbone_boundary_adapter.py`, lines 15–152: whole-member predictors, mean member CE, stochastic continuation and saved prediction timepoints.
- **S9:** `graph_init_mechanism_analysis_root_v1/ANALYSIS_v2.md`, existing local descent, same-alpha cancellation and Jensen limitations, reused and sharpened here.
- **S10:** `graph_initializer_cache_literature_followup_v1/REPORT.md`, `PAPER_CONCLUSIONS.json` and retained `REUSED_CONCLUSIONS.json` scopes; no primary reread.
