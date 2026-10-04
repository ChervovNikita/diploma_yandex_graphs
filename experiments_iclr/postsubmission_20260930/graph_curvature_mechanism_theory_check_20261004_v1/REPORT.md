# Source-only mechanism check: graph-conditioned private-head initialization

**Disposition.** The source supports a testable initialization-selection hypothesis. It supplies no new CE identity, negative-curvature splitting theorem, persistent-diversity guarantee, or predictive gain. Equal initial mean logits and equal mean-member CE gap leave enough response geometry free to alter the next full-model Adam update. The graph-specific part is the restricted candidate bank; the mechanism by which differing initializations alter learning is generic.

This assessment reads local source and saved literature conclusions only. No project module or numerical dependency was imported or executed; no runtime array, checkpoint, or project prediction-score payload was opened. No server, manuscript, existing index, or publication was changed. Adjacent saved literature result statements incidentally appeared in index excerpts; their numbers are not used. Input hashes and read scopes are in `SOURCES.json`. The expressions below concern exact arithmetic unless qualified. They do not establish actual warm-state numerical qualification.

## 1. Bound construction and constants

The v3 selector changes only the final private input factor `head.R` (Squirrel) or `global_head.R` (Photo) at initialization. The boundary source computes `W(h * R_m) * S_m + B_m`. Every upstream factor starts at identity; final `S_m=1` and copied biases agree. With the common features and shared head held fixed, logits therefore have the exact affine form

\[
z_{mn}=z_{0n}+A_n\delta_m,\qquad A_n=W\operatorname{diag}(h_n),\quad
\delta_m\in\{s v_i,-s v_i,s v_j,-s v_j\}.
\]

The common center is the same safeguarded head displacement in every arm. The vectors are orthonormal and orthogonal to the common head TRAIN gradient at that center. Thus the initial mean logits equal `z0` on every target node. Source affinity does not substitute for checking the actual loaded wrapper, precision, and warm state.

The separate root prospective freeze fixes radius `0.5`, cap `1.0`, positive-D floor `1e-9`, match tolerance `1e-10 + 0.001*|D0|`, 40 bisections, mean-logit tolerances `1e-5/1e-6`, rank tolerances `1e-10/1e-6`, sign tolerance `1e-12`, and tie/abstention tolerances `1e-6`. The bound v3 protocol still contains null placeholders, and index v55 preserves the prior report's corresponding statement. These are different source records; this check uses the separate frozen values without rewriting either record. A freeze is not qualification or execution authority.

## 2. Generic known identity: own-member CE includes a spread cost

For one node let `A(z)=log sum_c exp(z_c)`, `p_m=softmax(z_m)`, `bar z=mean_m z_m`, and `p*=softmax(bar z)`. With one-hot or normalized soft labels `y`, `CE(z,y)=A(z)-y^T z`. Then exactly

\[
\frac1M\sum_m CE(z_m,y)=CE(\bar z,y)+D,
\quad D=\frac1M\sum_m[A(z_m)-A(\bar z)]
       =\frac1M\sum_m KL(p_*\Vert p_m)\ge0.
\]

Equivalently, `D=mean_m B_A(z_m,bar z)` for the log-sum-exp Bregman divergence. The linear Bregman terms cancel because the offsets average to zero. Also `p*` is the normalized geometric probability mean and

\[
D=-\log\sum_c\prod_m p_{mc}^{1/M}.
\]

These are general algebraic identities, not graph results. TRAIN averaging preserves them. For fixed member logits the gap is independent of the labels; labels influence it indirectly through the supervised warm center and candidate construction. `D=0` exactly when all members have the same probabilities, allowing arbitrary all-class logit shifts. Class-centered response checks address this null direction.

The retained training objective is therefore `L_own=L_pool+D`, with the same geometric pool used for serving. Gradient descent on `L_own` can reduce `D`; the objective supplies no incentive to keep diversity large. The identity does not imply complementary correct decisions or improved accuracy. Increasing `D` while holding mean logits fixed increases mean member CE and leaves the deployed predictor unchanged. Probability averaging is a different predictor and cannot be substituted into this identity.

The saved JMLR record *A Unified Theory of Diversity in Ensemble Learning* already retains loss-matched centroid decompositions, the geometric/logit pool correspondence, and the dependence of benefits on member quality. The closest-prior report treats the exact identity as background algebra, and does not assert that GNCL prints this particular equation. Index v55 distinguishes GNCL (`2011.02952`) from *Gradient Starvation* (`2011.09468`); the latter supplies no mechanism evidence here.

## 3. What finite matching controls

The parameter covariance of this four-point construction is

\[
C_\theta=\frac{s^2}{2}(v_iv_i^T+v_jv_j^T),\qquad
C_{z,n}=A_n C_\theta A_n^T.
\]

Writing `H_n=diag(p0_n)-p0_n p0_n^T`, the small-radius expansion is

\[
D(s)=\frac1{2N}\sum_n\operatorname{tr}(H_n C_{z,n})+O(s^4).
\]

This is ordinary PSD softmax-loss curvature; antithetic symmetry removes odd terms. The exact finite gap is not just this quadratic approximation. Matching `D` controls one aggregate scalar, including higher even-order response terms. It does not match parameter radius, parameter covariance spectrum, per-node spread, class directions, individual member losses, gradients, or Adam preconditioning. Different matched pairs generally require different radii.

Even the phrase “covariance effect” needs scope: a finite four-point ensemble has higher moments. Rotating the two axes within the same plane preserves `C_theta` at a fixed radius but can change its fourth moments and finite CE responses. Any finite benefit concerns the complete initialization distribution unless a further approximation isolates covariance.

For an affine closure, `D(s)` is convex, even, zero at zero, and nondecreasing for `s>=0`. It is strictly increasing for positive `s` if at least one TRAIN response has a nonzero class contrast. This supplies the mathematical basis for bounded scalar bisection, subject to cap feasibility and finite-precision measurement. Checking approximate mean preservation at evaluated points alone is not proof of global affinity.

Projection also has a stronger immediate implication: because each offset is orthogonal to the gradient at the center and head CE is convex, every member's TRAIN CE is at least the common-center CE. A positive matched gap requires positive slack between the common-center loss and the original accepted Armijo bound. If this slack is zero or too small, all spread candidates must fail the member guard. A positive `D0` measured before guarding its reference pair does not establish reference eligibility. No instantaneous negative-curvature benefit exists in this affine head slice.

## 4. Generic learning response: why the next update can differ

For shared coordinates `phi` and member-private coordinates `a_m`, the exact initial gradients are

\[
g_\phi=\frac1{MN}\sum_{mn}J_{\phi,mn}^T(p_{mn}-y_n),\qquad
g_{a_m}=\frac1{MN}\sum_nJ_{a_m,n}^T(p_{mn}-y_n).
\]

Mean logits and `D` do not determine these gradients. In particular, `mean softmax(z_m)` generally differs from `softmax(mean z_m)`, and the shared Jacobians themselves depend on private head offsets.

The boundary algebra makes both channels explicit. For one node set `r_m=c+delta_m`, `e_m=p_m-y`, `e0=p0-y`, and `u0=h*c`. Omitting the common node average, the shared-head gradient is

\[
G_W=\mathbb E_m[e_m(h*r_m)^T],
\quad G_W-G_{W,0}=(\mathbb E e_m-e0)u0^T+
\mathbb E[(e_m-e0)(h*\delta_m)^T].
\]

The upstream feature gradient is

\[
g_h= c*(W^T\mathbb E e_m)+
\mathbb E[\delta_m*(W^T(e_m-e0))].
\]

Here `*` is elementwise multiplication. The residual-offset correlations survive antithetic averaging. Locally they are second order and depend on orientation through `H_n`, `A_n`, and `C_theta`; finite responses also include higher moments. The private final-factor gradient is `(1/M) A_n^T e_m`, before node averaging. Other live private factors also receive differing gradients through the model.

This is a generic shared-weight ensemble effect, not a graph theorem. Coupled Adam then applies its coordinate-dependent moment map and weight decay. The same transported state across arms makes comparison controlled, but does not make the update identical. Euclidean gradient orthogonality of offsets does not imply orthogonality to a momentum or preconditioned Adam step. The private `1/4` loss normalization is correctly present; it is not by itself a claim that Adam's effective step is one quarter of a single-model step.

The selector evaluates the complete actual update and pooled TRAIN CE, discards the trial, and returns the pretrial initialization, optimizer and RNG. Consequently it selects a finite local response. It does not transfer the winning trial's optimizer state, prove repeated-step descent, or establish later predictive quality. Dropout-off selection and native continuation can also have different learning maps.

## 5. Precisely graph-specific content and failure cases

The graph bank has columns `A^T M H_b(S) E r`, with TRAIN residual injection `E`, TRAIN remasking `M`, and the borrowed cubic Bernstein operators

\[
H_b(S)=\binom3b((I+S)/2)^{3-b}((I-S)/2)^b,\qquad\sum_bH_b=I.
\]

Thus its four columns sum to the common head gradient. Centering and projecting them yields a span of rank at most three. This is a graph-conditioned, supervised first-order VJP restriction. It is not an eigenspace of the head CE Hessian, a GGN/Laplace posterior, or a negative-curvature estimator. BernNet's filters and C&S's supervised residual transport are already attributed in the saved index. The shallow-ensemble record `2602.15747v1` already retains centered curvature-aware head samples, full-model continuation, and useful covariance orientation in regression/UQ; it supplies no graph-classification accuracy guarantee.

Concrete limits and collapse risks follow without an experiment:

- If the residual is confined to one eigenspace of `S`, every filtered residual is a scalar multiple of it. The VJPs are collinear with the common gradient, and projection collapses the bank. More generally, graph filters or the head pullback can leave rank below three. The source correctly retains a null arm rather than redrawing.
- Positive `D` guarantees initial probabilistic disagreement somewhere, not different correct class decisions. All four can retain the same incorrect argmax. Equal mean logits guarantee exactly equal initial pooled quality.
- In binary classification with zero initial logit margin, antithetic margins satisfy `sigmoid(t)+sigmoid(-t)=1`. Hence the mean member probabilities equal the common probabilities for every node, even with positive matched `D`. If only the affine private `R` slice is updated by plain gradient descent and every other coordinate is fixed, the mean head gradient and next pooled logits are identical to the common arm. Updates to other coordinates or Adam may break this special equality; the example disproves a universal “spread necessarily changes the useful mean-head step” assertion, without claiming identical full-model trials.
- If only the affine private `R` slice is trained, the local private contrasts under plain gradient descent contract along positive head Hessian modes for a sufficiently small step: `delta+ approximately (I-eta H_head/M) delta`. Null modes persist. No diversity-maintaining term counters this. Full nonlinear continuation, dropout or Adam can alter this behavior, so collapse is a risk rather than an established outcome.
- Graph spectral separation need not align with errors relevant to generalization. The topology-permuted control preserves spectrum while breaking alignment; selected random and permuted spans already test graph restriction against generic finite selection. TRAIN-trial selection can reward a transient or overfit response. Native validation-selected later NLL is itself selection-associated, as the prospective freeze states.

## 6. One proposed discriminating diagnostic, not an adopted change

**Exact one-step shared/private update attribution on the existing discarded trials.** Before discarding each trial, partition named coordinates into shared weights/body `phi` and private factors/biases `a`. At the same dropout-off evaluation state, evaluate two hybrid mean logits, `bar z(phi+,a0)` and `bar z(phi0,a+)`, in addition to the already evaluated initial and complete post-step logits. Define

\[
\Delta_s=\bar z(\phi^+,a^0)-\bar z_0,\quad
\Delta_p=\bar z(\phi^0,a^+)-\bar z_0,\quad
\Delta_{int}=\bar z(\phi^+,a^+)-\bar z_0-\Delta_s-\Delta_p.
\]

These sum exactly to the full mean-logit displacement; they need no linearization and retain a separate interaction term. Compare their class-centered responses and pooled-CE effects with the common, random, and permuted trials already in the protocol. This determines whether a local trial win arises through shared learning, private mean changes, or their interaction. It does not convert that win into a graph-specific or predictive guarantee. The diagnostic would require a prospective source amendment and two additional hybrid forwards per trial, fully charged; nothing is executed, frozen anew, or admitted by this report.

**Conclusion.** A defensible interpretation is that graph-error filtering proposes a restricted set of distinct finite head initializations, and a local native learning map chooses among them. Generic CE/Bregman algebra explains their initial spread cost and permits orientation-dependent learning responses. The sources do not imply that graph conditioning makes those responses useful or persistent; that remains the purpose of the retained comparisons.
