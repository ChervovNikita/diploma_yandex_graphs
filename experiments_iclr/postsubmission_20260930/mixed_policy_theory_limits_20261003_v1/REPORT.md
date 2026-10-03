# First-order pooled-loss limits of shared-pool/private-own training

3 October 2026. Independent mathematical critique of the frozen M4 policy. The useful unresolved question is whether its simultaneous step improves the pooled CE locally. The exact answer for raw SGD depends on one gradient inner product; the policy alone does not make that quantity positive. An exact affine, binary, two-example M4 witness below has nonzero shared and private gradients, improves every member's own CE to first order, and strictly increases pooled CE along every positive step on its specified SGD ray. This establishes a limit of the generic update rule, not a prediction for the frozen HGT learner or a requested acceptance verdict.

## 1. Assumptions and the exact first-order criterion

Let `theta=(W,phi_1,...,phi_4)`, with each private block affecting only its own member. Use the same fixed examples, weights `w_i>=0`, `sum_i w_i=1`, and fixed forward randomness for both losses. All gradients are evaluated at one pre-update state. CE is applied to mean raw logits, and own CE retains its original `1/4` member normalization. Assume differentiability locally; the finite-step bound below additionally requires a local Lipschitz gradient of pooled loss on the actual step segment. This is a conditional TRAIN-loss statement. It supplies no validation, generalization, stochastic-expectation, or full-trajectory guarantee.

Write

    a = grad_W L_pool,
    b = grad_phi L_pool,
    h = grad_phi L_own = b + d,
    d = grad_phi D,
    V = (a,h).

The identity `L_own=L_pool+D`, `D=mean_m KL(p_pool||p_m)`, is inherited from the scout and is not a new claim. The candidate supplies `V` to the optimizer. For simultaneous raw SGD with one positive scalar step `eta`,

    theta(eta) = theta - eta V,
    L_pool(theta(eta)) - L_pool(theta)
        = -eta A + o(eta),
    A = ||a||^2 + <b,h>
      = ||a||^2 + ||b||^2 + <b,d>.

Thus `A>0` is exactly strict pooled descent to first order; `A<0` is strict ascent; `A=0` leaves the leading nonzero term unresolved. This is the standard descent-direction inner-product criterion from numerical optimization, applied to the already proposed field. It does not require or repeat the scout's generic cross-partial/nonconservativity proof.

If `grad L_pool` is `L`-Lipschitz along the step segment, the ordinary descent lemma gives

    Delta L_pool <= -eta A + (L eta^2 / 2) ||V||^2.

For `A>0`, `V!=0`, `L>0`, and a segment on which that bound holds, any `0<eta<2A/(L||V||^2)` is sufficient for decrease. Differentiability alone still gives decrease for all sufficiently small positive steps when `A>0`, without an explicit step bound. A positive first-order coefficient does not guarantee decrease at the frozen native learning rate.

A sufficient alignment condition is `<b_m,h_m>>=0` for every private block and at least one strictly positive contribution among `||a||^2` and those products. More permissively, negative private alignment is allowed if its total magnitude is below the shared squared-gradient contribution plus positive private contributions. Cauchy--Schwarz gives the conservative sufficient bound

    ||b|| ||d|| < ||a||^2 + ||b||^2  =>  A>0.

These are mathematical conditions, not added checks, tuning rules, projections, or experimental gates. If fixed SGD parameter groups had distinct positive rates, the corresponding first-order expression would be the sum of the rate-weighted block products. No such change is proposed for the frozen learner.

## 2. Why logit alignment does not establish parameter alignment

For example `i`, let `s*_i=p_pool,i-e_yi`, `s_mi=p_mi-e_yi`, and `J_mi=partial z_mi/partial phi_m`. Then

    b_m = (1/4) sum_i w_i J_mi^T s*_i,
    h_m = (1/4) sum_i w_i J_mi^T s_mi,
    <b_m,h_m> = (1/16) sum_ij w_i w_j
                   (s*_i)^T J_mi J_mj^T s_mj.

At the **unrestricted logit level**, each CE residual has a negative target coordinate and positive nontarget coordinates for finite logits. Hence `(s*_i)^T s_mi>0` for every example with at least two classes. Treating every example/member logit vector as an independent parameter therefore gives positive private alignment. A real parameterization applies Jacobians and ties parameters across examples, so that elementary observation is insufficient.

For **one binary example**, a stronger statement does survive an arbitrary differentiable private parameterization. Let `t_m` be its class-1 minus class-0 margin, `y in {0,1}`, `u_m=(2y-1)t_m`, `ubar=mean_m u_m`, `f(u)=log(1+exp(-u))`, and `v_m=grad_phi_m u_m`. Put `r_m=sigmoid(-u_m)>0` and `r*=sigmoid(-ubar)>0`. Then

    b_m = -(r*/4) v_m,
    h_m = -(r_m/4) v_m = (r_m/r*) b_m,
    <b_m,h_m> = (r* r_m/16) ||v_m||^2 >= 0.

Private own and pooled gradients cannot oppose in this single binary-example setting. The whole mixed SGD direction is pooled descent to first order whenever the pooled gradient is nonzero. This statement is local, uses finite logits and the same example/forward state, and does not extend to arbitrary multiclass Jacobians or shared parameters across multiple examples.

For **multiple binary examples**, let `v_mi=grad_phi_m u_mi`, `r_mi=sigmoid(-u_mi)`, and `r*_i=sigmoid(-ubar_i)`. The private alignment becomes

    <b_m,h_m> = (1/16) sum_ij w_i w_j r*_i r_mj <v_mi,v_mj>.

Although all residual magnitudes are positive, the cross-example Jacobian products can be negative. The two losses weight those Jacobian directions differently and their sums can point in opposite directions. Sufficient conditions include all `<v_mi,v_mj>>=0` within each member, or a positive residual ratio `r_mi/r*_i` that is constant across that member's examples. Neither is implied by shared/private factorization. Independent per-example logit coordinates have no adverse off-example interactions; shared private factors generally do.

For **one multiclass example**, `J_m J_m^T` is positive semidefinite, but that does not force the bilinear product between two different residuals to be positive. For instance, with target class 1,

    p_pool=(1/2,3/8,1/8), p_m=(1/2,1/8,3/8),
    J_m=(0,1,-1)^T,
    J_m^T s*=1/4, J_m^T s_m=-1/4.

The raw residuals align, yet this scalar private parameter has opposite own and pooled gradients. These distributions are feasible under M4 mean-logit pooling: set `z_1=log p_m` coordinatewise and `z_2=z_3=z_4=(4 log p_pool-log p_m)/3` at the state. This is a local geometric illustration, not a separate predictive experiment. The binary one-example collinearity claim must not be generalized to the frozen three/four-class tasks.

## 3. Exact simultaneous-SGD witness with M4 and all blocks active

This affine structural witness uses two hypothetical examples, both with target class 1, one scalar shared parameter `W`, and one scalar private parameter per member. It is not fitted data, a fixture implementation, or a demonstrated realization of the native HGT architecture. Its purpose is to falsify a universal pooled-descent claim based only on the split and CE geometry.

Let `ell=log 3`, `r=sqrt 3`, and binary logits be `z_mi=(t_mi,0)`. The loss of a margin is `f(t)=log(1+exp(-t))`. Define

    t_11 = -ell + W/16 + phi_1,
    t_12 = +ell + W/16 - phi_1,
    t_m1 = +ell + W/16 + phi_m/16,  m=2,3,4,
    t_m2 = -ell + W/16 + phi_m/16,  m=2,3,4.

Evaluate at `W=phi_1=...=phi_4=0`. The member-1 probabilities across the examples are `(1/4,3/4)`; those of each other member are `(3/4,1/4)`. Pooled margins are `(ell/2,-ell/2)` and pooled probabilities are `(r/(1+r),1/(1+r))`. Put `c=(r-1)/(r+1)=2-r>0`. Including both the example mean and original member mean,

    a = -1/32,
    b_1 = c/8,        h_1 = -1/16,
    b_2=b_3=b_4 = -1/128,
    h_2=h_3=h_4 = -1/128.

The first private block opposes pooled descent despite correcting its own examples in aggregate. All other private blocks and the shared block have strictly positive pooled alignment. Nevertheless,

    A = 1/1024 - c/128 + 3/16384
      = (128 sqrt(3)-237)/16384 < 0,
    d/deta L_pool(theta-eta V)|eta=0
      = (237-128 sqrt(3))/16384 > 0.

The inequality is exact (`237^2 > 128^2 * 3`), not a floating-point threshold. The step is simultaneous:

    W(eta)=eta/32,
    phi_1(eta)=eta/16,
    phi_m(eta)=eta/128, m=2,3,4.

Therefore the pooled margins along that ray are

    tbar_1(eta)=ell/2 + 147 eta/8192,
    tbar_2(eta)=-ell/2 - 109 eta/8192.

The exact pooled loss is

    F(eta) = [f(ell/2+147 eta/8192)
              + f(-ell/2-109 eta/8192)] / 2.

Each affine slope is nonzero and `f''>0` at every finite argument, so `F` is strictly convex in `eta`. Since `F'(0)>0`, `F(eta)>F(0)` for every `eta>0` on this unbounded affine witness. The committee's already weaker second-example margin worsens enough to dominate improvement on the first example.

In contrast, at the initial state `grad_W L_own=a`, so the supplied field happens to equal `grad L_own` there. Thus

    d/deta L_own(theta-eta V)|0 = -83/16384 < 0.

Indeed, if `L_m` is a member's mean own CE,

    d/deta L_1(theta-eta V)|0 = -17/1024,
    d/deta L_m(theta-eta V)|0 = -5/4096, m=2,3,4.

Every member improves to first order while the served pool worsens. This witness does not establish that the candidate is worse than own/own, pool/pool, or the reverse policy on the native learner. The accidental equality of shared own/pool gradients at this state makes it an especially plain limit on translating member competence into pooled progress, without attributing the effect to nonconservativity.

`EXACT_WITNESS.json` records the expressions and normalization without an executable fixture or numerical pilot.

## 4. Why the SGD condition does not guarantee native AdamW improvement

For an AdamW group at time `t`, the supplied current gradient `g=V` enters

    m_t = beta1_t m_(t-1) + (1-beta1_t) g,
    v_t = beta2_t v_(t-1) + (1-beta2_t) g^2,
    q_t = mhat_t / (sqrt(vhat_t)+epsilon) + lambda theta,
    theta_plus = theta - alpha_t q_t,

with elementwise square/division and the optimizer's actual bias corrections, group state, and scheduler timing. The relevant first-order coefficient for that displacement is `<grad L_pool,q_t>` (summed with each group's learning rate), not `A=<grad L_pool,V>`.

Historical first moments can point against the current pooled gradient. Adaptive coordinate weights can emphasize negative private coordinate products even if their unweighted sum is positive. Decoupled weight decay adds a displacement whose pooled-loss inner product has no fixed sign. Already on a zero-moment first step, bias-corrected Adam's gradient part is `g/(|g|+epsilon)`, rather than the SGD field. Thus taking a small `alpha_t` does not recover the Euclidean alignment coefficient; it only scales the direction AdamW actually produces.

If there were no momentum or decay, the one-example binary positive-collinearity result would survive positive diagonal scaling of each current-gradient block. The native learner has optimizer state and decay, so that restricted observation is not its guarantee. Finite curvature still matters after actual-step alignment is established. OneCycle sets the step and possibly momentum schedule; it does not enforce a pooled-loss descent lemma or line search. No optimizer modification, alternate update order, new step multiplier, or guard is proposed here.

## 5. Attribution, scope, and remaining limits

The descent-direction criterion, Cauchy--Schwarz bound, and smoothness/descent lemma are standard numerical optimization facts (e.g. Nocedal and Wright, *Numerical Optimization*, 2nd ed., 2006). Negative task-gradient inner products as a conflict diagnostic are established in multi-task optimization, e.g. Yu et al., [Gradient Surgery for Multi-Task Learning, 2020](https://arxiv.org/abs/2001.06782); no PCGrad projection is used or proposed. Adam and decoupled AdamW are due to Kingma and Ba, [2014](https://arxiv.org/abs/1412.6980), and Loshchilov and Hutter, [2017](https://arxiv.org/abs/1711.05101). These are background attributions, not new primary-source reads or theorem-novelty claims.

The scout and closest-prior report retain the ancestry of own/pool objectives, block-dependent training, and the previously saved exact 32-step split. This packet adds a focused algebraic limit and sufficient conditions, not a new identity or learner. It neither certifies usefulness nor rejects the frozen complete comparison.

Only the four named immutable inputs were read. The frozen design's SHA256 was verified as `739160acd4ffb0380348f79496e88624db6163f4a3a93902b043d58eaacc18c5`. Paths and metadata printed inside that authorized design were not followed. No current run outputs, graph payloads, actual labels, fitted states, server, Desktop, author code, new primary papers, or canonical records were opened. No subagents, training, numerical implementation/fixture battery, learner amendment, gate amendment, or acceptance verdict was produced. Mathematical target labels in the witness are definitions, not accessed labels.

Architecture-specific Jacobians, actual gradient alignment, optimizer moments, finite native displacement, and predictive outcomes remain unmeasured. A universal guarantee for the abstract split is false; whether the frozen native policy improves served prediction remains the existing empirical utility question. Payloads and provenance are hash-bound in `MANIFEST.json`, with a separate `SHA256SUMS` seal and read-only permissions.
