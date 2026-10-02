# Local objective and limits of a graph-error refresh

Prospective mathematics only. No implementation, model, array or outcome computation is provided. This analysis extends the retained source-bound initializer interpretation and does not change that sealed packet. All projections below are elementary finite-dimensional operations, not new theory.

## 1. Define the state before defining the direction

Freeze shared dense weights, buffers, dropout and private parameters outside the admitted `d`-coordinate factor slice. The actual route closure is `z_m(theta_m)`, with TRAIN Jacobian `J_m`, `m=1,...,M`, `M=4`. It includes each route's own already trained outside-slice factors and biases. Averaging only the active slice does not create one common predictor when those outside parameters differ. The existing identity-only warm closure is therefore not a valid drop-in refresh API.

Let `Gamma` class-center every TRAIN logit row. All following Jacobians/cotangents can be understood in that gauge. Let

`r_m = (softmax(z_m)-Y)/n`, `g_m = J_m^T r_m`,

`z_pool = mean_m z_m`, `r_pool = (softmax(z_pool)-Y)/n`,

`k_m = J_m^T r_pool`, `gbar = mean_m g_m`, `kbar = mean_m k_m`.

The member gradient `g_m` and pooled-signal gradient `k_m` are different. With `E` injecting TRAIN rows into the released graph and `K_b=E^T H_b E`, use the initializer's fixed cubic Bernstein bands, `sum_b K_b=I`. Assign one fixed band to each route and choose the following **explicit** refreshed signal:

`q_m = (K_m-I/M) r_pool`, `h_m = J_m^T q_m`.

`sum_m q_m=0`. The signal is common pooled error filtered into route contrasts, then TRAIN-remasked. Using four separate `r_m` instead would define another procedure and generally lose this cotangent partition. None of these cotangents is differentiated through during the pulse.

## 2. Why the original simple projection breaks

For a common `g`, route centering and per-route projection onto `g`-orthogonal vectors commute. For distinct `g_m`, projecting each centered row onto its own `g_m` generally destroys the route mean; centering again generally destroys the separate orthogonalities. A fixed two-pass repetition does not certify the joint constraints. Alternating projections may converge under suitable conditions, but this is not the initializer's exact algebra and would require tolerances/iterations and cost to be declared.

A direct small Gram construction gives the exact joint projection. Stack route parameter vectors into `R^(Md)` and let

`C = (I_M - 11^T/M) tensor I_d`.

For each member define `v_m = C(e_m tensor g_m)`, and let `V` have these columns. With `h=stack_m h_m`, set

`a = C h - V(V^T V)^+ V^T C h`.   (1)

The Gram entries are

`(V^T V)_(m,l) = 1[m=l]||g_m||^2 - (g_m^T g_l)/M`.

Thus only a 4×4 pseudoinverse and `O(M^2 d)` vector operations are needed. It projects onto `sum_m u_m=0` and `g_m^T u_m=0` for every route. Rank degeneracy is expected when gradients agree; an ordinary inverse is incorrect. A prospective numerical contract must predeclare rank tolerance, residual checks and failure/zero-signal handling. No such contract is implemented here.

If extra first-order **pooled CE neutrality** is desired, append the column `C stack_m k_m` to `V` and use the same formula with a 5×5 Gram. This enforces `sum_m k_m^T u_m=0`. A factor `1/M` in that column changes only its scaling, not the exact constraint subspace. In finite arithmetic this scaling affects conditioning and must be fixed. A full pooled-logit constraint `sum_m J_m u_m=0` would instead involve many output dimensions; the five-column construction supplies only one scalar pooled-loss constraint.

## 3. What objective this exactly solves locally

For `a!=0` and joint radius `rho`,

`u* = -rho a/||a||`

solves

`min_u sum_m q_m^T J_m u_m`

subject to route centering, the chosen gradient-neutrality constraints, and `||u||<=rho`. Its optimum is `-rho||a||` by Cauchy–Schwarz after orthogonal projection. This is a linear prediction objective in the Euclidean private-factor metric. It is not a global classification loss or a maximum-disagreement objective. The output response is

`Delta = -rho/||a|| * J_block Pi J_block^T q`,

where `J_block=diag(J_1,...,J_M)` and `Pi` is the joint projector in (1). The induced block kernel is positive semidefinite, so this *same frozen q* decreases its own linear functional. That statement does not imply a decrease of unfiltered CE or useful held-out errors.

A simple common private direction is `v=-gbar` in every route. Choose `rho=0.5||gbar||`; this reduces to the initializer's radius at a common state. Add `u_m` to `v` for each candidate. The mean-member CE derivative is

`D L_member[v+u] = -||gbar||^2`,

because each `g_m^T u_m=0`. Each individual member's derivative is instead `-g_m^T gbar`, which can be positive. There is no guarantee that all members can descend under a common shift.

With the optional pooled constraint, the pooled CE derivative is

`D L_pool[v+u] = -kbar^T gbar`.

It need not be negative. Without that constraint, add `(1/M)sum_m k_m^T u_m`. A training-only finite safeguard must therefore check the pool against **its own previous baseline**, independently of the member mean. One could design a common multiobjective descent direction using `gbar` and `kbar`; doing so is an additional declared operation with possible opposing-gradient degeneracy, not an inherited initializer guarantee. The representative screen retains the simple common direction and records paired failures.

When all route functions, Jacobians and gradients agree, the pooled constraint becomes redundant, `q` sums to zero, and (1) reduces to the centered common-gradient-orthogonal initializer construction. This is the limited exact bridge between initialization and the proposed refresh.

## 4. Parameter centering is not prediction centering

Although `sum_m u_m=0`, generally

`mean_m J_m u_m != 0`.

For any reference Jacobian `J_ref`,

`||mean_m J_m u_m|| <= max_m ||J_m-J_ref|| * ||u||/sqrt(M)`.   (2)

This gives a first-order pooled-logit approximation only if actual Jacobian variation is small. A Hessian-Lipschitz argument could bound `||J_m-J_ref||` from full private-context differences under a smooth common parameterization, but the active slice alone does not control those differences. Neither a fixed low-dimensional slice nor a prior cold/warm AD check certifies that bound. The retained native models contain ReLUs.

If logit Hessians have bounds `B_m` along the relevant frozen-state segments, Taylor remainders satisfy

`||R_m|| <= (alpha^2 B_m/2)||v+u_m||^2`.

The pooled extra logits can therefore contain an `O(alpha)` Jacobian-mismatch term plus the bounded `O(alpha^2)` terms. A pooled-gradient constraint makes the *pooled CE directional derivative* of the extra step zero; it does not remove all extra first-order pooled logit or decision changes.

Jensen says `CE(mean z_m,Y) <= mean_m CE(z_m,Y)` at each fixed state. After divergence, the previous pool can already lie strictly below the member mean. Reducing every member or the mean does not establish that the pool improves from that lower previous CE.

## 5. Why no general fixed potential is inferred

Even before dynamic projections, a fixed symmetric graph filter applied to a CE cotangent is not generally a gradient field of a scalar objective in logits. For `r(z)=grad_z CE(z)`,

`D_z[K r(z)] = K H_CE(z)`.

The CE Hessian varies by node/class; `K H_CE` is generally nonsymmetric because the filter couples nodes. Unless the commutator vanishes on the relevant space, it is not the Hessian of a scalar potential. In squared-error regression `H=I` this obstruction disappears for fixed symmetric `K`, illustrating why SEA's regression target does not transfer unchanged. The gradient of `0.5 r^T K r` would include `H_CE K r`, not just `K r`. Restriction through a particular Jacobian can create exceptional cases; no general conservative-field claim is made.

PSD alone also gives no CE-descent sign for a neural kernel/filter product. For example, `B=diag(1,0)` and `K=[[1,2],[2,4]]/5` are both PSD, with `K` a contraction, yet for `r=(1,-1)`, `r^T B K r=-1/5`. This is elementary algebra showing why PSD/noncommutation arguments cannot supply a universal descent conclusion; it is not a graph/model experiment.

In the refresh, the residual, Jacobians, gradient constraints, normalization radius and projector change with the trained state. Thus the honest description is **repeated solutions of detached local linear prediction problems with finite quality safeguards**, or an intervention/approximation algorithm. It is not descent of an established time-invariant global CE objective. Calling it graph-gradient preconditioning does not create that missing proof.

## 6. The finite target must compare at the same common step

At an identical fork state and the same accepted alpha, compare a graph candidate to the common-only candidate:

`Psi_extra(alpha) = sum_m q_m^T Gamma[z_m(theta_m+alpha(v+u_m)) - z_m(theta_m+alpha v)]`.

Its first coefficient is `-rho||a||`. With smooth Hessian bounds as above,

`|Psi_extra(alpha)+alpha rho||a||| <= (alpha^2/2)sum_m ||q_m|| B_m (||v+u_m||^2+||v||^2)`.   (3)

Using only the candidate's change from the original state would also include `sum_m q_m^T J_m v`, which need not cancel when Jacobians differ. This is why the shared-alpha common comparison matters. No Hessian bound is measured here, so finite realization must be reported, not presumed from the derivative.

The existing common-Jacobian random Gram-coloring control is not directly reusable: mixing route directions can break `g_m`-orthogonality and the pooled constraint, while the `J_m` differ. A joint-projected fixed random control is possible at matched parameter radius, but its prediction Gram is not automatically matched. The proposed topology-permuted pulse is a representative complete-operation falsifier, with that attribution limit explicit.

## 7. What would stop the follow-up

Do not add persistent refreshes simply because ordinary CE does not enforce graph orientation. Stop if the already declared post-closure diagnostic supplies no useful finite graph-error orientation to retain; if the new actual-state closure/joint projection cannot be qualified without substantial extra machinery; if one paired pulse frequently fails the finite pool/member-mean gate; if its signed finite effect contradicts (3)'s intended direction; or if later pooled quality/cost does not beat the representative controls. No parameter-distance or spread statistic can substitute for those questions.
