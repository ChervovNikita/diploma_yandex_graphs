# Projection equivalence and boundaries

All functions below are evaluated at the fixed step reference `B=(theta_old,phi_plus)` with stopped support/graph weights. This is symbolic mathematics, not a source or trained-state check.

## Virtual-loss substitution

Set `ell_m=-S_m`, `q_m=grad ell_m=-h_m`, `g=-d0`, `v=-d` and `d=d0+c`. Then

```text
||d-d0||^2 = ||v-g||^2
h_m^T d = (-h_m)^T(-d) = q_m^T v.
```

Thus `min_d .5||d-d0||^2` with every `h_m^T d>=0` is exactly the GEM Euclidean inequality projection. Division of both displacement vectors by a common positive step size preserves this equivalence; it multiplies the objective by a positive constant. Neither the sign substitution nor the virtual-loss construction supplies a new projection method.

The proposal's additional equality gives `a^T d=a^T d0`. Its cap is a rejection condition on the minimum correction. If the minimum correction exceeds the cap, every feasible correction exceeds it. The cap does not require a new QP objective. Zero is always feasible for the inequality cone, but can violate the affine pooled-progress equality.

## Pooled-neutral dual

For nonnull `a`, let `b=-H^T d0` and `P_a=I-aa^T/(a^T a)`. The primal is

```text
min_c .5 c^T c, subject to H^T c>=b, a^T c=0.
```

Its Lagrangian, with `lambda>=0`, is

```text
L(c,lambda,mu) = .5 c^T c - lambda^T(H^T c-b) + mu a^T c.
```

Stationarity and the equality imply

```text
c=H lambda-mu a
mu=(a^T H lambda)/(a^T a)
c=P_a H lambda.
```

Substitution yields

```text
max_lambda>=0 lambda^T b-.5 lambda^T(H^T P_a H)lambda.
```

Feasibility and finite numerical rank/KKT checks remain necessary. A singular Gram does not justify ridge regularization or a changed contract. With at most four inequality columns, the existing fixed active-set enumeration has at most 16 subsets. These are established constrained-projection calculations.

## A-GEM boundary

With `h_ref=sum alpha_m h_m`, the single inequality `h_ref^T d>=0` maps to A-GEM's halfspace. With pooled equality, let `b_ref=-h_ref^T d0` and `r=P_a h_ref`. For `b_ref>0` and `r!=0`, the minimum correction is

```text
c = (b_ref/||r||^2) r.
```

Cauchy–Schwarz establishes minimality in the pooled-neutral subspace. If `b_ref<=0`, zero correction is feasible. If `b_ref>0` and `r=0`, feasibility fails. This equality-constrained extension is not vanilla A-GEM. With equal weights and `h_1=e_2,h_2=-e_2`, the reference gradient is zero despite erosion of one score along `-e_2`; averaging cannot replace every-member protection.

## Ordinary member-CE substitution

For `k_m=grad_theta L_m(B)`, local CE nonincrease is `k_m^T d<=0`. Therefore the proposed comparator uses `H_CE=[-k_1,...,-k_4]` in the identical inequality/equality QP. Its finite checks protect actual ordinary member CE, while the graph arm checks frozen weighted logit margins. Their finite guarded functionals, supports and units differ even when native proposals, moments, pooled guards, window and caps are matched.

## Literal saved-v1 sign ambiguity

A-GEM v1's description of GEM combines negative gradient rows `G` with a plus dual linear term and plus restoration. Taking its printed signs literally for `g=-1,g_1=1,G=-1` gives dual objective `.5v^2+v` for `v>=0`, hence `v=0` and restored gradient `-1`. The primal inner product with `g_1` is `-1`, violating the constraint. Positive gradient rows would instead give `.5v^2-v`, `v=1`, and restoration zero. This bounded scalar witness concerns the saved v1 printing only. The unambiguous primal and independently derived dual above are used for attribution; later versions/code remain uninspected.

`SYMBOLIC_WITNESS.json` and the stdlib verifier check sign mapping, objective/KKT equality, averaged-gradient cancellation and affine infeasibility using rational numbers. No graph-specific benefit follows from these identities.
