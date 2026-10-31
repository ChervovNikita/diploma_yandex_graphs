# Local forward and adjoint decomposition

Analytical derivation only. `P` and realized `Q` are fixed linear operators on the node axis. Their coefficients are independent of the differentiable member states. No numerical experiment or model execution is used here.

Let `Z_m` be the full normalized state of member `m`, `M` the member count, and `B = (1/M) sum_m Z_m`. Graph aggregation in the candidate is

```
G_m = P B + Q (Z_m - B).
```

## Forward identity

Because the same realized `Q` is used by every member,

```
mean_m G_m = P B.
```

This holds in exact arithmetic for every realized sample, not only in expectation. If `E[Q | current Z] = P`, each `G_m` is conditionally unbiased for `P Z_m` at this one aggregation. Neither identity implies that the next nonlinear state, pooled prediction, loss or optimizer step equals its exact-graph counterpart. Finite-precision reductions must be audited with declared tolerances.

## Candidate adjoint

For arbitrary upstream gradients `U_m = dL/dG_m`, write `Ubar = mean_m U_m`. Differentiating both mean appearances gives

```
dZ_j = Q^T U_j + (P^T - Q^T) (sum_m U_m)/M
     = P^T Ubar + Q^T (U_j - Ubar).
```

Thus the local gradient's common member component uses the full operator and its contrast components use the sampled operator. In particular,

```
mean_j dZ_j = P^T Ubar.
```

Equivalently, if `J = 11^T/M` is the projector onto the member mean, the local aggregation operator is `J ⊗ P + (I-J) ⊗ Q`, using a member-first vectorization convention; its adjoint is `J ⊗ P^T + (I-J) ⊗ Q^T`. This is standard projector algebra, not an originality claim. A shared parameterization is not needed for the local identity, but the declared pilot uses shared SAGE/FFN maps and separate full-width member states.

The upstream `U_m` usually depends on `Q` through nonlinear computation. Consequently `E[Q^T U_m] = P^T E[U_m]` cannot be assumed. The mean identity is local to the `Z` variables. If `Z_m = LayerNorm(H_m)`, different LayerNorm Jacobians can also prevent it from holding for the gradients of the raw `H_m`. Nothing here establishes unbiased shared-weight gradients, exact loss minimization, convergence to the exact model's optimum or a diminishing sampling error.

## Detached-mean control

Define `Bstop = stop_gradient(mean_m Z_m)` and use **that same tensor in both branches**:

```
Gstop_m = P Bstop + Q (Z_m - Bstop).
```

Its values agree with `G_m` at a fixed `Z` and `Q`, but its derivative is

```
dZstop_j = Q^T U_j,
mean_j dZstop_j = Q^T Ubar.
```

No sampled states, architecture, initialization, loss, current baseline or forward estimator have changed. Parameter trajectories can diverge after their first update because their local derivatives differ. A numerical equivalence test must freeze weights and RNG states; equality after independently training the arms is neither required nor expected.

Detaching only the exact `P B` branch while retaining a differentiable `B` in the sampled residual would give `Q^T(U_j-Ubar)`. It would suppress the common gradient entirely and is a different, unsuitable control.

This ablation is informed by the distinction between a historical/control reference and a current differentiable reference in Chen et al. (2018), but is not Chen's full algorithm. Their reference is historical, they update node histories, and their dropout-aware procedure includes additional mean/fluctuation terms. SGCN++ likewise corrects temporal layerwise gradients, which are absent here.

## Cost distinction

For `M=4`, width `D`, full operator with `E` effective entries and quarter-edge `Q`, both arms' forward sparse edge-channel work is

```
E D + (E/4)(4 D) = 2 E D.
```

The differentiable candidate's adjoint adds a full-width-`D` `P^T` multiplication and a quarter-edge-width-`4D` `Q^T` multiplication: another `2 E D`. The detached control uses only `Q^T`, costing `E D` in this leading sparse arithmetic. Its forward-plus-backward sparse arithmetic is therefore `3 E D` versus the candidate's `4 E D`, before all dense computations and packing/normalization/reduction overheads. There is no complete training-cost equality.

Count transpose storage, any use of a custom SpMM adjoint, actual allocation, CPU/GPU preparation and all other work separately in the engineering audit. Equal forward arithmetic does not establish equal latency or peak training memory. At deployment the stop-gradient instruction has no effect on predictions or the fixed-state operator workload.

## What a passed control would support

If the fixed paired study shows better quality than the detached control while satisfying the original cost and exact-model quality gates, it would support using a differentiable current mean for this architecture and these complete tasks. It would distinguish that effect from the candidate's forward control-variate reduction. It would not prove originality of the decomposition, superiority over temporal variance-reduction methods, or generalization to attention/nonlinear-edge models.
