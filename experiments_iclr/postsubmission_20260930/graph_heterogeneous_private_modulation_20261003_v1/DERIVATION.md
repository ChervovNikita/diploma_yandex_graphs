# Local operator, initialization and observation conditions

These are abstract calculations at fixed factor sites. They are not a theorem about the complete HGT function class or a trained-state check.

## Equivalences and local rank

For row vectors,

```text
h D(a_m alpha_r) W D(b_m beta_r)
= h D(a_m) [D(alpha_r) W D(beta_r)] D(b_m).
```

Products are elementwise. The equality uses commutation of diagonal matrices on each side, without commuting a diagonal through a dense map. It is ordinary BE on a shared typed map. Input/output diagonal residual activation adapters at exactly these sites give the same function.

At a fixed coordinate `(i,j)` of the effective linear map, with the source/type core incorporated into `W_r`, global factors give

```text
E_mr = a_mi b_mj W_rij.
```

This is an outer product over member and relation. Every 2×2 minor is zero. For the proposed output tensor,

```text
E_mr = a_mi (b_mj + c_m q_r u_j) W_rij,
```

the matrix is a sum of two outer products and has rank at most two. A nonzero minor requires independent member vectors `a_:i b_:j` and `a_:i c_:`, independent relation vectors `W_:ij` and `q_: W_:ij`, and `u_j!=0`. Zero map entries or constant `q` on relevant relations can remove the distinction. For parallel canonical relations with the same endpoint types, endpoint-only conditioning likewise cannot distinguish the printed DRSA operator `A_source B_target^T`; native R-GCN/HGT relation indexing is already prior.

The rational witness `[[2,3],[3,5]]` has determinant one. It demonstrates only this coefficient-family distinction. HGT attention can depend on private states, downstream nonlinearities can create other interactions, wider channels can emulate local maps, and a linear mean pool can absorb their average. None of those behaviors is ruled out by a coordinate minor.

## CP gradients

Write `s_mr=b_m+c_m q_r u` and let `G_mr=partial J/partial s_mr` include all downstream/graph derivatives of the chosen symmetric joint objective. Then

```text
partial J/partial c_m = sum_r q_r <u,G_mr>
partial J/partial q_r = sum_m c_m <u,G_mr>
partial J/partial u   = sum_m,r c_m q_r G_mr.
```

These expressions apply to uniform own-member CE or a symmetric pooled objective, with their corresponding `G`. They do not assume an independent source feature once downstream private trajectories differ.

1. **Function-preserving `c=0`.** The new term is zero and the `q,u` gradients are zero. The `c` gradient can be nonzero. If every route/stochastic state and base factor is identical, `G_mr=G_r` for each member and all `c_m` gradients are identical. With identical optimizer states and symmetric updates, the equality of members persists. If ordinary BE/dropout already makes `G_mr` different, the `c` gradients may differ, but useful distinctness is not guaranteed.
2. **All-three-zero factors.** All three product gradients vanish. With zero optimizer moments and no external perturbation, ordinary gradient updates/decay keep the new term dead.
3. **Zero `q`, antithetic `c`.** If `sum_m c_m=0` and routes are identical, `partial J/partial q_r=<u,G_r>sum_m c_m=0`. Both other gradients contain `q` and vanish. In particular the symmetric pooled loss cannot revive this term. Existing route/stochastic differences can break cancellation; they must be attributed separately.
4. **Nonzero products.** Nonzero `c_m,q_r,u_j` make the elementary Jacobians `q_r u_j`, `c_m u_j`, `c_m q_r` nonzero. Loss-gradient contractions can still cancel or be orthogonal. Nonzero parameter derivatives are not a training-progress theorem.

## Prospective initialization

For `M=4`, use `gamma=(-3,-1,1,3)/sqrt(5)`, `c_m=delta gamma_m`, `delta=0.01`; `gamma` has mean zero and RMS one. Use balanced ±1 relation values with at least two distinct values (`R>=2`) and ±1 output-channel values, permuted by the bound generic seed convention. Pair the same ordinary BE factors and shared backbone initialization across all matching arms. The added output perturbation is at most `3 delta/sqrt(5)` in absolute value per coordinate. If a baseline output factor has magnitude one, this cannot flip its sign.

The initialization is not the same global-BE function. It creates a generic member×relation perturbation, without TRAIN-label/graph-cotangent seeding. In a purely linear branch with common responses, its averaged contribution is exactly zero because `sum c=0`; `q,u` pooled gradients can initially cancel even with nonzero products. Nonlinear/private responses need not preserve this cancellation. Ordinary BE factor diversity and independent dropout can already supply private route distinction; report the additional CP contribution separately.

As in any CP parameterization, reciprocal rescaling of `c,q,u` creates gauge freedom and changes gradient conditioning. Fixed initialization/regularization coordinates and optimizer rules are part of the proposed contract, not evidence that one optimization is superior.

## Observation and cost boundary

An effective-map change becomes an observable message change only when the corresponding input and relation response are nonzero and survive summation/attention/nonlinearity. Opposite relation responses may cancel. All-zero inputs, irrelevant relations, zero base entries, saturated downstream paths or identical relation operators can produce nulls. HGT private attention already offers interaction through hidden states; a larger local coefficient family need not enlarge the served pooled family.

The parameter count per layer is `2Md+M+R+d`, versus `2Md` global BE and `Md+MRd` unrestricted relation-output BE at the same sites. Full private recurrence/attention is retained. There is no graph-channel arithmetic reduction. The wider-global-BE comparator is chosen before outcomes from actual source parameter counts; its budget gap is reported, not called exact equality.
