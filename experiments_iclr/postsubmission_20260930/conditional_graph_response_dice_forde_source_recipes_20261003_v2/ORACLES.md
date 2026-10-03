# Prospective equivalence oracles

None was executed in this source packet. Separate numerical authorization, runtime binding and root adoption of tolerances are required. Do not test a silently repaired predecessor recipe. The pinned author file at commit `e8f9d7418be43064ffdc247ba7e62a61149da542` remains the column-median force authority.

## O1: clustered M4 witness

With one target and normalized vectors `s1=s2=s3=(1,0)`, `s4=(0,1)`,

```
D = [[0,0,0,2],
     [0,0,0,2],
     [0,0,0,2],
     [2,2,2,0]]
```

Column medians over live i are `[0,0,0,2]`; the full-matrix median is 0. Author `h[j]=median_i(D[:,j])/log4+1e-12`. Clustered live members have cross kernels to reference 4 near 1/4 and a nonzero tangent force away from e2; the global-median variant suppresses every cross term. The ideal unit vectors are exact in float32 normalization with squared-norm additive 1e-24. Self and duplicate-direction terms retain kernel 1 and zero distance gradient.

## O2: asymmetric nondegenerate M4 witness

Use `s1=(1,0),s2=(0,1),s3=(-1,0),s4=(3/5,4/5)` in ideal unit-sphere arithmetic. Exact squared distances are

```
D = [[0,   2,   4,    4/5 ],
     [2,   0,   2,    2/5 ],
     [4,   2,   0,   16/5 ],
     [4/5, 2/5,16/5,  0   ]]
```

The author column medians are `[7/5,6/5,13/5,3/5]`; the full-matrix median is 7/5. The bandwidth shape must be `[4,1]`, and `K12 != K21`. Float32 vectors require normal rounding tolerance for 3/5 and 4/5. This fixture distinguishes axes and reference-dependent broadcasting even if a global median coincides with one column.

## O3: force, stop-gradients and self denominator

Let `S_i=sum_(j,b) exp(-D_ijb/h_jb)` and keep reference normalized vectors and h fixed. For `R_sum=(1/B)sum_i(log S_i-log B)`,

\[
\partial_{s_{ib}}R_{\rm sum}
=-\frac{2}{B S_i}\sum_j e^{-D_{ijb}/h_{jb}}
\frac{s_{ib}-s_{jb}^{\rm ref}}{h_{jb}}.
\]

Gradient descent subtracts this derivative, giving repulsion. Chain it through the epsilon-regularized input-gradient normalization. Compare analytic, pinned author synthetic formula and prepared Gram implementation. References and h must receive no gradient; self entries contribute1 to S but zero numerator. Removing self, doubling both argument derivatives, or choosing attraction must fail the oracle.

## O4: target-average placement and extra1/B

Use B=2 targets with differing pair distances to show `log(sum_j mean_b kernel)` differs from `mean_b log(sum_j kernel)`. Duplicate the target batch algebraically: K stays constant, but the source `R_sum` halves because its explicit1/B changes; mean CE stays constant. Mean-member conversion also divides both objective parts by M; it is used only for the separately adopted matched protocol. Do not change B for memory reasons while claiming unchanged force strength. A final partial minibatch uses its actual B, matching the author code's extracted batch size.

## O5: full-feature pullback versus Gram, including mixed private derivatives

Use a small non-symmetric fixed float32 COO P, max power2, two selected targets, F=2 and a smooth node-local nonlinear predictor with private parameters. Include duplicate COO records and source-consistent row orientation; do not coalesce the oracle into a different operator. For example, a rational matrix with rows `(1/2,1/2,0)`, `(0,1/2,1/2)`, `(1/3,0,2/3)` gives overlapping target supports.

1. Build `T_k=P^k X` with live X. For each member and target independently, compute `g_direct=grad_X raw_logit_y(v)` with `create_graph=True`.
2. From detached token values, compute selected token-row q with a live private graph, using a summed selected-row score VJP. Check target-row logits equal complete native logits; perturb one target token row and verify other target outputs are unaffected.
3. Compute `a[b,k,:]=row_v(P^k)` and G from the same unaltered P. Check direct `sum_k (P^k)^T g_Tk` gradients and inner products against q/G contraction.
4. Check normalized distances, author h[j,b], log-KDE values and **mixed private force gradients** against direct full-X evaluation. Keep exact target IDs/ordering and float32 dtype across paths. Run native PolyFormer/BE row-locality and mixed-autograd checks after its composition is separately authorized.
5. For finite differences, freeze the normalized reference particle values and h **from the base point across perturbations**, and perturb only the live particle/private parameter. Detaching in autograd while recomputing reference/h values in every finite-difference call is not the same derivative. The prepared `reference_coefficients` and `frozen_bandwidth` inputs support this oracle.
6. Include zero and small gradients, duplicate directions, identity P, non-symmetric P and one ReLU-away-from-boundary native case. Never mask nonfinite/negative Gram quadratic values with an undisclosed clamp. If float32 contraction cannot satisfy the adopted direct-oracle tolerance, retain exact direct pullback or stop; do not project/truncate the metric.

Root must freeze absolute/relative tolerances before numerical checks. Record maximum error and norm regimes rather than asserting bitwise Gram equality. Mathematical equivalence is exact; arithmetic equivalence is a runtime gate.

## O6: DICE narrow source/stochastic/state checks

Use synthetic labels with a two-ID class and an uneven larger class. Ensure negative partners always share class, differ from anchor, and preserve the target class distribution. Two negatives can use the same allowed ID but need independent noisy replicas. Check shared positive banks across pairs, exact first-auxiliary reuse, later bank redraws and all partner means frozen at base. Verify clipped joint-only score, unweighted BCE ratio2, and sum-member Eq6 factor `delta/(M-1)`; the matched mean-member factor is `delta/[M(M-1)]`.

Check no auxiliary gradient reaches predictor parameters; D remains frozen only during predictor score; successful-cycle gradients are None and stable flags are explicit. A source-style exception must stop without a partial-cycle checkpoint. For the separate guarded protocol, add accept/partial/reject/zero/each-auxiliary-exception cases with full exact joint restoration or commitment, distinguishing rejected base gradients from successful cleared gradients.
