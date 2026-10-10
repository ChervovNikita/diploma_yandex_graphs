# Coherent reciprocal fast rows: optimizer and prior triage

**Decision:** the mechanism is clear and different from the closed PubMed normalization control, but it is already a saved, known coordinate/rate-diversity hypothesis. Retain only an inactive coherent-start utility question. No new initialization principle, initial predictive diversity, graph-specific mechanism, implementation or fit admission follows.

## Actual map law first

The bound `portable_internal_be_public_interface_20261007_v2/core/factors.py` computes `linear(x*r_m,W)*s_m + b`. In column notation its effective map is `A_m=D_s W D_r`, with the bias **outside** `D_s`. Thus `r_m=c_m*u_m`, `s_m=v_m/c_m` keeps `A_m=D_v W D_u` and leaves `b` unchanged. At `u=v=1`, every eligible map is exactly the native affine map in real arithmetic. Cancellation is inside the map, before activations, graph attention, norms and residuals; it does not require moving a scaling through attention.

The parent proposal restricts changes to eligible bias-free native graph projections and leaves all other fast factors one. That scope is retained. The inspected outside bias would also cancel correctly, but this note does not expand placement. The input stem must be unit in **both** coherent conditions: retaining its current Rademacher signs would preserve a signed factorized start, not the native initial function. Native slow parameters and route dropout streams remain matched. The deterministic initial predictors coincide; distinct training dropout realizations still supply the existing stochastic diversity.

Power-of-two `c=(1/2,1,2,4)` avoids rounding of these constants and ordinary normal scalar multiplies; it is not a general bitwise equality guarantee for complete kernels, underflow or overflow. This is a starting-function statement, not a guarantee that the native predictor is competent or remains preserved after optimization.

## Exact optimizer interpretation

At function-matched states, with identical realized loss/RNG and no raw-factor penalty or clipping,

```text
g_r = g_u/c,  g_s = c*g_v,  g_W = g_W(canonical),  g_b = g_b(canonical).
```

For ordinary SGD at rate `eta`, canonical variables `u=r/c`, `v=c*s` receive rates `eta/c^2` and `eta*c^2`. This is an exact trajectory identity with mapped momentum, not merely a first-step claim. For Adam with epsilon outside the square root, matching beta values/clocks and transported moments,

```text
m_r=m_u/c, V_r=V_u/c^2;   m_s=c*m_v, V_s=c^2*V_v.
u+ = u - (eta/c) mhat_u/(sqrt(Vhat_u)+c*epsilon).
v+ = v - (eta*c) mhat_v/(sqrt(Vhat_v)+epsilon/c).
```

Shared/native options and histories are unchanged in this matched canonical implementation. Zero fresh moments satisfy the mapping. Continuing from an unrelated optimizer history, changing `c` during training, parameter clipping, raw penalties, unmatched missing-gradient semantics or different dropout streams would not be this identity. AdamW with zero decay uses this Adam law; the inspected SAGE runner explicitly sets `weight_decay=0` and applies mean own CE with one backward/step, without bank-gradient clipping.

| c | SGD input/output rate divided by eta | Adam input/output rate divided by eta | Canonical Adam input/output epsilon divided by epsilon |
|---:|---:|---:|---:|
| 1/2 | 4, 1/4 | 2, 1/2 | 1/2, 2 |
| 1 | 1, 1 | 1, 1 | 1, 1 |
| 2 | 1/4, 4 | 1/2, 2 | 2, 1/2 |
| 4 | 1/16, 16 | 1/4, 4 | 4, 1/4 |

Using ordinary epsilon with only the Adam rates is approximate, requiring the relevant square-root moments to dominate the changed epsilons. No moment evidence establishes that limit here. Multiplying raw gradients alone generally cancels in Adam except for epsilon; it does **not** implement these canonical learning rates. R/S banks are matrix parameters, so this equivalence is row-specific, not automatically expressible by ordinary scalar parameter groups.

At a unit effective start, define canonical SGD factor gradients `a=g_u`, `b=g_v` and the shared gradient `H`. The exact simultaneous update of one effective map is

```text
A+ = D_(1-eta*c^2*b) (W-eta*H) D_(1-eta*c^(-2)*a).
```

To first order its change is `-eta[H + c^2 D_b W + c^(-2) W D_a]`. For Adam, replace the factor terms by `c D_dv W` and `c^(-1) W D_du`, using the exact adjusted-epsilon directions. At fixed W the explicit factor-rate product in the cross term is `eta^2` for both laws; Adam directions can still depend on c through epsilon. The general finite-step law and decay boundaries are in `PROOF.json`.

The private Jacobian columns are rescaled, so their span is unchanged. This modifies optimization access/step geometry within the existing function class; it supplies no new graph evidence or initial correct alternatives. With a compensating optimizer—SGD raw rates `c^2 eta, eta/c^2`, or Adam raw rates `c eta, eta/c` and epsilons `epsilon/c, c*epsilon`—the scaled representation follows ordinary coherent-unit training exactly under mapped states. That is an algebraic duplicate, not another performance arm.

## Strong collision and limits of the proposed comparison

The local **8 October reciprocal scalar Adam assessment** already proves these same transformations and proposes an inactive unit-start versus reciprocal-start comparison with a modest log-symmetric schedule. The 7 October gauge note already names `r_m -> k_m r_m, s_m -> s_m/k_m`. The present schedule and restricted graph-map placement are recipe changes, not a new mechanism. A bounded saved-scope search did not identify a completed exact coherent-unit/power-of-two native comparison; this is not a global absence claim.

This differs from closed PubMed M-normalization: that keeps unit starts and native learning rates while uniformly matching private coupled-decay and epsilon to the `1/M` own-loss reduction. Here SAGE has zero decay, and reciprocal gauges induce unequal member/axis learning rates even at zero epsilon. It is also different from Rademacher/native-attention screens, which alter initial functions or attention parameter starts. Those screens do not resolve the coherent-function comparison.

The new schedule is not balanced in its average optimizer action. For Adam, `mean(c)=1.875` and `mean(1/c)=.9375`: mean output-factor rate increases while mean input-factor rate decreases slightly. For SGD these means are `5.3125` and `1.328125`. These are written-constant arithmetic, not observed update norms. With identical initial directions and negligible epsilon, coherent uniform axis rates at these means match the average first-order private displacement. Actual route dropout and later divergence prevent a general displacement equality. A reciprocal-versus-unit gain alone would support utility of this full rate recipe; it would not establish heterogeneity as its cause. Do not add an unqualified “diversity” explanation or a scale grid.

## Closest primary scopes

| Retained primary | Collision and boundary |
|---|---|
| [BatchEnsemble 2002.06715v2](https://arxiv.org/abs/2002.06715v2), §3.1 Eqs1–5 | Shared dense W with private rank-one input/output factors is the exact model ancestry. This scope does not certify a published reciprocal-coherent initializer. |
| [TabM 2410.24210v3](https://arxiv.org/pdf/2410.24210v3), method p6 | Later multiplicative adapters start at one and preserve their maps; the first adapter remains random. Complete coherent identity is a different recipe, not a new identity-adapter principle. |
| [LoRA-Ensemble 2405.14438v5](https://arxiv.org/abs/2405.14438v5), retained pinned author initializer | Private random A with zero B preserves each adapted base map while giving different B Jacobians. Function-preserving branches with different optimization directions are already explicit; its frozen base/private heads and additive factors differ. |
| [Path-SGD 1506.02617v1](https://arxiv.org/html/1506.02617v1), §2 | Function-equivalent parameter rescalings can change SGD trajectories. Its ReLU network symmetry is broader/different from this within-map gauge; no graph-data conclusion transfers. |
| [LoRA Done RITE 2410.20625v1](https://arxiv.org/html/2410.20625v1), §§2.1–2.2 | Reciprocal factor scaling, SGD/Adam noninvariance and effective-weight changes are directly analyzed. Additive low-rank factors and its invariant optimizer differ. |
| [Hyperparameter Ensembles 2006.13570v3](https://arxiv.org/html/2006.13570v3), saved §§3/4/D.1 scopes and primary excerpts | Fixed-initialization hyperparameter diversity, shared Hyper-Batch factors, and a separate fast-weight rate multiplier are established. This does not establish the exact coherent reciprocal recipe or graph utility. Its validation tuning is not adopted. |

Saved conclusions were checked before targeted primary reuse. No fresh primary method, whole-paper/proof/result reproduction or new author-code audit was performed. Incidental empirical prose in retained BE/TabM/Path-SGD passages is unused.

## What could change the next decision

The one falsifiable utility question is whether these known fixed row rates, from coherent native functions, preserve member competence while acquiring useful correct alternatives and improving the actual probability pool over coherent all-one/native-rate shared4. That coherent control is indispensable; preserve current Rademacher shared4 and strong genuinely independent4 as practical references. The exact canonical row-rate duplicate needs no separate performance fit. A gain with worse member quality, lost coverage, or worse final pool/NLL fails the intended role; parameter distance or optimizer divergence alone is insufficient. A heterogeneity-specific claim additionally needs a matched mean axis-rate contrast or a more justified schedule.

No quality prediction is established by the algebra. Root must decide whether this low-priority known-method recipe warrants source qualification after current family closures. This packet creates no initializer, optimizer, trainer, owner, job, schedule, model import, forward, payload read or partial 77/GCN/GAT outcome access. Existing packets, outcomes, paper scores and canonical state remain unchanged; TEST stays closed.
