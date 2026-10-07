# Shared Adam-step projection with private progress

7 October 2026. **The proposed update has a meaningful block distinction, but is exactly an established constrained-projection construction with private coordinates fixed. Treat it as an attributed optimizer control, not a new optimizer principle.** No literal complete published shared/private Adam recipe was established in the bounded sources; that unresolved implementation detail does not establish novelty. No computation is proposed for the frozen pilot.

## 1. What the rule actually solves

At one old state and one fixed supervised-loss/RNG realization, let `a_m=(g_m^P)ᵀΔφ_m`, let A have rows `(g_m^S)ᵀ`, and set `b_m=−a_m`. The native shared/private proposals must include the actual optimizer state, bias correction, clipping/decay and gradient scaling. They may include auxiliaries; the protected gradients are those of the stated **own supervised losses**.

The assumed dependence `L_m(θ,φ_m)` is essential. If own predictions exchange private states or use other cross-member forward dependencies, replace a_m by `Σ_j(∇_{φ_j}L_m)ᵀΔφ_j`. Auxiliary coupling changes proposals but does not itself invalidate own-loss separability when the predictive paths remain separate. No model-source audit was performed here.

For a declared positive-definite metric H, the rule is

`min_d ½(d−d₀)ᵀH(d−d₀), subject to Ad≤b`.

Euclidean H=I is the simplest specified version. Its dual is

`min_{λ≥0} ½λᵀGλ+(b−Ad₀)ᵀλ`,

`G=AH⁻¹Aᵀ`, and `d*=d₀−H⁻¹Aᵀλ*`.

K=4 gives a four-variable dual and at most16 active subsets. The Gram can be singular; the strictly convex primal still has a unique solution when feasible, while dual multipliers may be nonunique. Numerical rank/KKT tolerances are required; a hidden ridge changes the rule. The small dual does not make obtaining four full shared gradients or storing them free.

**Exact full-parameter equivalence:** set `q=(d,p₁,…,p_K)`, `q₀=(d₀,Δφ₁,…,Δφ_K)` and `ĝ_m=(g_m^S,0,…,g_m^P,…,0)`. Project q₀ to `ĝ_mᵀq≤0` while imposing `p_m=Δφ_m`. Eliminating those equality-fixed private coordinates yields precisely the QP above. The private offsets are therefore standard affine constraints arising from block elimination. An unconstrained full-parameter projection allows the private corrections to change too; its Euclidean Gram has extra diagonal terms `δ_mn||g_m^P||²`.

The offset matters operationally. If `a_m=−2` and `(g_m^S)ᵀd₀=1`, the total first-order change is−1 and the native proposal passes. A shared-only constraint would forbid that shared ascent. Conversely, Adam momentum/auxiliary terms can give `a_m>0`; the shared block must compensate rather than treating private updates as automatic descent. This is a difference from ignoring the private block, not a new optimization geometry.

When every `a_m≤0`, the feasible set contains the shared-only b=0 cone. Under the identical proposal/metric, the private-aware minimum correction is therefore no larger than the shared-only minimum correction. Mixed positive/negative private contributions remove this nesting. Smaller corrections in the all-descent case are an algebraic consequence, not evidence of better learning.

## 2. Closest prior and exact boundaries

| Prior | What is established | Relation to this rule |
|---|---|---|
| [GEM, §3](https://arxiv.org/html/1706.08840v1) | Minimum Euclidean correction of a proposal under simultaneous first-order loss-nonincrease halfspaces, solved with a small dual. Reused indexed method. | Closest projection ancestry. Current-member losses, actual Adam displacement and fixed private offsets change the contract; the projection algebra is the same. Gradient refresh and a small K-dimensional dual are already prior. |
| [Sener & Koltun, MGDA-MTL, §§3.1–3.3](https://arxiv.org/html/1810.04650v1#S3.SS1) | Explicit shared/private hypotheses; own gradient descent on task-specific parameters followed by minimum-norm convex combination of shared task gradients. | Shared/private optimization is already explicit. Its shared convex-hull objective does not credit actual private Adam progress as affine offsets or stay closest to a native Adam step. MGDA-UB assumes a common encoder representation; all-layer BE has member-dependent paths, so its cheap representation-gradient surrogate cannot be imported automatically. |
| [PCGrad](https://arxiv.org/abs/2001.06782v1) | Pairwise projection of conflicting task gradients in random order, then an optimizer update; shared-parameter method in the saved scope. | Does not solve the simultaneous closest-step affine QP. Passing a corrected gradient through Adam is different from constraining the resulting actual Adam displacement. Full-coordinate gradient methods can include private derivatives; the blanket claim that MTL methods ignore private progress is false. |
| [CAGrad, §3](https://arxiv.org/html/2110.14048v1) | Maximize worst task directional progress within a ball around the average gradient; small simplex dual; stated average-loss convergence for its SGD/smoothness regime. Root's saved scope reused. | Different objective and feasible set. With full coordinates, private blocks enter total directional derivatives; with shared coordinates only, they do not. Neither choice fixes a native private proposal and spends its measured progress through b. Its theorem does not transfer to projected Adam. |
| [FAMO, §3](https://arxiv.org/html/2306.03792v1) | Approximate balanced relative loss decrease through dynamically weighted log-loss gradients and loss-change-based weight adaptation. Root's saved scope reused. | Different amortized weighting rule; no simultaneous hard per-step closest-Adam constraints. Its generic full-parameter loss formulation can include private effects, but its loss-change history is not an exact current private-progress budget. No author implementation equivalence was audited. |
| [MT²O, 2024, §III-A and IV-A](https://arxiv.org/html/2403.16162v1#S3.SS1) | Explicitly drops task-specific parameters in its simplified multiobjective problem, then transfers parameters among different scalarized Pareto subproblems. | A direct shared/private locator checked and excluded: it does not supply the proposed private-offset Adam projection. This bounded exclusion is not an absence-of-prior certificate. |

There is also a close **existing project control**: `graph_shared_gradient_gem_closure_20261003_v1/PROSPECTIVE_MEMBER_CE_CONTROL.json`. It projects the shared displacement using member CE gradients at `B=(θ_old,φ_plus)`, protects member losses relative to B, and adds pooled equality/cap/finite guards. It is **not identical**: the current proposal instead protects the combined shared/private first-order change relative to the old joint state, allowing private progress to pay for shared ascent. Rebranding the old projection as new would be incorrect; the change of reference/budget must be stated.

## 3. Feasibility, moments and the actual guarantee

If every `a_m≤0`, d=0 is feasible. Private SGD on each own loss provides this sign; native Adam need not. For example, shared gradients+1 and−1 with private changes `a₁=a₂=+1` demand `d≤−1` and `d≥+1`, hence are infeasible. If `g_m^S=0` and `a_m>0`, the shared block cannot repair that member. In general, feasibility requires `λᵀb≥0` for every `λ≥0` with `Aᵀλ=0` (Farkas' condition).

Choose the failure contract before outcomes. A declared slack variant can solve

`min_{d,ξ≥0} ½||d−d₀||_H²+(ρ/2)Σ_m w_m ξ_m², subject to Ad≤b+ξ`.

Positive slack permits positive predicted loss change; it must be reported as such. ρ/weights/loss units change the tradeoff. Native fallback forfeits protection. Zeroing only the shared step need not protect a private-ascent proposal. Rejecting the **whole joint displacement** gives exact no parameter change on the fixed loss, but whether proposed moments/step clocks still advance must be bound explicitly. Shrinking only the shared correction is not a general repair for affine constraints.

On a feasible hard step, the exact promise is

`D L_m[(d*,Δφ_m)] = (g_m^S)ᵀd*+(g_m^P)ᵀΔφ_m ≤0`

for the stated old-state loss and parameterization. It is a **first-order directional statement**, not finite-step nonincrease. Under a β_m-Lipschitz gradient,

`L_m(new)−L_m(old) ≤ c_m+(β_m/2)(||d*||²+||Δφ_m||²)`.

A boundary constraint c_m=0 permits finite increase. Strictly negative c_m supports sufficiently small **joint** scaling under local smoothness; retaining the full private step while shrinking only the shared step need not work. ReLU kinks, stochastic view changes, stale gradients and mixed precision require their own qualification. Protection of sampled own CE does not imply lower full-TRAIN loss, served-pool NLL/accuracy, generalization, calibration or diversity.

Joint backtracking changes the accepted private update, so it is a separate contract from retaining the full native Δφ_m. A finite-loss rejection/fallback must likewise state what happens to both blocks and the proposed optimizer state.

Retaining native proposed moments defines a projected-Adam-style state convention. The carried moments describe the original proposal gradient, while the parameter displacement was changed. This is a valid declared algorithm, but not ordinary Adam on the same scalar objective, and standard CAGrad/MGDA convergence cannot be claimed. Recomputing moments from projected directions would be another algorithm. Different H choices, especially a moment-based metric, are also distinct contracts; Euclidean closeness is parameter-coordinate dependent.

Finally, this is a competence constraint, not a diversity generator. Identical members can stay identical; the single-map BE capacity restriction remains. Protecting every fitted member can obstruct useful shared learning or collective improvement, including regimes where individually weaker members help the pool.

## 4. Meaningful controls and decision

The essential mechanistic comparison is **the identical QP/state/failure contract with b=0 versus b=−a**, using the same old-state supervised gradients and actual native private proposals. This isolates credit for private progress. Add native Adam and the generic full-parameter halfspace projection, which can correct private blocks and remains feasible at the zero joint step. Match gradient opportunity and charge all backward passes, rejected proposals and finite checks. CAGrad/PCGrad/FAMO are broader optimizer competitors only after their parameter scopes and native-state integration are qualified; a weak arbitrary port is not a meaningful control.

A prospective audit should record every member's private contribution a_m, native/corrected shared contribution, total predicted change, actual finite own-loss change on fixed views, active constraints/rank/KKT residuals, slack/fallback rate, correction norms and optimizer cost. Later predictive testing must use the actual mean-probability pool and competent untied/single controls. A useful result would show that private-budget credit avoids unnecessary shared corrections **and** improves pooled utility; a smaller correction or more feasible steps alone is insufficient. No graph-specific property is used by this QP.

**Decision: defer projected-optimizer implementation and compute.** The current internal-BE full family continues unchanged. First require observed, reproducible member harm from the actual native shared/joint step after accounting for private progress, on fixed loss/view realizations. Pairwise gradient conflict, weak standalone members, or a shared contribution paid for by private descent do not establish that need. Even observed fitted-loss harm does not prove that preventing it improves the served pool.

Retain the rule as an attributed fixed-private constrained projection control, and reject a claim of a new projection/private-aware MTL principle. The literal Adam/moment/private-offset recipe remains unverified as a published complete algorithm, and its possible utility remains untested. No frozen-suite changes or compute admission follow from this note.

Scope: root's CAGrad/FAMO scopes and indexed GEM/PCGrad conclusions reused; two new bounded primary leads (MGDA-MTL and MT²O formulation/method boundary). Zero full-paper/proof/code/result audits, model/data/score/checkpoint/GPU/server actions or scientific-source edits. Exact passages, hashes, retrievals and limits are saved alongside this note.
