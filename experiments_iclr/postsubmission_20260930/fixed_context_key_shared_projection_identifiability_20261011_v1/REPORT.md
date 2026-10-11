# Common auxiliary projection: identifiability boundary

## Scope

This is an interpretation of the existing fixed-key construction for future results. It makes no novelty, acceptance or quality verdict. No current outcome, prediction array, model, fit or scientific remote operation was accessed. The construction, first-moment reduction and scoped primary conclusions were reused first. The saved G-CRD method was not reopened. No new primary source was needed for the following matrix and chain-rule argument.

## One bounded theoretical claim

**Distinct assigned targets can change auxiliary geometry without necessarily changing native serving predictions. A common A limits which query differences it can absorb at fixed hidden states, but failure to absorb a difference does not force a useful native prediction change. Consequently, the existing shared-versus-I4 contrasts identify complete training packages; they do not isolate native backbone sharing because projection ownership changes too.**

All local derivatives below assume nonzero `Ah` outside the normalization clamp. No actual hidden states, ranks, gradients or projectors were inspected.

### A. What the projector can absorb

Index the route/node pairs by `a`. Let `z_a=A h_a`, `q_a=z_a/||z_a||`, and

`d_a = (I−q_a q_aᵀ)(μ_pa−μ_Qa)/(τ||z_a||)`.

Then the already saved identities give

`∂L_a/∂h_a = Aᵀd_a`,

`∂L_aux/∂A = Σ_a w_a d_a h_aᵀ`,

where the weights are exactly those of the declared route/node mean. With different dictionaries, both `μ_p` and `μ_Q` differ; a target-mean difference alone is not the complete gradient difference.

Any update confined to A can change q and auxiliary loss while leaving the native predictor **exactly unchanged**, because A is removed at serving. An A-only fitting explanation is therefore possible even when targets and projected queries differ. This is an existence and identifiability statement, not a claim that simultaneous Adam training actually chooses A-only fitting.

There is a direct algebraic witness: a nonzero tangent `d_a` perpendicular to `range(A)` gives `Aᵀd_a=0`, while `d_a h_aᵀ` can be nonzero. Such a row can move the projector without an auxiliary hidden gradient. Opposite rows can also cancel the shared A gradient. Conversely, cancellation of the A gradient need not cancel route-private native gradients: the native Jacobians can differ. Neither nonzero query gradients nor aggregate cancellation settles the served effect.

At fixed H, the local query motions available through A alone are

`δq_a = T_a ΔA h_a`, with `T_a=(I−q_a q_aᵀ)/||Ah_a||`.

A specified query-motion field is absorbable to first order exactly when it lies in the image of this single shared linear map from `ΔA` to all `δq_a`. Query changes outside that image cannot be obtained by a projector-only perturbation at the current H. They may require hidden-state motion, or remain unfitted. Some fields remain unrealizable even if H changes.

There is also a finite condition. For a chosen nonzero unit query field `U=[u_a]` at fixed `H=[h_a]`, a common A can realize it iff there exist positive scales `s_a` such that

`A H = U diag(s)`.

Equivalently, for those scales, `ker(H) ⊆ ker(U diag(s))`. Necessity follows by multiplying `AH=U diag(s)` by any vector in `ker(H)`. For sufficiency, define A on `range(H)` by `A(Hv)=U diag(s)v`; the kernel condition makes this well-defined, and the map can be extended linearly to the remaining hidden space. Every dependence among hidden columns must therefore also hold among the scaled query columns. This is a condition for a specified query field, not a claim that the optimum of the full dictionary loss is `normalize(μ_Q)`.

In particular, identical hidden vectors give identical queries; positively collinear hidden vectors also give identical normalized queries. A common A cannot assign them distinct route queries. Realizing such a split requires changing H or changing projection ownership. The optimizer can instead retain a compromise and nonzero auxiliary error.

Across every route, `rank([q_a]) ≤ rank(A) ≤ 128` in the 300-dimensional key space. Changing H cannot overcome this common query-subspace cap. With separate route projectors, each route still has rank at most 128, but the union of their query subspaces can have rank up to 300. These are representational upper bounds, not measured ranks. A target-mean field of high rank does not prove that the full loss requires an equally high-rank optimal query field.

Common A also imposes one PSD hidden-space metric `M=AᵀA` for all projected query-query cosines. Alignment to the fixed keys additionally depends on `Aᵀk_j`; M alone does not identify the orientation relative to those keys. Separate A maps can vary both metric and key alignment by route.

### B. Which changes must reach serving

No difference in Q, target means or auxiliary queries alone forces a native prediction difference. The route's native hidden Jacobian first pulls back `Aᵀd_a`; shared-parameter sums can cancel, and route-private Jacobians can remove other components. The declared optimizer then maps the total gradient and its history to a native parameter update. A gradient-descent intuition is not an exact statement about a finite Adam transition.

At the prediction level, let `δθ_native` be the actual native parameter motion. A first-order change in served probabilities requires

`Dθ p_served · δθ_native ≠ 0`.

For the native probability mean, this derivative is the mean of the route probability derivatives. Hidden changes may lie in a serving-head null direction; logits may change only by a class-common offset; or nonzero member probability changes may cancel in the pool. Even a nonzero pooled probability change need not cross an argmax boundary or improve accuracy. For an affine final head, its kernel is a simple hidden-space example; the statement otherwise uses the actual serving-head Jacobian without assuming that head is affine.

If a desired query split cannot be fitted by common A at fixed H, then a realized split demonstrates H changed. It still does not demonstrate the change reached native probabilities. If future served accuracy differs under exact source/data/selection custody, the native serving function changed on the relevant cases; that observation does not identify which sharing component caused it.

## What the current contrasts identify

| Existing comparison | Held fixed or addressed | Identification limit |
|---|---|---|
| Private assignment versus common four complete-pair losses | Same native shared architecture and common A; contrasts objective allocation. | Also changes route objective exposure and auxiliary computation. It does not test whether common A is necessary or isolate its absorption. |
| Exact moment alignment versus full fixed-key loss | Same projection ownership and positive target steering. | Identifies the dictionary-term package, not native backbone sharing or where alignment is stored. |
| Fixed versus live projected peer keys | Same map dimensions and ownership. | A also receives key-side gradients in the live-key arm; key source and gradient paths change together. |
| Uniform/incidence controls | Same fixed keys and common A; changes the assigned target field. | The projector can respond to the changed field. These do not isolate native backbone sharing, a unique topology cause, or output transfer. |
| Shared assigned arm versus assigned genuine I4 | Same assigned key/target pairs. | Shared native parameters **and common A** become independent native bodies **and separate A maps**. Capacity, optimizer coupling and native selection also differ. Any quality gap is a package comparison. |
| Shared arm versus all-pair M1/I4 | Same dictionary/target information is available. | Objective exposure, route structure, projection ownership for I4 and native parameter/selection packages differ. This is a capable reference, not a pure sharing intervention. |

The current I4 maps are not merely copies of one common train-only map. Even with matching initial A bytes, separate training permits different learned subspaces and orientation. An eventual shared-versus-I4 advantage or disadvantage must therefore not be attributed solely to backbone sharing. Saved MLCL/AMCL auxiliary-head conclusions and HLCL's shared projection give relevant ancestry; they supply no theorem resolving this identifiability problem.

## One falsifiable implication

**An eventual shared-assigned versus assigned-I4 quality gap attributed to native backbone sharing alone should survive the comparison with projection ownership held at separate A maps.** If the gap disappears or reverses when the shared native architecture uses route-private A maps, the backbone-only account is undermined. If it persists, the evidence is conditional on private projection ownership and still concerns the native architecture/capacity/optimization/selection package. It does not establish a universal isolated backbone effect.

This implication specifies no numerical threshold or new gate. Use the parent-declared served quality, risk, paired seeds and complete acquisition-cost rules if the later control is elected. Auxiliary fitting or spread alone cannot satisfy it.

## At most one later decisive control

**Projection ownership within the same shared native architecture:** retain the existing assigned fixed `(K_m,Q_m,E_i)` objectives, native shared/private parameters, own CE, total scalar-loss weights, initial native bytes, streams, optimizer rules, schedules, selectors and serving pool. Replace the one common bare A by four independently trained bare maps `A_m`, one per route, initialized as exact copies of the current common `A_0`.

This preserves initial queries and native predictions. Each A receives only its route's contribution as dictated by the unchanged scalar objective; no special gradient scaling or protection is introduced. It adds three projector copies: train-only projection weights rise from 38,400 to 153,600, an increment of 115,200, with corresponding optimizer storage. Query projection operations remain four route projections, but measured total cost must include the extra parameter updates and memory. Serving remains the native predictor with no A maps.

The control holds native sharing fixed and tests the **projector ownership/capacity package**, not a pure geometry effect at equal trainable parameter count. It supplies one missing comparison; it does not complete a two-factor study. With score `Y` for a consistently evaluated metric, the three available settings would permit the arithmetic identity

`Y(shared native, common A)−Y(independent native, private A)`

`= [Y(shared native, common A)−Y(shared native, private A)]`

`+ [Y(shared native, private A)−Y(independent native, private A)]`.

The first difference holds native architecture fixed. The second holds projection ownership fixed while retaining the native architecture, parameter, optimizer and selector differences. This identity separates an observed contrast along that path; it neither estimates a general interaction nor converts the second term into a pure causal backbone effect.

This is a later interpretive control specification only. No arm, roster, source, qualifier, fit or execution permission is added here.

## Evidence and operation counts

**Zero new primary searches or method reads, zero whole papers, proofs, code audits, models, fits and scientific remote operations.** Existing primary conclusions and sealed theory/candidate files are hash-bound in `REUSED_BINDINGS.json`; raw G-CRD was not reread. All new files are confined to this packet. Canonical state, active source, rosters and manuscript remain untouched.
