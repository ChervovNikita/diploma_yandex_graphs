# Identity-start Householder pairs on live shared graph maps

**Decision:** this is a materially different private operator from scalar gauge/rate reweighting, with direct orthogonal-adapter and ensemble ancestry. Retain one inactive utility hypothesis, without implementation or fit nomination. A pair is a constrained plane rotation, not an arbitrary dense orthogonal adapter or a generally more expressive replacement for rank-one LoRA. The separate reciprocal-gauge packet remains inactive and unchanged.

## Exact operation and common component

For an eligible native square graph projection `W in R^(d*d)`, use column vectors and declare

```text
A_m = D_s,m W Q_m D_r,m,
Q_m = H(u_m1) H(u_m2),
H(u) = I - 2 u u^T/(u^T u),  u != 0.
```

This places the rotation between the existing input diagonal factor and native W. Different placement relative to a learned diagonal factor is a different operator. The original diagonal r/s factors remain a common component in the baseline, rotation and additive controls; the two normals are **additional**, separate private parameters. The additive control is `D_s,m(W+b_m a_m^T)D_r,m`, with unit incoming a and outgoing b=0. Each selected square site therefore adds `2d` coordinates per route; with the common diagonal bank each candidate has `4d` private coordinates at that site. This is not a parameter-neutral replacement for the `2d` diagonal baseline.

Initialize both normals to the same nonzero unit direction q within a route, with prospectively seeded directions across routes. Set every diagonal factor to one for coherent native starts. Since `H(q)^2=I`, each route initially reproduces the native affine map, including its outside bias. The inspected `FactorLinear` adds bias after output scaling. Consequently biased SAGE projections can satisfy this algebra too; root and the combination role own the exact square-map whitelist. Native attention, residuals and norms see the unchanged initial map. Training dropout keeps its existing per-route streams.

The two vectors must be distinct parameter objects with equal initial values. Aliasing them makes `H(u)H(u)=I` for every state and removes the proposed learning operation. Normalization is part of the forward law; nonzero-radius handling, raw radius versus explicit sphere updates, and any clamp/retraction must be specified before qualification. This memo does not provide an implementation.

The combination role's completed source-only design reports the native L2/H128 whitelist as two GCN/GAT `conv.lin` sites, or four SAGE `conv.lin_l/lin_r` sites; stem, FFN and head stay outside the added correction. This implies 2,048 or 4,096 extra normal/LoRA coordinates across four routes, respectively. These are source-derived counts, not a runtime qualification. The existing role/count reporters recognize only diagonal r/s, so extra normal/additive parameters need explicit ownership and cost reporting before any source is admitted.

## Actual tangent freedom

Let `a,b` be first-order changes of the normalized normals at the equal start, so `q^T a=q^T b=0`. Differentiating both reflections gives

```text
delta Q = 2[(a-b) q^T - q(a-b)^T].
```

It is skew-symmetric, has rank at most two, and depends only on `a-b`. At fixed q its live dimension is **d−1**, with `d+1` raw null directions: two radii and `d−1` common normal changes. Both normals can receive opposite task gradients immediately; this does not imply nonzero gradients in every task-visible direction. `q` and `−q` give the same reflection and tangent family. In d=2 all normals expose the same one-dimensional rotation tangent; merely assigning different directions cannot create different tangent spaces there.

Globally, a product of two reflections fixes every vector perpendicular to their span. It has determinant +1 and `rank(Q−I)<=2`. Away from degeneracies its family dimension is `2d−3` for d>=2: a two-plane has `2(d−2)` degrees and its rotation angle adds one. It can be dense in entries while still affecting only one plane. It does not cover general SO(d) for d>=4; the identity differential is smaller even in d=3. At d=128, the pair stores 256 scalars but exposes only 127 first-order directions at the stated start, compared with 8,128 directions of the full orthogonal group.

At unit diagonals the effective private perturbation is `delta A=W delta Q`. It can be killed or reduced by W's nullspace, the factual input span, normalization or the downstream predictor Jacobian. The tangent dimension above concerns Q, not guaranteed predictive freedom.

For a map cotangent `G=dL/dA`, define `C=W^T G`. At unit-radius initialization,

```text
grad_u1 = 2(C-C^T)q,   grad_u2 = -2(C-C^T)q.
```

Only the antisymmetric part of C drives this stage initially. If it is symmetric, or its skew part annihilates q, the pair has no initial gradient although other adapters may learn. One simultaneous plain SGD step gives

```text
delta Q = -8 eta[(C-C^T) q q^T + q q^T(C-C^T)] + O(eta^2).
```

Thus ordinary equal parameter learning rates do not match the physical update size of a zero-output LoRA branch. This identity is for SGD and the declared unit-radius start. Native Adam's coordinate moments and epsilon alter the displacement; they are not rotation-invariant. A normalized forward preserves orthogonality under Adam, but no Adam descent, convergence, conditioning or competence theorem is supplied.

## What the norm constraint does—and does not—preserve

`Q` preserves the Euclidean norm of the factor-weighted input `D_r x`. For fixed W, `WQ` has the same row Gram matrix `WW^T`, singular values, rank and Frobenius norm as W. It can still change `||WQx||` for a particular input because W can weight directions unequally. With learned diagonal factors, the complete `D_s W Q D_r` need not preserve the baseline singular values or row Gram; shared W also continues to learn. No initial predictor, calibration, member competence or graph-evidence guarantee follows from orthogonality. A pair can reach a half-turn, so it is not a small-step constraint.

A member-common Q can be absorbed into the free shared W. Route-specific Q values generally cannot all be absorbed into one W. Leaving the native residual/head/norm paths fixed also prevents assuming a complete hidden-coordinate relabeling. A node-shared feature rotation commutes with fixed scalar channelwise graph aggregation, but feature-dependent attention and the surrounding nonlinear network can change. It is a feature adapter, not a new edge-dependent graph transport or a new information source.

## Distinction from the closed and retained alternatives

| Comparison | Actual distinction and boundary |
|---|---|
| Reciprocal scalar rows / closed private normalization | Those rescale existing optimizer coordinates/rates or uniformly match private regularization/epsilon. The pair introduces off-diagonal skew directions in the effective map. It is not another scalar rescaling of the diagonal-factor tangent. Both remain known optimization/adapter ingredients. |
| Diagonal BE at a fixed square map | At W=I the BE perturbations are diagonal, whereas a nonzero pair perturbation is off-diagonal skew. Adding Q can leave the common diagonal orbit. This local witness is not a universal nonlinear GNN expressivity or quality theorem. |
| Cold rank-one LoRA with incoming a and outgoing b=0 | Its initial tangent is `delta b a^T`, with d live outgoing directions and no initial incoming gradient. The pair instead exposes d−1 coupled skew directions, usually rank two. Equal `2d` storage does not equate tangent dimensions, regularizers, step size or usefulness. |
| General additive family | `W(Q−I)` is exactly an additive rank-at-most-two correction with nonlinear ties and orthogonality constraints. A nontrivial full-rank-base plane rotation generally is not rank-one addition; rank-one stretching generally is not a right-orthogonal rotation. The isolated families are incomparable, not a general rank/expressivity advantage. Common diagonal factors and live W further limit whole-network conclusions. |

Both this pair and zero-output/distinct-dictionary LoRA preserve initial maps while choosing different private tangent directions across members. That principle is already in saved primary methods. The possible useful difference is the **orthogonal constraint and its coupled tangent**, rather than a new neutral-initialization idea. Existing complete errors do not establish that missing off-diagonal freedom or unwanted private norm growth caused the failures.

## Closest primary collisions, saved first

| Source and actual scope | Collision; limit |
|---|---|
| **New scoped** [OFT 2306.07280v1](https://arxiv.org/html/2306.07280v1), §§3.2–3.3 | Multiplicative orthogonal transforms, identity starts, Cayley/skew coordinates and parameter-efficient block structure are explicit. Its frozen pretrained base/preservation motivation differs from a jointly learned fresh shared W. |
| **New scoped** [BOFT 2311.06243v1](https://arxiv.org/html/2311.06243v1), §§3–4 | Products of structured orthogonal factors, dense mixing, and identity initialization are explicit. Butterfly structure has different freedom/work; its capacity discussion is not inherited by a two-reflection pair. |
| Retained [ETHER 2405.20271v1](https://arxiv.org/html/2405.20271v1), targeted §3.2–3.3 and saved OFT paragraph | Learned Householder transforms and the neutral two-vector `I-uu^T+vv^T` are explicit. ETHER+ is generally nonorthogonal, and its equal-normal tangent is symmetric; the ordered product here is orthogonal with a skew tangent. The primitive and neutral-branch principle remain direct prior. |
| Retained [HousE 2202.07919v1](https://arxiv.org/html/2202.07919v1), §3.1 Eqs2–5 | Even products of Householder reflections parameterize relation rotations using vector operations. Our one pair is a restricted instance; its member-global GNN placement differs from relation-specific entity scoring. |
| Saved [LoRA-Ensemble 2405.14438v5](https://arxiv.org/abs/2405.14438v5), pinned initializer scopes; [TabLoRA 2607.10077v1](https://arxiv.org/html/2607.10077v1), saved §§3.3–3.5 | Private different dictionaries/zero outgoing maps and jointly learned shared W/private additive maps already establish the ensemble and cold-sharing ingredients. No teacher acquisition is necessary. |
| Saved [Superposition of many models into one 1902.05522v2](https://arxiv.org/html/1902.05522v2), §§2–3 in the structured-coordinate memo | A shared matrix applied after private orthogonal/permutation contexts is already a published operation. Its fixed contexts and sequential superposition contract differ from these learned same-task rotations. |

Saved HTA, GoldE and geometry-sharing notes were also checked; no new scope or performance is credited to them. The intended **DELTA** ensemble identity remains unresolved after two bounded, noisy arXiv metadata queries. Unrelated Delta-LoRA/DeltaLLM titles were not substituted. This limits prior coverage and supports no absence or originality claim. No complete-paper, author-code, proof/convergence or empirical reproduction is claimed. Incidental source result prose is unused.

## One inactive, falsifiable question

At coherent native starts and fixed square graph sites, can the known orthogonal pair add useful correct alternatives and improve the restored probability pool while retaining mean/worst member competence, beyond the same diagonal bank and the equal-extra-coordinate cold rank-one LoRA adapter? Root must qualify exact sites, normal parameterization/radius, objective reduction, Adam state/rates, native RNG, stopping and serving before any implementation decision. Copied versus distinct q values would distinguish private-rotation capacity from the distinct-tangent initializer if that attribution becomes necessary; it is not a proposed additional fitted grid.

The baseline has the same learned diagonal component and no Q; rank-one LoRA adds its `2d` coordinates at the identical sites and coherent start. Preserve current Rademacher shared4 and capable genuine I4 as practical competence references. A same-operation capable single/independent reference is necessary before claiming a benefit from shared ensembling. The orthogonal-preservation claim additionally needs evidence in the actual operator; changed hidden distance, more parameter motion or reduced pooling loss with destroyed coverage cannot pass.

Each Householder can be applied as a dot product and rank-one vector subtraction, without forming a dense Q. Two reflections add O(Nd) work per invoked route/site, plus backward, normal storage and optimizer states; all existing live graph trajectories remain paid. Storing or premerging private effective matrices changes the storage contract. No wall-time/memory advantage is qualified.

**No fit nomination follows.** This is a distinct attributed local-operator hypothesis with substantial prior, low-dimensional initial access and no demonstrated need or utility on the current graph. The scalar-gauge schedule is separate and inactive. No model import, dataset/logit/checkpoint/label read, numerical fixture, new trainer/source/owner, remote scientific action, partial family outcome or TEST access occurred. Root owns canonical state and publication.
