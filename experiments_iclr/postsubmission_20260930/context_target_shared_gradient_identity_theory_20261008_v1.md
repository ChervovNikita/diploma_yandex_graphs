# Frozen Context9: target linearity and instantaneous shared gradients

This note explains the actual frozen `shared_route` versus `shared_common` assignment at a fixed, identical-path unit-factor state. It gives no training result or adoption verdict. The ingredients are elementary soft-target cross-entropy, differentiation, and covariance bilinearity; this is explanatory calculus, not a new ML theorem or method. Only static source and the frozen protocol were inspected. No model, target array, checkpoint, numeric fixture, or comparative outcome was loaded or executed.

## What the source actually computes

Let \(M=4\), \(B=512\), and let the ordered auxiliary panel be the original `torch.linspace(0,579,steps=512,...).long()` panel. Four TRAIN-label masks come from the signatures \(X,X-PX,PX,P^2X\): directed destination aggregation uses row-normalized \(A+I\), deduplicated ordered edges, and one self-loop; each mask selects up to 16 other same-class rows by cosine similarity and includes self. The masks are restricted to the panel **before** row normalization. Thus

\[
Q_{m,ij}=\frac{\mathbf1\{(i,j)\text{ is selected in restricted mask }m\}}
 {\sum_k\mathbf1\{(i,k)\text{ is selected in restricted mask }m\}},\qquad
\bar Q=\frac1M\sum_m Q_m,\qquad \Delta_m=Q_m-\bar Q.
\]

Every mathematical target row has mass one and \(\sum_m\Delta_m=0\) entry by entry. `route` assigns \(Q_m\) to route \(m\); `common` broadcasts the arithmetic mean of these already normalized matrices to every route. It does not normalize a union of masks. These definitions are `H/context_positive_masks.py:9–45,48–85,127–147`, byte-identical in `U`. `R/FROZEN_PROTOCOL.json:43–59` freezes the same convention. `I/frozen_targets.py:85–128` consumes and validates the prospectively bound bundle rather than regenerating it; `U/train_targets.py:77–116` prepares a caller TRAIN-only bundle before any model or optimizer exists.

Both dispatchers map these methods to the same `be_unit_contrastive` arm, with alignment coefficient \(\lambda=.05\), temperature \(\tau=.2\), and residual weight zero (`I` and `U/context_dispatch.py:10–17,29–68,81–89`). The facade caches fixed weights, retains ordinary own supervision, and returns zero residual contrast (`H/session_objectives_adapter.py:19–59,61–74`; corresponding executable behavior in `U:19–58,60–73`).

For one route, write \(a_i,b_j\) for its two captured representations after `F.normalize`, and \(S_{ij}=a_i^Tb_j/\tau\). The actual alignment is

\[
\ell(S,Q)=-\frac1{2B}\sum_{ij}Q_{ij}
 \left[\log\operatorname{softmax}_{\rm row}(S)_{ij}
       +\log\operatorname{softmax}_{\rm row}(S^T)_{ij}\right].
\]

It is **linear in fixed \(Q\)** (`H` and `U/context_alignment_objective.py:20–36`). The reverse view reuses \(Q\), not \(Q^T\). Denominators include all panel rows, including unselected same-class distractors, and do not depend on which positives \(Q\) selects. If \(P=\operatorname{softmax}_{\rm row}(S)\) and \(T=\operatorname{softmax}_{\rm row}(S^T)\), row normalization gives

\[
\nabla_S\ell(S,Q)=\frac{P+T^T-Q-Q^T}{2B},\qquad
\nabla_S\ell(S,Q_m)-\nabla_S\ell(S,\bar Q)
=-\frac{\Delta_m+\Delta_m^T}{2B}. \tag{1}
\]

Consequently only the **symmetrized** target difference changes the score gradient in this particular symmetric loss. A nonzero directed target difference does not by itself guarantee a nonzero parameter-gradient difference.

The implemented shared objective is

\[
L=\frac1M\sum_m\left[C_m+\lambda\ell(S_m,Q_m)\right],\qquad
C_m=\tfrac12\{\mathrm{CE}(z_m^a,y)+\mathrm{CE}(z_m^b,y)\}.
\tag{2}
\]

Own CE uses all 580 TRAIN rows, whereas alignment uses the 512 panel. `I/context_recompute.py:43–50` constructs (2), then accumulates all member/view VJPs before one Adam step (`:55–98`). Its bytes equal `U/context_recompute.py`. For the untied arm it instead uses a sum of member objectives, so each separate predictor gets its own unscaled gradient. The standalone helper `context_alignment_objective.objective` expresses the same scaling (`:51–58`), but the frozen driver actually installs the facade and replay (`I/train_context.py:9–20,152–156`).

## Identical paths: exact mathematical shared-gradient equality

Fix the model state, graph, labels, panel, and targets. Let \(\theta\) be any shared parameter block and \(\phi_m\) the private factor row of member \(m\). Include normalization and both views in the score Jacobian. If all routes have the **same score path and shared score Jacobian** \(J_\theta=\partial\operatorname{vec}S/\partial\theta\), (1) gives

\[
\nabla_\theta L_{\rm route}-\nabla_\theta L_{\rm common}
=-\frac{\lambda}{2BM}J_\theta^T
 \operatorname{vec}\!\left[\sum_m(\Delta_m+\Delta_m^T)\right]=0. \tag{3}
\]

Ordinary CE is identical between assignments at that same state and same views, so (3) covers the **total** shared gradient. Equal scores alone are insufficient if their shared parameter Jacobians differ. Identical routes with unit factors and identical stochastic views give the required path symmetry: `C/core/factors.py:18–26` implements \(\operatorname{diag}(s_m)W\operatorname{diag}(r_m)\), initializes every \(r_m,s_m\) to one, and retains shared \(W,b\). `C/core/models.py:181–192,200–207` uses one shared native body and does not apply the alternative sign initialization to `be_unit`.

For a private row, cancellation across routes does not occur in its own coordinate block:

\[
\nabla_{\phi_m}L_{\rm route}-\nabla_{\phi_m}L_{\rm common}
=-\frac{\lambda}{2BM}J_{\phi_m}^T
 \operatorname{vec}(\Delta_m+\Delta_m^T). \tag{4}
\]

This **can** change private factor gradients; it need not do so if the symmetrized difference vanishes or lies in the Jacobian's null space. If corresponding private Jacobians are identical, their numerical differences sum to zero after identifying matching rows, but those rows are distinct trainable coordinates. In the shared arm each row receives the \(1/M\) scaling in (4); the source does not rescale private rows to an unscaled loss.

All live shared weights, biases, normalizations, and attention parameters obey (3) when its assumptions hold; sharing a head does not defeat the identity. The Wiki representation is captured **before** the active final prediction head (`C/core/models.py:72–83`). Thus final-head parameters/factors have no direct alignment gradient; their immediate own-CE gradients are unchanged by target assignment. Internal factors upstream of that representation may receive (4). No new projection head or selective gradient permission is present.

## Independent identical route-view laws: mean and covariance

Now keep the state and targets fixed and let \(\xi_m\) denote route \(m\)'s **entire two-view stochastic path**. Require these paths to be independent across routes and identically distributed, with a common forward/Jacobian law, independently of the frozen targets. Require finite second moments and the relevant Jacobians to exist. Independence between the two views *within* a route is unnecessary. Couple the two assignments by using the same \(\xi_m\) for each corresponding route.

Let \(c(\xi)=\nabla_\theta[C(\xi)+\lambda\ell(S(\xi),\bar Q)]\), define the linear operator

\[
A(\xi)\Delta=-\frac{\lambda}{2B}J_\theta(\xi)^T
 \operatorname{vec}(\Delta+\Delta^T),\quad
G_C=\frac1M\sum_m c(\xi_m),\quad
D=\frac1M\sum_m A(\xi_m)\Delta_m,\quad G_R=G_C+D.
\]

Linearity and the identical law give

\[
\mathbb E D=\frac1M(\mathbb E A)\sum_m\Delta_m=0,
\qquad \mathbb E G_R=\mathbb E G_C. \tag{5}
\]

Mean equality needs identical marginal path/Jacobian laws; cross-route independence is needed for the covariance calculation below. Using \(\operatorname{Cov}(X,Y)=\mathbb E[(X-\mathbb EX)(Y-\mathbb EY)^T]\), independence eliminates terms with different route indices, and bilinearity eliminates the remaining sum:

\[
\operatorname{Cov}(G_C,D)
=\frac1{M^2}\sum_m\operatorname{Cov}(c(\xi),A(\xi)\Delta_m)
=\frac1{M^2}\operatorname{Cov}\!\left(c(\xi),A(\xi)\sum_m\Delta_m\right)=0.
\]

The transpose cross term also vanishes. Therefore, under these stated conditions,

\[
\boxed{\operatorname{Cov}(G_R)=\operatorname{Cov}(G_C)+\operatorname{Cov}(D)},\qquad
\operatorname{Cov}(D)=\frac1{M^2}\sum_m
 \operatorname{Cov}(A(\xi)\Delta_m)\succeq0. \tag{6}
\]

The calculation includes the own-CE term inside \(c\): CE and target perturbations may be correlated within one route; their cross-covariances cancel **after summing over routes**. Equation (6) means route assignment adds a positive-semidefinite instantaneous shared-gradient covariance under this idealized law, potentially zero. It does not establish lower noise. No strictly positive variance is guaranteed in any direction.

The same identities hold conditional on a common state/graph/panel/other shared random variable \(H\), if paths are conditionally iid, targets are fixed given \(H\) and independent of the remaining path randomness, and \(\sum_m\Delta_m=0\) conditional on \(H\). Since \(\mathbb E[D\mid H]=0\) and the conditional cross-covariance is zero, the law of total covariance gives the unconditional version too, with \(\operatorname{Cov}(D)=\mathbb E[\operatorname{Cov}(D\mid H)]\). Arbitrary correlated route paths, unequal path/Jacobian laws, or targets chosen using their current dropout do not justify (6); in general both cross-covariance terms remain. Fully identical route paths instead give \(D=0\) pathwise by (3).

## Limits in the actual frozen implementation

The frozen code uses distinct persistent member RNG streams, seeded `seed + 1009*member + 300001` (`C/portable.py:123–150`), and retains two views. Unit factors therefore establish identical deterministic routes, **not identical realized training dropout paths**. Separate seeded pseudorandom streams are a source fact; mathematical conditional iid is an assumption for (5)–(6), not a measured property established here. After private factors or states diverge, even fresh iid masks generally have unequal parameter-Jacobian laws, so the mean identity no longer follows.

Equations (1)–(6) are exact real-arithmetic identities for the specified target construction. The facade constructs the NumPy mean and then casts route/common caches separately to the model dtype (`H/session_objectives_adapter.py:37–49`; `U:36–48`), and the loss casts weights again and checks row sums with tolerance (`context_alignment_objective.py:20–28`). Casting can make the actual cached common matrix differ from the exact mean of the cached route matrices; floating-point reductions and VJPs also round. Thus the source supplies no guarantee of **bitwise** shared-gradient equality, even with identical views. This does not break mathematical target linearity, but a literal runtime exactness claim would be too strong.

The covariance identity is specific to equal route coefficients, equal anchor/member averaging, fixed target-independent score denominators, and the same two-view objective. Changing those rules requires a fresh derivation. The registered permuted control preserves classes and scored degrees, but generally has a different entrywise aggregate target from the factual masks (`context_positive_masks.py:88–124`); (3)–(6) do not compare it with the factual common arm unless their means match. The available `cycle` helper is not a frozen Context9 dispatcher arm and does not preserve per-update aggregate mass.

Adam keeps first and second gradient moments and applies an elementwise ratio (`C/portable.py:117–122`; replay steps after full accumulation). Equal expected gradients do not imply equal gradient second moments, equal expected Adam updates, or equal optimizer trajectories. Exact current shared gradients together with identical shared optimizer history suffice for equality of that one shared update in mathematical arithmetic; changed private rows can alter the next path. Nothing here proves competence, correctness acquisition, convergence, superiority, or novelty.

In plain English: common and factual targets supply the same aggregate positive supervision. When all routes follow the same path, their changes cancel in shared coordinates while they can redirect separate private coordinates. With idealized independent identical stochastic paths, cancellation holds for the shared **mean**, while the route assignment can add shared gradient variability. Learning changes the paths and optimizer histories. That is why aggregate supervision equality still requires the frozen representative complete-training comparison; this calculation cannot replace it.

## Exact static source custody

All prefixes below expand under `P = /Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930`:

- `R = context_positive_staged_scientific_adoption_root_20261008_v1`
- `H = contrastive_BE_steering_continuing_research_20261007_v1`
- `I = H/integration_successor_v2`
- `U = portable_context_steering_public_interface_20261008_v1`
- `C = portable_internal_be_public_interface_20261007_v2`

The following SHA-256 values were computed directly from the inspected static source bytes on 2026-10-08. The target archive hash `80d30ed946a1408f861a773952b174713c487765c75e95b8e25456b30478eae6` is only a **declared binding read in `I/frozen_targets.py:9`**; its archive was not opened or rehashed.

| Exact file under the prefixes above | SHA-256 actually computed |
|---|---|
| `R/FROZEN_PROTOCOL.json` | `2b62738d48e651bb329eae6f289d8c225bd6163b349de8a3440eacef2adc7f58` |
| `I/SOURCE_BINDINGS.json` | `6a5234eb3e1f548b063d5d128b95784066a20d3a43a2763f62543d961f80f6e4` |
| `I/context_dispatch.py` | `1f2634d51c3313585538c35867b3a3ecc3aae7f0f5a19e8cd7a3bc429a729b4e` |
| `I/frozen_targets.py` | `ce14c6b6b887dd060e681dd6f94981b0a2f2db96879f2147566958238fa89149` |
| `I/context_recompute.py` and `U/context_recompute.py` | `120d6863fe57da53e115c78d8adc9ff39864ae932b0479df84686479f19f86a6` |
| `I/train_context.py` | `8362a8fcaed090511f65076277d3f1ed297d37800d4e3b91d8674617e74ca7d1` |
| `H/context_alignment_objective.py` and `U/context_alignment_objective.py` | `3fe36a626b3e408247cf557a1bd89a75d70147e400325043a8a3d7d47f81adfd` |
| `H/context_positive_masks.py` and `U/context_positive_masks.py` | `7615d8441a699f297314e7c838b9eb897b9e3dbfe22d1be350742e83f21405a9` |
| `H/session_objectives_adapter.py` | `23c58328c9b4ff9317576fca7c30ae5c3a93deb1a4c41e69187a1799eb0a4a24` |
| `U/session_objectives_adapter.py` | `fde9092c2e3775b2dcdb0fdf76e1ecde632b775082bafc3c55c1d5d1ddac9009` |
| `U/context_dispatch.py` | `1fccb25d548031e426c3f325449861f1a736cae4b3c11f40b865363fc7a81b32` |
| `U/train_targets.py` | `84798ab339baf89598d43d3a2d87430c84982602ce11c479d03e244fcb3bd88b` |
| `U/SOURCE_ORIGIN.json` | `ec7aa13d5eaa61e7aacf26c834b3542f97154c46a4286950ff6d668affe28777` |
| `U/MANIFEST.json` | `2222beb59a2279602cc72723a6cd924ec6978fbd6a02e46a73db781b31736ed5` |
| `C/portable.py` | `29beab567eeb4c070c432e2ce31990c88ec2d8c8618aa5dfb85ebedcf4f0ca14` |
| `C/core/factors.py` | `9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3` |
| `C/core/models.py` | `2be6d872e962d9fdd341883b0d1060170fe0a176f7a8d6829d988fc6f8e7f9c3` |
| `C/core/objectives.py` | `9208e677ab5cfb286862216991453f8592a1d373120019a6e2ff14c38fad5bcf` |

`I/SOURCE_BINDINGS.json` identifies the helper and shared-core files used by the frozen successor. The directly computed helper/core hashes agree with those bindings. `U/SOURCE_ORIGIN.json` and `U/MANIFEST.json` identify the public copies; direct hash equality above confirms the mask, alignment, and replay copies. The public facade has a different module docstring, while the inspected objective/cache behavior is the same.
