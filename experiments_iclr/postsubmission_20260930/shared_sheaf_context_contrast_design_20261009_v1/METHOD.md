# Learn complementary graph corrections to a common local predictor

**Disposition:** retain one exploratory learning-rule hypothesis; reject novelty and superiority claims at present. Sharing learned weights while ensembling private geometry already has BSNN precedent. Global correlated latents, masked-context supervision, residual/product-of-experts learning, balanced expert assignment and BatchEnsemble factors are established ingredients. The specific operation below is absent from inspected BSNN §3, but its priority outside that scope is unestablished.

## 1. The operation

Train each persistent member's geometry to supply a **label-correct graph-context correction** to the same frozen feature-only predictor. Give members different fixed graph contexts, allocated globally rather than separately per edge or query. Preserve complete full-graph supervised prediction as the primary task. There is no member-vector repulsion, no requirement to make another correct member wrong and no posterior-sample interpretation.

Let θ contain every shared native stem, feature map, incidence slow weight, epsilon vector and head. Let η_m contain **only private factors in the ordered incidence learners**, one set per layer. Heads/feature maps have no private factors in this design. Each member executes its own complete native general-map path and has its own current states. Main training and serving remain

`F(θ,η) = mean_m mean_(q∈TRAIN) [-log p_(θ,η_m)(y_q | G,X)]`,

`p_pool = mean_m p_(θ,η_m)(· | G,X)`.

The factor parameterization is an implementation choice; the learning operation is the context-conditioned contrast and its recipient. It can also be applied to a structured global-latent sheaf generator.

### Fixed contexts

Before fitting, partition the full unlabelled support into four deterministic balanced graph patches P_k. Labels do not choose the partition. Define evidence sources `A_k = P_k ∩ TRAIN`. Select one fixed TRAIN query panel Q by a seeded, class-stratified hash (64 queries, 32 per binary class), and remove Q from all evidence-source sets. No query's raw features are an auxiliary evidence source.

View `X^(k)` retains original feature rows only for A_k; every other feature row, including Q, is zero. The **same complete graph and original native propagation** are used. Labels are never feature inputs. No edge deletion, degree rewrite or new mask-token channel is introduced. For the auxiliary loss use the predeclared structurally reachable queries `Q_k = {q∈Q : dist_G(q,A_k)≤L}`. All controls use the same panels/eligibility; primary F still uses every TRAIN row. Reject this representative design if any Q_k has fewer than eight queries or lacks either class; do not choose a more favorable panel after fitting.

These are observational training views, not claimed causal interventions or guaranteed label-preserving worlds. A member must learn to predict the observed labels from restricted context. Structural priors or memorization can still explain success.

### Contextual logits with the local shortcut removed

Run the actual native model on X^(k), giving logits `z_m^k(q)`. Define `z_θ^0(q;X^(k))` as the same model's **zero-message reference**: its captured stem, native epsilon residual scales and shared head, with every propagated message zero. In original general NSD this reference is analytic:

`H^0 = D_ε H_stem`, where `D_ε = product_l diag(1+tanh ε_l)`;

`z_θ^0 = W_head vec(H^0) + b_head`.

The native residual recurrence makes this exact for that reference. Per-layer dropout is applied to message paths, not the stored residual; the stem realization must be the one captured from the context forward. No second graph construction or dense graph matrix is needed. The reference is independent of η_m. With shared geometry-only ownership it cannot be sabotaged by the auxiliary private update.

Set `d_m^k(q) = z_m^k(q) - z_θ^0(q;X^(k))`. Bias and the no-message residual cancel. Native log probabilities can be subtracted instead: the additional class-constant normalization shift cancels in the subsequent softmax. This is a **predictive graph contribution**, not a distance between raw restriction maps.

Use one competent, restored, frozen feature-only MLP from the required ordinary baselines as a common local predictor `r(c|x_q)`. It receives only x_q and the same TRAIN label opportunity. It is not trained on extra truth. Cache its query probabilities; clip only to `[10^-4,1-10^-4]` and renormalize. No teacher-strength search is proposed.

The contrast is

`p_corr,m^k(c|q) = softmax_c [ log r(c|x_q) + d_m^k(q)_c / τ ]`,

`J_mk = mean_(q∈Q_k) [-log p_corr,m^k(y_q|q)]`, with `τ=1`.

The correct class is contrasted against the other classes, **not against another member's correct prediction**. This is a product-of-experts/residual likelihood used as a training signal, not a Bayesian factorization theorem. Local and graph evidence need not be conditionally independent.

### Coherent ownership and recipients

After 100 full-task F-only warmup epochs, evaluate the four context tasks on TRAIN Q only. Solve one deterministic balanced assignment

`Π* = argmin_(Π permutation matrix) sum_(m,k) Π_mk stopgrad(J_mk)`.

Use fixed hashed ties, freeze Π, and give each member its assigned context for the remaining training. A whole view belongs to one mode over the whole graph and all subsequent iterations; no incidence-wise lottery or per-query winner routing is used. All members continue to learn F on all TRAIN nodes and serve the complete graph.

Updates are explicitly

`g_θ = ∂_θ F`,

`g_ηm = ∂_ηm F + (λ/M) ∂_ηm J_(m,k(m))`, with `λ=0.1`.

For J, θ and the teacher are held constant. Gradients through current states and the native map/normalization recurrence into η_m remain intact. There is no auxiliary gradient to shared heads/features or a learned free probe decoder. This is a separate recipient rule, not the gradient of an ordinary θ/η joint ELBO.

## 2. What the contrast adds—and what it does not

With `d=0`, `p_corr=r`: a confidently correct local prediction supplies little auxiliary gradient, while a wrong local prediction supplies a strong correct-class gradient. Specifically,

`∂J/∂d_c = [p_corr(c)-1(c=y)] / τ`.

Ordinary masked-view CE instead gives `[p_ctx(c)-1(c=y)]`. Primary full-task or pool loss can exploit the local feature path without teaching a mode to correct it from a specified graph source. J asks a restricted graph context to supply the missing class correction, and sends that request specifically to the private geometry.

The added signal is nevertheless close to **residual boosting, teacher-error weighting and supervised view augmentation**. If r is uniform, J becomes ordinary CE on d. A hardness-weighted ordinary context loss may explain the whole gain. A teacher that already solves TRAIN can leave negligible useful residual signal; do not force artificial diversity in that case. A poorly calibrated teacher can bias the correction. These are decisive controls, not reasons to invent an additional penalty.

More precisely, `p_corr ∝ r * p_native_context / p_native_null`. If `r=p_native_null`, J is **exactly ordinary context CE**. Otherwise it adds an explicit query-dependent class-logit adjustment `log r - log p_native_null`; it does not create a new evidence statistic. This algebra is why a matched ordinary/teacher-hardness context arm is required, and why the contrast itself receives no novelty clearance.

Topology/index tensors and the four context masks are built once. Each post-warmup step needs the M ordinary full-input paths plus one native context path per member; the analytic null reference and cached teacher add no graph path. Native state-conditioned maps/SVD are recomputed within those forwards as required. There is no per-step graph reconstruction, dense `N×N` diagnostic, path-energy probe or latent-distance objective. Report the extra context forward/backward work and frozen teacher storage.

**Native-compatible sanity separation:** if every native right feature matrix is zero, propagated messages are zero regardless of the private sheaf. Then `z_m^k=z_θ^0`, `d=0` and `∂J/∂η=0`. Pure geometric differences invisible to prediction receive no reward. This addresses the immediate failure of an operator-norm contrast that could improve while the native residual/head ignored the graph. It does not guarantee generalization, independent evidence or useful ensembling.

Ownership can still degenerate into regional specialization, node-frame adaptation, graph-structural class priors or memorized masked queries. Full-input pooled quality, context transfer to VALID and matched replacements must decide utility. Training context competence or different matrices alone is insufficient.

## 3. BSNN collision and gauge/native semantics

BSNN §3.2 explicitly factorizes conditional draws over incidences; §3.3 constructs distributions from input features and samples independent sheaves per layer. This limits **joint stochastic uncertainty**, not the ability of its learned conditional mean to be one coherent graph operator. Shared distribution parameters already couple learning across edges. State feedback or a global latent `q(z) product_(u,e) q(F_(u,e)|h,z)` can supply graph-wide sampled modes; this is an existing structured-posterior route, not our invention.

Changing that representation while retaining its ordinary ELBO does not introduce the context task, common local reference, global context assignment or geometry-only J recipient. Conversely a structured BSNN can receive the **same J**. If it replaces the proposed finite bank, persistence/BE has no unique scientific role. Deterministic learned η_m are not valid posterior samples or automatically finite-KL Dirac approximations. No Bayesian uncertainty claim is made.

Use original general NSD exactly: one ordered endpoint learner; possibly singular tanh matrices; reverse incidence pairing; block-degree SVD normalization with native augmentation, training jitter and clamping; ELU and native epsilon/residual updates. Every context view recomputes the operator from its own current states using the same static support. Keep training jitter for differentiable J; evaluating the zero-jitter SVD with gradients can be unstable. The detached unnormalized `.L` diagnostic is not a substitute for live propagation.

The contrast compares class predictions of the actual context path and its well-defined zero-message reference. Under a **fully compensated** hidden-coordinate reparameterization preserving both functions, it is unchanged. Rotating only restriction maps while holding native ELU, feature maps, head or normalization fixed is not such a reparameterization. Arbitrary general-map blocks are not orthogonal transports, and transpose is not inverse; no generic holonomy/curvature claim follows. Flat/node-frame and matched node-capacity replacements remain necessary to distinguish useful graph corrections from feature-frame alignment.

## 4. Claim boundary

This is one concrete **predictive context steering operation** relative to the inspected BSNN objective, not merely a new latent covariance or BE placement. Its pieces collide with familiar masked prediction, residual experts, balanced specialization and geometry sharing. A substantive novelty claim is currently unsupported and rejected. A useful result would be a bounded sharing/learning-rule result only after the paired controls in `EXPERIMENT.json` and decisive falsifiers pass. No model, data, outcome or server was accessed for this design; no frozen/live packet changed.
