# Learnable contrast strength inside a shared BatchEnsemble GNN

7 October 2026. Technical follow-up to the sealed operator-diversity report. **A learnable coefficient needs an objective tied to task utility; making the current coefficients ordinary trainable parameters does not provide that.** No novelty or outcome claim is made, and the frozen 24-cell suite is unchanged.

## 1. Why ordinary coefficient learning fails

The inspected frozen WikiCS source is `objectives.py` and `run.py`. Every arm receives two own-label stochastic views. Enabled arms add

`L = L_own + λ_A A + λ_R R`, with `λ_A=λ_R=.05`, temperature `.2`, and at most512 TRAIN objects for the auxiliaries.

Both A and R are nonnegative cross-entropies. R is exactly zero for one member. For ordinary joint minimization with `λ_j≥0`, `∂L/∂λ_j=R_j≥0`: the optimal coefficient is zero whenever the loss is positive. `softplus(a)` merely drives a toward−∞; a lower bound forces the bound. A zero auxiliary loss leaves the coefficient unidentified by that term. When valid centered objects remain and M>1, normalized scores lie in `[−1/τ,1/τ]`, giving `R≥log[1+(M−1)exp(−2/τ)]>0`. This makes the zero-coefficient issue especially direct. The all-skipped residual case instead returns zero.

Changing R to a signed diversity reward does not solve this. If the objective is `L_own−λD`, D>0 and λ is unbounded, minimization drives λ upward without bound; a bounded λ generally saturates at its upper bound. Letting λ take either sign is worse for a nonnegative loss. Adding a constant to R leaves a fixed-weight network update unchanged but changes a freely learned coefficient's gradient. Such coefficient learning is not invariant even to an arbitrary loss offset. Entropy/log-barrier or uncertainty-weighting terms can prevent the trivial endpoint, but choose another balancing objective; they do not certify graph-task compatibility.

Strong repulsion can harm useful representations even with λ fixed: members must share information about Y; centering class means removes a first moment, not every label-relevant feature. Centered vectors can encode member identity or head-nullspace directions without changing predictions. Conversely, forcing their separation can distort useful within-class geometry. The frozen residual loss differentiates through live centering, shared W and private factors. Zero-sum gradients in representation coordinates do not freeze learned class means after a shared parameter update. A coefficient on this loss therefore does not isolate steering to private factors.

[DICE, §2](https://arxiv.org/html/2101.05544v1#S2.SS1.SSS2.p2) is the directly relevant task-information precedent: reduce redundancy conditional on Y while retaining class relevance. Its adversarial/bottleneck estimator is different from this centered cosine-based CE; the latter supplies no conditional-MI or label-preservation theorem.

## 2. A supported mechanism: learn strength from a primary-task response

[Auto-Lambda](https://arxiv.org/abs/2202.03091v1) is close prior for learning auxiliary-loss weights through a primary-task meta-loss. Its pinned [author implementation](https://github.com/lorenmt/auto-lambda/blob/bdba6365953f7ebb3a8ae33fabfecd3f933ecc27/auto_lambda.py#L13) forms a virtual network update from weighted training losses, evaluates only the chosen primary tasks, then computes a finite-difference hypergradient for the weights. The coefficients are not minimized against their own auxiliary losses.

For a prospective graph adaptation, θ includes shared W and all trainable factors; α denotes initialization strength, if later made a differentiable parameter. Let A/B be prospectively fixed permitted TRAIN roles, or a fixed minibatch-role schedule, with B supplying primary labels independently of each inner update:

`θ⁺ = Update(θ; L_own,A + λ_A A_A + λ_R R_A)`

`min_{0≤λ_A,λ_R≤λ_max} F_pool,B(θ⁺)`,

where F is the NLL of the actual mean-softmax pool, with own-member losses monitored or given a declared task-loss budget. For a simple SGD step of size η, the one-step hypergradient is

`∂F(θ⁺)/∂λ_R = −η ∇F(θ⁺) · ∇R_A(θ)`.

Thus an auxiliary direction that improves the primary task is rewarded; a harmful one is downweighted. Zero remains a valid outcome. This learns whether steering helps, rather than guaranteeing nonzero diversity. The same logic can optimize initialization strength through the training trajectory, but α must change actual initial factor values; adding a multiplier to an already detached/reset random initialization has no such gradient. Keep α fixed first to avoid confusing initializer and steering effects.

This is an **attributed prospective adaptation**, not native Auto-Lambda or a guaranteed safe rule. Its CIFAR author source uses another shuffled loader over the same TRAIN set, not a heldout split, and does not clamp its meta-weights. Bounded weights, separated TRAIN roles and a served-pool outer objective would be explicit modifications. Graph-neighbor dependence does not become iid through splitting node IDs. One-step evidence may miss later harm; short-lookahead meta-gradients, full Adam state, shared/private coupling and graph passes all require cost and source qualification. No official VALID/TEST labels should be repurposed to learn coefficients in the present pilot.

A stronger **local task-loss safeguard** is possible, without promising generalization. For `g=∇L_task`, `h=∇R`, an SGD step `d=−η(g+λh)` preserves a fraction of first-order task descent when

`g·h<0 ⇒ λ ≤ κ||g||²/(−g·h)`, with fixed `0≤κ<1`.

For `g·h≥0`, a separate step/trust bound is still needed. This is ordinary gradient-compatibility reasoning, adjacent to the already indexed [PCGrad](https://arxiv.org/abs/2001.06782v1), not a new mechanism. It protects only the stated local task loss to first order. Native Adam needs the actual optimizer-induced direction, and finite-step checking needs the true post-step loss; an Euclidean bound cannot be silently transferred. Pool NLL and own CE are different objectives, so state which task is protected.

## 3. Closest priors and the initialization boundary

DICE supplies conditional task-compatible diversity motivation; Auto-Lambda supplies primary-task-driven weight learning; PCGrad supplies conflict-aware gradient ancestry. Previously inspected **σN-Ens (2026)** supplies learned compact internal modulation with continuing regularization, but its inspected temperature/coefficient rule is not evidence of meta-learned graph contrast strength. CDLG, SuGAr and attention disagreement regularization already supply contrastive/graph-evidence diversity objectives. TabM supplies early randomized adapters plus own-member losses. None of these scoped findings establishes the complete proposed combination's novelty or its utility.

The sealed report's distinction remains: factors can change actual evidence dependence through private nonlinear gates, recomputed attention, and root/hop weighting. Learned λ controls the training incentive; it cannot create graph capacity absent from the architecture. Initialization establishes a starting asymmetry, while continuing steering may maintain or redirect it. They are distinct interventions, not guaranteed substitutes.

**A compact capacity check:** for one linear BE map `W_m=diag(r_m) W diag(s_m)`, with the relevant factors and entries nonzero, every member shares the same entry cross-ratios:

`W_m,ij W_m,kl / (W_m,il W_m,kj) = W_ij W_kl / (W_il W_kj)`.

Row/column factors cancel. Equivalently, the elementwise ratio of two member maps is a rank-one row/column scaling. For example, the full-rank target maps `[[1,1],[1,2]]` and `[[1,1],[1,3]]` have cross-ratios2 and3 and cannot both be realized by that single shared map in those fixed coordinates. Learning W changes the common ratio over training but does not give each member a different ratio at the same step. Larger hidden/factor distances cannot remove this compatibility restriction. It distinguishes **insufficient independent map capacity** from merely insufficient separation of representable maps.

This is a single-map algebraic fact, not an expressivity theorem for deep nonlinear/residual GNNs, private readouts, coordinate changes or matrices with zeros. Zero factors change support and make the displayed ratios undefined. Capacity remedies are also prior: indexed [LoRA-Ensemble](https://arxiv.org/abs/2405.14438v5) uses private additive low-rank attention adapters around a frozen shared transformer. Such a family can leave the row/column-scaling constraint, while retaining its own rank restrictions. It is an attributed, different capacity control; stronger λ alone is not that control.

## 4. What the fixed-coefficient 24-cell pilot identifies

The frozen design is eight arms × seeds6101/6203/6307,1100epochs. Its primary contrast is `be_init_contrastive−be_init`. The four BE arms permit a fixed-dose initialization/contrast interaction:

`(be_init_contrastive−be_init)−(be_unit_contrastive−be_unit)`.

That can test whether the chosen contrast package's effect depends on the chosen initializer. It cannot estimate a learned-strength benefit, a dose-response curve, an optimal λ, or an initialization-amplitude optimum. `single_contrastive−single` estimates single-model alignment, because its member residual loss is zero; it does not isolate alignment's contribution inside BE. Independent arms are useful competitors, but ordinary independent4 uses own-selected member checkpoints while the contrastive packed bank is jointly selected; their difference is not a clean contrast-loss-only intervention. The later eight-view single control remains necessary for a claim about sharing: shared W receives eight stochastic path gradients, versus the ordinary single's two.

**Predictive pilot results alone cannot separate functional from cosmetic diversity.** The separately frozen TRAIN panel adds four fixed graph/feature interventions, first/last local attention distributions, true-class margin/probability responses, shared errors and pool rescues/harm. Changed attention with unchanged served probabilities/responses is consistent with compensation; changed margin responses establish functional differences on those inputs. Embedding distance alone establishes neither.

The panel's580 targets were fitted; thinning/zeroing is not certified label-preserving. Five input variants sample only a finite set of responses. The panel can falsify a claimed functional change on its probes, or show that a hidden/attention difference reaches predictions; it cannot prove useful causal specialization, heldout gain or task preservation. Combine it with predeclared pooled utility and later independent predictive confirmation. Keep the present suite intact and evaluate this fixed package first; any adaptive-strength successor needs a separate protocol and matched label/update/compute controls.

Scope: reused sealed source conclusions without rereading their papers; read frozen config/objective/run slices and panel protocol as text only. One new primary abstract and one bounded pinned author-code scope support Auto-Lambda. Full-paper reads, imports/execution of model code, model/data/checkpoint access, GPU/server actions and frozen-suite edits: zero. Exact excerpts, hashes, rejected locator and limitations are saved alongside this note.
