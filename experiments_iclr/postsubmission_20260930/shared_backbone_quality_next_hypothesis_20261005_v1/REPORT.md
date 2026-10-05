# One next hypothesis: shared features that support private learning across endpoints

## Scientific claim to test

Train the shared core to make the **served ensemble useful after private learners update on different endpoints**. The hypothesis is that this criterion can favor transferable graph features over representations that cheaply fit repeatedly supervised endpoints, while retaining competent private compatibility/context routes. It changes ongoing learning credit; it adds no reflection, embedding repulsion, inference gate, or external labels.

This is an unproved finite-sample/optimization hypothesis. Independent ensembles can represent the shared solution by tying parameters. A capable single can also use meta-training. Superiority to either requires empirical evidence under the strong controls below.

## Why current sharing can fail at the served predictor

For binary cross-entropy on raw logits, with \(\bar z=M^{-1}\sum_m z_m\),

\[
M^{-1}\sum_m\ell(y,z_m)=\ell(y,\bar z)+D,
\qquad D=M^{-1}\sum_m\operatorname{softplus}(z_m)-\operatorname{softplus}(\bar z)\ge0.
\]

Thus own-member supervision optimizes served loss plus a Jensen penalty. Its shared gradient can suppress differences that help the mean prediction. The retained shared-gradient assessment also shows that aligned member loss gradients can erase a useful relative margin when route Jacobians differ; conflict between loss gradients is not the only relevant failure. Conversely, pooled loss can reward compensating, individually poor outputs—the retained learner-collusion prior. More disagreement alone cannot diagnose success.

Graph queries repeatedly reuse endpoints and neighborhoods. A shared representation may encode shortcuts that all heads exploit on overlapping supervision, while failing to support useful corrections elsewhere. The proposed criterion asks whether private learning transfers to queries whose endpoint labels were excluded from that **current** inner update. These are mechanisms to test, not diagnosed causes of the frozen cohort's behavior.

The finite-sample rationale for beating an independent ensemble is that one constrained core must support several private learning trajectories on endpoint-separated queries. This may reduce encoder-specific shortcut variance without removing private context corrections. It supplies no extra labels; all independent members can see the same data and can use the same transfer rule. If their additional freedom generalizes well, sharing offers no quality advantage.

The rationale for beating a capable single is that separately adapted nonlinear context routes may make complementary, competent corrections whose mean transfers better than one adaptation path. A single with sufficient capacity and the same schedule may learn those interactions equally well. This possibility motivates the ensemble-specific objective and the adapted-single control; it is not an expressivity separation. The joint meta-gradient can select features supporting useful cancellation on outer queries, while the own-loss anchor limits reliance on poor compensating heads.

## One specified learning rule

Keep a fixed source-qualified **NCN** architecture with shared parameters \(\theta\) and private route parameters \(\phi_m\). The partition must cover every trainable parameter, including tied decoder matrices and private factors. Existing frames may be retained unchanged; the learning rule does not require them.

At each training episode:

1. Draw an outer query batch \(O\) using the declared **TRAIN-only native positive/negative query generator**, independently of model scores. Let \(V_O\) be its endpoints. For each member draw an inner batch \(I_m\) from TRAIN queries having neither endpoint in \(V_O\), with fixed sizes and class balance. Use fixed member-specific streams.
2. Build the episode support graph by removing positive target edges in \(O\cup\bigcup_m I_m\), applying native masking conventions consistently. All routes use this same observed support. Other TRAIN edges, including other edges incident to \(V_O\), remain available as graph context. The removed query labels remain available only to their assigned supervised loss.
3. Compute **one virtual** private optimizer step on that member's own inner BCE, normalized within member: \(\phi'_m(\theta)=U_m(\phi_m,\nabla_{\phi_m}L^{I_m}_{\rm own};s_m)\). Starting private parameters and previous optimizer state are constants for this episode's differentiation. Differentiate through this update, including its specified optimizer transformation. Preserve the initial parameters/moments; do not commit the virtual step.
4. Evaluate the virtual updated routes on \(O\) using the **dropout-off serving map** on the episode support graph. Update shared parameters to \(\theta^+\) using the full live-path derivative of

\[
F_O(\theta)=\tfrac12 L_O(y,\bar z')+	frac12M^{-1}\sum_m L_O(y,z'_m),
\quad z'_m=f(\theta,\phi'_m(\theta)),\quad
\bar z'=M^{-1}\sum_mz'_m.
\]

5. **Recompute** the private optimizer step at \(\theta^+\), starting from the original \(\phi_m,s_m\), on the same inner batch, support graph and captured inner randomness. Commit \(\phi_m^+=\phi'_m(\theta^+)\) and its corresponding moments once. Discard all virtual state changes; advance native RNG/buffer state only under the declared committed-step convention.

The equal weighting is fixed, with no proposed sweep. The first term scores the raw-logit mean; the second anchors member competence. The objective equals served BCE plus half the Jensen penalty; this scalar tradeoff is prior. Shared optimizer state advances once on the meta-gradient; private parameters receive the recomputed inner own-loss update only. Private gradient normalization is an explicit convention to reproduce in every matched control, rather than assuming Adam makes rescaling immaterial. Exact higher-order support must be qualified before implementation. An NCNC extension with detached completion needs an explicit surrogate-derivative contract; it is not silently covered by the exact NCN statement.

The extra credit is

\[
\nabla_\theta F_O=\partial_\theta F_O+
\sum_m(D_\theta\phi'_m)^T\nabla_{\phi'_m}F_O.
\]

For an SGD private step the second term contains the mixed inner-loss Hessian. It assesses how changing shared features changes what a private learner can learn. A stop-gradient through \(\phi'_m\) removes precisely this term, furnishing a matched causal control. This differs from the retained direct shared-pooled/private-own policy and the retained response-preserving projection.

The persistent state after an episode is \((\theta^+,\phi_1^+,\ldots,\phi_M^+)\), including the recomputed private moments. With fixed episode state, these weights realize the criterion at \(\theta^+\); the unadapted starting heads are not served. After training, serve this committed state through the ordinary frozen predictor \(M^{-1}\sum_m f_{\rm eval}(\theta,\phi_m)\) on the allowed inference support graph. There is **no inference adaptation**, label request, member selection, or router. If probabilities are needed, apply sigmoid to the mean logit; do not average member probabilities. Official validation may supply the same fixed checkpoint selector across arms, but no non-TRAIN label enters an update.

## Is the graph task distribution defensible?

The outer queries follow the declared query generator. Endpoint separation changes the source of the private learning signal; it does not create a certified distribution of new graphs or new nodes. It is a conditional transfer regularizer on one observed graph. Surrounding support, features, historical private states and shared parameters remain correlated with the outer endpoints. Rotating TRAIN queries have influenced earlier updates, so this is **not an unbiased cross-fitted estimate of generalization**.

An edge spanning an outer endpoint and another block cannot enter an inner supervised batch for that episode. An observed spanning edge outside the target-query set can remain graph context; its existence must not be misdescribed as hidden. All validation/test edges, reverse-edge duplicates and equivalent target facts must obey the native label-visibility contract. Query-positive edges cannot remain in support. No future label may be consulted to repair sampled-negative noise.

Filtering inner queries changes their degree distribution and can remove many candidates around hubs. Fix \(b_{\rm in},b_{\rm out}\) before fitting, count rejection/coverage, and predeclare an ordinary matched-update fallback for infeasible episodes. Resampling away difficult outer endpoints would silently alter the target distribution. The fallback rate is evidence about applicability. Labels are temporarily excluded, not globally discarded, but unequal exposure must be reported.

This rationale is strongest when deployment includes sparse endpoint supervision and correlated repeated-node queries. Ordinary transductive HeaRT evaluation does not automatically meet that premise. The degree/common-neighbor-matched random-split control is essential: an arbitrary carving benefit or additional target masking would not establish graph endpoint-transfer utility.

## Closest prior and remaining difference

Retained MAML supplies differentiation through learning; GNCL/TreeNets and learner collusion supply own/pool objective ancestry; CAMERO/TabM supply shared/private ensemble supervision. The retained graph-gradient and curvature packets already propose contrast protection and one-step initialization selection. None should be relabeled as this proposal's discovery.

Three targeted ambiguities were resolved:

| Primary scope | What is already established | Difference of this proposal |
|---|---|---|
| **ANIL**, [1909.09157v2](https://arxiv.org/abs/1909.09157v2), §4 and AppendixC.1 | Head-only inner adaptation with second-order outer learning of reusable features. | Persistent NCN route learners; endpoint-separated query supervision; a competence-anchored served ensemble objective; no adaptation at inference. The meta operator is attributed. |
| **Meta-Graph**, [1912.09867v2](https://arxiv.org/abs/1912.09867v2), §2, §§3.1–3.2, §3.4 | Meta-learning link predictors across a distribution of related graphs, with local adaptation and held-edge outer updates. | This proposal uses one graph and correlated query episodes. Meta-Graph's multiple-graph assumption does not justify arbitrary endpoint blocks. Graph meta-learning is attributed. |
| **BMAML**, [1806.03836v4](https://arxiv.org/abs/1806.03836v4), §§3.1–3.2 and exact sharing paragraph | Ensemble meta-learning, likelihood/chaser objectives and interacting particles. Its image setup explicitly shares the feature extractor, has private classifiers, adapts classifiers internally, and meta-updates shared features/classifiers; EMAML uses that sharing scheme too. | The proposed raw-logit objective, private inner-only updates, persistent routes, NCN detach/context semantics and endpoint-conditioned supervision differ. Shared-backbone ensemble meta-learning itself is direct prior. |

The new item **in this research program** is this complete graph training criterion and its testable endpoint-transfer rationale. No new bilevel, ensemble, sharing, or negative-correlation principle is claimed. Broader novelty remains unresolved. A positive result would establish utility of this exact recipe; attributable sharing benefit also requires the corresponding untied control.

## Precise limits and decisive controls

If all private routes, optimizer states, effective update maps and data/RNG paths are identical, their adapted logits coincide. Then both terms in \(F_O\) equal single-route BCE. The recipe reduces to the corresponding single learner. Different routes are therefore necessary for ensemble-specific benefit, and this method does not force useful differences.

For fixed episode batches, support, inner randomness and starting optimizer state, suppose the live-path NCN/update composition makes \(F_O\) differentiable with a \(\beta\)-Lipschitz gradient. A plain shared gradient step of size \(0<\eta<2/\beta\), followed by the specified private recomputation, obeys

\[
F_O(\theta^+)\le F_O(\theta)-\eta(1-\beta\eta/2)\|\nabla F_O(\theta)\|^2.
\]

This is standard local descent for this episode's **committed post-adaptation state**, not a new theorem. It does not cover native Adam automatically, non-smooth crossings, later episodes, final support-distribution changes, generalization or ranking. Without private recomputation, even this state alignment would fail.

Require a prospectively separate comparison including:

- Ordinary competent single and independent four-member ensemble, plus a capable single using **the identical inner/outer schedule and objective**.
- Four independent encoders using the identical ensemble transfer rule. This is the direct sharing control and has additional representational freedom.
- Shared routes with the same batches, support graph, private recomputation, competence anchor and outer forward, but detached virtual private update in the shared derivative.
- Degree/common-neighbor/exposure-matched random query separation, and ordinary training with the same target masking. These isolate endpoint separation and graph exposure.

Predeclare total episodes \(U\), one **committed** private step and one shared step per episode, query counts, optimizer/state conventions, checkpoints and selector. Every episode pays for the virtual all-member inner predictions/gradients, dropout-off adapted outer predictions, full meta backward with higher-order live paths, and all-member inner recomputation at the new shared state. The same inner examples are computed twice; count that exposure and cost. Charge native context work, retained-graph memory, failures, fallbacks and full training time. Encoder reuse is conditional on compatible support/RNG/buffer states; no free or fixed-factor cost claim is made. Match update/exposure schedules in diagnostic arms, and give competent ordinary alternatives comparable complete training/selection budgets. Serving still pays one shared encoder plus all native route work. No hidden warm stage or final refit is proposed.

**Falsification:** no replicated served ranking gain over both competent single and independent ensemble rejects the main accuracy claim. Matching the detached-update control rejects the extra adaptation-credit explanation. Matching the random separation control rejects endpoint-specific attribution. Matching the adapted single supports generic meta-regularization rather than ensemble benefit. Better TRAIN/outer BCE, extra disagreement, competence preservation, or lower storage cannot rescue absent served quality. Extensive fallback, unstable meta-gradients, or support-distribution mismatch would identify a practical failure of this defined recipe.

No outcomes, numerical/model execution, GPU jobs, server actions, canonical edits, agents, or changes to the frozen Citeseer cohort were performed. Sources, retained conclusions and exact reading scopes are confined to this packet.
