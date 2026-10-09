# Joint proper source-context credit with additive rank-one routes

9 October 2026. **One preferred, inactive candidate:** train full-input and source-ablated predictions with proper supervised BCE, and give each assigned route additional proper BCE credit for restoring its source to an absent-peer committee. Optimize no negative absent loss. Use additive rank-one adapters as the known foundation, with multiplicative BE and genuine independent4 controls.

This is contextual augmentation plus recipient-restricted GNCL-style ensemble credit, with LoRA/TabLoRA parameterization ancestry. It is not a new proper score, low-rank primitive, attention rule or novelty claim. A positive representative quality result could still make this attributed foundation scientifically useful. No source implementation or run is authorized by this design.

## 1. Model, data and serving

Use the existing complete IMDB source/role/backbone contract, all TRAIN rows/five observed Bernoulli labels, the native full dimensions and six qualified affine interfaces. Let m=0,1,2,3. Slow native parameters theta are genuinely shared. Member-private eta_m consists only of per-group `u_m,v_m` at the two grouped projections and per-site rank-one factors at semantic Q/K/V and fc_after_concat:

`W_m=W+u_m v_m^T`, shared outside bias b.

Every u starts at0; v is independently seeded finite Gaussian input direction normalized to Euclidean norm1, once per native group/site/member. No redraw, radius/search, learned extra scale or private bias is introduced. Zero/nonfinite norm is a failed initialization. This is a LoRA-style native-function-preserving start; it initially exposes only outgoing-factor derivatives. Shared/nonsteered native weights, heads, biases, LayerNorm, PReLU, dropout and zero-start semantic gamma remain intact. Additive integration is not yet qualified merely because these native sites were qualified for BE.

Each member produces logits z_m^v and p_m^v=sigmoid(z_m^v) for views `v∈{full,−actor,−director,−keyword}`. The three ablations come from removal of actual raw typed supports **before** native feature/TRAIN-label propagation/normalization, with node counts and all unaffected support retained. A semantic-cache mask is not this intervention. Inputs/roles/labels are identical across views; VALID labels never enter propagated label inputs.

Assignments are fixed: a_0=actor, a_1=director, a_2=keyword; member3 has no extra source credit. All four remain full-label predictors. Serve every restored member on full inputs and take the strict native `probability>0.5` decision from the FP32 arithmetic probability mean. No source gate, learned router, ablated prediction or subset is served.

## 2. Exact positive objectives and recipients

Write B(p,y) for complete TRAIN marginal BCE over every row/label, with stable native logit/event-probability arithmetic. Define

`P=(1/16) sum_(m=0..3) sum_(v in four views) B(p_m^v,y)`.

This equal-weight four-view own-supervised objective gives every member a factual task loss and proper supervision on all source-absent inputs. It is ordinary source-ablation augmentation; it can trade factual quality for robustness and gives no finite nonworsening guarantee.

At the same old parameter state and declared native stochastic realizations, collect all16 predictions before any optimizer step. For an active recipient m and its source a_m, let

`C_m^a=(p_m^full + sum_(k!=m) stopgrad(p_k^(−a)))/4`,

`S_m^a=B(C_m^a,y)`.

The absence peers are detached **only for this extra credit**; their own P losses remain live. Use one fixed relative coefficient1. The actual joint update vector is

`g_theta=partial_theta P`,

`g_eta_m=partial_eta_m P + (1/3) partial_eta_m S_m^(a_m)` for m=0,1,2,

`g_eta_3=partial_eta_3 P`.

All derivatives of S terminate in that recipient's eligible private factors. Its full live nonlinear path, including earlier private factors, remains differentiable; blocking shared parameter gradients must not detach hidden activations. Accumulate every gradient at unchanged old parameters and perform one deduplicated ordinary native Adam update per epoch. This begins with the first update; there is no own-step/post-step correction, rejection loop, clipping replacement, scheduled dose, teacher, learned coefficient or strength grid. Shared parameters receive P alone, so this is an explicitly restricted update, not a globally differentiated scalar `P+mean S` over every parameter.

For one observed label event, use q_m=p_m for y=1 and q_m=1−p_m for y=0. With frozen peer sum c,

`S_m=−log[(q_m^full+c)/4]`,

`partial_eta S_m = rho_m partial_eta BCE(z_m^full,y)`,

`rho_m=q_m^full/(q_m^full+c)`.

BCE is proper for the pooled Bernoulli probability; a frozen-peer constrained component is not thereby calibrated as an individual predictor. The extra credit can still weaken or miscalibrate a member, so P and the actual member gates remain necessary.

This is positive responsibility-weighted factual supervision. It has **zero extra gradient through the recipient's absent prediction** and no negatively weighted BCE term. A recipient gets more contextual responsibility when absent peers supply little correct-event mass. A very weak recipient can also get little responsibility; its proper P loss remains the competence foundation. The coefficient/reductions are explicit rather than inferred from parameter count or compared through raw L2 dose.

Source assignment changes the peer context used for credit; it does not force the recipient to use that named family. Movie-own features, other retained sources, biases and semantic attention can supply the correct probability. This is precisely what the neutral-assignment and current U/D diagnostics test.

## 3. Gaming counterexample and truthful innovation

The old signed objective is J=S−A, where A is all-absent pool BCE. Hold the supplied correct-event probability at0.8. If the absent correct-event probability falls from0.6 to0.3, then

`J changes from log(0.6/0.8) to log(0.3/0.8)`.

J improves while the supplied prediction is unchanged and absence becomes worse. This is a symbolic probability counterexample, not a recomputed project outcome.

The new source term is S alone; lowering the recipient's absent-event probability cannot lower that term. P penalizes the absent predictions through proper supervised loss. Nevertheless a joint shared-parameter/Adam tradeoff can indirectly worsen an absent risk. **No unconditional per-step absence/own-risk theorem is claimed:** removing the old finite guards sacrifices that protection. This design prevents a direct negative-absence incentive; its reporting/admission rule prevents claiming innovation from indirect absence harm.

At selected, freshly restored current FP32 members, reuse the existing exact diagnostics:

`A_a=B(mean_k p_k^(−a),y)`,

`S_(m,a)=B([p_m^full+sum_(k!=m) p_k^(−a)]/4,y)`,

`U_(m,a)=A_a−S_(m,a)`,

`D_(m,a)=B([p_m^(−a)+sum_(k!=m)p_k^full]/4,y)−B(mean_k p_k^full,y)`.

All4×3 cells, native specificity and both repair/harm contexts remain visible. **U/D is a measurement, not the optimized scalar or selector.** For an assigned cell, compare candidate with its same-parameterization P-only reference and report separately delta A, delta S and delta U. Credit a source-restoration innovation only if S improves and A is nonworse. A positive delta U obtained with worse A is explicitly classified as absence-harm-confounded and cannot support source-specialization advancement. Retain every pair/family, including unavailable/negative cells; never choose favorable sources. This criterion must be prospectively frozen before any new fit and does not alter the closed pilot's gate.

Full all-input task quality, member competence and deployed net repairs remain required. Positive TRAIN context credit or U alone cannot pass. A source-specific claim additionally needs positive deployed D/specificity and useful assigned-versus-neutral source responses; an improvement explained by ordinary augmentation or generic contextual weighting is reported under that narrower name.

## 4. Minimal complete representative pilot

Use one complete IMDB graph, the already established paired roles/base seeds(1,1),(2,2),(3,3), source assignments and native full architecture. These roles/VALID outcomes have been consumed, so this is explicitly exploratory paired development evidence. Every condition starts freshly from its paired native reset; all views/labels, horizons, precision, updates and paid source outputs are matched. No hidden reduced proxy or new dataset is substituted.

The core interaction has four shared committees per pair:

| Parameterization | P only | P plus assigned contextual source credit |
|---|---|---|
| Multiplicative BE, unit r/s | BE_P | BE_PS |
| Additive rank1, random v/zero u | ADD_P | **ADD_PS**, the preferred candidate |

Report both parameterization contrasts and both credit contrasts, including

`interaction=[Q(ADD_PS)−Q(ADD_P)]−[Q(BE_PS)−Q(BE_P)]`.

A compound win is not credited to either main effect without these contrasts. The candidate must first be useful under full task/member metrics, not merely increase the interaction statistic.

Six other required controls/anchors across the paired family are defined narrowly:

- **Genuine independent4 ADD_P and ADD_PS:** four fresh full native bodies/optimizers, identical rank-one sites/start, own P supervision and same restricted contextual credit. All body storage is disjoint; detached committee losses do not make it a shared bank. Each body retains its own selected checkpoint; pool all four. These bound the preferred additive sharing claim. An independent BE crossing is required only for a separate BE-sharing/three-way sharing-interaction claim, which this pilot makes no attempt to establish.
- **ADD_P with one common copied v dictionary:** same zero-u native start and parameter budget. This tests independently randomized tangent coverage rather than attributing that package to the affine family alone. Sign-flipped dictionaries are a known functional-null first-step equivalence and are not another fitted arm.
- **ADD_PS neutral credit:** in every update each active recipient uses the average of S_m^a over all three source families, with total source coefficient unchanged. All16 native predictions are already present; no cycling/exposure shortfall or extra source forward is hidden. This is the capable COMMON/all-source credit control for persistent assignment.
- **Capable native single P and rank4-adapter single P:** same complete graph, four-view proper supervision and factual selector. Rank4 supplies the four rank-one private-vector capacity budget at each site without a probability committee. These test native competence and generic low-rank capacity. They are P-trained anchors, not falsely labeled identical three-peer source objectives. The same-operation genuine independent4 controls handle that learning-policy comparison. A broader ensemble-necessity claim later needs a capable joint four-path model/readout, since rank4 alone cannot emulate separately evolving nonlinear trajectories.

This is10 conditions per pair: six shared committees, two genuine independent4 committees and two singles. Across three pairs it declares18 shared-bank fits, six independent4 groups/24 genuinely independent bodies and six singles. No condition is selected/replaced after outcomes. The comparisons are controls for one preferred candidate, not an adapter/rank/coefficient/initializer search.

Keep the native200-epoch maximum, literal epoch-minus-best>50 stopping, strict full factual VALID BCE selection/earliest ties and fresh full-input serving. Shared banks use their coherent factual-pool selector; independent bodies use own factual BCE selectors. All four TRAIN views receive supervision, but ablated VALID views, U/D and source credit never choose the checkpoint. Same data/role/buffer/cache/RNG semantics and admitted precision must hold; source/context losses use FP32 observed-event arithmetic through native differentiable scores. Reduced precision or one-step tapes need separate source qualification, not assumption.

Before fitting, root must create a new prospective freeze with complete quality criteria, counts/costs, custody and the absence-harm classification above. The old practical micro/member/supporting criteria can inform that draft; they are not silently reissued as a new gate. For analysis use all paired deltas/mean/sample SD/range/leave-one-out. Three role/optimizer pairs on one graph with reused VALID are not confirmation or graph generalization. Positive exploratory results can justify a separately frozen unused-outcome confirmation despite established ancestry.

## 5. Handoff and cost boundary

Reuse the exact native model/source-view machinery listed in IMPLEMENTATION_HANDOFF.md. Extend only a separately versioned inactive source when root authorizes it. Do not rewrite the sealed post-step pilot or preserve its name while changing training. The existing saved diagnostic implementation is the only U/D/F1/pooling/repair-harm authority; duplicate formulas/tests are not needed for readout.

Each update pays16 complete member/view forwards and their proper-loss backwards. Context credit reuses those predictions or a charged native replay at unchanged parameters if memory requires it. No model step occurs between view/member gradient contributions. Count all source-view construction/repropagation, repeated cached-input state trajectories, replay/backward/Adam, checkpoint/restore/selection/serving, CUDA/wall/CPU/RSS and storage. Source views can share immutable input caches; member learned states/normalization/dropout remain separate. Independent4 receives exactly matched information/training/serving, and its setup/own-body costs are retained rather than declared free.

No speed, resource admission or quality result is established. If source credit harms factual or absent competence, random dictionaries explain a foundation gain, neutral credit explains assignment, singles explain capacity, or independent4 matches the quality/cost frontier, narrow or close the corresponding claim. Do not force route diversity when the correct task has no useful separate source evidence.
