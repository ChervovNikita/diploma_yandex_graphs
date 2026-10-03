# Cardinality-aware single NCNC comparator: source plan

3 October 2026. Preparation only. This packet defines one prospective comparator and its qualification gates. It contains no model implementation, fit release, executed likelihood test, dataset read, selected state, prediction, scientific outcome or remote operation. The frozen J/F representative and the running NCNC family retain their existing definitions and releases.

## Purpose and interpretation

The scoped literature scout identified a missing control: a single completion model can learn the distribution of the number of residual observation bits as well as relative slot preferences. MaskGAE supplies masked-degree supervision ancestry; CAM supplies coherent linkset and graph-context ancestry. GRAN already supplies the mixture likelihood ancestry. This comparator tests whether a competent single with a count law can account for effects attributed to the J observation-pattern auxiliary.

The new arm is **C64-D4**: one width64 NCNC encoder and one native width64 completion scorer/decoder, a count head, and four joint completion draws decoded by the same decoder. Its auxiliary reconstructs the same TRAIN observation pattern used by J/F. A future comparison uses the frozen M4/N64, seed0, 100-epoch, lambda1 J/F representative. This preparation proposes one additional seed0 comparator fit only after implementation, qualification and a separate root release. It does not release that fit or add a search.

This is a model-competence comparison with a disclosed serving change. Frozen J/F decode each component's soft conditional weights. C64-D4 decodes four full joint binary draws. Their loss, model and computational differences must be reported. A C64-D4 comparison alone cannot isolate the effect of count supervision, sampling, sharing or likelihood placement. The fixed-bank C64-M4 contrast below holds the model and four-draw nonlinear route constant while removing the predicted law's dependence. It cannot establish superiority over a richer single, a covariance-aware control or a full-mode grouped single.

## Predicted quantities

For a target query e, enumerate the native residual slots from its record-masked graph. Let R be their number, Z_r indicate whether the missing counterpart edge was observed in complete supplied TRAIN, and K=sum_r Z_r. The single recursive NCNC scorer outputs s_r and the native clamp logit

    t_r = 2.5 * (s_r - 6) + log(.1),       q_r = sigmoid(t_r).

The count head outputs one logit a_k for every feasible k=0,...,R and pi_k=softmax(a)_k. The pattern density is

    P_C(Z | usable masked context)
      = pi_K * exp(sum_r t_r Z_r) / e_K(exp(t_1),...,exp(t_R)),

where e_K is the elementary symmetric polynomial. It predicts the full count law, a conditional distribution over size-K subsets, E[K], and actual slot marginals mu_r=sum_k pi_k P(Z_r=1 | K=k,t). q_r is an affinity supplied to this density; it is generally **not** mu_r. Neither q, pi nor mu is certified as a posterior over true unobserved links.

The count law is unrestricted over feasible counts at the mathematical interface. The fixed neural head below is one finite parametrization. Within a given count, subset scores are additive in t: this control cannot express every same-count higher-order law. A richer graph context can still improve its target score. The six-slot witness in the scout is an information example, not evidence that the graph contains useful versions of those laws.

## Exact graph and observation contract

- Construct the minibatch graph by removing selected positive **record IDs** before symmetric coalescing. Surviving duplicates can retain the same edge. Do not replace this with a unique-edge mask or an extra masking distribution.
- Enumerate common, left-only and right-only neighbors from that graph before encoder edge dropout. Slots are exactly the concatenated left-only and right-only counterpart queries. Both-endpoint-unobserved neighbors remain outside support; no candidate cap, sampling or extension is admitted.
- Retained common neighbors stay in the native common-feature sum with weight one and are outside Z. A synthetic positive is counted only when its counterpart edge actually disappeared after duplicate coalescing.
- For every residual counterpart query, Z=1 means observed in complete supplied TRAIN before the record mask. Z=0 means not observed in that snapshot; it is not a verified latent nonlink. All residual zeros are supervised by the same explicit observation proxy as J/F.
- The complete TRAIN teacher is a membership oracle for labels only. Build the support, masked encoder input, count context and forward predictions before invoking that oracle. Encoder dropout does not create additional teacher slots. VALID/TEST edges never supply auxiliary labels.
- Native sampled negative target pairs retain their conventional y=0 target supervision. Target y, positive/negative-query flags, selected-record membership and synthetic-removal indicators are unavailable to the count head and completion forward path.
- R=0 contributes auxiliary zero and remains in the positive/negative all-query means. pi_0=1 by definition. Native empty recursive-score paths retain their source call/dropout schedule; empty support is not a reason to skip the native second feature transform.

## Concrete count-head context

The encoder width d is64. Let h be the shared masked-graph encoder output, h'=h+xlin(h) the single outer native transform, C the retained common set and U the union of residual slots. Means of empty sets are zero. The vector part of the context is

    [h_i+h_j, h_i*h_j, mean_{v in C} h'_v, mean_{r in U} h'_{node(r)}]

and has4d entries. Append eight usable context scalars:

1. log1p(deg_i)+log1p(deg_j);
2. abs(log1p(deg_i)-log1p(deg_j));
3. log1p(|C|);
4. log1p(R);
5. abs(R_left-R_right)/max(R,1);
6. mean_r q_r;
7. mean_r q_r^2;
8. max_r q_r, with empty maximum zero.

All degrees and sets come from the usable record-masked graph. The q summaries use the one native candidate-score pass, before labels are looked up. Context is symmetric under endpoint exchange and invariant to residual enumeration order. It combines learned endpoint, common-neighbor and residual-set representations with observed structural sizes and scorer summaries. It is a concrete limited set-context head, not certification of all graph-context alternatives.

For each candidate k append six count features:

    [k/R, (k/R)^2, log1p(k), log1p(R-k), 1[k=0], 1[k=R]].

For R>0, a shared MLP with input270, hidden16, ReLU, hidden16, ReLU, output1 produces a_k. No head dropout, normalization, learned embeddings indexed by degree, graph tokens, mask identifier or count truncation is added. All R+1 logits are evaluated. The k in this operation is a candidate class index; the teacher K is used only after all logits exist in the auxiliary loss.

The count head has4625 trainable active parameters. Native N64 has38147 total/33922 active parameters including its unused fixed-pt ptlin; C64-D4 has42772 total/38547 active parameters. The frozen factorized M4/N64 has43790 total/38793 active parameters. These are source-formula counts, not instantiated-model receipts. C64-D4 is smaller by1018 total and246 active parameters. Qualification must audit actual parameter tensors, including retained unused ptlin. This is close capacity matching with the same encoder width, not exact parameter equality.

## Training and prediction

Use the stable algorithm in `LIKELIHOOD_AND_SAMPLING.md`. For R>0,

    L_C(e) = -log P_C(Z_e | context_e) / R_e.

The total objective is the unchanged native positive target log-sigmoid mean plus negative target log-sigmoid mean over four decoded draws, plus lambda1 times the positive-query mean and negative-query mean of L_C. Empty-support auxiliary zeros stay in those means. No source-zero reweighting, positive-only reconstruction, coefficient tuning or winner assignment is introduced.

Sample four independent full vectors B^(d) from P_C using **detached** t and pi. For each draw use residual weights w_r^(d)=1.05*B_r^(d), native weight1 for common neighbors, the differentiable outer h' features, and the same single nonlinear NCNC decoder. Serve the mean of the four raw logits. Training averages native BCE over all four draw logits; it does not apply BCE to the served mean. There is no score-function estimator, straight-through estimator, relaxation or teacher-conditioned sample.

The preserved native author source was inspected at lines703–705 and745–784: it first sets p0=sigmoid(scale*(score-offset)), then computes alpha*pt*p0/(pt*p0+1-p0), then multiplies by the residual-adjacency membership values before summing features. Thus q=pt*p0/(pt*p0+1-p0)=sigmoid(t), with alpha applied outside that bounded ratio. There is no subsequent clamp of alpha*q to1. This preserves the score-to-t transformation and alpha's scale/order, while changing the final completion weights from soft1.05*q to sampled1.05*B. In expectation the residual weights are1.05*mu, not1.05*q. Do not call the original soft clamp route preserved. No clamp is applied to1.05*B; a selected bit has weight1.05. Report this routing change explicitly. Using native soft weights alone would collapse the structured law before the nonlinear decoder and would not implement the chosen comparator.

Auxiliary gradients train the complete depth-zero scorer, count head, decoder maps used inside that scorer, both native feature transforms and shared encoder. Main target gradients train the outer feature route and decoder conditioned on sampled completions. Sampling parameters are detached; main target loss does not directly train t or pi through sampling. Shared parameters can still receive main gradients through their ordinary feature/decoder uses. The auxiliary changes subsequent target features and samples as the shared parameters and predicted law learn. The derivative contracts are in `LIKELIHOOD_AND_SAMPLING.md`.

### Fixed-bank marginalization contrast: C64-M4

Prebind a second serving calculation at the **same C64-D4-selected checkpoint**, with no refit and no separate checkpoint/seed selection. Compute that bank's actual marginals mu from t/pi and draw four vectors independently per slot:

    P_M(B | context) = product_r mu_r^B_r * (1-mu_r)^(1-B_r).

Decode each with the same1.05*B weighting, outer features and native nonlinear decoder, then average the four raw logits exactly as C64-D4. Use the same keyed slot uniforms/draw indices for both routes; C64-D4 additionally uses its count uniform. C64-M4 has no coupling between slots in its sampled law. Its implied count law is the Poisson-binomial law of mu, while C64-D4 retains pi. Both share the full trained encoder/scorer/count head, pointwise marginals, scale and four-draw decoder computation. C64-M4 is more informative than replacing mu by the original sigmoid(t), which would also change marginals.

At a fixed bank, a difference between the **expected** scores is due to retained dependence entering the nonlinear decoder, rather than a change from soft decoding to sampling or from one decoder pass to four. The declared four-draw score is still a noisy estimate: report that finite-draw limitation and the common-uniform pairing; do not equate one observed ranking difference with an exact expected-score difference. This contrast cannot isolate how count-law supervision changed the trained representation, because both routes use the same count-trained bank. It is a prediction-route ablation, not an independently trained weaker single or a new scientific fit.

Compute C64-M4 once at the primary selected checkpoint on complete VALID and on the fixed diagnostic set; it cannot become a second winner search. Charge marginal-DP work, four extra samples/decodes and complete query scoring. The structured route's selection still uses its own four fixed draws at each epoch. No additional fit, draw-count search or alteration of J/F is introduced.

## Fixed prospective recipe and fairness

Keep the native width64 recipe, input/encoder/edge/decoder dropout .25/.1/.25/.3, depth1, no residual-neighbor subsampling, candidate splitsize=-1, alpha1.05, scale2.5, offset6, pt.1, encoder/decoder Adam learning rates .0082/.0037 and native Adam defaults. Count-head parameters belong to the decoder optimizer group. Native model initialization follows the seed0 constructor contract; initialize the added count head with a separate CPU generator using the SHA256 domain `ncnc-cardinality-single-v1|seed=0|domain=count-init`, Xavier-uniform matrices and zero biases. It must not advance the native initialization/dropout/sampler stream.

The future representative is100 epochs, seed0, TRAIN batch65536 with the native17 full batches and drop-last rule, official VALID batch131072, lambda1 and four draws. Use the pinned native negative-sampler and permutation calls with seed0, the same sampling law, counts and drop-last semantics. The count head has no stochastic layers. The single recursive scorer executes one outer xlin, then left recursion and right recursion with the native second full-node transform each time; candidate scoring has auxiliary autograd. Four downstream decodes reuse outer features and use independent native decoder dropout in draw order0,...,3. This is a different native RNG/work schedule from four scorers. Later permutations can consequently differ under the global native RNG; report every epoch's label-free negative/permutation/mask receipts. The plan matches masking/sampling laws and supervision, and does not claim byte-identical paired streams or paired dropout with J/F.

Use a separate counter-based uniform stream for completion draws. Training keys contain arm version, seed, epoch, native batch index, query ordinal, draw index and count/slot purpose. Evaluation keys contain arm version, seed, canonical unordered endpoint pair, draw index and count/slot purpose; they exclude labels, dataset-pool tags, epoch and batch partition. The same query receives the same base uniforms in each evaluation/checkpoint and regardless of batching. Slots have a deterministic counterpart-pair order. Fix draw count4 before outcomes; no draw-count or sampling-seed search.

The final target score is a four-draw estimate of a nonlinear-decoder expectation, not an exact expectation. Its finite-draw noise is a limitation. Qualification checks reproducibility/batch invariance; any later extra-draw estimate is a separately charged diagnostic and cannot silently replace serving or selection.

Selection remains the first strict best complete official VALID Hits@50 over100 epoch states, on the complete TRAIN-only graph. Use60084 positive and100000 shared-negative scores, mean raw logits and strict `positive > 50th largest negative`. Do not select on count likelihood, synthetic masked labels, partial VALID, TEST or a new sample realization. No project values are read in this preparation.

Charge one full encoder, outer feature transforms, both complete candidate-score passes and auxiliary backward, all count contexts/logits, teacher label lookups, FP64 count DP forward/recomputation/backward, four joint draws, four nonlinear decodes, all validation and selection work, transfer/synchronization, peak memory and failed attempts. DP isO(R^2) per query and can dominate; four decoder draws are paid work. Parameter counts and equal epochs do not establish equal wall time or FLOPs. Preserve full supports; resource failure closes this arm as specified, rather than permitting an unreported support cap, lower degree, changed precision, extra fit or smaller evaluation.

## Prospective qualification and decision boundary

`QUALIFICATION_PLAN.json` binds the later engineering checks. No qualification is executed here. First qualify density normalization, DP gradients, exact-count sampling, graph/teacher separation, duplicate masking, detached target routing, model dimensions, RNG custody and raw-logit serving on fabricated inputs. Only a separate root engineering release may permit a complete TRAIN resource epoch and evaluation-coverage qualification. Engineering states are disposable and cannot initialize science.

If implementation and complete-support resource gates pass, a separate scientific release may admit the single frozen comparator fit. Interpretation must include target ranking, auxiliary positive-versus-source-zero competence, count-stratified likelihood and prediction, and complete resource costs. Diagnostic masking is a label-only observation reconstruction audit with separately bound TRAIN holdout/mask keys; it is not a source of model inputs or a second selection rule.

A positive J/F result remains evidence for that specific frozen observation-pattern regularizer. This later comparator can narrow a count-based explanation at a representative setting. A null/adverse result closes only the declared arm/setting. Neither outcome proves useful higher-order ambiguity generally, calibrated missing-link uncertainty, global novelty, or superiority over rich-context/full-mode single models.

## Custody and scope

`SOURCE_BINDINGS.json` records exact local preserved source and prior-packet bytes. Those modules were read as text and were not imported or executed. The new literature entries were already scoped in the sealed scout; this plan performs no additional primary-paper method read. The successor index_v39 preserves158 prior records and appends five scoped identities, with blocked direct leads held as metadata. No canonical ledger, status, release, dataset, checkpoint, running source or job is edited by this packet.
