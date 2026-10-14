# Scoped literature and retained conclusions

This is a bounded method/design reading, not full-paper certification, a
systematic absence search, a performance transfer or novelty clearance.
Primary texts are saved under `primary/`; exact scopes/hashes are recorded in
`READ_SCOPES.json`. Numerical results are not adopted. Locator searches
incidentally exposed some experiment text; it is not outcome evidence here.

## New primary method scopes

1. **Li, Zhang and Yuan, Edge Contrastive Learning: An Augmentation-Free Graph
   Contrastive Learning Model**, [arXiv:2412.11075v1](https://arxiv.org/html/2412.11075v1),
   posted December 2024. Read §3.1–3.3, Eq.1–8 and Algorithm 1. AFECL uses a
   multi-head GAT, endpoint-derived edge embeddings, and same-edge plus
   adjacent-edge positives. Endpoint non-neighbor edge pairs supply negatives;
   large-graph edge sampling bounds its otherwise quadratic contrast work.
   **Consequence:** an edge/query objective needs an explicit relational target.
   Our same-query cross-route positives are narrower than adjacent-edge positives;
   native source query alternatives replace a new graph/edge population. We do
   not reproduce its encoder, edge sampling, or complete unsupervised objective.

2. **Chang, Hung, Lu and Shen, Enhancing Contrastive Link Prediction With Edge
   Balancing Augmentation**, [arXiv:2508.14808v1](https://arxiv.org/html/2508.14808v1),
   posted August 2025; arXiv comment says accepted by CIKM 2025. Read §4.1–4.3,
   Eq.6–12 and Algorithm 1. CoEBA changes topology by pruning predicted noisy
   edges and adding embedding-similar neighbors, masks features, and runs the
   same encoder on original/augmented graphs. Original-graph neighbor-concentrated
   contrast includes one-hop neighbors as positives to avoid repelling valid
   neighbors; the augmented-view rule differs. **Consequence:** generic
   instance negatives are not automatically suitable link targets. We retain
   native topology/sampling and test endpoint-conditioned source alternatives;
   no CoEBA rewiring, variational autoencoder, warmup or extra graph pass is
   imported. Its analysis/theorem assumptions are not audited or claimed for us.

3. **Li, Gao, Zhang and Torkamani, Ensembles of Low-Rank Expert Adapters**,
   [arXiv:2502.00089v1](https://arxiv.org/html/2502.00089v1), posted January 2025;
   arXiv comment says ICLR 2025 proceedings. Read §3.1–3.4, Eq.5–10. ELREA first
   fits a base LoRA, computes projected Adam/instruction-token gradient features,
   clusters examples, fine-tunes cluster adapters, and performs late weighted
   logit aggregation using input-gradient/centroid similarity and a base adapter.
   **Consequence:** specialization and conditional late aggregation have direct
   prior ancestry. We use fixed endpoint weights rather than gradient clustering,
   and an optional TRAIN-fitted prehead/logit gate rather than inference-time
   input gradients. No source LM, frozen-base result, route-calibration or graph
   performance claim transfers. A gradient of the unknown link target must not
   be used as a serving-time routing signal.

## Retained evidence reused; no duplicate primary-read credit

- **LoRA-Ensemble**, arXiv:2405.14438v5, is already in literature index v72,
  paper record 126, with pinned author-source findings. Reuse its conclusion:
  frozen transformer body plus private additive low-rank attention factors and
  classifiers; inspected source uses mean own CE and probability averaging,
  with a separate raw-logit artifact path. This is adjacent efficient-ensemble
  ancestry, not the same trainable-shared-body/mean-logit protocol. The current
  retrieval happened before identifier reconciliation; the saved HTML adds no
  new method/read credit and is not reread as a new paper here.
- The retained **conditional fast-factor** conclusion identifies conditional
  diagonal maps' GNN-FiLM/edge-conditioned-filter ancestry and keeps earlier
  task-specific negative evidence scoped. Endpoint conditioning is therefore
  a utility hypothesis, not a new primitive or an inference that an old Tolokers
  result predicts Collab behavior.
- Retained **NCNC coherent completion** conclusions distinguish observed TRAIN
  membership from verified latent absence and require capable same-information
  controls. Sampled negatives here retain exactly the source meaning.
- Retained **label-conditioned latent-link** conclusions reject broad
  generative/Bayesian/graph-contrastive novelty claims for soft compatibility
  scores. This package makes no latent-posterior claim.
- Retained **member-depth late aggregation** conclusions require a nonlinear
  concatenation/bottleneck comparator with the same bank, fit roles, context
  and parameter opportunity. That control remains necessary for the optional
  late gate. Fitting a second head on already-selected VALID is not an honest
  new validation-only pilot; our proposed gate is fitted jointly on TRAIN.
- Retained **contrastive BE complementarity** scope stresses native member
  quality and useful common-error repair; a representation-distance or generic
  diversity effect does not establish a served task gain.

## Resulting finite questions

1. Does fixed endpoint-correlated private credit improve native member/pool
   ranking beyond an equally weighted and query-IID private-credit control?
2. Does the endpoint-conditioned denominator add utility beyond same-query
   cosine alignment and a matched global opposite-label denominator, while
   preserving member competence and source query/loss laws?
3. Do nonlinear task-prehead private adapters or conditional late aggregation
   add utility beyond matched shared capacity and an ordinary nonlinear fusion
   comparator? These are optional follow-ups, not silently combined mechanisms.

No paper's reported superiority is transferred to these new hypotheses. The
270 reproduction fits remain a separate obligation; no pilot, TEST operation,
full-benchmark launch or original-score edit was performed by this work.
