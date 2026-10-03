# Joint completion/decoder uncertainty: scoped literature assessment

## Decision

**No additional defensible learner novelty gap was found.** Preserving member association between learned completion weights and nonlinear decoders is already supplied by independent NCNC. Joint graph/model uncertainty and averaging nonlinear predictions also have direct primary precedents. A finite shared-factor implementation may have useful quality or cost behavior, but those properties remain empirical; they do not establish a new uncertainty principle.

This packet adds two newly inspected identities, **LDS (1903.11960)** and **CORE (2404.11032)**, and reinspects three older identities, **BGCN (1811.11103), node copying (1911.04965), and adaptive connection sampling (2006.04064)**. All five are scoped method reads, not full-paper reads. The latter three are already discussed in older saved packets despite being absent from v33's normalized identity list. They must not be counted as three newly discovered papers.

**No distinct new quality hypothesis is proposed.** The existing own-versus-pooled NCNC question can be retained as a falsifier of whether a particular shared-factor architecture preserves a known useful mechanism. A positive result would support that implementation's preservation claim; pairing itself would still be prior.

## What was checked first

Index v33 has 136 conclusion records, 90 normalized paper identities and two software identities; these are not full-paper-read totals. The existing NCNC priors and graph-link structural-uncertainty conclusions establish:

- A native independent NCNC member retains its own scorer, clamp, completion weights, contextual features and decoder.
- The already proposed architecture difference is a shared encoder/factorized decoder bank using its own post-clamp completion weights instead of their member mean.
- IECNC explicitly uses learned completion probabilities or `-p log p` with contextual features. E-GAE pools embeddings before decoding. PENCIL reconstructs propagation adjacency from observed token fields; it does not specify learned missing-edge probabilities.
- Link-MoE mixes complete expert scores. Bayesian/NTK/factorized ensemble antecedents prevent identifying ordinary factor members as posterior samples without an inference argument.
- The CN=0 confidence-controlled disagreement/error-retention diagnostic already exists. It is not a fresh hypothesis here.

The older resolvent packet additionally already rejects generic graph-uncertainty plus shared-solve/BatchEnsemble novelty and records VGCN's learned low-rank/dot-product graph posterior. Low-rank graph probabilities and shared computation therefore supply no fresh gap by themselves.

## Primary evidence

| Source and exact scope | Mechanism verified | Consequence and scope limit |
|---|---|---|
| **BGCN**, [1811.11103v1](https://arxiv.org/html/1811.11103v1), §4 and Algorithm 1 | Eq. (5) uses `p(W | Y_L,X,G)`; Eq. (6) averages full predictions from graph-conditioned weights. The implementation substitutes an a-MMSBM MAP estimate for integrating graph parameters, then trains/uses MC dropout for each sampled graph and averages Eq. (8). | Joint topology/model association and nonlinear marginal prediction are direct prior. It is approximate semi-supervised node classification under a community graph model, not an NCNC link-completion posterior. |
| **LDS**, [1903.11960v4](https://arxiv.org/html/1903.11960v4), §§3.1–3.2, Eqs. (5)–(15), footnote 2 and Algorithm 1 | Learns independent Bernoulli graph probabilities through a validation-label outer objective and a shared GCN through expected training loss. Eqs. (8)–(9) average predictions over sampled graphs. Footnote 2 explicitly prints `E f_w(X,A) != f_w(X,E A) = f_w(X,theta)`, because the predictor is nonlinear. | Averaging after nonlinear prediction instead of applying a predictor to mean topology is an explicit prior principle. Its shared deterministic weights do not constitute graph-conditioned weight-posterior samples. Its STE and truncated hypergradients are biased. |
| **Node copying**, [1911.04965v1](https://arxiv.org/html/1911.04965v1), §3 and Algorithm 1 | Samples same-predicted-class neighborhood donors using observed graph, features and training labels; keeps node features unchanged. Fits GCNN weights conditional on each sampled graph, using MC dropout. | Task-conditioned graph uncertainty and correlated neighborhood replacement are prior. Preserve a printed v1 caveat: Eqs. (6)–(7) evaluate the predictive likelihood on `G_obs`, although fitting conditions on sampled graphs. This packet does not silently substitute a sampled graph into that likelihood. |
| **Adaptive connection sampling**, [2006.04064v1](https://arxiv.org/html/2006.04064v1), §§3–4 | Learns beta-Bernoulli masks with effective edge-weight interpretation. The inspected family factorizes `q(W,Z)=q(W)q(Z)` and sets `q(W)=delta(W-M)`. Masks multiply observed adjacency and have its sparsity. | Learned random connection masks and model training are prior. This scope does not preserve arbitrary joint posterior dependence and cannot add a missing edge where adjacency is zero. §5's random-walk variant was not freshly read. |
| **CORE**, [2404.11032v1](https://arxiv.org/html/2404.11032v1), §3 and Appendix B.2–B.4 | Inflates top-k missing edges from scored nonedges with a common neighbor; learns target-link-specific Bernoulli reduction using subgraph/edge attention; trains a nonlinear GNN/pooling/MLP predictor and can share the encoder. Inflated edges are topology inputs rather than training labels. Appendix B.3 uses expected edge probabilities as inference weights. | Completion plus learned uncertain topology plus nonlinear link prediction and encoder sharing already have a direct link-task prior. Native inference differs from MC graph prediction averaging. It does not supply a finite private completion/head ensemble or an exact joint posterior. |

CORE's theorem assumes an invertible deterministic local-neighborhood-to-link-probability mapping and sufficient coverage by the inflated graph. Its proof was not audited. The theorem does not establish general posterior identification, calibrated uncertainty or a quality gain from replacing native inference with sampling. The 2026 journal record, DOI 10.1145/3789200, is metadata title/author-matched to the inspected 2024 preprint; journal-version method changes were not checked.

These distinctions matter: an observed-edge drop mask, a sampled hard graph, and NCNC's soft post-clamp missing-neighbor completion weights are different objects. The broad principles transfer as novelty blockers; these methods are not claimed to be identical algorithms.

## Narrow architecture question and next falsifier

The already retained question is:

> Under a shared encoder and finite factorized NCNC bank, does retaining each member's own completion representation preserve useful held-out predictions that are lost by pooling completion weights before the private nonlinear decoder?

This is a question about a particular compression/sharing choice. It is not evidence that independent ensembles fail to preserve completion/decoder association.

A prospective fixed-bank comparison can clarify the mechanism without inventing a new hypothesis. Let `q_m` be aligned, post-clamp completion weights for one target pair; let `C(q)` form the completion/context representation with common features and support; let `f_m` be the complete private nonlinear decoder. At the same frozen bank, compare:

- `D = (1/M) sum_m f_m(C(q_m))`: existing own-completion prediction.
- `X = (1/M²) sum_m sum_k f_m(C(q_k))`: all crossed completion/head predictions.
- `P = (1/M) sum_m f_m(C(mean_k q_k))`: existing pooled-completion prediction.

`D-X` isolates empirical member association at that bank; `X-P` isolates averaging across completion representations versus applying the decoder to their mean. These are diagnostic controls and algebra, not a novel learning method. Candidate positions, features, clamps and decoders must be compatible across members for crossing to be meaningful. Proper probability pooling must be preserved in every arm.

**Falsifier:** if own weights fail to improve a predeclared held-out proper predictive score over competent pooled weights, there is no quality support for preserving them in this compressed architecture. If an own-versus-pooled difference survives but disappears against the crossed comparison, the effect is consistent with nonlinear representation averaging rather than private member association. If it survives crossing, it still supports coadaptation/association, not Bayesian calibration. A separately trained pooled architecture is also needed to avoid treating an inference-time mismatch as evidence that a competent pooled learner cannot work.

A resource-matched independent NCNC ensemble remains the capability reference because it already preserves member association. Any positive result must disclose shared-encoder capacity, trainable parameter family, full work and predictive sample count. Evidence of better quality at a fixed shared budget would establish an empirical factorization tradeoff. It would not establish a new completion primitive, new joint marginalization, or calibrated posterior sampling.

No experiment or code execution is authorized or performed by this packet.

## Search disposition and custody

Public OpenAlex title/topic searches covered Bayesian graph learning, link uncertainty, completion ensembles and NCNC successors. An NCNC citation query returned 11 indexed citing records; that is not an exhaustive successor census. It led to CORE's arXiv preprint. Bayesian GNNs for Interpretable Link Prediction (2025; DOI 10.1109/icaci65340.2025.11096302) remains metadata-only; no primary method claim is adopted. Nonparametric BGCN (1910.12132) was discovered but not read.

`READ_SCOPES.json` records exact method scopes and incidental public-results exposure. Some floating published benchmark tables/captions appeared inside extracted method sections; none was used to decide quality or novelty. Root clarified that the task restriction concerns unopened project study/heldout outcomes, not published paper results. No project benchmark outcomes, heldout files, datasets, checkpoints or predictions were opened. No project or numerical code was executed/imported, and no remote state was mutated.

Only this fresh packet was written. Index v33 and the previous engineering critique were preserved with their original hashes. `PAPER_CONCLUSIONS.json` is prepared for root adoption; `PRIMARY_EXCERPTS.json` labels exact prose, normalized math and equation transcriptions separately. Raw primary HTML, mechanical text extracts and discovery responses are retained. Retrieval/extraction is not counted as a full read.

**Recommendation:** adopt the two new scoped method identities and the three reused source-specific conclusions, while reconciling legacy identity omissions. Do not promote pairing, the covariance identity, nonlinear prediction averaging or shared low-rank parameterization into a learner novelty claim. The current literature supports an empirical implementation-preservation test, with no additional quality gap asserted.

