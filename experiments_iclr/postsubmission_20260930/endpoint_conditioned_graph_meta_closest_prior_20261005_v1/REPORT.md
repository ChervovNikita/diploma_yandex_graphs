# Endpoint-conditioned graph meta-training: bounded closest-prior check

## Decision

The four newly inspected primary method scopes do **not establish an exact prior match** to the candidate's complete rule. They also do not clear the rule as globally novel. The remaining item is a specific training recipe and utility hypothesis: condition current private supervision on excluding both endpoints of every positive and negative outer query, while retaining shared graph context; train the shared representation through persistent private adaptation; recompute and commit each private update at the new shared state; and serve the final ordinary predictor.

**The strongest new finding is SELAR. Its Algorithm 1 and Eq. (8) already implement virtual fast update → slow meta update → recomputed, persistent fast update in graph training. Generic recomputation order cannot support a novelty claim.** SELAR's slow variable is a loss-weighting network, and its committed fast update trains the shared encoder and task heads on the complete primary minibatch plus auxiliary examples. Those differences leave the candidate's exact parameter partition and supervision eligibility unmatched in this bounded inspection; they do not make recomputation itself new.

The broad ingredients already covered by retained prior conclusions remain covered: head-only inner adaptation and representation meta-gradients (ANIL, OML), ensemble meta-learning with a shared representation (BMAML/EMAML), online parameter persistence (OML/La-MAML/FTML), and train-only meta optimization followed by ordinary deployment (MLDG/MetaReg). These retained records earn zero new reading credit here. No existing ranking outcome is used as novelty evidence.

## Exact object compared

The bound candidate is `shared_backbone_private_transfer_training_source_20261005_v2/REPORT.md`, with the technical limits in `endpoint_private_transfer_technical_challenge_memo_20261005_v1/REPORT.md`. Input hashes are retained in `INPUT_BINDINGS.json`.

1. Sample positive and negative outer TRAIN queries and form the union of all their endpoint nodes.
2. Each of four inner supervision streams excludes both endpoints of every outer query. This changes eligibility of **current labels**; it does not remove endpoint information from historical parameters or all context.
3. Use a common paired support graph, masking outer positive edges and both endpoint/random arms' inner positive edges, while retaining other TRAIN context.
4. Starting from persistent private parameters and Adam moments, form one virtual private step. Earlier state is treated as constant.
5. Update the shared core with `0.5 * BCE(mean raw logits) + 0.5 * mean(member BCE)`, differentiating through the current private update.
6. Discard the virtual private state. Recompute from the original private parameters and moments at the newly updated shared core, using the same inner data, mask and RNG, then commit once.
7. Private committed updates use their own inner labels; outer labels do not directly update private parameters. Serve the committed ordinary predictor without inference adaptation.

F4's private block comprises factors, LayerNorm affine parameters and the mixture coefficient; shared parameters include the GNN and dense predictor matrices. The capable single control adapts its complete nonlinear NCN predictor. The meta derivative is one-step and truncated with respect to earlier private-state history.

## What each new primary scope supplies

| Primary source | Direct prior established in inspected scope | Material difference from the exact candidate |
|---|---|---|
| **SELAR** (Hwang et al., arXiv:2103.00771v1), §III and §III-B, Eqs. (1)–(9), Algorithm 1 | Shared GNN with task-specific primary/auxiliary predictors; differentiable virtual model update; primary meta loss updates a slow loss-weighting function; actual model update is recomputed from the original model with the new weighting parameters and persists. Algorithm output is the trained primary network. | The slow variable is loss weighting, not the representation. The fast block contains encoder and task-model weights. Its actual commit uses the complete primary minibatch, including the meta fold, plus auxiliary examples. Task heads predict different tasks, not members of a served same-task ensemble. No positive-and-negative endpoint exclusion is specified. |
| **G-Meta** (Huang and Zitnik, arXiv:2006.07889v1), §3, §5, Appendix E Algorithm 1, LP protocol in §6 and Appendix I | Local-subgraph graph meta-learning; support prototypes and GNN inner adaptation followed by query meta loss. The single-graph formulation transfers to disjoint label sets; LP uses node-pair examples. The LP preprocessing explicitly reserves a fixed 30% of edges for support and 70% for query in each graph. | Edge-disjoint support/query sets are not the candidate's endpoint-excluded supervision. Random negative sampling has no stated exclusion of both endpoints of all outer positives and negatives. Inner GNN copies start from the same meta parameters, and meta-testing repeats adaptation. No persistent private Adam ensemble or candidate commit partition is specified. |
| **MLDGG** (Tian et al., arXiv:2411.12913v1), §3–§4, Eqs. (1)–(13), Algorithm 1 | Graph-domain meta-learning with shared structure/representation learners and domain-specific GNN parameters; support/query sampling, inner adaptation and aggregate query outer update. | Tasks are source graphs/domains. Support/query sets are sampled from semantic encoder outputs. The output is a parameter initialization that is fine-tuned on the target graph. No same-task served ensemble, current-query endpoint exclusion, or private inner-only recomputation rule is specified. Its domain/causal assumptions do not certify artificial endpoint blocks within one graph. |
| **Mochi** (Mattos and Silva, arXiv:2604.22031v1), §3 Eqs. (1)–(2), Appendix H.1 | Recent graph representation meta-training through an episode-fitted differentiable ridge readout. Node, link and graph examples share a readout interface. Link embeddings use endpoint Hadamard products; positive edges and uniform negatives from a same-graph non-edge pool are sampled for support/query. | The readout is solved per episode, not a persistent private Adam learner. Disjoint examples are specified, but no endpoint-union exclusion is stated. No same-task persistent ensemble or recomputed private commitment appears in the method equations. H.1 contains inconsistent evaluation-readout descriptions; deployment comparisons are therefore bounded below. |

### SELAR's update is the decisive ordering precedent

In the notation of SELAR, Eq. (6) builds `w_hat^k(Theta^k)` from `w^k`. Eq. (7) updates `Theta` using the primary meta loss through that virtual step. Eq. (8) then computes

`w^{k+1} = w^k - alpha * grad_w L^{pr+au}(w^k; Theta^{k+1})`.

Algorithm 1 lines 6–8 specify the virtual split update; lines 10–12 specify the new slow parameter and recomputed actual update. Crucially, line 12 uses `D_m^pr union D_m^au`, whereas the virtual update uses `D_m^{pr(train)} union D_m^au`. Thus SELAR both establishes the order and differs from the candidate's private inner-only label commitment. Merely changing SGD to a stateful Adam step does not establish a new meta-learning principle.

### G-Meta and Mochi do not establish endpoint-disjoint episodes

G-Meta Appendix I states that the fixed edge split ensures support/query distinction. That statement is about examples/edges; it does not state that the support endpoint set is disjoint from the endpoints of every positive and negative query. The local-subgraph construction also does not establish disjoint context, and this packet does not interpret the paper's wording about independent tasks as a proof of independence for the candidate.

Mochi §3.3 specifies a disjoint query example set. H.1 specifies uniform non-edges from a fixed pool within the same graph, with positive/negative balance in both sets. No endpoint exclusion rule is specified. H.1 first describes frozen embeddings with the original readouts (ridge for Mochi), then describes prototypes/logistic regression for NC/GC and dot-product LP scoring. The clear representation-through-ridge training equations can be cited; the exact LP inference protocol is unresolved by this text alone and is not relied on to claim a deployment distinction.

## Component-level claim boundary

| Candidate component | Assessment after this check |
|---|---|
| Shared representation differentiated through private/head learning | Established generic prior; ANIL/OML/BMAML retained, Mochi adds recent graph-specific evidence. |
| Shared representation and multiple private predictors | Established generic prior; retained BMAML/MetaReg plus SELAR's graph task-specific heads. Same-task served ensemble and auxiliary-task heads must be distinguished accurately. |
| Persistent state across training steps | Established generic prior. No inspected new scope states the exact candidate's private Adam-state partition, but optimizer state is not a cleared conceptual innovation. |
| Virtual update, slow meta update, recompute actual fast update | Established direct graph prior in SELAR Eq. (8)/Algorithm 1. |
| Ordinary fixed deployment after train-only meta optimization | Established retained MLDG/MetaReg prior; SELAR outputs its trained primary network. This is not a new general principle. |
| Within-one-graph support/query episodes | Established graph episode ancestry. G-Meta's single-graph formulation uses disjoint labels; its LP protocol and Mochi use separated examples. Neither inspected protocol specifies the complete positive-and-negative endpoint-union exclusion. |
| Current private label eligibility excludes every outer query endpoint while context/history remain | Exact rule not located in these four scopes. A narrow conditional supervision choice remains; independence, unseen-node generalization and domain transfer do not follow. |
| Exact half aggregate-logit BCE plus half member BCE, paired support masks, live/truncated Adam derivative and inner-only recomputed persistence | Complete combination not located here. Exact objective/schedule matching is not a global novelty clearance, and these implementation choices require a utility test. |

## Defensible claim and concrete ways it can fail

A defensible provisional description is: **“We evaluate whether excluding all positive and negative outer-query endpoints from current private supervision improves train-only transfer in a persistent shared-GNN predictor, using an ordinary final predictor.”** If the same-task ensemble earns its added cost against the capable single control, the description may specify that ensemble. The bounded search supports describing the complete configuration precisely, not “a new meta-learning principle,” “the first graph meta-ensemble,” or a guarantee of adaptation to new nodes/domains.

The technical memo's objections remain active. Historical state and unmasked context retain endpoint information; the candidate defines correlated episodes rather than independent tasks. Paired masking changes support substantially. Recomputing at the new shared state does not eliminate the episode/full-support shift. A useful effect may arise from altered training geometry or effective budget rather than the intended transfer mechanism.

The existing planned contrasts are therefore discriminating: endpoint versus matched random eligibility; live versus detached current-update derivative; F4 versus capable single; tied versus untied four; and paid recomputed versus stale committed updates. An endpoint/random tie weakens the endpoint-transfer claim. A live/detached tie weakens the differentiable-transfer mechanism claim. Failure against the capable single weakens an ensemble-specific utility claim. Failure against paid stale commit weakens the need for the recomputation schedule. No such contrasts were executed or their outcomes inspected for this literature packet.

## Direct citation targets

- Hwang et al. **Self-supervised Auxiliary Learning for Graph Neural Networks via Meta-Learning**, arXiv:2103.00771v1 (2021), [§III-B](https://arxiv.org/html/2103.00771v1#S3.SS2), especially Eqs. (6)–(8) and [Algorithm 1](https://arxiv.org/html/2103.00771v1#alg1). Cite for graph auxiliary meta-learning and recomputed actual updates.
- Huang and Zitnik. **Graph Meta Learning via Local Subgraphs**, arXiv:2006.07889v1 (2020), [§5](https://arxiv.org/html/2006.07889v1#S5), [Appendix E](https://arxiv.org/html/2006.07889v1#A5), and [Appendix I](https://arxiv.org/html/2006.07889v1#A9). Cite for local-subgraph support/query graph meta-learning, and accurately describe its LP edge separation.
- Tian et al. **MLDGG: Meta-Learning for Domain Generalization on Graphs**, arXiv:2411.12913v1 (2024), [§3](https://arxiv.org/html/2411.12913v1#S3) and [§4.3](https://arxiv.org/html/2411.12913v1#S4.SS3). Discovery located a 2025 ACM record, DOI `10.1145/3690624.3709188`; only the retained arXiv version was read, and text equivalence to the ACM version is unverified.
- Mattos and Silva. **Mochi: Aligning Pre-training and Inference for Efficient Graph Foundation Models via Meta-Learning**, arXiv:2604.22031v1 (2026), [§3](https://arxiv.org/html/2604.22031v1#S3) and [Appendix H.1](https://arxiv.org/html/2604.22031v1#A8.SS1). Cite for graph representation learning through differentiable ridge episode fitting; do not resolve its inconsistent evaluation description by assumption.

Retained direct-prior citation roles and exact source/scope references are in `REUSED_CONCLUSIONS.json`. In particular OML, ANIL, BMAML, MLDG and MetaReg should accompany any broad representation/head/ensemble/deployment discussion, even though they were not reread here.

## Search and scope limits

Six OpenAlex metadata searches, including date-restricted 2024–2026 searches, are saved with responses and receipts under `discovery/`. The four selected identities were checked against the retained v63/v64 index before primary retrieval; the v64 descriptor was reconstructed through its compatible loader. `DEDUPLICATION_BEFORE_RETRIEVAL.json` records the absence of identity matches. Selection prioritized exact graph update order, local support/query sampling, source-domain parameter separation, and a recent graph head-adaptation method. Other hits are metadata-only leads, not method evidence.

There are four new bounded primary method identities and zero full-paper reads. Sections/proofs/results outside the declared method/protocol scopes were not substantively inspected. Heading inventories and limited keyword locators supplied navigation only. Some protocol paragraphs contain incidental hardware, timing, shot counts and published performance language; no numerical outcome was adopted. No author code was inspected. Metadata truncation, ranking and indexing can miss relevant work, and failure to locate the conjunction in four scopes cannot prove its absence from the literature.

All work in this packet is literature retrieval, text extraction and artifact hashing. It performs no numerical/model execution, server actions, canonical edits, predictive adoption or reviewer acceptance decision. `MANIFEST.json` and `SEAL.json` bind the retained artifacts; each retained artifact is under 2 MB.
