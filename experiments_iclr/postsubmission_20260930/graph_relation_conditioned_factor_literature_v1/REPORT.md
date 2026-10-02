# Relation-conditioned factors with shared heterogeneous graph transport

2 October 2026. Local literature/source assessment; index v17 and saved idea packets consulted first. **Three new scoped primary methods; zero full-paper reads.** No runner, model execution, scientific imports, datasets, labels, checkpoints, SSH, GPU work or active-study changes.

## Decision

**No-go for opening a distinct method/pilot lane.** Relation-conditioned rank-one maps can preserve a common linear aggregation when their inputs and transport operators are common. That is an exact, useful implementation fact. SeHGNN already explicitly separates typed/metapath aggregation from learned semantic transforms; R-GCN supplies shared relation bases; HGT supplies typed projections and relation-conditioned attention/messages. Adding member factors after those aggregates creates a compact ensemble of established semantic adapters.

Private nonlinear histories across layers generally require private transport. Pooling them into one common source representation makes a recurrent shared-transport architecture feasible, but that changes the member messages and is the existing common-message/mixture question. Training-error-directed initialization, relation-indexed factors or a generic repulsion/coupling loss do not supply a further operation. No specific mechanism was found that retains useful member-specific relational information at reduced edge-channel cost and distinguishes this branch from those known choices. No pilot, tune grid or source runner is recommended.

An exact publication matching the complete typed-factor/common-message assembly was not established. This bounded absence is not novelty evidence. The conclusion concerns promotion under the requested distinct-mechanism goal, not a claim that the architecture cannot improve a particular task.

## New primary evidence

| Source | Necessary methods inspected | What it resolves |
|---|---|---|
| Schlichtkrull et al., [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/html/1703.06103v4), v4, 26 October 2017 | §§2.1–3: relation-specific linear messages, normalization, self map, nonlinear stacking and CE; basis/block decompositions. Selected entity-classification setup and supplementary preprocessing paragraphs. | \(W_r=\sum_b a_{rb}V_b\) explicitly shares dense bases across relations; block decomposition restricts within-map channel coupling. Neither makes separately evolving member states share their transport automatically. |
| Hu et al., [Heterogeneous Graph Transformer](https://arxiv.org/html/2003.01332v1), v1, 3 March 2020 | §§3–4 method and sampling/timestamp prose; selected schema/task/setup paragraphs. | Type-specific K/Q/message/output maps, relation attention/message matrices and target-neighbor softmax are direct typed-sharing prior. Changing private histories or Q/K factors generally changes edge attention, so the common operator needed for exact sharing is lost. |
| Yang et al., [Simple and Efficient Heterogeneous Graph Neural Network](https://arxiv.org/html/2207.02547v3), v3, 1 September 2023 | Motivation and complete main method; Appendix A projection/aggregation commutation and selected native recipe. | Fixed relation/metapath mean aggregation is precomputed once; each semantic channel then has an MLP and shared Q/K/V semantic fusion. Training-label propagation with removed self contribution is an additional input. This directly supplies the feasible decoupled aggregation boundary. |

The SeHGNN source's empirical findings about neighbor attention and semantic depth are source-specific results, not universal theorems. Its neighbor attention is distinct from attention over cached semantic vectors. Keeping graph transport common can still allow private semantic fusion; the latter is node-local dense work and must be charged for each member.

The new readings are not a survey of all heterogeneous ensembles. Recent methods are covered by reused HG-Adapter (2024), HGEN (2025), CHoE (2026) and temporal heterogeneous graph pretraining (September 2026), rather than reacquired as new papers.

## Exact feasible computation boundary

Use row-vector notation. For relation r from source type s to target type t, let \(P_r\in\mathbb R^{N_t\times N_s}\) be a fixed linear normalized neighbor operator, \(H_s\) a common source state, and \(W_r\) a shared dense map. A private relation/channel map is

\[
W_{m,r}=D(a_{m,r})W_rD(b_{m,r}).
\]

For factors constant across source nodes of that relation,

\[
P_r\big(H_sD(a_{m,r})\big)W_rD(b_{m,r})
=(P_rH_s)D(a_{m,r})W_rD(b_{m,r}).
\]

Thus compute \(Z_r=P_rH_s\) once and reuse it across members. Type/relation dimensions and factor sites must match. Arbitrary node-specific source gates cannot be pulled through \(P_r\); target-specific post-aggregation gates can be applied locally. Member-specific edge attention, normalization, graph masks or sampling likewise invalidate the common-operator assumption unless their differences are explicitly retained.

This is **one shared relation sweep**, not necessarily one graph multiplication: a heterogeneous layer may require separate \(P_r\) operations and relation-resolved buffers. A union aggregate \(\sum_rP_rH_s\) generally loses the information needed for arbitrary relation-specific private responses. Dense maps, private activations, semantic fusion, normalization, output heads and backward work remain.

R-GCN basis sharing also must be interpreted correctly. A fixed basis allows

\[
\sum_r P_rH W_r
=\sum_b\left(\sum_r a_{rb}P_rH\right)V_b.
\]

This changes the order of linear sums/transforms; it does not erase relation work. Private relation coefficients or channel factors can require relation-resolved caches or member-specific combinations. Basis/block decomposition is a storage/inductive-bias choice, not a generic M-fold reduction in transport arithmetic.

## What happens at the next nonlinear layer

If the private update gives different \(H_m^{\ell+1}\), exact member r-messages at the next layer require \(P_rH_m^{\ell+1}\). Shared dense weights or small factor vectors do not make these arrays equal.

Writing \(H_m=\mu+\Delta_m\), with \(\sum_m\Delta_m=0\), gives

\[
P_rH_m=P_r\mu+P_r\Delta_m.
\]

Transporting only \(\mu\) discards every relation's transported member-contrast term. An output-side diagonal factor generally cannot reconstruct neighboring contrasts from \(P_r\mu\). Private local/root state still can support different predictions, but it does not restore the omitted neighbor information.

Three coherent choices follow:

| Choice | Shared graph work | Scientific interpretation |
|---|---|---|
| Fixed typed/metapath cache followed by private factors/MLPs | Cache acquisition shared; eligible dense transforms repeated. | SeHGNN/SIGN-style decoupled semantic ensemble or adapters. No new transport operation. |
| Recurrent relation aggregation of a common member pool \(\mu\), with private root/local updates | One relation sweep over the declared pool each layer. | Common-message ensemble; relational member contrast is removed from that branch. Relation-conditioned source pooling is a graph expert mixture. |
| Exact recurrent member states, packed as channels | One API call can transport \([H_1,\ldots,H_M]\), but processes M times the feature channels. | Vectorized independent transport; fewer calls do not establish fewer edge-channel operations. |

A special constrained family may retain a small common feature basis across depth. Such closure must be demonstrated with its nonlinearities, biases, normalization and factor placement. Positively homogeneous activations can commute with eligible positive output scalings in limited cases, but arbitrary two-sided relation factors and relation sums do not preserve that closure. A compact shared-basis model is not automatically an exact representation of arbitrary private GNN trajectories.

## Initialization and coupling do not yet create a distinct gap

The saved typed-factor assessment already proves equivalence to equally located diagonal residual activation adapters. Adding an r index preserves that identity. Generic posterior sampling of rank-one factors is covered by Rank-1 BNN; generic class-aware diversity/repulsion and heterogeneous meta-path ensemble decorrelation are covered by the saved DICE/CDLG/HGEN/SuGAr conclusions. None guarantees useful predictive complementarity.

The saved PreGS/CHoE/GraphMoRE conclusions cover warm supervised graph transfer, structure-conditioned expert selection and deliberately different initial expert regimes. The existing graph-band/VJP initializer and its separately frozen Gram-matched random control cover the current error-directed seeding question. Partitioning its cotangents/directions by relation is a granularity change within that family until a new criterion is supplied.

In a common-transport model, useful relation-response diversity would have to improve predictions using retained relation-resolved information. Different factor norms, different relation labels, lower embedding cosine similarity or equal utilization do not show that. A common-message gradient also remains ordinary differentiation through the pool and transpose relation operators; no new gradient principle is provided by the r index.

An operation that would distinguish this branch must specify **which member-specific relational contrast is retained, why it is task-useful, how it is transported at lower total cost, and what approximation/error consequence follows**. The present proposal supplies neither a contrast-retention rule beyond known mixtures/caches nor a qualified nonlinear closure. An elementary commutation identity alone cannot fill that gap.

## Pinned source findings

Author sources were read as text, never imported or executed. Commits are retrieval-time snapshots, not certified manuscript-version commits.

- **R-GCN coauthor repository**, [tkipf/relational-gcn](https://github.com/tkipf/relational-gcn), commit 4bec1341dd46b72bf482f7ed26c2dca4533577f6 (15 March 2018). README identifies the entity-classification implementation; the inspected primary HTML did not link this repository. graph.py lines 81–113 aggregate each support first, concatenate the results and apply a composed basis/dense map before activation. Lines 53–63 construct bases and relation coefficients. This is direct code evidence for the linear ordering, not a current runtime qualification. README pins Keras 1.2.1/Theano 0.9.0 and describes substantial compilation; it is not a drop-in modern runner.
- **HGT paper-linked repository**, [acbull/pyHGT](https://github.com/acbull/pyHGT), commit 85eaccd482bc1d1af56c2de297b6e3a88b96d5cd (8 September 2023). HGTConv lines 60–111 computes Q/K and relation scores from current endpoint states, softmaxes over target-node neighborhoods, then weights relation-transformed messages. Lines 114–134 apply nonlinear, typed output/skip/norm updates. Holding only relation matrices common cannot make private attention common. Other convolution classes were not method-audited. The pinned code's relation-prior shape differs from the paper's printed full meta-relation prior; no exact-version reproduction is claimed.
- **SeHGNN paper-linked repository**, [ICT-GIMLab/SeHGNN](https://github.com/ICT-GIMLab/SeHGNN), commit e92bd37d0b803457339555684f139b4c8f3e160d (16 October 2023). ogbn/utils.py lines 49–84 builds metapath features by repeated typed mean aggregation; ogbn/main.py lines 77–105 extracts/reorders the cache before stage training. HGB model.py lines 71–92 gives separate semantic projection matrices; lines 196–228 consumes semantic inputs and fuses them locally.

SeHGNN has important cost/protocol distinctions. The HGB model can consume sparse metapath matrices and multiply learned embeddings inside the forward path (lines 196–201); that work must not be described as a free dense feature cache. The OGB driver shares fixed feature propagation before stages, while label/pseudo-label inputs are propagated again within stages (main.py lines 129–185 and 250–291). The README's stronger ogbn-mag variant adds 500,000-step ComplEx acquisition. A hypothetical ensemble must charge those choices and cannot claim published competence while silently removing their input/pretraining budget. The source prints validation/test diagnostics; this review does not infer test-tuned selection or certify a new execution protocol.

These source branches are sufficient to resolve the sharing assumptions. No dataset, artifact weights, external embeddings, notebooks or personal files were retrieved.

## Why no representative pilot is proposed

The task asks for a distinct learning direction before a runner or compute lane. The three feasible choices above reduce to an existing decoupled semantic ensemble, the current common-message architecture with typed maps, or packed member work. Error-directed relation seeding and generic coupling add already assessed ingredients. A full-data pilot would therefore study known-component utility without isolating an additional mechanism.

No dataset subset, tune grid, training recipe or amended active protocol is supplied. Reopen only after a concrete contrast-retention/closure mechanism addresses the stated information/cost boundary and survives comparison with a capable cached semantic model, an equally located diagonal-adapter ensemble and exact packed relation transport. That would be a new assessment. It does not authorize changing the active graph-init or BUDDY study.

## Custody and limits

READ_SCOPES.json and INSPECTED_PASSAGES.json retain exact scoped block/source-line locators and hashes. REUSED_CONCLUSIONS.json preserves saved scopes/statuses and no new read credit. INPUT_BINDINGS.json binds index v17 and the saved idea reports. Bounded discovery returns are metadata only; mostly off-topic results provide no absence-of-prior evidence.

An initial guessed SeHGNN identifier, 2211.12740, resolved to an unrelated masked-autoencoding paper. Only its abstract-page metadata was retrieved; it was excluded without a method read. Exact-title discovery corrected the identifier to 2207.02547v3. A README-case 404 was corrected using the pinned tree's Readme.md spelling. Both outcomes remain in retrieval logs.

The three-new-primary cap is exhausted. No first-use claim, superiority result, full-paper read or current-host qualification follows from this bounded packet. Predecessor sealed packets, index v17 and active protocols are unchanged.
