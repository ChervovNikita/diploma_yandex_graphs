# Heterogeneous graph private-factor gap assessment

**Decision: no credible distinct method gap is established. Do not open another method or training lane for type-indexed rank-one factors.** This bounded assessment used the literature memory first, reused prior conclusions, and read exactly two new primary papers. No exact duplicate of the complete proposed typed-factor ensemble was verified. That is a retrieval limit, not evidence of novelty.

The useful result is a sharper comparator boundary: heterogeneous graph ensembles, graph-layer BatchEnsemble, typed graph adaptation, structural expert initialization, and supervised graph-route initialization already have precedents. Giving the private vectors an extra node-type or relation-type index does not supply a distinct learning operation.

## New primary evidence

| Source | Exact method evidence | Consequence |
|---|---|---|
| [HG-Adapter, 2411.01155v1](https://arxiv.org/html/2411.01155v1), submitted 2 November 2024 | §2.3 factorizes adapter maps into low-rank matrices, learns same-type k-nearest-neighbor structure, and learns a softmax over neighbor/relation types for each target node. Adapted messages are added to frozen graph embeddings. §2.4 uses training-label propagation, class-subgraph contrast, feature reconstruction and a margin objective. | Compact graph adapters, typed structural weighting and label-informed graph adaptation are direct prior. This source does **not** establish an exact rank-one ensemble. Its theorem statements were inspected for context; their proofs and assumptions were not validated. |
| [Temporal Heterogeneous Graph Pretraining for Relational Deep Learning, 2609.35219v1](https://arxiv.org/html/2609.35219v1), submitted 28 September 2026 | TimeMix has shared temporal scales and table-specific projections. TimeRoPE constructs temporal messages with relation-specific neighbor/self maps. Typed pooling, relation-specific decoders, structural contrast and staged pretraining are explicit. THGFM has relation-specific attention plus a shared-space branch with endpoint adapters and type-conditioned gated fusion. | This very recent predictive framework supplies competent typed and temporal comparators. It is a preprint, and the scoped text does not define the endpoint adapters' rank. Neither ensemble uncertainty nor a runtime benefit from rank-one factors was demonstrated here. |

Exact inspected line ranges and retained extracts are in [INSPECTED_PASSAGES.json](INSPECTED_PASSAGES.json); machine-readable takeaways are in [PAPER_CONCLUSIONS.json](PAPER_CONCLUSIONS.json). Primary HTML bytes, versions, request timestamps and hashes are retained. No author-linked implementation URL appeared in either retained paper HTML; author code and dataset exports were not retrieved.

## Reused evidence and the remaining distinction

The saved Kim 2023 official-viewer assessment already verifies graph-layer shared matrices with private rank-one modulation in GCN/GIN/GAT, separate member states, full-graph duplication and probability averaging. HGEN 2025 already ensembles heterogeneous meta-path graph GNNs and penalizes pooled representation similarity. CHoE 2026 already routes frozen meta-path experts using structural similarity and load penalties. TabM already supplies deliberate private-factor initialization and independent member losses.

GraphMoRE already varies expert curvature at initialization and uses topology-conditioned routing; its topological heterogeneity must not be confused with node/relation types. PreGS already initializes graph experts by supervised parameter transfer. The existing round 15 graph-band residual/VJP factor-seeding proposal already addresses a one-time graph-informed expansion from a warm shared predictor. A typed partition of that same operation belongs to its existing question until a further mechanism is specified. These are reused conclusions, bound in [REUSED_CONCLUSIONS.json](REUSED_CONCLUSIONS.json), without rereading unchanged papers.

The strongest obstruction is analytical. For any eligible type/relation map, a member transform

`D_s W_r D_a h`

is exactly the diagonal residual path

`h' = h + (a−1)⊙h; y = W_r h'; output = y + (s−1)⊙y`.

Type or relation indexing changes which vectors are selected, and preserves this equality. Keeping adapters at the exact sites also preserves private states, nonlinearities, normalization and readouts. Matched initialization, regularization and optimizer coordinates are essential. This is a derived identity using the saved path-equivalence result, not a claim that HG-Adapter is the identical family.

Typed factors may improve an empirical inductive bias when one global channel adjustment is inappropriate for different relations. That is a conjecture about adaptation utility. They also increase private parameter count with the number of eligible maps. Separate nonlinear member paths still require their complete graph states, relation aggregation, temporal operations and dense transforms; sharing stored matrices does not establish fewer graph computations. Storage savings could matter under a measured storage constraint.

## Baseline admission and stop rule

No new extension is proposed. If an independently distinct heterogeneous mechanism later warrants a utility study, the retained recent reference is HeteroGNN with TimeMix/TimeRoPE and Temp-Sub→Rel-Hist; THGFM with Temp-Sub→Rel-Future is its stronger Transformer-family reference. The paper compares five backbones and chooses its two families by mean validation rank. Its native recipe has three layers, width 128, fanouts 128/64/32, 100,000 pretraining steps and 100,000 downstream steps per run. All preparation, pretraining and complete member work must be charged; this is a reference recipe awaiting exact implementation custody.

Representative full settings are RelBench **rel-arxiv** author-category/paper-citation (validation/test cutoffs 2022-01-01/2023-01-01) and **rel-avito** user-clicks/user-visits (2015-05-08/2015-05-14), preserving their original chronological splits and cutoff-valid sampling. Their sizes and splits here are source metadata. A future claim needs the same native single model, one-computation shared-trunk heads, global versus typed factors at matched sites, capable untied/low-rank-adapter ensembles, and an exact diagonal-path audit. Predictive NLL and a predeclared abstention-risk decision should accompany measured storage, peak memory and end-to-end cost.

**The current stop rule is met:** no additional learning operation survives beyond known type conditioning, adaptation, ensemble factors or the existing initialization lane. Do not turn absence of a search hit into promotion. If separately reopened for adaptation utility, stop promotion when a benefit disappears under competent matched controls or comes solely from storage savings without a storage-constrained workload.

Search coverage is bounded: OpenAlex metadata, the newest 50 arXiv heterogeneous-graph titles, and targeted term queries. Two targeted arXiv queries timed out; Bing returned irrelevant dictionary results and Google returned a redirect page. These failures are preserved and provide no negative literature evidence. A September 29 transport-map BatchEnsemble paper was discovered only at abstract level; its graph is a transport DAG, so it is not typed-graph method evidence. [SEARCH_SCOPE.json](SEARCH_SCOPE.json) records these limits. No models, data, training, SSH, installations, manuscripts, PDF compilation or original scores were touched.
