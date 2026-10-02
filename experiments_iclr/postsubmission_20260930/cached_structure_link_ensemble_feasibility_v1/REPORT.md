# Cached-structure link prediction with four factorized predictors

## Decision

**Source-level go for a practical shared-cache BUDDY predictor comparison; no-go for promoting a new learner or launching a graph-residual-guidance pilot.** BUDDY supplies query-specific structural information that the previous pooled dot-decoder proposal lacked. Its fixed preprocessing is a credible boundary for sharing work. Replacing its predictor with a parameter-efficient ensemble is an attributed architecture port, and the evidence does not establish an additional graph learning mechanism.

This packet prepares literature, source and protocol feasibility only. No implementation, data acquisition, fitting, model/checkpoint/label access, scientific execution, SSH or GPU work occurred. Current studies and earlier packets are unchanged. The prospective comparison below is not admitted or launched; native parity and measured complete-task resource qualification remain necessary.

## Evidence reused before acquisition

`literature_memory/index_v15` was consulted first, followed by the existing `link_covariance_idea_v1` and `gine_closest_control_v1/link_recommendation_gap_v1` reports and source maps. Retain their conclusions:

- Averaged member dot scores can collapse exactly to a deterministic wider embedding; member-loss training supplies an ordinary auxiliary-loss/regularization difference. Do not reopen that proposal.
- BUDDY/ELPH, NCN/NCNC, Neo-GNN and SEAL already introduce query-specific structure. Factorized independent-node embeddings do not supply it by themselves.
- Shared heterogeneous heads and consensus transfer are direct prior in ConCF. Member-aware negative selection must confront MixGCF, SRNS, AHNS and DivNS rather than claim novelty from disagreement.
- BatchEnsemble and TabM establish shared dense weights, private modulation, member losses and ensemble-aware selection. Packed-Ensembles/MIMO and deterministic grouped predictors are relevant architecture controls, with their actual adaptations declared.

The old LP scout's deferred broad family is not reused as a launch plan. New acquisition was restricted to unresolved current cached-LP discovery, a direct mixture-of-link-predictors primary, and the released BUDDY source boundary/defaults. Exact scopes and failures are retained. Searches are bounded, and misses do not certify novelty.

## Why BUDDY is the useful boundary

The retained primary [Graph Neural Networks for Link Prediction with Subgraph Sketching](https://arxiv.org/abs/2209.15486), §5, precomputes propagated node features and MinHash/HyperLogLog neighborhood sketches. Query pairs receive approximate intersection/distance-label counts and endpoint feature interactions. Its predictor learns from those features without a trainable graph pass each batch. The paper gives preprocessing O(kE(d+h)) and pair scoring O(k²h+kd²); caching a fixed query set can avoid repeated structural-feature extraction.

The changed question required a targeted reread of §5, Appendix B.2–B.4 and caching passages, plus newly pinned source inspection. The source commit is **3562d94a07d1166faa0949030824bf75ad9bb2c4**, 5 December 2023. Source files were read as text, never imported or executed.

| Component | Shareable in the proposed comparison | Limit |
|---|---|---|
| Training topology, deterministic propagated supplied features, node sketches/cards, degree and fixed-pair structural caches | One versioned cache for all arms and members | Changes in topology, query/negative identity, hash settings, direction/weights or preprocessing require a different cache. Shape alone does not establish cache identity. |
| Learned native BUDDY feature/structure predictor | Dense W can be shared with private factors | Private factors before endpoint products or nonlinearities still require member computations. These learned transformations are not fixed preprocessing. |
| Trainable node embeddings and their SIGN propagation | Can be one explicitly shared trainable component | Released training recomputes propagated embeddings per edge batch. They cannot be cached forever while being optimized. Parameter, optimizer and graph-pass costs remain. |
| ELPH learnable feature encoding | One explicitly common encoder is possible | Private encoder modulation destroys common-state reuse; a common encoder with private heads is already a shared-trunk model. ELPH is a more expensive separate branch, not the preferred cache pilot. |

Crucially, **four independent BUDDY predictors can share the same deterministic caches too**. Compare one preprocessing build plus four independent predictor fits with one build plus a jointly trained M4 factorized predictor. Savings from avoiding four redundant cache builds cannot be attributed to M4. Sharing sketches also shares their approximation errors; predictor diversity does not independently resample or correct the graph structure.

For native-structure parity, factorize the existing learned predictor branches while preserving their endpoint products, feature/structure concatenation and scalar output. Member-specific BatchNorm statistics and affine parameters need an explicit contract; merging member rows into one BatchNorm changes the model. A frozen common representation followed by four *linear* heads is a smaller alternative, but mean-logit pooling then collapses to one linear head. Introducing nonlinear private heads creates an ordinary shared-trunk multi-head predictor and requires the corresponding deterministic/capacity control.

## Direct mixture/ensemble prior: Link-MoE

[Mixture of Link Predictors on Graphs, 2402.08583v2](https://arxiv.org/html/2402.08583v2), NeurIPS 2024, was read at §§4.1–4.3, Eqs.1–3, Algorithm 1, §5.1 setup, Appendix A.2, and scoped G/H/J/K passages.

Its prediction is

\[
p(u,v)=\sigma\!\left(\sum_o G_o(x_u\odot x_v,s_{uv})\,z_o(u,v)\right).
\]

The gate has separate MLP branches for endpoint Hadamard features and structural heuristics, concatenates them, and uses an MLP/softmax to produce pair-specific expert weights. Its features include degree, CN, AA, RA, shortest path, Katz and PPR. Experts include BUDDY, NCN/NCNC, Neo-GNN, SEAL, GCN, MLP, Node2Vec and others; large-graph experts differ where memory limits apply.

**Training is two stage:** train each expert separately with its own recipe, infer link scores, then optimize the gate on those fixed scores using BCE. Algorithm 1 updates gate parameters in its second stage. This amortizes repeated router tuning and avoids having every expert resident simultaneously; it does not erase expert training/inference cost. Appendix G discusses an end-to-end variant and expert collapse as a reported outcome, not a guarantee against collapse for a new model.

Appendix A.2 repurposes **80% of the official collab validation set for gate training and 20% for gate selection**. This extra supervised-label use is a material protocol difference. The proposed training-only M4 screen must not silently compare against its headline numbers. A future native Link-MoE comparator requires its declared validation split and equally available supervision, or a clearly labeled adaptation with an internal training calibration split. The test set remains held out.

The paper explicitly includes uniform Mean-Ensemble and learned Global-Ensemble controls. Appendix H compares prior LP stacking: Ghasemian et al., *Stacking models for nearly optimal link prediction in complex networks* (PNAS 2020), and Chen et al., *An ensemble model for link prediction based on graph embedding* (DSS 2022). Their names/mechanisms here are bibliographic and author-described leads, not new independent full-primary reads. Link-MoE itself is direct primary evidence that pair-structural expert combination and cached expert-score training are established. M4's homogeneous factor sharing could reduce predictor storage relative to its heterogeneous experts, but supplies no new routing or graph-learning principle.

## Recent competence reference

[PENCIL, Plain Transformers are Surprisingly Powerful Link Predictors, 2602.01553v4](https://arxiv.org/html/2602.01553v4), revised 28 September 2026, was read only at §§3.1–3.2 and the abstract/version record. It tokenizes a sampled query subgraph with local one-hot IDs, adjacency rows and endpoint task tokens; bidirectional attention and an adjacency-propagation residual learn query representations, followed by endpoint concatenation and a scalar BCE head.

This is a current structural LP reference, not a cached-node-feature replacement: query subgraphs and trainable Transformer computations remain. Its reported benchmark superiority was not independently checked or transferred here. NCN/NCNC and PENCIL are later confirmation controls if a strong LP claim is sought. A small factorized BUDDY comparison alone establishes neither state of the art nor superiority to these methods.

## Can residual guidance act in edge/task space?

Yes, the dimensions are well defined. For a fixed training-positive/negative candidate set Q, let z_Q be warm predictor logits, r_Q=σ(z_Q)−y_Q, and J_Q the Jacobian of those logits with respect to private factors. A candidate direction could be

\[
d=-J_Q^\top F_Qr_Q,
\]

where F_Q is an explicitly declared operator on candidate edges. For example, an incidence matrix connects candidate edges to their endpoints, allowing endpoint-sharing smoothing without materializing a dense line graph; a training-graph propagation operator can additionally connect those endpoints. Matrix-free products can cost O(|Q|+|E_train|) per pass, while materializing pair adjacency can cost Σᵥd_Q(v)² and be prohibitive at hubs. Negatives belong to a task relation set for this calculation, not to the graph used for BUDDY sketches or propagation.

This is an implementable research description, **not a supported methodological delta**:

- Using BUDDY structural features in a predictor gradient or in J already makes the gradient structure-informed; it remains ordinary supervised learning on fixed graph-derived features. TabM substitution supplies no additional graph operation.
- Extra edge filtering changes the optimization direction, but standard filtering plus a VJP does not prove useful diversity. Even PSD F_Q does not generally make −JᵀF_Qr a descent direction for the original BCE, since JJᵀ and F_Q need not commute.
- Endpoint-sharing candidate relations depend on the sampled negative set and its artificial label mixture. Sharing an endpoint does not establish that two missing-edge labels or residuals should agree. A task-space operator must separate directed source/target roles where appropriate and avoid future positives and held-out labels.
- M4 contrast directions cannot be assumed to improve individual or pooled losses. Their relationship to the existing graph-residual initialization remains an application of that prior operation until a distinct edge-specific target and advantage survive a direct assessment.
- Link-MoE already uses structural heuristics to choose per-pair expert contributions. A proposed factor gate or structure-conditioned residual reweighting must specify a difference from this direct prior, generic gradient preconditioning, raw-residual initialization and a deterministic grouped predictor. The current source evidence supplies no such advantage.

The narrow discovery did not resolve a separate full-primary edge-residual method. This uncertainty is preserved; it is not interpreted as an open novelty gap. No residual filter, sampler, factor controller or warm-start training machinery is built or recommended for a model pilot.

## Representative protocol and prospective pilot

Use **complete ogbl-collab** as the first bounded practical task: 235,868 nodes, 1,285,465 documented edges before inverse-edge processing, supplied 128-dimensional features, training through 2017, validation 2018 and test 2019. Retain official Hits@50 against the common official 100,000-negative pool and its strict threshold tie rule. These are saved OGB documentation/source metadata, not accessed dataset exports. This is a fixed temporal split on a static forecasting benchmark, not an event-stream temporal-GNN evaluation.

Protocol requirements:

1. Freeze training-only topology for preprocessing, validation and test. Adding validation edges for collab test is OGB-permitted but a separate declared protocol. Released BUDDY `get_ogb_data` currently adds them unconditionally; the presence of a CLI flag does not disable that branch. A prospective port must make the policy explicit and verify it before execution.
2. Preserve all official nodes, training supervision and held-out pairs. Released code disables largest-component filtering for OGB despite the paper's broader LCC statement. Set year cutoff to zero; the saved README's collab command uses 2007 and therefore differs. Freeze edge coalescing, weights, inverse directions and degree normalization.
3. Generate one declared training-negative set using training topology and non-self-pairs only, with one negative per positive for the bounded screen. Do not reject candidates using future/test-positive identities. Cache identity must include negative seed, pair order and graph hash; the released negative-cache filename alone does not. All arms use the same positive/negative examples and batches. This fixes an empirical sampled BCE objective; it does not establish calibrated edge probabilities.
4. Retain official validation/test negative arrays unchanged. Do not substitute per-positive pools, AUC or HeaRT negatives. Link-MoE's HeaRT appendix describes a separate harder, endpoint-constrained negative protocol, not an interchangeable score.
5. Select checkpoints using validation only; expose final test scores once the family is locked. The released runner evaluates test each epoch, so it is not used unchanged for this research contract.

**Proposed practical family, pending qualification:** one native-structure BUDDY predictor, its M4 factorized port, four independently trained predictors with the same shared cache, and an analytically parameter-matched single predictor. Use three fixed seeds for an exploratory paired comparison; this is 12 task/arm/seed cells and 21 optimizer fits when independent members are counted. No graph-residual arm is included.

For a modest GPU comparison, a single fixed hidden width **256**, two hash hops, HLL p=8, MinHash 128, no trainable node-ID embeddings, the collab README's lr=0.02/feature dropout=0.05/label dropout=0.1/normalized structure features, BCE/Adam, one fixed negative per positive and 100 complete epochs is a reasonable declared bounded port. Width 256 is an adaptation, not the released collab CLI default 1024 or a native published-score reproduction. Native-structure parity precedes factorization; BatchNorm/member-state semantics and exact total parameters must be checked when code is separately authorized. Batch/feature-extraction chunks may be reduced for memory without omitting rows; they must be fixed across matched arms because training batch size affects optimization.

The first qualification should be CPU cache construction plus representative complete training/validation passes, using a small GPU only for dense predictor work if needed. Public dimensions imply about **115 MiB** for one float32 N×128 node-feature matrix, about **864 MiB** for three depths of N×(128 int64 MinHash + 256 int8 HLL) sketches, and **32 bytes per candidate pair** for eight float32 structural counts. Temporary buffers, separate graph policies, Adam, BatchNorm and four member activations are additional. The supplied graph is moderate in size, but these byte counts do not prove peak memory, CPU completion time or GPU hours. Do not use the source default 11-million-pair extraction chunk without profiling.

Measure cold preprocessing, cache bytes and reads, all predictor fits, negative generation, checkpointing, full validation/final official scoring, warm latency and peak memory. Compare independent predictors with cache reuse on equal terms. Report all seeds and paired uncertainty; a one-task pilot is conditional fixed-split evidence. DDI is a later non-temporal confirmation with no supplied features and dense topology; its learned-ID/propagation path needs a separate resource assessment. PPA/citation2 are not initial small-pilot substitutes.

**Final go/no-go:** retain this as a practical benchmark feasibility packet. Proceed only to separately authorized native parity and resource qualification if that application benchmark is useful. Do not claim a new graph ensemble method, launch guided factors, or build training machinery on the present evidence.
