# Efficient graph ensemble controls: final literature comparison

**Version 1, 1 October 2026. Source inspection and symbolic arithmetic only.** No neural network, fit, dataset, label, checkpoint, GPU or server was accessed. This is a comparator recommendation, not execution admission or a novelty certificate. New work is confined to `graph_ensemble_gap_skeptic_v1/efficient_controls_v1`.

## Decision

1. **First direct control: packed independent SAGE, with both parameter and sparse-channel budgets stated.** Its graph computation is a straightforward port of Packed-Ensembles when the neighbor operator is fixed and linear. Audit and reuse the existing implementation; its default width44 targets an older architecture.
2. **MIMO is an established efficient ensemble method.** A sound one-graph control is MIMO on intact, label-free SIGN feature rows, under a separate contract for example pairing, capacity, repetition and pooling. The existing deferred SIGN comparison is a different frozen model and must not be silently converted or repeated.
3. **Rank-1 Bayesian factors are a secondary storage/uncertainty control.** A valid port needs actual variational inference and coherent GNN weight samples. Its small posterior storage does not imply one graph propagation or cheap inference.
4. **Reuse the existing graph-specific dispositions.** Audit the completed all-layer BE control against Kim's recovered thesis; retain shallow heads as a cheap common-backbone control and E2GNN as a later deployment control. N-Ens is additional direct prior for trainable member normalization/diversity, with separate member trajectories.

None of these methods alone tests the own-only adjoint intervention in the shared-mean model. That intervention needs the separately specified mechanism comparison. Generic input/output concatenation, grouped independent subnetworks, trainable member factors, and normalization-based diversity are not new contributions here.

## What is faithful on each graph regime?

| Family | One transductive graph | Graph-level mini-batches | Essential distinction |
|---|---|---|---|
| MIMO, Havasi et al. | Sample complete cached rows `q_v=[X_v,(PX)_v,…]` with their training node IDs/labels; concatenate M rows before an ordinary hidden map, predict the M corresponding labels. Raw row shuffling against unchanged P is invalid. | Straightforward if samples are graph signals with a common semantically aligned vertex set and common P. Arbitrary unordered, differently sized graphs require a declared invariant encoding or a bespoke multi-graph construction. | Native independent input tuples are different from repeating one graph throughout training or adding heads to a shared GNN embedding. |
| Packed-Ensembles | M independent narrow GNN states/parameters on the same graph. For fixed P, pack states into `[N,Md]` and apply P once. | Every member sees each intact graph; a block-diagonal batch operator can act on packed channels. Different graph sizes need no cross-graph node alignment. | One sparse call still processes Md edge channels. Parameter independence does not require independently shuffled data. |
| Rank-1 BNN | Draw effective layer weights and keep each sampled weight fixed throughout the full graph's dependency computation. Score only permitted training labels. | Each intact graph may receive its own complete weight draw, as the paper permits sampling per independent example. | Node-specific random factors in one coupled message-passing pass define stochastic node gates, not an ordinary shared-weight posterior sample. |
| Kim-inspired graph BE / N-Ens | Separate member graph states remain necessary; vectorization is possible. | Intact graphs can be batched/vectorized for all members. | Shared stored weights do not establish a single shared graph trajectory. |
| Shallow heads / E2GNN | Shallow heads compute a common GNN representation once; the distilled E2GNN MLP needs no graph at deployment. | The same distinction applies after task-appropriate readout. | These offer deployment references, with different member capacity or multi-stage training. |

### MIMO: the example contract is decisive

The primary [2010.06610v2](https://arxiv.org/html/2010.06610v2), §2, says inputs are “sampled independently from the training set”; the loss sums the M matched NLLs, and inference tiles the same example M times then averages predictive distributions. These are the key semantics, not the acronym or simply having M output heads. The paper's 0.03% parameter and 0.01% FLOP overhead are for a 36.479M-parameter ResNet28-10. They do not transfer to a small GNN whose expanded input/output maps can dominate its budget.

For a transductive SIGN port, shuffle **whole cached rows and their corresponding training labels together**. Build the fixed cache without labels, with the same complete graph visibility and node ordering as the admitted graph recipe. Cached rows may be correlated through shared neighborhoods; sampling row IDs independently reproduces MIMO's tuple-allocation rule, not an assertion that graph observations are statistically independent. Training tuples must draw from permitted training IDs; validation/inference repeats each held-out row.

For recurrent raw-X message passing, a permutation of node features must also preserve the correct graph context. Leaving P fixed while slot m uses `X[π_m]` attaches features to the wrong vertices. Permuting the graph as `P_m=Π_m P Π_mᵀ` preserves each copy, but mixing slot channels then requires a declared multi-operator architecture. Rooted subgraphs or full graph copies preserve context at additional acquisition, propagation and memory cost; they are not an automatically cheap native MIMO port. Repeating identical `(X,P)` throughout training corresponds to the fully repeated-input end of the family, where §3.5 warns that heads become correlated.

For arbitrary graph-level examples, padding alone does not supply a meaningful node correspondence, and a disjoint union simply batches M intact graphs while retaining their graph work. A permutation-invariant GNN encoder followed by MIMO on fixed-length graph embeddings is feasible, but is a **late-fusion adaptation**. Charge its encoder and disclose which part actually sees independently paired inputs. Fixed pretrained/cached graph embeddings additionally change the training problem. Do not claim an arbitrary multi-graph construction inherits image MIMO's one-trunk cost.

Capacity and repetition must be competent. §3.5/Appendix B use input correlation ρ to help limited-capacity networks, M3/batch repetition4 for CIFAR, and M2/ρ=.6/repetition2 for ImageNet. A forced M4/D64/ρ=0/no-repetition arm can handicap MIMO. Declare a small equal validation-selection budget across methods, or label a fixed zero-search port as such; do not import published test-tuned settings as guaranteed graph optima. Charge all tuple presentations and repetitions. The copied author training source divides unique batch size by repetitions, repeats/shuffles indices, gathers features and labels with the same indices, sums member CE, and increases steps per epoch accordingly.

MIMO initializes an ordinary wider-input network and independent head coefficients, not BE factors: the retained author model uses He-normal convolution/head kernels and BatchNorm. A SAGE/LayerNorm/AdamW port is an explicit backbone/protocol adaptation. Sum versus mean member loss changes its scale relative to regularization; record the reduction and compatible optimizer/decay choice rather than equating recipes by numeric learning rate alone. At repeated-input inference an initial affine map can be folded exactly: `[q,…,q]W=q∑_m W_m`, with the original bias. Permit that optimization and verify its assumptions before future timing.

### Packed-Ensembles: clean graph port, two different budgets

The primary [2210.09184v4](https://arxiv.org/html/2210.09184v4), §3, propagates independent groups through the network. Per-member width is `αC/M`; hidden matrix cost scales as `α²/(Mγ)`. With γ1, `α=√M` matches **hidden** matrix parameters, while αM corresponds to full-width independent models. Boundary maps, biases and normalization prevent a generic whole-model width match. The paper permits the same batch for all members and simplifies the first layer to one ordinary map with distinct output-channel filters.

The existing `PackedIndependentSAGEEnsemble` has independent input/root/neighbor/FFN/output weights and private per-member LayerNorm. Its fixed incoming-neighbor mean can be packed exactly because `P[H1,…,HM]=[PH1,…,PHM]`. This claim concerns fixed linear SAGE/GCN aggregation. GAT/member-dependent edge weights require member-specific attention/edge computation and different accounting.

For the current M4/D64/L2 source, use the exact stored parameter formulas from source, not the old D128+r16 default. With features F and classes C:

`T = 68F + 72C + 42,496` for the tied current model;

`PE(d) = 4[10d² + d(F+C+13) + C]` for independent source-structure SAGE models.

| Existing task dimensions | T | Closest integer d / PE(d) | Signed mismatch | Sparse channels Md | Channel match d16 / parameters |
|---|---:|---:|---:|---:|---:|
| Products F100/C47 | 52,680 | 29 / 52,388 | −0.554% | 116 | 64 / 20,668 |
| Reddit2 F602/C41 | 86,384 | 24 / 86,180 | −0.236% | 96 | 64 / 52,388 |

These are symbolic recommendations, not proposed repeat fits or measured feasibility. For another task calculate from its public F/C. The parameter-matched arm has more than64 sparse channels; the channel-matched arm has fewer parameters. Both help interpret a utility claim. Width44 processes176 channels and is stale for this target. `PackedSharedSAGEEnsemble` instead packs tied full-width trajectories and is an execution control, not the Packed-Ensembles architecture.

Avoid artificial loss of diversity: use distinct member initializations and private normalization; identical initial weights plus completely synchronized deterministic training can remain identical (Appendix F). Independent data pairing is unnecessary. The retained current TorchUncertainty `PackedLinear` initializes each group with its own fan-in Kaiming-uniform draw; the local graph prototype uses Xavier weights and zero biases. These are different choices. For a source-matched independent-SAGE control, initialize/transfer each source member's tensors under a declared seed policy, preserving the source's per-map rules. Do not flatten all member weights before fan-in initialization. Also audit normalization: the retained library `PackedLayerNorm` defaults to one group unless `first=True`; applying that default across all packed member channels would mix their statistics. The local `GroupedLayerNorm` normalizes only the member width and has private affine tensors.

Use γ1 initially. Appendix B reduces γ when subgroups are too narrow (minimum64 channels in the image experiments) and omits subgroups at some boundaries. Those image thresholds are not a universal graph requirement, but silently applying heavy subgroup sparsity at width16/24/29 would be an additional handicap. No theoretical grouped-kernel FLOP reduction certifies wall speed: Appendix H reports bandwidth limitations and hardware/precision dependence.

### Rank-1 Bayesian factors: retain the Bayesian recipe

The primary [2005.07186v2](https://arxiv.org/html/2005.07186v2), §§3.1–3.4, puts distributions on r/s in every effective weight `W′=W⊙rsᵀ`, point-estimates W, and optimizes expected likelihood plus factor KLs and a W prior/L2 term. Replacing deterministic BE factors by occasional noise without this objective is not a reproduction.

Appendices B/C specify ordinary-network initialization for W; posterior means initialized with random ±1 signs or normal draws centered at1; small initial Gaussian posterior scales and distribution-specific Cauchy settings; all-layer factors excluding normalization (and certain embedding layers); and linear KL annealing for2/3 of training. The native image recipe uses four mixture components. Gaussian evaluation defaults to one draw per component; Cauchy evaluation uses four per component, and additional Gaussian draws improve results at additional compute. Training horizons also exceed deterministic baselines. A graph recipe must declare family/prior/scales, KL normalization by the **training-label count**, schedule, component assignment, all factored maps, normalization/bias treatment and inference samples. Do not silently use total node count to dilute the KL.

For each sampled GNN, hold its sampled effective layer weight constant over all nodes participating in that graph computation and over uses of a tied parameter. Different layers may have different factors; the full draw and any mixture association must follow a declared joint distribution. Resampling factors independently at each node inside one coupled pass changes the stochastic function. Independently sampling a full coherent rooted-graph forward per target can estimate expected node loss, but incurs those dependency computations. The paper's per-example sampling shortcut is directly applicable to independent graph examples or fixed cached-row MLPs, not automatically to interacting node states.

K components × S draws require corresponding nonlinear graph trajectories, even if vectorized and even though shared W is stored once. Record posterior means/scales, all component vectors, optimizer storage, active states, sparse channels and every draw. The default image Gaussian K4/S1 case is four GNN samples, not one common message pass. Cauchy K4/S4 is16. This makes the method valuable for storage/UQ comparisons; it is not evidence of graph inference speedup.

## Graph-specific prior and existing disposition

- **Kim 2023 thesis:** the existing official-viewer recovery verifies graph-layer rank-one factors, separate hidden states, full-graph duplication and probability pooling. Factor initialization, some normalization details and exact author code remain unresolved. Audit the already completed all-layer study; call any newly repaired source a Kim-inspired port, not exact reproduction.
- **Shallow graph ensembles / DPOSE:** verified sources already establish a common graph trunk with private output heads, chiefly regression energy/force uncertainty. A classifier requires an explicitly adapted objective. Reuse this disposition as a cheap backbone/head reference; do not infer classification success from atomistic regression.
- **E2GNN:** the existing later-control recipe preserves five teachers, policy/student training, extra validation-label supervision and native test-access/seed issues. Its graph-free student lacks M member predictions. Retain it as a separate deployment-efficiency control and charge teachers/policy/student/cache; no launch or repeat is recommended here.
- **N-Ens, [2607.23860v1](https://arxiv.org/html/2607.23860v1), §§3.1–3.2:** root-retrieved primary text directly verifies shared conv/attention weights, private sigmoid-bounded normalization scales and fresh private heads; each member takes its own forward. Its minimized regularizer is `+λ∑log softmaxτ(sigmoid γ)`, **not Shannon entropy**. This is close prior against broad novelty claims for learned member allocation or controlled factor diversity. Reported parameter-cosine/output correlations in vision do not guarantee predictive diversity on a graph. A SAGE/LayerNorm port needs separate graph accounting.
- **Graph MIMO search collisions:** [2505.11346v1](https://arxiv.org/html/2505.11346v1), §3, calls d input feature channels and c output feature channels of one graph signal “MIMO”; it studies spectral/localized graph convolution. [2305.11389v2](https://arxiv.org/html/2305.11389v2), §§3–4, predicts graph modes through meta-information-generated encoders/decoders and link completion. Neither inspected method supplies Havasi-style independent-tuple ensemble training. Wireless MIMO detector hits are likewise not this comparator.

Six bounded arXiv searches are retained. The packed-graph query returned zero entries; this does not establish absent graph prior. GEENI and the Information Fusion article keep their existing primary-access limitations. No claim of exhaustive coverage follows.

## Prospective fairness and cost contract

Use native probability pooling for new MIMO/PE/rank-1 Bayesian ports. The existing shared-mean and cached-SIGN recipes freeze raw-logit pooling. For a new comparison, predeclare native and common-pooling views and which selects checkpoints; charge every validation candidate. Preserve old scores. Probability pooling and softmax of average logits generally differ.

Before any future execution, seal architecture/init/loss/sampling/pooling, complete graph coverage/split visibility, finite validation-selection budget, fresh seed policy and full cost cap. Audit existing definitions before adding fits. Record:

1. Trainable storage **and** optimizer/posterior storage, shared/private maps, parameter mismatch, per-member width and active node-state bytes.
2. Sparse calls **and edge-channel work**, dense operations, graph copies/readout work, backward/recomputation, tuple slots, unique IDs and total example presentations, repetitions, K/S and normalization semantics.
3. Total training/preparation/selection wall, failures, cache construction/verification/storage, teacher acquisition and fitting, policy/student work and any reused upstream training cost.
4. End-to-end cold-from-raw and warm-representation deployment, graph/cache loads, output pooling/transfers, complete-node coverage, peak host/device memory and measured hardware/precision/runtime. Amortize a cache once over declared reuse; also report a single-deployment cost.

Permit exact affine folding, first-layer repeated-input simplification, valid channel packing, chunking and checkpointing for every eligible method. These optimizations need later source/numeric qualification; this report executed none. Parameter matching, sparse-channel matching, equal updates and equal unique examples are different comparison axes. A competent comparison states their mismatches rather than claiming all budgets are equal.

## Primary passages and integrity

`INSPECTED_PASSAGES.json` retains exact extracted passages with zero-based block indices/HTML IDs or source line ranges, claim dispositions and source hashes. `CONTEXT_BINDINGS.json` hashes existing source/report/receipt files; `REUSE.json` binds byte-for-byte reuse. `PRIMARY_FETCH.json` and `SEARCH_RECEIPTS.json` retain this pass's exact endpoint/time/status/hash receipts. `RETRIEVAL_BINDINGS.json` also preserves the original MIMO/PE/author-code and root N-Ens retrieval provenance. Later default-branch author code is a hash-pinned snapshot, not an asserted paper-version commit. `PARAMETER_COUNTS.json` contains stdlib arithmetic, with no model import. `ARTIFACT_MANIFEST.json` and `SHA256SUMS` seal the packet; `verify_packet.py` checks hashes, reuse equality, version markers and passage bindings.
