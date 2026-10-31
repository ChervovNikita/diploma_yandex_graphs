# Propagation-cost scout: one ensemble sampling hypothesis

30 September 2026. **Source-only proposal; no implementation, data access, compute launch, or adoption authorization.** Work products are confined to this directory. Earlier allocation-audit work is complete and is outside this scout.

## Decision

Retain one narrow hypothesis: **use the current mean of the normalized member states as an exact graph control variate, and apply one shared sampled operator to their full-width deviations**. It is an empirical efficiency hypothesis with standard control-variate algebra. The inspected literature does not establish the exact current-member construction as a prior method, but this bounded search cannot certify novelty.

The plausible operating regime is an edge-heavy graph where member states have a large common component and their smaller disagreements still improve pooled predictions. The method keeps all member states and nonlinear updates at full width. It therefore addresses sparse propagation cost, with no promised activation-memory or dense-computation reduction. If member disagreement is large, correlated sampling errors survive nonlinear updates, or dense work dominates, it can fail.

## The precise intervention

Start from source TABM/SAGE: four full-width member states, shared SAGE/FFN weights and shared LayerNorm affine parameters, boundary BatchEnsemble factors, mean own-member cross-entropy, and deployment by averaging raw logits. At each layer, for the states actually entering neighbor aggregation, define

```
Z_m     = LayerNorm(H_m)
Zbar    = mean_m Z_m
Delta_m = Z_m - Zbar
G_m     = P Zbar + Q Delta_m
H_m     <- H_m + source_SAGE_FFN(Z_m, G_m)
```

`P` is the exact, fixed incoming-neighbor mean operator. No implicit self-loops are introduced. The only change is the expression supplying neighbor aggregation. Mean/centering remain differentiable; detaching the mean would be a different algorithm. Do not center before LayerNorm and assume centering survives normalization.

Every member receives the **same realized `Q`** in a layer. In exact arithmetic, `sum_m Delta_m = 0`, so `mean_m G_m = P Zbar` for every mask. With independent current-layer sampling and `E[Q | current states] = P`, each member's linear aggregation is conditionally unbiased relative to exact aggregation of those *current approximate states*. These are local linear identities. They do not make a multi-layer nonlinear network an unbiased replay of the exact-member network.

Member mean aggregation is preserved **only before the nonlinear update**. For aggregation errors `e_m`, `sum_m e_m = 0` does not imply `sum_m J_m e_m = 0`: member-dependent Jacobians, arising even with shared weights, can destroy pooled first-order cancellation. LayerNorm, FFNs, dropout, losses, gradients, and AdamW updates have no exactness or unbiasedness guarantee here. The mathematical artifact gives explicit counterexamples and a limited linear MSE decomposition.

### Fixed sampling and normalization contract

Partition each receiver's incoming edge entries into four label-free groups `C_0,...,C_3`. Preserve each original `P` weight, so `sum_c C_c = P`. Within the canonical receiver/source ordering, assign incoming edge rank `r` to color `(r + offset(receiver)) mod 4`; the fixed offset is a SHA256-derived value modulo four from the edge-only canonical `P` fingerprint, receiver ID, and the declared salt `20260930`. Each row's group sizes differ by at most one. This structured partition can produce correlated errors across nodes; no independent-edge variance formula is asserted.

In each training layer/forward choose a uniform color `c`, independent of current states and of the dropout RNG, and use `Q = 4 C_c` for all members. Four fixed CSR groups avoid constructing new edge masks every epoch. **Do not divide by sampled degree**: use the original full-graph denominator and inclusion-probability correction. A receiver with degree below four can have empty sampled groups. Isolates keep zero neighbor aggregation and the original root branch.

Evaluation is deterministic: use color 0 in layer 0 and color 2 in layer 1, fixed before validation or test outcomes, with dropout disabled. There is exactly one evaluation view. Choosing the best color, using all four colors, averaging repeated evaluations, or refining historical caches would change the recipe and must charge every propagation. Fixed-mask evaluation is biased relative to exact member propagation in general.

The simple identity covers fixed linear mean/SAGE and full-degree-normalized GCN operators. The pilot is SAGE only. In attention, member-dependent edge weights and a sampled softmax denominator change the operator; correcting inclusion probabilities alone does not recover this construction. Degree-sensitive attention, nonlinear edge messages, and GINE are outside this proposal.

## Cost and exact comparator

The exact execution baseline is `PackedSharedSAGEEnsemble` in `../efficient_paths/centered_paths.py`, lines 385–429 in the inspected source. It retains all member states and uses one sparse multiplication on `normalized.flatten(1)` per layer. Its `copy_from_source_tabm` transfer is supported by previously recorded source-equivalence checks. This scout does not rerun or adopt old numerical results. The future comparison must use the fastest valid source-equivalent packed or serial implementation, chosen by score-blind engineering measurements; deliberately slow member loops are insufficient.

Let `E` be the actual directed edge-entry workload of the frozen operator, `N` the complete node count, `D=64`, and `M=4`.

| Arm | Expected edge-channel work per layer | Full-width member nonlinear work |
|---|---:|---:|
| Exact member TABM/SAGE | `4 E D` | four members |
| Candidate | `E D + 4 (E/4) D = 2 E D` | four members |
| Naive common half-edge sampler | `4 (E/2) D = 2 E D` | four members |
| Ordinary BASE/SAGE | `E D` | one model |

The candidate replaces one width-`4D` sparse call with a full-graph width-`D` call and a quarter-edge width-`4D` call. Its expected sparse arithmetic is half the exact baseline; individual group sizes and actual work must be logged. **This is not a twofold latency promise.** Charge reductions, subtraction, broadcast/add, packing or gather/scatter, any dense materialization of `MND` tensors, autograd and transpose sparse calls, all normalizations, dense FFNs, optimizer steps, and raw-logit pooling. An implementation that materializes an `E x M x D` edge tensor is inadmissible under the proposed memory cap.

Charge full CSR normalization, all four color CSR groups, naive half-edge CSR groups, row pointers, indices, weights, host/device copies, preparation wall time, and cold start. Keep selected evaluation graphs resident only if their storage is included. If all four quarter groups and four adjacent half-color unions are retained, their edge-index/weight storage sums to approximately `E + 2E` in addition to full `P`, before row-pointer overhead. A faster lazy union strategy is allowed only if its runtime and peak storage are measured. Full member activations remain `O(MND)`; exact serial inference/backward may use less memory.

## Two complete new graph nominations

| Nomination | Primary provider metadata | Fixed graph-access policy and limit |
|---|---|---|
| **Reddit2** | PyG: 232,965 nodes, 23,213,838 edge entries, 602 features, 41 classes. | Complete GraphSAINT sparsified Reddit provider graph; official masks; transductive full-graph features/edges for every arm, train labels only for optimization. It is related to Reddit, not an independent corpus from another Reddit variant. Do not compare this transductive recipe to published inductive scores. |
| **ogbn-products** | OGB: 2,449,029 nodes, metadata 61,859,140 edges, 100 PCA features, 47 classes; popularity split top 8% train, next 2% validation, remaining 90% test. | Complete official provider graph and official splits; transductive features/edges shared by all arms. Freeze and report the actual directed/coalesced workload; OGB's metadata edge count must not be blindly multiplied or used as the CSR count. |

A bounded read of phase research notes and a text search of accessible source/protocol/README/ledger documents found no existing experiment record for these two names. That is **not a complete registry audit**. Both are nominations pending the coordinator's study-history check and outcome protection. No datasets, labels, checkpoints, or result tensors were opened. If either is already development-exposed, stop and register a new task in a new protocol; do not silently replace a failed graph or turn a subset into a new task.

Preserve provider edge multiplicity and existing self edges without inserting/removing edges. Canonical CSR coalescing must sum original normalized weights, retaining the same mean operator. Do not retain only the largest component, prune low-degree nodes, sample a small induced graph, or count the two graphs' nodes as independent experimental units.

## One small paired study

`PILOT_SPEC.json` is the machine-readable proposal. It is not executable admission material.

Four arms only: candidate; exact-member TABM/SAGE; ordinary single-model BASE/SAGE; naive common sampled operator directly on member states. The naive control chooses the same uniform training color `c` as the candidate but uses `Q_naive = 2(C_c + C_(c+1 mod 4))`, giving half-edge propagation and equal expected edge-channel arithmetic. Different sample fractions are explicit: a same-quarter-mask naive arm would be cheaper than the candidate and is not the proposed cost comparator. The cost-matched control does not isolate a same-mask theorem; the limited theorem concerns equal masks only.

Use one recipe on both graphs: four members where applicable, width 64, two source LayerNorm/SAGE/FFN residual blocks, FFN multiplier 1, dropout 0.2, source boundary factor initialization, raw-logit pooling; AdamW learning rate 0.001, weight decay 0.0001, default betas/epsilon; 100 full-graph epochs, validation after each epoch, lowest complete pooled validation CE checkpoint, earliest epoch on a tie. No early stopping, scheduler, temperature fitting, width sweep, sample-fraction sweep, or large HPO grid. Three paired optimizer seeds `0,1,2`: **2 graphs x 3 seeds x 4 arms = 24 fits**. Exact/candidate/naive start from identical parameter tensors and use matched member dropout streams; sampler randomness has its own stream. The single model uses the same source block structure without boundary ensemble factors.

Proposed future cap: one A100 80GB, **8 GPU-hours total**, peak reserved memory at most **60 GiB**, at most 16 CPU-hours of graph preparation, and 200 GiB of additional disk storage. These are caps, not measured feasibility or ETAs. A later authorized engineering gate, using complete graphs and all four arms without validation/test scores, must project the full family including 100 validations per fit, final evaluation and profiling to at most 6.5 GPU-hours; 1.5 hours remain for preparation, startup and contingency. Its profiling consumes the same 8-hour cap. If full products execution or the complete family cannot fit, the study is incomplete. No minibatch/smaller-graph fallback or selective successful-cell reporting is authorized by this proposal.

## Decisive gates before a method claim

All three seeds and all four arms must finish on **both** complete graphs. Gate definitions are prospective choices, not learned from current outcomes.

1. **Full quality noninferiority.** At the chosen checkpoint, evaluate every official validation/test target using the arm's declared deployment operator. Relative to exact TABM/SAGE, candidate pooled CE must increase by at most 0.01, accuracy decrease by at most 0.5 percentage points, and unnormalized multiclass Brier score increase by at most 0.01, for every paired seed and both graphs. The corresponding one-sided 95% paired-seed t bounds must also satisfy those margins on each graph/split. Three seeds only screen optimizer variability; they do not establish population noninferiority. Do not bootstrap nodes as independent graphs or pool the two datasets into one significance test. Fixed 15-bin ECE is diagnostic, without post-hoc calibration.
2. **Useful diversity.** Exact pooled test CE must improve by at least 0.005 on the three-seed mean over both the separately trained single model and its own validation-chosen member. Candidate must retain at least half of each exact CE gain and have positive mean pooled CE gain over its own validation-chosen member. One-sided 95% paired-seed bounds for each required positive gain must exceed zero. This gate is applied separately to both graphs. Report member CE/accuracy, prediction disagreement, pairwise probability distances, normalized contrast energy, pooled-versus-mean-member CE, and ensemble gains. Disagreement or a nonzero contrast norm alone is insufficient.
3. **Measured execution benefit.** On the intended GPU and each trained paired seed, complete-graph warm median inference wall latency, including raw-logit pooling, must be at most 75% of the fastest source-equivalent exact execution; p90 must be at most 90%. Use ten warmups and fifty synchronized measured forwards, with paired shuffled order and raw timing samples. Median graph-propagation time must fall at least 30%; report all component times without summing asynchronous overlaps as if independent. Also require candidate's aggregate complete fit/validation wall time over three seeds, including its graph preparation once, to be no greater than exact on each graph. Report cold first-prediction time, H2D, total GPU-hours, peak allocated/reserved memory and preparation amortization. No broad cold-deployment or memory advantage follows from a warm-latency pass.
4. **Mechanism control.** The candidate must beat the cost-matched naive sampler's test pooled CE by at least 0.005 on each graph's paired-seed mean, with a positive one-sided 95% paired-seed bound; accuracy may not be lower by more than 0.5 points. Report actual latency/memory for this control too: equal expected sparse arithmetic does not mean equal latency. If sampling alone explains quality/cost, drop the member-mean mechanism claim.

At three seeds, the one-sided t critical value is `t_(0.95,2)=2.91998558`; these bounds describe seed variation on the two fixed graphs only. A failed or ambiguous gate rejects promotion in this pilot. Passing supports a scoped feasibility result and independent confirmation planning, **not an originality certification**. Exact baseline lacking a useful ensemble gain also fails the rationale, even if the candidate matches it.

## Primary literature and novelty limits

Exact inspected passages and URLs are retained in `INSPECTED_PASSAGES.json`; retrieval manifests preserve successful responses and failures.

| Primary work | Already established; limit of the retained hypothesis |
|---|---|
| **Chen, Zhu, Song, ICML 2018**, [Stochastic Training of Graph Convolutional Networks with Variance Reduction](https://proceedings.mlr.press/v80/chen18p.html), [full version](https://arxiv.org/html/1710.10568v3), §3.1 Eqs. 5–6 and §5.1 | Uses `P H_history + P_hat(H-H_history)`; explicitly distinguishes unbiased aggregation from nonlinear/logit/gradient bias. Its dropout section also separates a mean baseline from stochastic fluctuations. The retained construction substitutes a *current across-member mean* and couples the residual mask across members; control variates and mean/residual separation are not new. Member disagreements need not vanish under fixed weights, so Chen's history-refresh exact-testing and variance-elimination guarantees do not transfer. |
| **Cong et al., KDD 2020**, [Minimal Variance Sampling with Provable Guarantees for Fast Training of Graph Neural Networks](https://arxiv.org/html/2006.13866v2), [publisher DOI](https://doi.org/10.1145/3394486.3403192), §4.1–4.2 | Checkpoint/inner loops, historical embedding correction, and gradient-norm adaptive sampling separate forward approximation from backward gradient variance. Publication metadata is independently verified through Crossref. It does not establish a current-member-mean identity or its nonlinear quality. |
| **Cong, Ramezani, Mahdavi**, [On the Importance of Sampling in Training GCNs: Tighter Analysis and Variance Reduction](https://arxiv.org/html/2103.02696v2), 2021 arXiv v2, §§5–6, Algorithms 2–3, Appendix E | SGCN+ recursively corrects historical preactivations; SGCN++ additionally corrects layerwise gradients with periodic exact/large snapshots and staleness safeguards. Coupled sampled differences and multilevel/composite variance reduction are established prior. The retained method has no historical gradients, snapshots, or guaranteed diminishing error. Do not transfer the paper's convergence rate. A guessed PMLR endpoint failed and is not venue evidence. |
| **Fey et al., ICML 2021**, [GNNAutoScale](https://proceedings.mlr.press/v139/fey21a.html), [full text](https://arxiv.org/html/2106.05609v1), §2 | Historical embeddings prune recursive computation; staleness and Lipschitz errors are explicit. Its processing/storage argument differs from retaining four full-width members on the complete graph. |
| **Neighbor-Sampling Based Momentum Stochastic Methods for Training Graph Neural Networks**, [2025 preprint](https://arxiv.org/html/2508.00267v1), §2.2 and §4 | Analyzes historical control-variate methods with Adam-type/momentum updates under assumptions and explicitly warns of nonlinear biased gradients. Its convergence hypotheses are not established for persistent cross-member disagreement. |
| **GRAND**, [full text](https://arxiv.org/html/2005.11079v3); **DropEdge**, [full text](https://arxiv.org/html/1907.10903v3); **Scalable Consistency Training via Self-Ensemble Self-Distillation**, [full text](https://arxiv.org/html/2110.06290v1) | Stochastic graph views, multi-view consistency and neighborhood-sampling self-ensembles are prior. Randomly sampling graph views or calling them members cannot be the contribution. Charge all views. |
| **Wang et al., SSGCN**, [primary publisher preview](https://doi.org/10.1007/s13042-023-02013-2), online 2023, vol. 15 (2024) | The inspected abstract describes sequential transfer of weights across sampled minibatch graphs and integrated learner predictions. HTTP 200 returned a subscription preview, not the full method. Precise propagation reuse remains an access gap. |

The bounded multilevel query found mainly coarsening, spectral/PDE and trace-estimation methods; its absent direct hit is not novelty evidence. SGCN++ provides a concrete inspected link to multilevel composite optimization. No claim of a new multilevel Monte Carlo principle is made.

Existing phase work already covers centered narrow paths, shared cached hop/filter banks, layer selection, and low-rank/all-layer factors. The retained proposal differs by keeping full-width nonlinear members and sampling only their graph residual aggregation. Standard control-variate substitution and channel packing remain engineering ingredients. GEENI's full procedure and *Graph Ensemble Neural Network* (Information Fusion 2024) remain unresolved nearest-prior gaps from the earlier primary-source audit. A credible broader novelty claim would require those methods' full inspection and stronger comparators/confirmation beyond this deliberately small pilot.

## Artifacts and evidence boundary

- `PILOT_SPEC.json`: fixed proposal and gates; explicitly unregistered/unimplemented/unlaunched.
- `MATHEMATICAL_LIMITS.md` and `ARITHMETIC_CHECK.json`: local linear proof, same-mask MSE decomposition, complementary-mask failure, and nonlinear Jacobian counterexample. Hand-constructed arithmetic only.
- `INSPECTED_PASSAGES.json`, `CITATIONS.bib`, `RETRIEVAL_FINAL_MANIFEST.json`: primary evidence and access limits.
- `CONTEXT_BINDINGS.json`, `TASK_NOVELTY_SEARCH.json`, `ARTIFACT_MANIFEST.json`: source inputs, bounded study-history search, and output hashes.
- `RESULTS_LEDGER.json`: explicitly empty experiment ledger.

No predictive or latency result is supplied. No implementation or launch was authorized or performed.
