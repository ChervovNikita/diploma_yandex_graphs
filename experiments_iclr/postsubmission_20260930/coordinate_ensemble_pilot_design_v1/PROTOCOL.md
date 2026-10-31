# Fixed coordinate ensemble: proposed development screen

1 October 2026. **Design only; root adoption is required before data acquisition, model construction, profiling or fitting.** This is a bounded test of a composition, with no novelty claim. Original dataset scores remain frozen. This document does not authorize execution.

## 1. Dataset and label contract

Use **LastFMAsia** (social graph: published 7,624 nodes, 55,612 edge entries, 128 features, 18 classes) and **Amazon Photo** (co-purchase graph: 7,650 nodes, 238,162 edge entries, 745 features, 8 classes), through the official PyG 2.7.0 providers. Published counts are planning inputs, not verified tensors. These datasets are disjoint from the original five and recorded phase studies, as checked by root; this is not an exhaustive absence claim about every old file.

The original source registry includes Roman Empire, Amazon Ratings, Minesweeper, Tolokers and Questions. None is a new development or confirmation dataset. In particular, Tolokers has possible prior execution artifacts reported by root; their contents were not inspected here. **Coauthor CS or Physics** is a later independent confirmation option, requiring a separate source/scope check and adoption. Questions is excluded.

Neither chosen PyG provider constructs official masks. Use **prospective custom stratified partitions**, never label them official splits. A custodian first assigns a common 20% per-class test holdout with seed **4100** and seals its labels. Calibration seed **4101** and development seeds **4102, 4103, 4104** split only the remaining 80% into 75% training and 25% validation: approximately 60/20/20 overall. The exact common rounding rule is **floor_test_and_validation_per_class_train_remainder**: test=`floor(0.2*n_class)`, validation=`floor(0.25*n_remaining)`, training is the remainder. Require at least one node in each partition per class, and reject infeasible class counts before fitting. The same global test nodes remain excluded from every run's train/validation labels.

Freeze raw/provider/processed hashes, node order, masks and feature/edge definitions before training. The common runner definitions are **features='none'** (original features, no normalization) and **edges='as_pyg'** (as acquired by the pinned providers, no additional edge/coalescing/loop transform). LastFM features may be signed, so naive row-sum normalization is unsuitable. Before fit admission, metadata must verify bidirectionality and report duplicate/self-loop counts; an unexpected topology requires source/data reconciliation before outcomes. Do not automatically double edges or alter loops. Full graph features/edges are transductive inputs; only training labels enter gradients. No label propagation, target-derived features or outcome-selected split seeds. Acquisition/stratification is a separate custodial operation, not a training process reading test labels.

## 2. Models and controlled operation

The pinned `TABMModel` has private BE factors/biases only in its input/output projectors, with shared residual propagation, FF maps and normalization. Preserve it as **G0**, using the same prospective SAGE recipe. The main family has K=4, width h=256, FF multiplier m=1, pre-LayerNorm residual SAGE and GELU. Preserve original SAGE bias flags: neighbor map biased, root map bias-free; both FF maps biased. LayerNorm affine parameters stay shared in the shared models.

| Arm | Operation |
|---|---|
| S | Competent ordinary single SAGE, same depth, width matched to PF's complete deployed bytes |
| G0 | Original boundary-projector GNNM (`TABMModel`) |
| B0 | G0 plus private interior biases; interior factors fixed at 1, permutations identity |
| F | All-map factor-only control: original boundary BE plus learnable interior input/output scales and the same private biases |
| PF | F plus fixed private hidden-coordinate permutations, independently assigned per eligible map/member |
| P0 | **Interior permutation-only** control: identical PF permutations/biases, interior scales fixed at 1; original boundary BE remains learnable |
| U | K=4 compact, fully untied SAGE members, same depth, complete deployed bytes matched to PF |
| H | Shared SAGE trunk with K private competent `w→w→C` GELU MLP heads, complete deployed bytes matched to PF |
| T | Strong heterophily-capable reference: author-style residual GT-sep, LayerNorm, width512, depth5, 8 heads, multiplier1, dropout0.2, AdamW lr3e-5 |

PF's maps use `D_s P_out W P_in D_r`, with fixed hidden-axis permutations. Raw feature/class axes remain in their declared order. Source preparation must specify eligible maps and composition orientation; the separate coordinate audit handles gauge equivalence. Coherently transported relabelings do not count as new functions. P0 is not a completely factor-free ensemble.

Use model seeds **17, 29, 43** for every development split, and independent recorded factor/permutation/dropout streams derived from those seeds. Same-width G0/B0/F/PF/P0 clone one original base state: identical W, boundary BE state and normalization; private interior biases copy the original bias. Interior factors start at 1. Thus PF/P0 and F/B0 respectively match at initialization; PF/F share factor and base seeds but differ in their declared permutations. Couple member dropout masks for paired same-shape comparisons. S/U/H have different widths, so use paired seed streams without equal-function or identical-tensor claims. Untied members receive independent ordinary initializations, not identical copied members. The single remains an ordinary model without private boundary R/S.

### Deployed-byte matching

Count unique trainable parameters, deployed dense values and fixed permutation buffers separately; do not count frozen identity factors as trainable. For this SAGE topology with m=1, shared affine norms and K members, analytic counts are:

`P_PF(h)=5Lh²+[d+C+2(L+1)+3K+12KL]h+Kd+2KC`.

`P_U(w)=K[5Lw²+(d+C+5L+3)w+C]`.

`P_H(w)=(5L+K)w²+[d+5L+3+K(C+1)]w+KC`.

Fix deployed parameter dtype FP32 and stored indices int64; no index compression, quantization or seed-based reconstruction is credited. PF has `I=9KLh=27,648` stored hidden-map indices at L=3,h=256,K=4, adding **221,184 bytes**. Its total model bytes are `B_PF=4P_PF+8I`. S/U/H have no such index buffers, so match `4P_arm(w)` to that complete target, minimizing absolute mismatch with ≤1% tolerance. Width choice is deterministic arithmetic from frozen F/C and source storage definitions, not HPO. Future admitted `storage_report` must agree before fits.

| Dataset | PF parameters / complete bytes | S width / bytes | U width / bytes | H width / bytes |
|---|---:|---:|---:|---:|
| LastFMAsia | 1,063,056 / 4,473,408 | 268 / 4,485,320 | 131 / 4,462,672 | 237 / 4,479,588 |
| Amazon Photo | 1,220,836 / 5,104,528 | 267 / 5,100,800 | 122 / 5,077,280 | 239 / 5,105,168 |

The equal-byte controls therefore have more trainable scalars than PF: report both quantities. F/G0 retain h256 for same-width attribution; F omits identity index arrays, so PF carries an explicit deployment-storage increment over F. Exact same width and byte budget are generally incompatible. Nine-arm utility screening prioritizes this stronger S over an additional same-width single. A full width256 untied ensemble is a later quality ceiling, not a required raw-quality victory for PF or part of this cap.

## 3. Competent training and checkpoint selection

The inspected author recipe performs **1,000 full-batch updates**, defaults to width512/depth5, dropout0.2, AdamW lr3e-5 and a constant scheduler without warmup. Its 1,000-step recipe is evidence for a normal training horizon; it is not proof that every new variant converges. The earlier 64-update screen supplies no convergence recipe.

Fix main depth **3**, one of the depths in the author's experiment commands. Before the core screen, calibrate the byte-matched S on seed4101/model17 for only **lr {3e-5,3e-4}**, separately for each dataset: four fits total. Select minimum validation NLL; ties lower lr. Freeze that dataset's common recipe for G0/B0/F/PF/P0/U/H/S. T retains the stated author-style recipe. No candidate-specific search or outcome-driven recipe change.

For every fit: full-batch AdamW, weight decay0, constant lr, dropout0.2; **minimum1,000, maximum2,000 updates**, evaluate every10. Stop only after update1,000 and 200 updates without a strict validation-NLL improvement; patience means updates, not evaluation calls. Keep the minimum **pooled validation NLL** checkpoint, ties earliest. Apply mean per-member CE to shared ensembles; each untied member receives ordinary independent CE optimization with its own optimizer, while its reported objective is the mean. No bootstrapping or class-weight search. Synchronize untied member epochs and select one pooled-ensemble checkpoint epoch. Persist training/validation curves and best epoch. Nonfinite training, capacity/memory failures, or a best checkpoint at the final evaluation with improving loss make the scientific result **inconclusive**, not evidence against the mechanism. No tiny truncated fit may satisfy the screen.

Core schedule: **9 arms × 2 datasets × 3 train/validation subdivisions × 3 model seeds =162 fits**, with every K-member arm counted as a complete ensemble fit.

## 4. Prediction, profiling and fixed decision

Primary deployed prediction is the uniform arithmetic pool `p(y|v)=K⁻¹Σ softmax(z_k(v))`; S/T use their single probabilities. Compute NLL with a stable log-probability reduction. Member identity stays fixed through the whole propagation. Use the same selected checkpoint for a secondary `softmax(mean logits)` sensitivity; do not switch pooling or checkpoint rules after outcomes. Report paired validation NLL and accuracy, macro-F1 and member disagreement; the latter cannot substitute for predictive benefit. No post-hoc member weights, pruning or calibration.

Profile actual complete fits and deployment: synchronized cold and warm full-graph inference median/p95, graph/feature/setup time, all-member forward/backward and optimizer time, gathers/permutation work, validation/checkpoint time, peak allocated/reserved GPU memory, deployed parameters and buffers. Exclude labels from prediction timing but include common graph transfer/setup in cold latency. U must include all four members. Reuse the same sparse graph representation; charge H's shared trunk advantage. Report both best-quality checkpoints and storage/quality/latency comparisons, not expanded nodes or FLOPs alone.

**Prospective go rule, based only on development validation:**

1. On each dataset, PF's mean paired NLL is at least **2% lower than F and G0**, and PF beats F in at least **6/9** cells.
2. Pooled across the two datasets with equal dataset weight, PF improves NLL by at least **1% relative to both P0 and B0**. This attributes useful benefit beyond fixed interior permutations and private biases.
3. PF improves NLL by at least **1% over the best equal-byte S/U/H control on one dataset**, with no more than1% NLL degradation on the other. Mean accuracy loses at most **0.5 percentage point** against F, G0 and that equal-byte control on either dataset.
4. PF/F deployed-parameter mismatch is zero apart from fixed permutation buffers; PF's complete training time and warm all-member inference latency are each ≤**1.25× F**. Report the frontier against S/U/H/T; do not require victory over a four-times-larger full ensemble.

For each dataset, relative gain is `1−mean_cell(NLL_PF)/mean_cell(NLL_control)` over its nine paired cells. Cross-dataset summaries give the two datasets equal weight, not weight proportional to node count. The strongest byte-matched comparator is the minimum mean-NLL member of the fixed S/U/H set on each dataset. These are effect-size screens, not node-independent significance tests. Failure of a properly completed screen is a no-go for this composition under this recipe; incomplete or incompetent training is inconclusive. Passing qualifies a separately adopted confirmation, not a novelty claim. All test labels stay sealed throughout development; any later one-time test evaluation requires a frozen analysis/checkpoint manifest and root's explicit adoption. No re-selection follows that evaluation.

## 5. Resource proposal and adoption boundary

Propose **at most six aggregate A100 GPU-hours** on the authorized single GPU, including qualification/profiling, calibration, validation, complete fits and deployment measurements. This is a planning cap, not an inherited user limit. Root's idle-GPU observation is not execution admission.

After root adopts qualification, obtain representative train/evaluation timings without test labels; forecast the entire nine-arm schedule at maximum2,000 updates, with 20% timing contingency and a 0.5 GPU-hour reserve. Approximate qualification is 100 training updates plus10 evaluations per dataset/arm. The training ceiling is 324,000 core updates +8,000 calibration updates +1,800 qualification updates. Sum each cell's measured setup + training + every10-update evaluation + checkpoint + deployed profiling cost; do not substitute one cheap arm's timing for all arms. A sensible admission requires `1.2 × forecast_GPU_hours + 0.5 ≤ 6` and sufficient actual memory. Roughly, amortizing all such work over 333,800 training updates gives a complete-step feasibility threshold near49 ms; this is not a measured runtime claim.

The published small graph sizes make a qualification worth considering, but no unconditional runtime forecast is established. If that forecast fails, do not launch the core schedule or shorten its competent training; record the proposed schedule as resource-infeasible and let root adopt a separately specified reduced design. No dataset/model work, SSH, source execution, manuscript edit or original score inspection was performed to write this protocol.

Sources: author repository commit `a431395582e929d88271309716bea4fe24ce6318` (README, train/dataset/util recipe and experiment commands); official PyG2.7.0 provider snapshot (LastFMAsia/Amazon). Exactly two public source families were inspected. arXiv metadata was discovery only. Source bytes, hashes and inherited-source bindings are retained in this packet.
