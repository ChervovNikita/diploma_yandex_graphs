# Two concrete modern backbone integrations

2026-10-02. Source research and unexecuted adapter draft only. No model imports, data/tensor loads, fits, tests, compilation, GPU work or SSH occurred in this lane. The sealed conformal packet, its five score methods, and the closed coordinate/factor screen are unchanged.

## Recommendation

Use **PolyFormer-Mono for filtered Squirrel** and **Polynormer-r for Amazon Photo** in a separate prospective strong-backbone amendment. These are reproducible, expressive 2024 primary models with exact released schedules for the respective datasets. The 2025 Tokenphormer paper and 2026 heterophily reassessment were checked before narrowing this choice. This is a practical source-supported pair, not a claim of current universal SOTA.

The proposed experiment extends the existing GNNM boundary projector mechanism to strong bases: one shared backbone, four private input/output rank-one projectors, and four complete nonlinear trajectories. Attribute the boundary algebra and input replication to BatchEnsemble. Precedent is a foundation for this extension. The failed coordinate/factor branch remains NO_GO because added factors failed its empirical utility requirement; this recommendation does not reopen it or introduce hidden-layer permutations/interior factors.

| Dataset | Pinned author release | Primary evidence | Published result/protocol |
|---|---|---|---|
| Filtered Squirrel | [air029/PolyFormer](https://github.com/air029/PolyFormer/tree/d390f39e88d0eaac80318fdc7704bd3bf3cf8b13), commit `d390f39e88d0eaac80318fdc7704bd3bf3cf8b13` | [2407.14459v1](https://arxiv.org/html/2407.14459v1), KDD 2024 DOI `10.1145/3637528.3671849`; §4.2.1, Table 5, Appendix D.2.1 | Mono 42.56 ± 0.96, mean with 95% CI over 10 official splits; N=2,223/F=2,089/C=5; official 50/25/25 train/validation/test. Released loader and script explicitly name `squirrel_filtered`. |
| Amazon Photo | [cornell-zhang/Polynormer](https://github.com/cornell-zhang/Polynormer/tree/fc8c276c9c5dfbd616d83f65338a3392188a5e08), commit `fc8c276c9c5dfbd616d83f65338a3392188a5e08` | [2403.01232v2](https://arxiv.org/html/2403.01232v2), ICLR 2024; Table 1, Appendices E/H, released `run.sh/main.py` | Polynormer-r 96.46 ± 0.26 (std, 10 runs), random 60/20/20. README 96.67 is a first run, not the paper mean. N=7,650/F=745/C=8. |

Both current teacher role schemes use approximately 20% training and reserve extra source-label sets before the final pool. Therefore these are **new role protocols**, not reproductions of the reported paper scores. Squirrel keeps the verified filtered provider and derived official-split roles; Photo keeps its verified provider and fixed row order. Do not use these papers' accuracies as expected scores under the new masks.

## Full schedules and preprocessing

**PolyFormer-Mono / filtered Squirrel.** The released command fixes hidden=256, order K=12 (13 tokens), 2 blocks, 4 heads, FFN dimension=128, q=1.4, order-MLP multiplier=1.0, ordinary dropout=0.3 and attention/FFN dropout `dprate`=0.8. Adam uses lr=1e-4/wd=0 for ordinary parameters and lr=1e-3/wd=1e-7 for every `attnmodule` parameter. Maximum 2,000 epochs; published validation-accuracy patience=250. Ten official mask rows, with the released fixed seed list, produced paper results.

The filtered NPZ loader takes `node_features`, `node_labels`, `edges`, `train_masks`, `val_masks`, `test_masks`; it symmetrizes edges and performs no feature normalization in this branch. For the monomial basis, `utils.mono_base` applies PyG `gcn_norm` with default self-loop handling, builds a SciPy sparse matrix and FP32 sparse Torch matrix, and recursively caches X,Ahat X,...,Ahat^12 X. Preserve that orientation, normalization, duplicate/self-loop semantics and dtype in the amendment. Its filename-only pickle cache is unsuitable for provenance: use a key binding graph/features/preprocessing/runtime/order hashes and never load a cache for another graph merely because its dataset name matches.

**Polynormer-r / Photo.** Use the released schedule: hidden_channels=64 per head, 8 heads, effective width=512; 7 local layers, 2 global layers; Adam lr=1e-3/wd=5e-5; input dropout=0.2; local/global dropout=0.7; beta=-1 (learned sigmoid beta); q/k shared; pre-LN disabled. Run **200 local warmup + 1,000 global-stage epochs = 1,200 optimizer updates**, full batch, no early stopping. At the transition the author driver restores the selected local model and Adam state, then enables global attention.

Important source discrepancy: paper Table 5 calls Photo warmup=200/total=1,000; released `run.sh` supplies local=200/global=1,000 and `main.py` sums them. The recommendation explicitly uses the released 1,200-update schedule and records the discrepancy. Do not silently call it the paper's 1,000-total schedule.

Photo's author loader adds `NormalizeFeatures()` to the PyG Amazon provider. Normalize the already verified feature rows once using the same transform semantics, prospectively, for every arm of this new lane. The driver symmetrizes edges, removes self-loops and adds exactly the chosen native self-loop representation for all N nodes. The local GAT layers set `add_self_loops=False`, because the driver has already added them. This is additional backbone preprocessing beyond the acquisition provider; do not misdescribe the sealed packet's earlier provider as already row-normalized.

For new teachers, selection uses **predictor-validation labels only**. The prospective amendment should retain the full source schedules above but explicitly change checkpoint selection to lowest pooled validation NLL (earliest epoch wins exact ties), and PolyFormer patience to 250 non-improving NLL epochs. Restore the best local NLL checkpoint at Photo epoch 200, reset the global-stage selector, and choose the final teacher among global-stage checkpoints. This avoids ambiguous cross-stage checkpoint reuse; `_global` is not stored in the native state dict, so save the stage explicitly. These are declared teacher-selection adaptations, not exact author training reproduction. Native scripts read test labels every epoch; use neither native executable as the scientific driver.

## Executable interface draft

`backbone_boundary_adapter.py` is a small, **unexecuted** PyTorch source draft. It accepts a freshly constructed pinned author model, moves its input/class projection matrices into shared boundary modules, and preserves the native body ordering. Interface:

* PolyFormer: `model(tokens[N,13,F]) -> logits[4,N,C]`, with label-free tokens prepared once.
* Polynormer: `model(x[N,F], edge_index[2,E]) -> logits[4,N,C]`, plus explicit `set_global_stage(bool)`.
* No labels, masks, acquisition, optimizer or scientific orchestration live inside the adapter.

Input R is independent Rademacher by member, S starts at one, head R/S start at one. Initial boundary biases copy the author projection biases to each private B row. Copying those biases is a declared integration choice differing from the old frozen GNNM zero-B initialization; it preserves native single-member/unit-factor parity algebraically. All interior weights, normalization affine parameters and native beta parameters remain common. Each fit constructs fresh author modules and fresh projectors; do not call the native reset method after wrapping. Fit loss is mean member cross-entropy, while predictive pooling is mean **probabilities**, not mean logits. Save all four member logits/probabilities for member-aware uncertainty methods.

The draft deliberately loops complete members because native PolyFormer attention and Polynormer GAT/global einsums assume their original axes. Applying them directly to `[M,N,...]` would mix or misinterpret axes. Vectorization is a later qualified execution improvement, offered fairly to independent controls. This draft has not been imported, syntax-compiled, tested or numerically qualified. Before use, a separate permitted engineering step must check unit-factor native parity, non-unit member separation, gradient accumulation, stage/checkpoint replay and final probability shapes on representative admitted inputs. No fit should start on source inspection alone.

## Bounded fair validation tuning

Freeze one four-configuration list per dataset before labels are exposed:

1. Released architecture and source learning rates/dropouts.
2. Same architecture/dropouts, all learning rates ×0.5.
3. Same architecture/dropouts, all learning rates ×2.
4. Source learning rates; ordinary/local/global dropout reduced by 0.2, floored at zero; input dropout and PolyFormer `dprate` unchanged.

Use the same configurations, full schedules, paired role maps and seeds **17/29/43** for (a) author single model, (b) four-member GNNM adaptation, and (c) four independently trained author models at the same native width. Each family chooses one configuration per graph by arithmetic mean validation pooled NLL across all three seeds; earliest listed configuration wins an exact tie. An independent family gets four full member fits per configuration/seed, and those costs are charged. Select and save a joint independent-family checkpoint using the same pooled NLL criterion at synchronized epoch boundaries; use family-level patience for PolyFormer. Do not give it a larger hidden hyperparameter search or choose favorable member seeds.

This is 4 configurations × 3 seeds × 3 families × 2 graphs = **72 family cells**, including all failed configurations. A family cell for the independent comparator contains four complete fits. No new configurations, longer schedule, seed replacement, graph replacement or train-label additions can be added in response to final-pool outcomes. If the budget cannot fund this, reduce the number of configurations prospectively and identically for every family, retaining the source configuration; do not abbreviate the published training schedules. Calibration A/B/D and the final pool are excluded from all teacher configuration/checkpoint selection. Freeze the selected teachers and score methods before final-pool labels are made available.

If a fit is poor, the above bounded validation opportunities address it. A continued inadequacy is an informative result for the declared roles; it cannot be rescued by expanding only the candidate's search. The 2026 reassessment `2409.05755v3` provides primary support for adequate tuning and explicitly identifies filtered Squirrel as ambiguous heterophily. Its enormous grid is evidence about tuning sensitivity, not a required or affordable tuning budget here.

## Parameter, cache and compute fairness

For four members, replacing a native affine F→H stem and H→C head removes H+C common bias scalars and adds `4*(F+3H+2C)` private scalars. Polynormer has **two** class heads, so its addition is `4*(F+4H+4C)` and its removed biases H+2C. All common norm, GAT attention, beta, FFN and token-order parameters count.

The authors specify PyTorch 2.0/PyG 2.3. The current project runtime was reported as Torch 2.1.2/PyG 2.7.0. Neither this adapter nor the source arithmetic qualifies cross-version operator equivalence. Bind actual Torch/PyG/scatter/sparse sources and runtime versions in the new amendment. Native PolyFormer model files also import unused utilities that transitively import dataset loaders and DGL, and seed global RNGs at import time; a clean class extraction with attributed source diffs is preferable to executing its original training entry point. The new source owner must account for that adaptation and controlled initialization.

Static source arithmetic for the selected releases (FP32 parameters; no model was constructed):

| Full training model | Single author | Four-member GNNM | Four independent models |
|---|---:|---:|---:|
| PolyFormer-Mono H256 | 4,419,437 scalars / 17,677,748 B | 4,430,644 / 17,722,576 B | 17,677,748 / 70,710,992 B |
| Polynormer-r H512 | 7,762,960 / 31,051,840 B | 7,773,732 / 31,094,928 B | 31,051,840 / 124,207,360 B |

These are analytical estimates requiring an admitted runtime inventory, not measured tensor inventories. They assume homogeneous PyG GAT's single H×H matrix plus two H attention vectors and no bias, as invoked by pinned source. Source `PolyAttn.bias` is an unregistered fixed tensor: add its 13 FP32 entries per block (104 B for two blocks), device copies and temporary sqrt tensors to actual runtime/cache accounting rather than pretending state-dict bytes describe everything.

The four independent models are a strong same-width quality comparator. They are not parameter/compute matched to GNNM. Sharing matrices establishes a storage saving, not equal functional capacity or fourfold arithmetic saving. A parameter-matched quartet may be added only under a separate prospective question; it is not needed to describe the observed cost/quality tradeoff honestly. Do not make a matched-capacity superiority claim from this three-family screen.

PolyFormer's common label-free token cache is substantial: FP32 N×13×F is **241,480,044 B** on filtered Squirrel. Share it once for every family where the actual implementation does so, including independent models. Native `torch.stack(list_mat)` can transiently duplicate it; a pre-stacked cache is an equivalent prospective execution improvement after qualification, charged to all arms. Four private hidden token states require at least **118,370,304 B** for one M×N×13×H array, before attention, FFN and autograd intermediates. They cannot be replaced by a single shared hidden trajectory.

Polynormer retains four private local GAT trajectories and four global-attention numerator/denominator paths. Shared graph topology does not share member-dependent attention scores. One M×N×H array alone is **62,668,800 B**; lifted edge tensors, attention buffers and autograd can dominate. Native linear global attention avoids N² node attention but still performs each member's projections, key/value reductions and query products. No speed, memory feasibility or uncertainty-diversity benefit is established here.

Record trainable weights, fixed buffers, graph/features, token caches, logits/probabilities, optimizer moments/gradients, selected/full checkpoints and peak allocations separately. Charge every tuning fit, failed cell, full update, validation forward, checkpoint, restore, preprocessing/cache build, probability pooling and score calibration. Report cold process+provider/cache+checkpoint setup and warm all-member full-graph serving separately. Pruning Photo's unused local training head at deployment is allowed only after parity; record both full training and actual deployed storage. Apply equivalent cache reuse, chunking and deployment optimizations fairly to independent models. A single trunk with four heads is a different cheaper model, not the independent full-trajectory comparator.

## Predictive models and uncertainty controls

The strong bases answer whether original GNNM remains useful with competent predictors. HeAD and CF remain attributed uncertainty-score controls from the sealed review; their presence does not qualify the predictive backbone as SOTA. For each selected family, preserve the five scores and their native source-label selection/calibration separation, charge all member-aware costs, and assess point quality, coverage, set size and cost on the same retained final population. Coverage claims retain the sealed graph-dependence qualifications. New teacher models require a new prospective binding, not edits to the sealed packet.

No additional heterogeneous, link or graph task is recommended in this first amendment. Neither selected author release supplies a matched heterogeneous task recipe, and two complete contrasting node tasks plus strong independent models are already necessary. A later task-type extension should have its own exact source/split and budget.

## Recent-paper disposition and reuse

* **Tokenphormer, AAAI 2025**, `2412.15302v2`: exact author repo verified by primary paper. It combines SGPM pretraining, mixed-walk tokens and hop tokens; paper Photo=96.14±0.14 on 60/20/20. Its primary Appendix training settings and released training defaults were read. The released script is Cora-specific; Photo SGPM settings are commented. Full graph pretraining/token custody and all associated costs would need another integration. Keep as a documented later predictive alternative, not an immediate uncertainty control or proof of better current-role accuracy.
* **GRAIN, AAAI 2025**, DOI `10.1609/aaai.v39i12.33461`: primary PDF retrieval failed twice. Pinned README/source/tree were inspected. The released default driver trains a TD3/meta-policy on Cora; inspected loader accepts Planetoid datasets, and final loop reports peak test accuracy. This does not supply a verified filtered-Squirrel full recipe. Do not assert its paper mechanism/results from this limited source read.
* **AutoSGNN, AAAI 2025**, DOI `10.1609/aaai.v39i18.34146`: metadata, README and repository tree only; no primary method read. Architecture-search directories are visible, but no performance/method judgment is inferred. It is discovery evidence, not a verified strong-base recommendation.

`PAPER_CONCLUSIONS.json` is the reusable per-paper record, including exact scope, versions, decisions and evidence. Consult it and the existing BatchEnsemble `PRIMARY_CONCLUSIONS.json` before rereading. `SOURCE_INVENTORY.json` binds saved source bytes. Only changed primary versions or named unresolved passages justify a further read.
