# Amazon Ratings baseline context and one fixed next comparison

Status: literature/source review completed; proposed comparison only. No fit, dataset, project result, checkpoint, server, install, or subagent was used. The ongoing NCNC queues were not inspected or duplicated. Prior packets and `literature_memory/index_v34` were preserved.

## 1. What this establishes

The current Amazon pilot fits a class-stratified, hash-ordered **80% of each official TRAIN mask**, leaving 20% as TRAIN control, on official splits 0/1/2 with block seeds 17/29/43. Published official 50/25/25 results use the entire official TRAIN mask. Their test means are context, not comparable pilot validation/control results, competence gates, or pass thresholds. The pilot's two PolyFormer recipes are source defaults and a Roman-empire transfer, not authored Amazon settings.

The strongest directly relevant combination of a primary five-graph table and an authored Amazon release command in this bounded review is **Polynormer-r**. This is a choice of a sourced comparison anchor, not a claim that it is the current global SOTA. The search is not an exhaustive leaderboard audit. Recent ATLAS v6 contributes a useful topology/channel comparison, but its public source and revised paper do not close the same label-use protocol.

## 2. Published context on the five Yandex graphs

Numbers below are percentages. Roman-empire and Amazon Ratings use accuracy; Minesweeper, Tolokers, and Questions use ROC AUC. A dash means the inspected primary table does not report that graph. Different rows are not paired experimental runs.

| Primary table/model | Roman | Amazon Ratings | Minesweeper | Tolokers | Questions | Uncertainty and protocol |
|---|---:|---:|---:|---:|---:|---|
| Platonov et al., GraphSAGE | 85.74 ± 0.67 | 53.63 ± 0.39 | 93.51 ± 0.57 | 82.43 ± 0.44 | 76.44 ± 0.62 | SD; one fit on each of ten official 50/25/25 masks |
| Platonov et al., GAT-sep | 88.75 ± 0.41 | 52.70 ± 0.62 | 93.91 ± 0.35 | 83.78 ± 0.43 | 76.79 ± 0.71 | Same benchmark; ego/neighbor separation, residual/FFN/LayerNorm baseline |
| Platonov et al., FSGNN | 79.92 ± 0.56 | 52.74 ± 0.83 | 90.08 ± 0.70 | 82.76 ± 0.61 | 78.86 ± 0.92 | Same benchmark; feature/hop selection is an established comparator |
| Polynormer | 92.13 ± 0.50 | 54.46 ± 0.40 | 96.96 ± 0.52 | 84.83 ± 0.72 | 77.95 ± 1.06 | Ten runs; official masks; native logger reports SD |
| Polynormer-r | 92.55 ± 0.37 | **54.81 ± 0.49** | 97.46 ± 0.36 | 85.91 ± 0.74 | 78.92 ± 0.89 | ReLU variant; ten runs; official masks |
| PolyFormer (Mono) | 78.89 ± 0.39 | — | 90.69 ± 0.38 | 84.00 ± 0.45 | 77.46 ± 0.65 | Paper reports mean and 95% CI over ten runs, not SD; binary ROC AUC verified in training source |
| ATLAS v6 | 66.22 ± 0.53 | 53.17 ± 0.81 | — | 82.19 ± 0.73 | 73.26 ± 0.83 | Reported SD; claims Platonov settings where applicable; public source caveat below |
| ATLAS-NF v6 | 78.10 ± 0.59 | 52.30 ± 0.64 | — | 82.88 ± 0.78 | 78.17 ± 1.12 | Adds one-hop neighbor attributes |
| ATLAS-LPF v6 | 70.22 ± 0.37 | 52.81 ± 0.37 | — | 82.30 ± 0.76 | 74.13 ± 0.92 | Adds masked training-label diffusion |
| ATLAS-LPF-NF v6 | 79.04 ± 0.72 | 51.91 ± 0.46 | — | 82.65 ± 0.92 | 77.35 ± 0.83 | Both local channels; no across-variant test-set selection proposed |
| Spexphormer | 87.54 ± 0.14 | 50.48 ± 0.34 | 90.71 ± 0.17 | 83.34 ± 0.31 | 73.25 ± 0.41 | ± type/run count not resolved by the scoped primary table/setup; release example repeats five seeds and loader retains mask 0 |
| ES-MLP | 65.44 ± 0.92 | 47.85 ± 1.23 | 50.87 ± 2.03 | — | — | Paper: ten runs/SD, Platonov setting for these three; release replaces masks with random per-class 60/20/20 |

Primary sources: [Yandex benchmark v2](https://arxiv.org/html/2302.11640v2), Table 4/Appendix A; [Polynormer v2](https://arxiv.org/html/2403.01232v2), Table 2/Appendices E,H; [PolyFormer v1](https://arxiv.org/html/2407.14459v1), Table 5/Appendix D.2.1; [ATLAS v6](https://arxiv.org/html/2512.14908v6), Table 2/Section 3/Appendix A.22; [Spexphormer v1](https://arxiv.org/html/2411.16278v1), Table 2/Sections 5.1–5.2/Appendix D.1; [ES-MLP v1](https://arxiv.org/html/2412.08310v1), Table 2/Section 4/Appendix A. Local line scopes and hashes are in `READ_SCOPES.json` and `SOURCE_BINDINGS.json`.

GNNFormer is not another Amazon row. The relevant paper is [Rethinking Graph Transformer Architecture Design for Node Classification, 2410.11189v1](https://arxiv.org/html/2410.11189v1), not the similarly named cytopathology model. It reports all 12 graphs on **48/32/20** splits. Of the five Yandex graphs, it includes only Roman-empire and Tolokers. It omits Amazon Ratings, Minesweeper, and Questions. Its Tolokers endpoint is accuracy, not the Yandex ROC AUC endpoint. No authored Amazon recipe or official author implementation was established. A retrieved GitHub repository explicitly calls itself an independent Actor reproduction; it is not promoted to author-source evidence.

## 3. Exact protocol and source limits

### Native benchmark baselines remain useful

The original benchmark's strong GraphSAGE is a residual network with a two-layer GELU MLP after every aggregation and LayerNorm. It is not a bare two-layer PyG GraphSAGE. Its paper fixes width 512, dropout 0.2, 1,000 steps, learning rate 3e-5, and searches depth 1–5 using validation. Attention baselines use eight heads. The pinned source uses AdamW with weight decay 0; the paper calls the optimizer Adam. The source preserves raw features, bidirects the graph, supervises only TRAIN nodes, and evaluates validation/test separately. The winning dataset-specific depth was not established; the authored search commands are recorded as a grid, not an invented winning setting.

### Polynormer closes an Amazon recipe, with explicit differences

Pinned author commit `fc8c276c9c5dfbd616d83f65338a3392188a5e08` releases the ReLU variant. The README's Amazon 55.04 is **one first run**, not the paper's ten-run 54.81 ± 0.49.

The Amazon command specifies: hidden width **256 per head × 2 heads = 512**, 10 local layers, 1 global layer, 200 local updates plus 2,500 global updates, Adam lr 0.001, weight decay 0, input dropout 0.2, local dropout 0.3, and global dropout inherited as 0.3. Default beta is -1, selecting learned sigmoid gates; Q/K are shared; pre-LN is false. There is no early stopping or scheduler in this driver. It uses raw PyG heterophilous features, all official mask rows when runs is increased, symmetrizes the graph, removes then adds self-loops, and supervises only TRAIN indices. A best validation-accuracy local checkpoint (model and optimizer) is restored at the global-stage transition. The strict validation selector and logger cover both stages; `_global` is not in `state_dict` and must be persisted explicitly in any derived implementation.

The paper describes 2,500 **total** epochs, eight heads and effective width 512. The release sums local and global counts to **2,700** Amazon updates and uses two heads. Tolokers and Questions also have paper/release differences, recorded in `AUTHOR_HYPERPARAMETERS.json`. The proposed comparison below fixes the release, not a blend of paper and code settings.

### PolyFormer does not establish Amazon competence

Its Table 5 covers four of the five Yandex graphs; Amazon is absent. The pinned CLI excludes Amazon, although the loader and training branches support it. No Amazon shell recipe was found in that pinned release. Source defaults and Roman Mono are legitimate prospective transfers, but neither can be described as the authored Amazon best setting. The original paper uses validation-selected Optuna with up to 400 complete trials; that budget cannot be transferred to a two-recipe pilot or used as a post-outcome tuning license.

### ATLAS paper and source do not close the same label-use contract

ATLAS v6, revised 26 August 2026, exposes community structure, one-hop attributes, and training-label diffusion as separate MLP input channels. Its paper says community-resolution search uses modularity to avoid labels, and LPF uses masked TRAIN labels only. These are established priors for topology/channel diversification; they do not show predictive member complementarity.

The inspected public source commit `9b2319765ca57749257405fdca80aaabef510df0` is dated **19 February 2026**, preceding v6. Amazon `run.sh` uses `--res None`. This calls `find_adaptive_resolutions` with default `enforce_theorem1=True`; the function reads full `data.label`, computes label/community NMI and a refinement test, and actually filters the retained resolutions by that test. Community preprocessing runs before split selection. This is a direct source path that would consult held-out labels, not merely a diagnostic print. The source and revised paper must be reconciled before any native reproduction. This observation does **not** prove which source produced the published v6 table.

LPF itself initializes its diffusion seed from `train_idx` only. For the 80%-TRAIN pilot, that index must be the fit subset, not the entire official TRAIN mask; the remaining TRAIN-control labels must stay out of LPF and any ANOVA selection. Raw features are used by the heterophilous loader. The authored Amazon source command/defaults are recorded separately from a verified paper reproduction.

### Spexphormer and ES-MLP require mask reconciliation

Spexphormer trains a narrow attention estimator, chooses it using validation, and reuses its learned attention scores for later wide-model runs. The two-stage cost belongs to the method. Its current heterophilous loader keeps `[:,0]` for all three masks; the README example repeats five initialization seeds on that mask. Thus the release does not independently demonstrate ten official-mask evaluation. The primary paper supplies an Amazon Table 8 recipe, but the inspected tree contains no Amazon YAML. Raw features and bidirected edges are verified in the loader; full preprocessing and the uncertainty convention remain unclosed.

ES-MLP learns task-relevant/irrelevant embeddings and edge weights through neighborhood contrastive and predicted-label consistency objectives. At inference it uses the feature model without adjacency. It reports only three of the five graphs. Its current `one_run` calls a loader that overwrites Amazon masks with per-class random 60/20/20, despite the paper saying it adopts Platonov settings. The release uses raw PyG features; its early-stop helper selects strict validation accuracy with patience 20. Paper and release are recorded independently.

### The current reassessment is a protocol warning, not a ranking certificate

The saved June 2026 v3 [heterophily reassessment](https://arxiv.org/html/2409.05755v3) uses ten fixed 50/25/25 masks for these graphs and a much larger Adam grid (three learning rates, ten weight decays, five dropouts; width 512). It calls Amazon Ratings “benign” relative to graph-aware versus graph-agnostic controls. This supports strong baseline and tuning controls; it neither establishes a native Amazon threshold for this pilot nor makes these numeric rows a current SOTA certificate. Its Springer discovery title/DOI differs; publisher full-text equivalence was not established.

## 4. Exactly one prospectively fixed next comparison

**Compare a four-member boundary GNNM Polynormer-r against four independently trained native Polynormer-r models on the existing Amazon 80%-TRAIN protocol.** This is an empirical extension of the saved backbone layout, not a new method claim. It tests whether GNNM's sharing preserves useful predictive quality on a source-qualified local-to-global Amazon backbone, beyond the current fixed-token PolyFormer transfers.

| Item | Fixed choice |
|---|---|
| Data roles | Existing official splits 0/1/2 and exact hash80% fit/20% TRAIN-control derivation; official VAL is the checkpoint selector; official TEST remains outside this proposed comparison |
| Block/member seeds | Blocks 17/29/43; independent member seeds `block_seed + 1009*m`, m=0,1,2,3; member 0 and the GNNM common-body initialization use the block seed; archive factor and dropout RNG states |
| Backbone | Pinned author ReLU model and Amazon release parameters above; raw features; bidirected edges; native self-loops; no LPF or extra structural channel |
| GNNM sharing | Saved `PolynormerBoundaryFamily` layout: shared native interior; private R/S/B only at `lin_in`, `pred_local`, `pred_global`; native biases copied to private B; Rademacher stem R, S=1, other R=1; complete private local hidden, GAT scores, global Q/K/V and reductions for every member |
| Training | Mean of four member TRAIN CE losses for GNNM; each independent model uses its own TRAIN CE. Same 200-local +2,500-global source schedule and Adam settings; no additional diversity/KD/error-allocation objective; no recipe search |
| Stage/selector | Strict official-VAL accuracy, earliest tie, spanning both stages as in native source. GNNM uses arithmetic-mean member probabilities for its VAL accuracy; each independent fit uses its own native selector. Restore the selected local-stage model/optimizer at the transition; explicitly serialize local/global flag. Report chosen stage for every fit |
| Pooling | Arithmetic mean of four class probability vectors for both arms; no learned weighting, temperature fitting, or member exclusion |
| Primary endpoint | Paired TRAIN-control NLL difference at selected checkpoints, reported per block and mean; report control accuracy, Brier score, and each member's NLL as descriptive companions |
| Cost evidence | Trainable/stored bytes, total preparation and fit time, peak memory, all four inference trajectories, and local/global stage count. Same recipe/trajectory count does not imply equal wall time or parameter count |
| Interpretation | Three block pairs give a bounded diagnostic, not a universal efficacy claim. No published-result pass threshold or numerical competence gate. Do not choose a new recipe or change the objective after control outcomes |

The saved adapter is an **unexecuted source draft**. Before an independently authorized fit, source parity and stage/checkpoint custody must be qualified; this packet does not certify them. This comparison is not launched or queued here.

Closest priors: Polynormer's local-to-global polynomial architecture; BatchEnsemble's shared W with member input/output scaling ([2002.06715v2](https://arxiv.org/abs/2002.06715v2)); the existing GNNM boundary-projector layout and its saved Photo backbone amendment. PolyFormer already supplies node-specific hop/order filters; ATLAS supplies topology/neighbor/label channels; Spexphormer supplies learned structural sampling. The remaining empirical question is the quality/cost effect of **parameter sharing within four complete trajectories on this Amazon-specific native backbone and fit protocol**. None of these inspected single-predictor tables measures that contrast. No global absence or scientific novelty is claimed. This does not duplicate NCNC decoder-completion work or reopen the FoRDE engineering packet.

## 5. Custody and read accounting

`REUSED_CONCLUSIONS.json` preserves the consulted conclusions and origins. `READ_SCOPES.json` distinguishes retained-paper incremental reads from new scoped reads and discovery-only exclusions. `PAPER_CONCLUSIONS.json` contains per-paper limitations and dispositions. `AUTHOR_HYPERPARAMETERS.json` separates paper settings, actual released commands, general defaults, search ranges, and missing settings. No whole-paper certification follows from downloading/extracting full HTML, and no figure/proof/full-author-source audit is claimed.

`SOURCE_BINDINGS.json` binds sources by URL/version/commit and SHA-256, including unchanged external files. `MANIFEST.json` and `REVIEW_SEAL.json` seal this new packet only. Index adoption is left to the parent task.
