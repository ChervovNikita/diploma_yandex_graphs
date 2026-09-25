# GNNM: BatchEnsemble projectors for graph neural networks

This repository contains the graph node classification implementation, the five benchmark graph files, archived result CSVs, and scripts for the GNNM study. GNNM trains several predictions that share a graph backbone. Each prediction uses a different low rank BatchEnsemble modulation of the input and output projectors. At evaluation time, the member logits are averaged before computing a class prediction or binary score.

[Kim (2023)](https://koasas.kaist.ac.kr/handle/10203/308201) previously applied BatchEnsemble factors inside GNN layers. The construction studied here places factors at the input and output projectors while tying every graph-layer weight tensor. The paper examines this placement and its limits; it does not claim the first use of BatchEnsemble in GNNs.

## What the completed comparisons show

The archived eight-backbone comparison covers five graphs, but its ENS and GNNM checkpoint rules differ. It is useful for describing where scores changed, not for attributing the changes to shared propagation. A later Roman Empire SAGE study uses the same five official masks and pooled validation checkpoint rule for its four-member models. GNNM averaged **89.502%** test accuracy, an ensemble of four width-512 ordinary networks averaged **87.515%**, and an ensemble selected to be close in trainable parameter count averaged **88.143%**. The masks overlap on one graph, so these are descriptive paired results rather than five independent tasks.

The separate `ogbn-arxiv` pilot gives a contrary result. At a fixed 300-epoch budget, BASE, the four-network ensemble, and GNNM averaged **70.639%**, **71.096%**, and **69.027%** test accuracy across three optimization seeds on one official temporal split. All nine validation-selected checkpoints were at epochs 294–300 and every run made a new validation best after epoch 275. The pilot therefore establishes neither convergence nor an accuracy advantage for GNNM on this graph. The 300-epoch budget was chosen after the 100-epoch pilot was inspected. Both budgets and the exact protocols are documented below.

A separate, post hoc matched-initialization control on the same split found **70.588%** mean test accuracy with untied propagation versus **69.027%** in its new tied rerun; its three paired differences all favored untied. That control changes parameter count and has selected checkpoints near the 300-epoch cap.

## Environment and data

A runnable environment uses Python 3.11, PyTorch 2.1.2 with CUDA 11.8, PyTorch Geometric 2.7.0, and DGL 2.4.0. Package versions are in `pyproject.toml` and `uv.lock`. The graph files used here are under `data/`. The training scripts load the ten official masks in those files. Run the commands below from the repository root, with a compatible NVIDIA GPU and Python environment.

## Recompute the archived main comparison

```
python experiments_iclr/reproduce_main.py
```

The script reads the archived CSVs from `results2/roman_empire`, `results2/amazon_ratings`, `results/minesweeper`, `results/questions`, and `results3/tolokers`. It checks that every dataset, backbone, method, and official split has depths 1 through 5, hidden width 512, and learning rate 3e-5. For each split it chooses the depth with the highest validation metric. Validation scores within 1e-12 count as tied, and the shallower depth wins. It then reads the test metric from the chosen row. The reported mean and population standard deviation use the ten selected test scores. No test score is used to choose depth. The 1e-12 tie tolerance is a retrospective reconstruction rule, since the original run manifest does not record a tie policy.

Outputs are under `experiments_iclr/recomputed_main/`: `per_split_selected.csv` lists 1,200 selected rows, `table_recomputed.csv` contains 120 method score summaries, `all_eight_candidate_summary.csv` contains 15 dataset-by-comparison backbone summaries, `all_eight_candidate_dataset_spread.csv` contains the five-dataset GNNM–ENS mask spread, `backbone_paired_effects.csv` and `dataset_level_effects.csv` contain paired effects, `audit.json` records ties and exploratory random-sign calculations for the five selected graphs, and `source_manifest.json` records the SHA256 hash of every input CSV. Those calculations do not establish a population-wide advantage. Accuracy is used for Roman Empire and Amazon Ratings. ROC AUC is used for Minesweeper, Questions, and Tolokers. The 40 dataset by backbone comparisons share graphs and splits, so a test treating all 40 cells as independent is not valid.

A retrospective check of `results/`, `results2/`, and `results3/` found exactly one eligible folder for each dataset: each has all three method CSVs with exactly 400 distinct rows. The selected paths are recorded in `SOURCE` in `experiments_iclr/reproduce_main.py`. That script validates the chosen CSVs but does not search the other candidate folders when run. These rows cover eight backbones (including TAG), five depths, ten official splits, width 512, and learning rate 3e-5. This rule uses only setup and completeness, not test scores. `results_grid/` is a separate broader search. The rule was identified retrospectively, and this repository does not establish whether the original authors used it before seeing test results. The chosen files match 102 of 105 printed cells in the earlier seven-backbone manuscript table to two decimals. The three Tolokers differences are GCN ENS 84.11 ± 0.81 here versus 84.10 ± 0.81 in that table, ResNet ENS 73.25 ± 1.02 versus 73.25 ± 1.03, and GAT GNNM 83.97 ± 0.75 versus 83.94 ± 0.71. The recomputed tables use the coherent manifest values and record this provenance.

The archived ENS procedure trains four separately seeded networks and chooses each member checkpoint from its own validation metric before averaging their logits. GNNM trains four members with one shared backbone and chooses one joint checkpoint from the pooled validation metric. Both then compare depths using the pooled validation metric. Consequently, the archived ENS versus GNNM score difference combines parameterization, training, and checkpoint selection. It cannot be attributed to parameter sharing alone. The additional four-member projector experiments below use pooled validation selection for every variant.

The current `run_base.py`, `run_base_ensemble.py`, and `run_tabm.py` default to choosing a hyperparameter setting on split 0 and reusing it. That is different from the archived main comparison. To collect new results with its depth selection protocol, pass `--search_each_split` to each runner while fixing width 512, learning rate 3e-5, and candidate depths 1 through 5. The older multi GPU shell scripts are historical launch scripts, not the source of the reconstructed table.

An optional retrospective sensitivity check is in `experiments_iclr/selection_sensitivity.py`. It compares per-split validation depth choice with one depth chosen by mean validation across masks or by split-0 validation. The note `experiments_iclr/recomputed_main/depth_policy_note.md` gives the measured effects and explains the cross-mask overlap caveat.

TAG is included in the eight-backbone main comparison. `python experiments_iclr/tag_sensitivity.py` applies the same validation-based depth rule to its rows independently and writes `experiments_iclr/recomputed_main/tag_sensitivity.csv` as a cross-check. It uses the same archived graphs and does not add an independent task.

## Recompute archived resource measurements

```
python experiments_iclr/reproduce_cost.py
```

This reads `ablation/results_cost/` and writes `experiments_iclr/recomputed_main/cost_table_recomputed.csv`. The main comparison and cost summary both use all eight backbones, including TAG, at five depths each. The output also gives a `without_TAG` subset to reconstruct the earlier seven-backbone table. The measured training step time in those CSVs used 50 warmup steps followed by 200 CUDA synchronized timed steps. Peak memory is PyTorch peak allocated memory in MiB with the graph already resident on the GPU. These are different quantities from an entire training run's elapsed time or GPU reserved memory. The archived profiler used AdamW weight decay 0.01; the archived predictive runs and the new control runs used zero weight decay. The archived cost figures therefore do not isolate the effect of model structure under identical optimization settings.

## Recompute the archived ablations

Run `python experiments_iclr/reproduce_ablations.py`. It reads the archived Roman Empire member-count and initialization CSVs and writes `experiments_iclr/recomputed_main/ablations_recomputed.csv`. These tables use the sample standard deviation over five splits, unlike the main table’s population standard deviation over ten splits. Two nominally identical GAT-sep, M=4, default-initialization collections yield different archived means (90.85% in the member-count table and 90.98% in the initialization table). The source logs do not establish the cause, so the tables should remain tied to their own CSVs.

## New attribution experiments

`experiments_iclr/projector_controls.py` runs a post hoc Roman Empire case study with five graph layers, width 512, learning rate 3e-5, four members where applicable, and official splits 0 through 4. It includes GNNM, ordinary independently learned projectors with a shared backbone, input-only and output-only BatchEnsemble projectors, independent output heads, a single ordinary network, one member with BatchEnsemble parameterization, an untied-backbone GNNM, frozen output factors, and an optional four-network ensemble selected by pooled validation accuracy. These settings were fixed before the full five-mask matrix, after inspection of the archived benchmark and split-0 pilots.

The untied-backbone variant copies the same seeded GNNM propagation stack four times while keeping both BatchEnsemble projectors and the output normalization shared. The frozen-output variant retains the same GNNM model but holds output factors R/S/B at their initial identity/zero values while training its shared output weight matrix. Both variants match GNNM's four initial member logits and random-generator state in the CPU test. The four-network pooled-checkpoint variant optimizes the mean of its four member cross entropies, as GNNM does, while its network parameters remain disjoint. The ordinary input-only control has a different initial output weight matrix despite matching GNNM's seeded backbone, so its comparison does not isolate output-factor learning. The one-member variant is also a parameterization and initialization check, not an ordinary network with identical initialization. Each run selects checkpoints using validation performance and evaluates test labels afterward.

```
python experiments_iclr/projector_controls.py --model SAGE \
  --variants gnnm independent_projectors heads_only input_only output_only base gnnm_m1 \
  --splits 0 1 2 3 4
python experiments_iclr/projector_controls.py --model SAGE \
  --variants untied_backbone freeze_output_factors ens_pooled \
  --splits 0 1 2 3 4
python experiments_iclr/analyze_decisions.py
python experiments_iclr/summarize_controls.py
```

The first GNNM split-0 pilot predates the completed control matrix. Before its separate rerun, the verification procedure requires exactly 35 unique first-stage SAGE rows with the expected settings and their checkpoint and prediction files. It freezes hashes of the matrix, control runner, and verifier. `experiments_iclr/verify_gnnm_split0.py` keeps the pilot only if the selected validation step matches, the validation metric differs by at most 1e-7, and the maximum checkpoint-tensor difference is at most 1e-6; otherwise it selects the rerun regardless of test score. It retains the original pilot row, matrix, checkpoint, and prediction in `experiments_iclr/gnnm_split0_verification/pilot_archive/`. When the rerun is selected, its checkpoint and prediction paths remain under the verification directory. The selected CSV row identifies the artifacts used by later analyses.

For the completed matrix, the main SAGE process loaded `projector_controls.py` before a later source edit confined to the `ens_pooled` variant, which was not part of its initial seven-variant matrix. The recorded `runner.sha256` identifies the source for the fresh GNNM split-0 rerun; it does not prove the exact source bytes already loaded by the original process. No launch-time source hash of that process was preserved. `experiments_iclr/README.md` gives the sequence for reproducing the 35-row gate, pilot check, and remaining controls from a clean output directory.

All 50 selected SAGE rows passed the complete artifact gate and a separate checkpoint-to-logit CPU replay. GNNM and the untied-backbone model begin with the same seeded member logits. They differ in whether their four graph propagation stacks are tied during training. GNNM averaged **89.502%** versus **86.562%** for the untied model, a **+2.940 percentage-point** paired difference across the five masks. Its mean member accuracy was **+3.079 points** higher, while its gain from pooling the members was **0.139 points** lower. In this setting, stronger member predictions account arithmetically for the pooled difference. Untying also increases trainable parameters from 6.74 million to 26.44 million and changes optimization after the first update. The frozen-output-factor control averaged **89.485%**, close to GNNM, so these masks do not show a material gain from learning the output multiplicative factors.

To check the selected artifacts without training, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/verify_control_artifacts.py --require-complete --check-checkpoints
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/verify_control_inference.py --require-complete --threads 2
```

The control CSV includes `train_seconds`, which is elapsed wall time for optimization, validation, and checkpoint writing together. It is not the synchronized training step measurement in the archived cost CSVs. Its legacy `nll` diagnostic clips probabilities at 1e-12 and can slightly understate cross entropy for very confident errors; use `test_loss` or the stable pooled cross entropy in `decision_analysis.csv` instead. Prediction archives contain member logits and post hoc descriptive variables. Local homophily is calculated from ground truth labels of neighboring nodes, including test neighbors. It is used only to describe errors after evaluation, never as a training or inference input. Because official test masks overlap across splits, per node rows across splits are not independent samples.

`experiments_iclr/lambda_objective.py` defines an optional, predeclared loss experiment at lambda values 0, 0.5, and 1. It should only be described as a measured experiment after its result CSV has been produced. Lambda 1 matches the original mean member loss. Lambda 0 optimizes the cross entropy of pooled logits.

`experiments_iclr/profile_strong_base.py` times four-member GNNM at width 512 and ordinary SAGE at widths 512, 896, 1024, and 1152 on Roman Empire split 0. It selects one of the three wider ordinary widths by closest synchronized training-step time, using the same zero AdamW weight decay as the new training controls and without reading validation or test scores. After the selected network is trained on five official masks, `experiments_iclr/summarize_strong_base.py` writes the paired accuracy and full-training-time comparison. The selection manifest stores all measured candidate times and parameter counts.

This comparison is complete and passed `python experiments_iclr/verify_strong_base.py --require-complete`. Width 1152 was selected at 310.21 ms per profiled step versus 321.93 ms for GNNM. Across the five overlapping Roman Empire masks, GNNM averaged 89.50% test accuracy and the wider BASE averaged 85.20%. Full training took longer for GNNM (984 versus 585 seconds on average), and the wider BASE learning rate was not separately tuned. These are local results on one graph, not a general compute-efficiency comparison.

## Post hoc added in-layer factors control

A completed Roman Empire SAGE control adds identity-initialized, member-specific
rank-one factors to the neighbor, root, and two feed-forward linear maps in
each hidden layer, while retaining GNNM's BatchEnsemble input and output
projectors and shared weights. The five official masks 0–4 use seed equal to
mask index, five layers of width 512, four members, learning rate 3e-5, zero
weight decay, at most 5,000 steps, and mean member cross entropy. One joint
checkpoint is selected by pooled validation accuracy. The hidden factors add
133,120 trainable parameters: 6,870,848 total versus GNNM's 6,737,728.

The all-layer control averaged **89.665%** selected test accuracy versus
**89.502%** for projector-only GNNM, a **+0.162 percentage-point** mean paired
difference; three masks favored the added factors and two favored boundary-only GNNM. This small,
mixed result is a post hoc within-code added-factor control, **not an exact
Kim (2023) reproduction**. The five masks overlap on one graph, and the
additional factors increase parameter count, so the result does not establish a
general or parameter-isolated advantage. The full five-mask checkpoint and
saved-logit replay passed. Run the read-only replay from this repository
with the complete selected artifact tree:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments_iclr/verify_all_layer_be_sage_frozen.py --threads 2
```

The anonymous ZIP omits the selected checkpoints and raw predictions, so
this server replay requires the full result tree. The frozen manifest records
that GAT was incomplete or failed after its launcher exited when this run
began; a later GAT rerun does not alter that historical observation. See
`experiments_iclr/all_layer_be_sage_protocol.md` for the full fixed protocol.

## Parameter-matched pooled ensemble

`experiments_iclr/select_parameter_matched_ens.py` chooses among four-network SAGE ensemble widths 192, 224, 256, 288, and 320 using only trainable parameter count. It chose width 256: **6,907,976** trainable parameters against GNNM's **6,737,728** (2.53% more). This ensemble uses the same five Roman Empire masks, five graph layers, learning rate, 5,000-step cap, and pooled validation checkpoint rule as GNNM and the width-512 pooled ENS control. The width-256 ensemble averaged **88.143%**, below GNNM on each mask by **1.094–1.571 percentage points**. Several selected ensemble checkpoints occurred late in the budget, so the comparison does not establish a converged ordering.

The selected split-0 checkpoints were also profiled for full-graph inference on one A100, with graph and model resident on the device. Median time was **109.696 ms** for GNNM and **39.877 ms** for width-256 ENS. The latter had lower accuracy in the five-mask study but lower measured latency in this one ordered timing run. Checkpoint loading is excluded. The raw timing samples were not saved, so this is a device-specific observation rather than a repeat-run latency estimate. Scores and configuration are in `experiments_iclr/parameter_matched_ens_results/paired_scores.csv`, `comparison.json`, and `inference_profile.json`.

To rerun the width-256 ensemble training, use a new result root. The runner skips a split when it finds an existing matching row, so the archived result directory must not be used for a fresh run:

```bash
.venv/bin/python experiments_iclr/projector_controls.py --model SAGE --variants ens_pooled --splits 0 1 2 3 4 --hidden_dim 256 --result_root experiments_iclr/parameter_matched_ens_reproduction
```

The checked-in `select_parameter_matched_ens.py`, `summarize_parameter_matched_ens.py`, and `verify_parameter_matched_ens.py` target the archived `parameter_matched_ens_results` directory. Their default invocations audit that archived study, not the new directory. The selection itself uses only parameter counts. To check the archived study with its complete selected artifacts, run:

```bash
.venv/bin/python experiments_iclr/select_parameter_matched_ens.py
.venv/bin/python experiments_iclr/summarize_parameter_matched_ens.py
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/verify_parameter_matched_ens.py --require-complete
```

To reproduce the recorded latency protocol, run `python experiments_iclr/profile_ensemble_inference.py` on a CUDA device after the three archived selected checkpoint sets exist, then repeat the verifier with `--require-inference-profile`. The verifier checks model and result structure, score arithmetic, and profile metadata. It does not retrain models or reproduce the raw timing samples.

## External graph pilot: `ogbn-arxiv`

`experiments_iclr/ogbn_arxiv_protocol.md` records the official graph and temporal split, two-layer width-128 SAGE setup, optimization seeds 0–2, and pooled validation checkpoint rule. BASE trains one ordinary network. ENS trains four independent ordinary networks and pools their logits. GNNM trains four members with shared propagation and pools their logits. All three use the same optimization settings within a budget. Their parameter counts and training cost differ. Each test score is read only after the validation-selected checkpoint has been restored.

| Fixed budget | BASE test % | ENS test % | GNNM test % |
| --- | ---: | ---: | ---: |
| 100 epochs | 67.783 | 68.172 | 64.375 |
| 300 epochs | 70.639 | 71.096 | 69.027 |

Entries are means across three seeds on the same graph and split. GNNM trailed ENS on every seed at both budgets. Its mean deficit narrowed from **3.797** to **2.070 percentage points**. The 300-epoch repeat was planned after viewing the 100-epoch results, and it still showed validation improvement through the last 25 epochs. Do not select between budgets by test score or treat seeds as independent graph tasks. The 300-epoch run starts from scratch, rather than continuing the 100-epoch checkpoints.

The following commands rerun the two fixed budgets into new output roots when the official OGB files are available under `experiments_iclr/data/`. Existing roots with selected records cause the runner to skip those runs:

```bash
.venv/bin/python experiments_iclr/ogbn_arxiv_pilot.py --device cuda:0 --seeds 0 1 2 --variants base ens gnnm --output-root experiments_iclr/ogbn_arxiv_reproduction_100
.venv/bin/python experiments_iclr/ogbn_arxiv_pilot.py --device cuda:0 --seeds 0 1 2 --variants base ens gnnm --max-epochs 300 --min-epochs 300 --output-root experiments_iclr/ogbn_arxiv_reproduction_300
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/verify_ogbn_arxiv_results.py --profile pilot-100 --result-root experiments_iclr/ogbn_arxiv_reproduction_100 --require-complete
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/verify_ogbn_arxiv_results.py --profile fixed-300 --result-root experiments_iclr/ogbn_arxiv_reproduction_300 --require-complete
```

The verifiers require all nine seed and variant pairs, check the official split and data fingerprints, replay selection from validation traces, and recalculate scores from saved logits. See `experiments_iclr/README.md` for the limits of those checks.

### Post hoc tied-versus-untied propagation control

A separate fixed 300-epoch control pairs four-member GNNM with a model whose four propagation stacks are independent deep copies of the same initial stack. Both arms use the official temporal split, the same training objective, and validation-only checkpoint selection. The tied arm here is a **new matched rerun**: its selected epochs and validation/test accuracies equal those of the earlier 300-epoch GNNM arms, but its checkpoint tensors and member logits are not bitwise identical. Use the new tied records for this paired comparison.

| Seed | New tied test accuracy | Untied test accuracy | Tied minus untied |
| ---: | ---: | ---: | ---: |
| 0 | 0.6941752433776855 | 0.7063761353492737 | -0.012200891971588135 |
| 1 | 0.6879205107688904 | 0.7049564719200134 | -0.017035961151123047 |
| 2 | 0.6887023448944092 | 0.7063144445419312 | -0.017612099647521973 |
| **Mean** | **0.6902660330136617** | **0.7058823506037394** | **-0.015616317590077719** |

The mean paired untied advantage is **1.561631759 percentage points** (sample SD of paired differences **0.297184000 points**). Untying increases trainable parameters from 189,248 to 684,608, so this is not a capacity-matched estimate of the effect of weight sharing. Selected epochs were 293–300 of 300; finishing the budget does not establish convergence or performance under longer training. All three seeds use one graph and one split, so the difference is descriptive optimization-seed variation. The protocol, selected records, summary, and verification amendment are under `experiments_iclr/ogbn_arxiv_untied_*`.

The original frozen verifier stopped on a field-name lookup for the untied training-start RNG hash. A disclosed verification-only copy changed that lookup and passed the full six-arm CPU checkpoint-to-logit replay. The original source lock and training artifacts remain unchanged. With the full result tree and official OGB data present, repeat the amended read-only gate from the repository root:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments_iclr/verify_ogbn_arxiv_untied_v2.py
```

See `experiments_iclr/ogbn_arxiv_untied_verification_amendment.md` for the original failure, one-line correction, and artifact hashes.

## Scope and provenance

`experiments_iclr/data_manifest.json` pins the five public graph NPZ files to one source commit and records their checksums. The tracked Amazon Ratings file had been an HTML page saved with an NPZ extension. It has been replaced with the authentic NPZ, and the old and new hashes are in the manifest. Run `python experiments_iclr/fetch_datasets.py` to verify all five files or download any that are missing.

The archived CSVs support the five node classification benchmarks and resource measurements described above. This clone does not contain raw PROTEINS folds, training code, or per run results, so a graph classification result from an older supplement cannot be independently reconstructed here. The scripts under `experiments_iclr/` keep new results separate from the archived CSVs.

## Anonymous code and data supplement

Run `python experiments_iclr/build_anonymous_bundle.py` from the repository root after checking the studies selected for an anonymous release. It writes `experiments_iclr/gnnm_anonymous_code.zip`. The builder checks the fixed-grid source CSV hashes, the exact 50-row control matrix, the split-0 verification audit and selected prediction arrays, the five timing-matched masks, and the checksums of the three public graph files included in the ZIP. It omits Git history, model checkpoints, the two largest public graph files, and author metadata, and scans included text for identity and private paths. The archive contains instructions for fetching and verifying the two omitted public graph files.

Use this ZIP as the anonymous code supplement for double-blind review. The public repository and its Git remote identify the authors, so the anonymous manuscript and supplementary material should refer to the uploaded anonymous archive rather than linking to that remote. For the completed OGB, untied-propagation, and matched-ensemble studies reported in the paper, pass `--include-ogb --include-ogb-300 --include-ogb-untied --include-parameter-matched-ens`. Each option invokes its complete artifact gate. The untied control is default-off: `--include-ogb-untied` requires `--include-ogb-300`, which in turn requires `--include-ogb`. It adds only compact source and result records after the corrected six-arm CPU replay succeeds.

## Upstream graph data notice

The five heterophily graph files are pinned to a Yandex Research source commit in `experiments_iclr/data_manifest.json`. The upstream repository publishes an MIT license. Its full notice is reproduced in `experiments_iclr/UPSTREAM_DATA_LICENCE.txt`.
