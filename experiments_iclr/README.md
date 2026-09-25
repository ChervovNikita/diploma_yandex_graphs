# ICLR study audit and additional experiments

This directory separates calculations from archived runs from subsequent experiments. None of the scripts chooses a model using test labels.

## Archived main table

Run `python experiments_iclr/reproduce_main.py` from the repository root. The script reads 15 CSVs named in `recomputed_main/source_manifest.json`. Each yields 400 distinct valid rows. The parser keeps the leading data row before the header in Minesweeper BASE and skips repeated header rows in Tolokers BASE and GNNM. Within each dataset, backbone, method, and official split, it checks five depths with width 512 and learning rate 3e-5. It selects the depth with the best validation metric and reads that depth's test metric. Values within 1e-12 of the best validation score count as a tie, broken toward the smaller depth. This rule changes the selected row for Minesweeper SAGE ENS split 7 relative to an exact floating point maximum. It makes that cell 93.87 ± 0.44, as printed in the earlier manuscript table. The tolerance is a retrospective reconstruction rule. The original run manifest does not document a tie policy.

The resulting `recomputed_main/` files are:

- `per_split_selected.csv`: 1,200 validation-selected rows, one for each dataset, backbone, method, and split.
- `table_recomputed.csv`: 120 means and population standard deviations across the ten official test splits.
- `all_eight_candidate_summary.csv`: 15 dataset-by-comparison summaries across all eight backbones.
- `all_eight_candidate_dataset_spread.csv`: five dataset rows showing the official-mask spread of the eight-backbone GNNM–ENS comparison.
- `backbone_paired_effects.csv`: paired method differences within each backbone and dataset, with descriptive counts of split wins, ties, and losses.
- `dataset_level_effects.csv`: each comparison averaged over the eight backbones of one dataset.
- `audit.json`: tie cases and exact one-sided random-sign tests over the five dataset-level differences.
- `source_manifest.json`: SHA256 checksums, byte sizes, and row counts for all input CSVs, plus a provenance disclosure.

The source rule selects, among `results/`, `results2/`, and `results3/`, the unique folder per dataset whose three method CSVs each contain 400 distinct rows for eight backbones, five depths, ten splits, width 512, and learning rate 3e-5. This selection uses only setup and completeness, not test scores. `results_grid/` contains a separate broader search. The rule was identified retrospectively, and the archive does not show whether the original authors used it before test inspection. The resulting manifest agrees with 102 of the 105 earlier main-table cells to two decimals. The three differences are Tolokers GCN ENS, ResNet ENS, and GAT GNNM. Use the coherent recalculated values with this disclosure.

The five datasets are the independent benchmark units for the dataset-level sign test. Backbones within a dataset and official splits within a graph are correlated. A Friedman test over all 40 dataset by backbone cells does not provide a valid independent-task p-value.

The optional `selection_sensitivity.py` recomputes the same raw CSVs under three validation-only depth rules. It writes full selected rows, cell summaries, and dataset effects. `recomputed_main/depth_policy_note.md` states the exact protocols and the caveat that official masks overlap. These were calculated retrospectively and do not replace the primary per-split rule.

Run `python experiments_iclr/tag_sensitivity.py` to check the five TAG rows of the eight-backbone main analysis independently. It writes `recomputed_main/tag_sensitivity.csv` using the same per-mask validation-only depth rule. TAG comes from the same archive and does not add an independent graph task.

The archived comparison also uses different checkpoint rules. ENS trains four independent networks, chooses the best validation checkpoint for each one separately, averages their logits, and then compares depths using the pooled validation metric. GNNM trains four members with one shared backbone and chooses one checkpoint using the pooled validation metric. The archived accuracy difference therefore includes checkpoint policy as well as model and optimization differences.

## Archived compute table

Run `python experiments_iclr/reproduce_cost.py`. The input `ablation/results_cost/` CSVs contain 200 measured configurations per dataset. The main table averages all eight backbones, including TAG, across five depths per backbone. The output `recomputed_main/cost_table_recomputed.csv` also provides a `without_TAG` subset to reconstruct the earlier seven-backbone table. The source profiler used 50 warmup and 200 CUDA synchronized training steps, and measured peak PyTorch allocated memory in MiB with the graph already resident. These figures are not full training run times or total GPU reserved memory. The archived profiler used AdamW weight decay 0.01; the archived predictive runs and the new control runs used zero weight decay, so the old cost table is not an optimizer-matched estimate of the new controls.

## Archived Roman Empire ablations

Run `python experiments_iclr/reproduce_ablations.py`. It validates the five official splits, depth 5, width 512, and learning rate 3e-5 for each source CSV, then writes `recomputed_main/ablations_recomputed.csv`. The older K and initialization tables used sample SD over five splits. The GAT-sep M=4 default rows differ between the two archived collections (90.85% and 90.98% mean test accuracy), even though the recorded configuration matches. The archive does not explain that run variation.

## Roman Empire projector case study

Run `python experiments_iclr/projector_controls.py --model SAGE --variants gnnm independent_projectors heads_only input_only output_only base gnnm_m1 --splits 0 1 2 3 4`. This is a post hoc case study of Roman Empire using five layers, width 512, learning rate 3e-5, four members where appropriate, and the first five official masks. Those settings were fixed before the full five-mask matrix, after inspecting the archived benchmark and split-0 pilots. Every four-member variant saves a checkpoint selected by the accuracy of its pooled validation logits. The test mask is evaluated only after selection.

For a fresh reproduction, run the first seven variants on masks 0 through
4. When their 35 rows are complete, record the matrix, runner, and verifier
hashes, rerun GNNM split 0 in a separate directory, then apply the fixed
verification rule. The verifier checks every first-stage row's configuration
and checkpoint and prediction files before it can adopt either run. The
controlled queue performs the same 35-row gate before starting the rerun.
Run the remaining three variants and recompute both summaries after all 50
rows exist. These commands assume a clean output directory and a compatible
GPU environment.

```bash
python experiments_iclr/projector_controls.py --model SAGE \
  --variants gnnm independent_projectors heads_only input_only output_only base gnnm_m1 \
  --splits 0 1 2 3 4
mkdir -p experiments_iclr/gnnm_split0_verification
sha256sum experiments_iclr/results/projector_controls.csv | cut -d ' ' -f1 > experiments_iclr/gnnm_split0_verification/matrix.sha256
sha256sum experiments_iclr/projector_controls.py | cut -d ' ' -f1 > experiments_iclr/gnnm_split0_verification/runner.sha256
sha256sum experiments_iclr/verify_gnnm_split0.py | cut -d ' ' -f1 > experiments_iclr/gnnm_split0_verification/verifier.sha256
python experiments_iclr/projector_controls.py --model SAGE \
  --variants gnnm --splits 0 \
  --result_root experiments_iclr/gnnm_split0_verification
python experiments_iclr/verify_gnnm_split0.py
python experiments_iclr/projector_controls.py --model SAGE \
  --variants untied_backbone freeze_output_factors ens_pooled \
  --splits 0 1 2 3 4
python experiments_iclr/analyze_decisions.py
python experiments_iclr/summarize_controls.py
```

The variants change these parts of one setup:

- `gnnm`: BatchEnsemble modulation of both input and output projectors with a shared graph backbone.
- `untied_backbone`: the same BatchEnsemble input and output projectors and the same shared output normalization as GNNM, but one propagation stack per member. Each stack is a deep copy of the same seeded GNNM stack, so all initial member logits and the post-construction training random-generator state match GNNM exactly on the CPU test graph. The four stacks then update independently from their own member losses. This directly tests the effect of tying the propagation parameters, with pooled validation checkpoint selection in both variants. It does not make their optimization trajectories identical after the first update.
- `freeze_output_factors`: the same seeded GNNM model, with its output projector member factors R and S fixed at one and B fixed at zero. Its shared output weight matrix remains trainable. All initial parameters, member logits, and the post-construction training random-generator state match GNNM exactly on the CPU test graph. This isolates whether learning member-specific output factors changes the result under the same pooled checkpoint rule.
- `independent_projectors`: four ordinary learned input projectors and output projectors with a shared graph backbone. Each ordinary linear projector starts with its own Xavier weights and zero bias. Both its input and output maps are diverse at initialization. GNNM shares one Xavier weight matrix for each projector, uses random ±1 input factors, and starts all output factors at the identity. This matches the weight scale but not the complete initialization.
- `heads_only`: one shared ordinary input projector, four ordinary output heads, and one shared graph backbone.
- `input_only`: BatchEnsemble input projector, one shared ordinary output projector, and one shared graph backbone. Although its input projector, propagation stack, and output normalization match GNNM at initialization, its output weight matrix differs because the two constructors consume random numbers differently. Its comparison with GNNM does not isolate output-factor learning.
- `output_only`: one shared ordinary input projector, BatchEnsemble output projector, and one shared graph backbone. Its output modulation starts at the identity, as in GNNM.
- `base`: one ordinary graph network.
- `gnnm_m1`: one member with BatchEnsemble parameterization. Its initialization is not identical to `base`.
- `ens_pooled`: four independent ordinary networks trained on the mean of their member cross entropy losses, with a joint checkpoint chosen by pooled validation accuracy. This addresses the archived checkpoint difference on the masks where it is run. It does not reproduce the archived member seeds or per-step validation cadence.

All new variants use the same graph, masks, backbone type, depth, width, learning rate, dropout, maximum steps, early stop rule, and validation frequency. Matching seed numbers do not make the backbone initial weights identical across variants because their constructors consume random numbers in different orders. The CPU audit `recomputed_main/init_audit.json` finds identical residual and output-normalization tensors for GNNM and input-only, but different learned backbone tensors for independent projectors, heads-only, output-only, and BASE. The untied-backbone and frozen-output variants separately verify exact initial member functions against GNNM. Comparisons against ordinary projectors remain paired by official mask but do not isolate projector design from initialization.

The first GNNM split-0 pilot was recorded before the full control matrix. A separate rerun is stored under `gnnm_split0_verification/`. Before comparing runs, the verification gate requires exactly 35 distinct combinations of the seven initial variants and five masks, the expected SAGE configuration, and existing checkpoint and prediction files for every row. It checks the frozen `matrix.sha256`, `runner.sha256`, and `verifier.sha256` manifests. The verifier compares the selected validation step, validation metric, and checkpoint tensors under fixed tolerances of exact step equality, 1e-7, and 1e-6, respectively. It adopts the fresh row whenever any check fails, regardless of test score. The audit is written before a row switch; the original pilot row, 35-row matrix, checkpoint, and prediction are copied into `pilot_archive/`. The selected main CSV row points to the retained verification artifacts if the fresh run is adopted. Both `analyze_decisions.py` and the anonymous bundle builder follow the selected row's `prediction_file` path. Regenerate the summaries after a row switch.

In the live original matrix, the main SAGE process loaded `projector_controls.py` before an ens_pooled-only source edit. That variant was not in the first seven-variant matrix. The `runner.sha256` manifest freezes the file used for the later fresh split-0 rerun, not the exact bytes previously loaded into the already-running main process. The original process had no preserved launch-time source hash. The matrix hash, configuration gate, checkpoint files, and pilot rerun audit establish result and artifact provenance within those limits; they do not reconstruct the original process's in-memory source.

The CSV `results/projector_controls.csv` records test performance and descriptive diversity measures. Its legacy `nll` field clips probabilities at 1e-12 and may slightly understate cross entropy for very confident wrong predictions; `test_loss` and the stable pooled cross entropy in `decision_analysis.csv` are the appropriate loss values. `train_seconds` includes training, validation, and checkpoint writes. It is not the synchronized step time in the archived compute table. Saved prediction archives include test member logits and labels. Most are under `results/predictions/`; an adopted fresh GNNM split-0 array remains under `gnnm_split0_verification/predictions/`. `python experiments_iclr/analyze_decisions.py` produces per-run, per-number-of-correct-members, and per-variant descriptive summaries. It checks that mean member cross entropy minus pooled-logit cross entropy equals the mean KL from the pooled softmax to the member softmax, and counts member-node decisions helped or harmed by pooling. `python experiments_iclr/summarize_controls.py` writes means and paired differences across completed official masks. These are descriptive because the masks come from one graph. Nodes can occur in multiple official test masks, so pooled node counts across splits are not independent evidence. Local homophily is calculated afterward from true labels of neighboring nodes, including held-out nodes. It is never a training or inference input.

The 50-row SAGE matrix is complete. All selected checkpoints, saved predictions, and fixed settings passed the complete artifact gate. A separate CPU replay rebuilt all 50 models from selected checkpoints on the pinned graph and compared fresh logits with the saved arrays. Run both read-only gates from the repository root, with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/verify_control_artifacts.py --require-complete --check-checkpoints
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/verify_control_inference.py --require-complete --threads 2
```

Pooled test accuracy means across masks 0–4 were **89.502% for GNNM**, **87.515% for width-512 pooled ENS**, **86.562% for the initially matched untied-backbone variant**, and **89.485% for GNNM with frozen output factors**. GNNM exceeded the untied variant on every mask by 2.524–3.600 percentage points. That mean **+2.940-point** difference equals **+3.079 points in mean member accuracy** plus **−0.139 points in the gain from pooling members**. This arithmetic places the observed difference mainly in individual member accuracy under this protocol. The untied model has 26.44 million trainable parameters versus GNNM's 6.74 million. Both models start with the same seeded member logits, but their optimization paths diverge after the first update. The frozen-factor result gives little evidence that learning the output multiplicative factors matters in this local SAGE setting. The five masks overlap on one graph and were studied after the archived results, so these results do not establish a graph-general causal law.

A separate optional initialization diagnostic can use the same initial residual and output-normalization tensors for GNNM and independent projectors. It draws one anchor backbone with seed `100000 + split`, copies those tensors into both variants, then resets the training random generator to `200000 + split`. Projectors still have their respective initializations. Run one Roman split with:

```
python experiments_iclr/projector_controls.py --model SAGE \
  --variants gnnm independent_projectors --splits 0 \
  --common_backbone_init --result_root experiments_iclr/common_init_results
```

This diagnostic should be described as measured only after its two result rows exist. Matching the initial backbone does not force later weights or dropout masks to stay identical during training.

`lambda_objective.py` predeclares a separate training objective comparison at lambda 0, 0.5, and 1 for SAGE on the same five official masks. Lambda 1 is the original mean member cross entropy. Lambda 0 trains only the pooled logits. Describe this comparison as an empirical result only if its CSV has been generated.

## Post hoc added in-layer SAGE factors control

The separate `all_layer_be_sage.py` run compares projector-only GNNM with
BatchEnsemble factors added to every hidden SAGE neighbor/root and both
feed-forward linear maps in each residual block. The added factors start at
identity (R=S=1, B=0), retaining the original shared weights and the input
and output projectors. The CPU initialization audit confirmed equal initial
member logits and random-generator state on all five masks. This was fixed
after the archived results were inspected and before this five-mask run.

The protocol uses the official Roman Empire masks 0–4, seed equal to mask
index, five SAGE layers of width 512, four members, dropout 0.2, AdamW at
3e-5 with zero weight decay, mean of four member cross entropies, and at most
5,000 steps. Pooled validation accuracy is checked at step 1 and every ten
steps; one joint checkpoint is retained at its maximum, earliest tie, with
an early stop after 300 steps without improvement. Test labels are read
after restoring that checkpoint. The hidden factors add 133,120 trainable
parameters, making 6,870,848 versus GNNM's 6,737,728.

| Official mask | Projector-only GNNM test % | All-layer test % | All-layer minus GNNM, points |
| ---: | ---: | ---: | ---: |
| 0 | 89.234 | 89.587 | +0.353 |
| 1 | 89.940 | 89.428 | −0.512 |
| 2 | 89.711 | 89.781 | +0.071 |
| 3 | 89.463 | 89.746 | +0.282 |
| 4 | 89.163 | 89.781 | +0.618 |
| Mean | **89.502** | **89.665** | **+0.162** |

The mean paired difference was +0.162 percentage points, with a sample
standard deviation of 0.425 points; all-layer scored higher on three masks
and lower on two. For context, width-512 pooled ENS averaged 87.515% and
untied-backbone GNNM averaged 86.562%. Those contrasts change more than
the added factors.
Because the masks overlap on one graph and this was planned after archived
results were inspected, this is a descriptive, within-code added-factor
control, **not an exact Kim (2023) reproduction** or a graph-general
effect. The additional parameters prevent a parameter-isolated claim.

The complete selected-artifact audit and CPU checkpoint-to-test-logit replay
passed on all five masks. Maximum absolute replay error was 3.815e-05,
with no member or pooled test decision changes. The launch-time manifest
recorded the optional GAT study as incomplete or failed after its launcher
exited; a later GAT rerun is a separate event. From the repository root,
with the full selected checkpoint and prediction tree present, run:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments_iclr/verify_all_layer_be_sage_frozen.py --threads 2
```

The anonymous ZIP omits those selected binaries and server launch logs.
The exact design and frozen inputs are in
`experiments_iclr/all_layer_be_sage_protocol.md` and
`experiments_iclr/all_layer_be_sage_results/protocol.json`.

## Timing-selected wider ordinary network

`profile_strong_base.py` profiles Roman Empire split 0 with 50 warmup and 200 CUDA-synchronized training steps. It measures four-member GNNM at width 512 and ordinary SAGE at widths 512, 896, 1024, and 1152. Among the predeclared wider candidates, it selects the ordinary width whose step time is closest to GNNM on a multiplicative scale. The profiler uses zero AdamW weight decay, matching the new training controls. Validation and test labels do not enter this selection. The manifest `strong_base_results/strong_base_selection.json` records every measured width, its parameters, and its step time. The experiment then trains the selected width on the five official masks with the same fixed depth, learning rate, checkpoint rule, and stopping rule as the other new controls. `summarize_strong_base.py` produced the paired score CSV and JSON summary. Full training elapsed time includes validation and checkpoint I/O, while the synchronized step time measures only one training step. These are separate quantities.

The five-mask result and paired summary are complete and passed `python experiments_iclr/verify_strong_base.py --require-complete`. The timing rule selected width 1152. GNNM averaged 89.502% test accuracy versus 85.203% for the wider BASE on the five overlapping masks, with a mean paired difference of +4.299 percentage points. Profiled steps took 321.93 and 310.21 ms, respectively; full training averaged 984 and 585 seconds. The wider BASE learning rate was not separately tuned, and this one-graph result does not establish a general compute-efficiency ordering.

## Parameter matched pooled ensemble

`select_parameter_matched_ens.py` counts trainable parameters in the exact Roman Empire SAGE constructors used by `projector_controls.py`. It fixes four-network `ens_pooled` candidate widths 192, 224, 256, 288, and 320, then chooses the width with the smallest absolute parameter-count difference from four-member GNNM at width 512. No validation or test score enters this choice. The selected width is **256**, with **6,907,976** trainable parameters versus GNNM's **6,737,728**, a 2.53% difference. `parameter_matched_ens_results/selection.json` records the counts and choice.

The width-256 ENS uses the same five Roman Empire masks, SAGE depth 5, AdamW learning rate 3e-5 and zero weight decay, 5,000-step cap, and pooled validation checkpoint rule as the width-512 pooled ENS. Its five selected rows, checkpoints, and predictions are complete. The summary files are `parameter_matched_ens_results/paired_scores.csv` and `comparison.json`.

| Official mask | GNNM test % | ENS width 512 test % | ENS width 256 test % |
| ---: | ---: | ---: | ---: |
| 0 | 89.234 | 87.169 | 87.875 |
| 1 | 89.940 | 87.928 | 88.846 |
| 2 | 89.711 | 87.381 | 88.175 |
| 3 | 89.463 | 87.646 | 88.228 |
| 4 | 89.163 | 87.451 | 87.593 |
| Mean | **89.502** | **87.515** | **88.143** |

The width-256 ensemble is below GNNM on every mask, with a mean paired difference of **−1.359 percentage points**. It has 2.53% more parameters but a narrower hidden layer. Several of its checkpoints were selected late under the 5,000-step cap, so the fixed-budget result does not establish a converged ranking. The masks overlap on one graph, which makes the paired differences descriptive rather than independent task replicates.

To reproduce from a clean result directory, first run the score-blind width selector, then train the five chosen ENS rows and calculate the summary:

```bash
.venv/bin/python experiments_iclr/select_parameter_matched_ens.py
.venv/bin/python experiments_iclr/projector_controls.py --model SAGE \
  --variants ens_pooled --splits 0 1 2 3 4 --hidden_dim 256 \
  --result_root experiments_iclr/parameter_matched_ens_results
.venv/bin/python experiments_iclr/summarize_parameter_matched_ens.py
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/verify_parameter_matched_ens.py --require-complete
```

The verifier checks the selection manifest against fresh constructor counts, the five selected checkpoints and predictions, the official held-out masks, score arithmetic, and paired summaries. A fresh CPU replay of all five selected checkpoints matched their saved test decisions, with no member or pooled class-decision changes. The gate itself checks artifact structure and arithmetic rather than retraining the models.

`profile_ensemble_inference.py` measured selected checkpoint size across five masks and full-graph pooled inference latency on split 0 for GNNM, width-512 ENS, and width-256 ENS. On one A100, with 20 warmups and 100 CUDA-synchronized forward measurements in the listed order, median times were **109.696 ms**, **108.984 ms**, and **39.877 ms**, respectively. Model weights and the graph were resident on the GPU. Data and checkpoint loading were excluded. The timing samples were not saved, so `inference_profile.json` supports one recorded device-specific comparison rather than a repeat-run latency distribution. Reproduce that profile on a CUDA device, then require its metadata in the gate:

```bash
.venv/bin/python experiments_iclr/profile_ensemble_inference.py
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/verify_parameter_matched_ens.py --require-complete --require-inference-profile
```

## External graph pilot

`ogbn_arxiv_protocol.md` documents the official `ogbn-arxiv` temporal split and the fixed two-layer, width-128 SAGE setup. BASE trains one ordinary network. ENS trains four independent ordinary networks. GNNM trains four members with one shared propagation backbone. Both four-member models train on the mean of member cross entropies, choose one checkpoint using pooled validation logits, and average raw member logits at inference. For every variant, test labels are used only after restoring a checkpoint selected by validation accuracy, then validation cross entropy, then earliest epoch. Seeds 0, 1, and 2 vary optimization on the same graph and split. The three variants have different parameter counts and training costs.

The completed 100-epoch pilot and its 300-epoch fixed-budget repeat both passed the nine-run artifact gate. The repeat was planned after the 100-epoch results were known. It started from fresh initializations with the same three seeds. It set `--min-epochs 300` to disable early stopping, while checking validation after every epoch with the same checkpoint rule. It was not a continuation from the pilot checkpoints.

| Budget | BASE test mean ± sample SD % | ENS test mean ± sample SD % | GNNM test mean ± sample SD % |
| ---: | ---: | ---: | ---: |
| 100 epochs | 67.783 ± 0.252 | 68.172 ± 0.127 | 64.375 ± 0.472 |
| 300 epochs | 70.639 ± 0.320 | 71.096 ± 0.159 | 69.027 ± 0.341 |

GNNM scored below ENS on every seed at both budgets. Its three-seed mean test deficit narrowed from **3.797** to **2.070 percentage points**. Every 300-epoch run selected a checkpoint at epoch 294–300, and each established a new validation-accuracy best after epoch 275. Neither budget establishes convergence or a ranking under longer training. The sample SDs describe optimization variation on one graph and one temporal split, not graph-level uncertainty. The 300-epoch budget was selected after seeing the 100-epoch results, so report both budgets rather than choosing one by test accuracy.

The result roots are `ogbn_arxiv_results/` and `ogbn_arxiv_300_results/`. Each contains a dataset fingerprint, fixed run configuration, nine selected run records, validation traces, selected checkpoints and logits, and aggregate CSV/JSON summaries. `ogbn_arxiv_diagnostics.py` computes descriptive member and pooled predictions from the saved 100-epoch logits. From the repository root, with the official OGB files under `experiments_iclr/data/`, reproduce and check both budgets as follows:

```bash
.venv/bin/python experiments_iclr/ogbn_arxiv_pilot.py \
  --device cuda:0 --seeds 0 1 2 --variants base ens gnnm
.venv/bin/python experiments_iclr/ogbn_arxiv_pilot.py \
  --device cuda:0 --seeds 0 1 2 --variants base ens gnnm \
  --max-epochs 300 --min-epochs 300 \
  --output-root experiments_iclr/ogbn_arxiv_300_results
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/verify_ogbn_arxiv_results.py \
  --profile pilot-100 --require-complete
CUDA_VISIBLE_DEVICES= .venv/bin/python experiments_iclr/verify_ogbn_arxiv_results.py \
  --profile fixed-300 --require-complete
```

The read-only gate checks the nine required seed/variant pairs, source and dataset fingerprints, validation-only selection against epoch traces, selected checkpoint schema and hashes, split indices and labels, and recomputed metrics from saved logits. A separate CPU replay compared selected checkpoint outputs with saved logits for both budgets without changing any class decisions. The repository gate itself checks internal consistency and strict checkpoint loading, but does not perform the full-graph checkpoint-to-logit replay. See `ogbn_arxiv_verifier_protocol.md` for its limits.

## Post hoc `ogbn-arxiv` tied-versus-untied propagation control

`ogbn_arxiv_untied_protocol.md` fixes seeds 0–2, four members, SAGE depth 2 and width 128, and exactly 300 epochs for a new tied/untied pair per seed. Both arms start from one initialized GNNM model. The untied arm gives each member an independent deep copy of the propagation stack; the input/output BatchEnsemble projectors and output normalization retain their within-arm sharing. The model construction checks equal initial member logits and parameter values, disjoint storage, and identical training-start RNG state. Checkpoints are selected by pooled validation accuracy, then pooled validation cross entropy, then earliest epoch. Test labels enter scoring only after restoration and validation replay.

| Seed | New tied selected epoch | Untied selected epoch | New tied test accuracy | Untied test accuracy | Tied minus untied |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 296 | 299 | 0.6941752433776855 | 0.7063761353492737 | -0.012200891971588135 |
| 1 | 300 | 293 | 0.6879205107688904 | 0.7049564719200134 | -0.017035961151123047 |
| 2 | 300 | 299 | 0.6887023448944092 | 0.7063144445419312 | -0.017612099647521973 |
| **Mean** | | | **0.6902660330136617** | **0.7058823506037394** | **-0.015616317590077719** |

The mean paired difference is **−1.561631759 percentage points** (sample SD **0.297184000 points**), favoring untied propagation in every seed. This is a post hoc mechanism check on one graph and one official temporal split, not an independent benchmark replication. The untied arm has 684,608 trainable parameters versus 189,248 tied, so capacity changes with weight sharing. All six selected checkpoints lie at epochs 293–300 of 300; the fixed budget does not establish convergence or the ordering under longer training. The new tied arms exactly match the earlier 300-epoch GNNM selected epochs and validation/test accuracies, but their checkpoint tensors and saved member logits differ. They are separate matched reruns, and this contrast uses their own paired records.

The original frozen verifier failed at the untied RNG field lookup (`untied_propagation_rng_start_sha256` versus the recorded `untied_rng_start_sha256`) after training completed. Its original completion marker was not written. `verify_ogbn_arxiv_untied_v2.py` changes only that lookup. It then passed the full read-only CPU replay of all six selected checkpoints, validation/test member logits, official split labels, and result arithmetic; `logs/ogbn_arxiv_untied_v2.complete` marks this separate gate. The original source lock and result artifact hashes remain intact. The amendment and exact hashes are in `ogbn_arxiv_untied_verification_amendment.md`. To repeat verification with the complete `ogbn_arxiv_untied_results/` tree and official OGB data present, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments_iclr/verify_ogbn_arxiv_untied_v2.py
```

The three paired differences, mean, sample SD, initialization records, validation traces, and selected prediction archives are in `ogbn_arxiv_untied_results/`. The verifier amendment is post hoc and should accompany any reported result.

## Scope

`data_manifest.json` pins the five public Heterophilous Graphs NPZ files to one source commit and records their byte sizes, Git blob hashes, and SHA256 hashes. The checked-out `data/amazon_ratings.npz` had been an HTML page saved with an NPZ extension; it was replaced by the authentic source NPZ, with old and new hashes recorded in the manifest. `fetch_datasets.py` verifies present files and downloads any missing file from that pinned source. The archived result CSVs were not changed by this repair.

The available raw CSVs support the five node classification benchmarks and the compute profiler. This clone does not contain the raw folds, code, or per-run records needed to independently reconstruct the PROTEINS graph classification result from an earlier supplement. The upstream graph-data MIT notice is reproduced verbatim in `UPSTREAM_DATA_LICENCE.txt`.

## Anonymous code supplement

Run `python experiments_iclr/build_anonymous_bundle.py` after checking the exact studies selected for an anonymous release. It creates `experiments_iclr/gnnm_anonymous_code.zip` and checks fixed-grid source hashes, selected prediction paths and hashes, and the three included public graph files. It retains the selected verification prediction when the original GNNM split-0 pilot is adopted. The archive omits checkpoints, Git history, and the two largest public graph files; its README gives the pinned download and verification procedure for the omitted files. For the completed studies reported here, use `--include-ogb --include-ogb-300 --include-ogb-untied --include-parameter-matched-ens`. The default-off `--include-ogb-untied` flag requires `--include-ogb-300`, which requires `--include-ogb`. Its gate checks the frozen source and artifact hashes and replays all six selected arms on CPU before adding compact records; the archive omits the original checkpoints and prediction arrays. Each optional gate requires complete checked artifacts.

Before building the ZIP, run the read-only checks `python experiments_iclr/verify_control_artifacts.py --require-complete --check-checkpoints` and `python experiments_iclr/verify_control_inference.py --require-complete --threads 2`. The first checks the fixed training configuration, checkpoint readability, saved decisions, probabilities, losses, member statistics, and held-out node alignment. The second rebuilds the selected models on the pinned graph and checks that their test logits agree with the saved prediction arrays. The builder repeats both checks as required gates.

The builder also has opt-in gates for complete all-layer SAGE, GAT tying and decision analysis, decision strata, and SAGE/GAT layerwise diversity studies. Enable only studies reported in the paper, using --include-all-layer-sage, --include-gat-pair, --include-decision-strata, --include-layerwise-sage, or --include-layerwise-gat. The GAT layerwise option requires the GAT pair option. Each gate checks the complete five-mask result and its available provenance before inclusion. These options omit checkpoints, raw predictions, logs, and process-ID files. Source and compact metadata can be audited in the ZIP, but regenerating optional analyses requires recreating the selected raw artifacts and, for server-gated launchers, their queue provenance.

The public Git remote identifies the authors. For double-blind review, upload this ZIP as supplementary material and refer to that anonymous supplement in the manuscript, without a public-repository link. Rebuild the ZIP after any source or result change.
