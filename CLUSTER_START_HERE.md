# Rebuttal cluster handoff - owner update, 12 November 2026

Read this file first. It is the authoritative entry point for the updated campaign. Historical plans, queues, packages and Slurm examples are retained as evidence; they are not instructions to launch another campaign.

## First interaction on the destination

Ask the owner these two questions before preparing the production allocation:

1. How many GPUs should this campaign use?
2. What is the maximum number of simultaneous fits per GPU? Suggest **4**.

Then inspect the destination's allocation rules, GPU memory, CPU resources, shared filesystem and Slurm configuration. Create site-specific Slurm integration there. No new Slurm jobs or scripts were prepared on the single-GPU preparation host. Do not assume 100 GPUs or four simultaneous heavyweight fits will fit in memory. The launcher treats concurrency as a maximum; memory and exclusive timing tasks can reduce occupancy.

The owner rejected the approximately 54,370-fit proposal completely. Use only the active plan below. Keep the colleague's staged hyperparameter search unchanged. Add only the two new-task comparisons and fixed historical-setting node contrastive branch. Do not expand the model/task matrix without an owner request.

## Package and offline setup

Transfer the new package directory named in the external HANDOFF.md and transfer_receipt.json. Check SHA256SUMS before extraction. The repository archive expands to gnnm_rebuttal_cluster/. Extract the environment archive into a separate writable prefix, activate that prefix, and run its bin/conda-unpack once. Use Linux x86-64 with a compatible NVIDIA driver; this environment is not a Mac environment. No environment downloads are required.

Set GNNM_ENV to the relocated environment's absolute prefix and source ablation/campaign/environment.sh to keep runtime caches inside the repository. From the unpacked repository, with that Python:

    python -m ablation.campaign.offline check --manifest ablation/campaign/offline_manifest_owner_20261112.json

This verifies source/data/cache hashes, installed package versions and prepared datasets with an audit that rejects network use. Source data and precomputed BUDDY caches are packaged. Official validation and test partitions are preserved. Do not rebuild or download data as a first response to an integrity failure: identify the mismatch.

The preparation GPU was an A100-SXM4-80GB. Check CUDA imports and a bounded smoke on the destination before scientific training. The preparation host cannot certify the destination driver or cross-host filesystem locking.

## Exact active scope

Active manifest: ablation/campaign/plans/owner_20261112.json.
Its adjacent .counts.json gives machine-readable totals.

Maximum: **24,510 training fits**, plus scoring, decisions, profiling and reports; **36,277 total tasks**. Of those fits, **9,270** belong to the conditional expansion and **15,240** are outside that expansion. Validation readiness gates can prevent additional confirmations. These are queue upper bounds, not a requirement to rerun already completed fits in the same immutable campaign root.

| Group | Maximum physical training fits | Scope |
| --- | ---: | --- |
| Retained staged HPO | 18,690 | Five graphs, GAT first, gated expansion to six other backbones, full-search and equal-budget confirmations |
| Additional Roman Empire BASE/ENS/GNNM controls | 1,675 | Controls not already present in GAT pilots |
| Roman Empire requested variants | 3,500 | 10 variants x 7 backbones x 5 depths x 10 splits |
| Selected-depth diversity retraining | 175 | 35 cells, each one four-path GNNM and four independent ENS members |
| Fixed-setting node contrastive addition | 350 | 5 datasets x 7 backbones x 10 splits; GNNM only |
| Added tasks | 120 | 8 physical fits x (10 MolBACE cohorts + 5 BUDDY cohorts) |

The five original datasets are Roman Empire, Amazon Ratings, Minesweeper, Tolokers and Questions. The seven backbones are ResNet, GCN, SAGE, GAT, GAT-sep, GT and GT-sep. TAG, MolHIV, molecular GIN and Collab NCN are not in this active campaign. Their old code and data may remain in the package.

### Retained colleague experiments

- Roman Empire placement: input-only, output-only with original identical factors, output-only with random signs, and both end projectors plus ensemble linears inside convolution blocks. The last variant is not middle-only.
- Roman Empire full ablations: two/eight paths, Xavier-all, ones-all, and new RMS-normalized Xavier-all. RMS normalization matches default factor RMS magnitude; it is not BatchNorm.
- One shared ordinary input projector and backbone, four independent ordinary linear heads, averaged head training losses and inference logits. Input BatchEnsemble is removed in this control.
- Roman coverage is seven backbones, depths 1-5, all ten official splits. Native member zero supplies BASE and is reused in ENS when identical.
- Cost/convergence profiles, short windows at fixed depths for light scaling, initialization diagnostics, depth tables from per-depth score CSVs, and the selected-depth diversity study remain.
- Diversity uses all five graphs x seven backbones, official split 0. ENS and GNNM each retain their own historical validation-selected depth and original training settings. Checkpoints and member outputs are retained; reports include prediction disagreement/error correlation and representation similarities. The active analysis uses VALID nodes and is descriptive, not an independent test of a validation-selected relationship. One missing Tolokers/ResNet/GNNM split-zero selection uses the recorded other-split validation mean fallback in selected_depths.csv. Do not silently reselect using TEST.

### Unchanged staged HPO

GAT pilots: 24 configs x 5 depths x 3 official splits x 5 datasets; five physical fits per cell (four native members and one GNNM). The combined grid includes 16 dropout/attention/head combinations and nine LR/WD pairs with one overlap. LR values are 3e-5, 1e-4, 3e-4; WD values are 0, 1e-4, .01.

Expand only when independently validation-tuned GNNM exceeds independently tuned ENS on at least three of five datasets, the LR/WD pool is feasible, and selected fits satisfy convergence checks. Other backbones use the reference (3e-5,0) plus two pairs ranked from GAT validation across datasets/methods. Select independently per method/dataset/backbone and confirm on splits 3-9.

The separate budget replay uses the same predeclared candidate order and runtime of complete training runs. For each method, only candidates completed within the common tuning budget can supply its validation winner. Confirm those winners separately on splits 3-9. This tests prediction quality under equal tuning compute, not fastest convergence. No second pilot search is needed to replay the budget. Details: ablation/campaign/GRID_SEARCH_DECISION.md.

### Two added tasks: six result arms each

Primary pairs only: **OGBG-MolBACE / GCN** and **OGBL-Collab / BUDDY**.

Six arms: native single, independent four-model ENS, vanilla four-path GNNM, contrastive-only GNNM, private-BN-only GNNM, combined GNNM. Four fresh native members provide BASE and ENS, plus four GNNM fits: eight physical fits per cohort. Ten molecular cohorts and five BUDDY cohorts are the declared campaign design, not ten/five different OGB data splits.

Across each cohort keep data, architecture, ordinary loss, optimizer and checkpoint policy aligned. Different seeds from the authors are accepted; independent ENS members use distinct documented seeds. Full VALID after every epoch; patience starts immediately and stops after **50** consecutive checks without strict improvement. **No minimum 100 epochs.** Preserve counter/optimizer/RNG through the 1,000 -> 2,000 cap continuation. Select best VALID. A reference100 readout exists only when the run actually reaches 100, and is optional for completeness.

Private BN changes route running statistics/counters while retaining shared learned affine parameters. Contrastive-only retains ordinary shared BN. Contrastive settings are fixed: coefficient .05, temperature .2, up to 1,024 anchors and 16 alternatives, same TRAIN object positive across routes, different-class TRAIN objects as negatives, 12 ordered route pairs. Extra gradients affect declared private input-boundary factors only. The source-specific BUDDY same-query sampling remains intact.

Native source/data audit: research_audits/transfer_reference_fidelity_20261112/README.md. Official archives/splits and native-function parity are checked. Exact published predictive-score replication is not certified. In particular, the pinned BUDDY repository calls itself a reimplementation; its Collab defaults (width 1024, LR .02) differ from the paper's stated search range, and the paper's exact repetition count remains unestablished. Describe these facts instead of claiming exact paper-protocol replication.

### Node contrastive transfer on the original five

350 GNNM-only fits. Historical validation-selected depth/config per dataset/backbone, all original ten splits and paired seeds. No new depth/HPO sweep, BASE or ENS fits under this addition.

Frozen references: ablation/p13_node_contrast/historical_baseline.csv and .json. Preserve original LR 3e-5, WD 0, width 512, dropout/attention/head settings and LayerNorm architecture. Preserve original checkpoint/stopping recipe: original selector every ten steps, patience 300 with archived controller semantics, cap 5,000 (20,000 Tolokers), no convergence extension beyond the historical cap. Do not apply the added-task 50-check controller here. Add only the fixed contrastive objective/gradient policy, indexed to TRAIN nodes. Report differences against frozen archived scores, including losses. Do not overwrite the published results.

## Portable manager and independent fits

Every manifest task has an immutable ID, visible argv, dependencies, resource needs, retry policy and output contract. Each fit owns its optimizer/RNG/checkpoint and can run independently when its dependencies are ready. Selection, budget and aggregation tasks wait for their inputs; they are not independent scientific fits.

The generic manager is already implemented: ablation/campaign/launch.py, worker.py, store.py. Use a single identical active plan and a single **new shared campaign root inside the unpacked repository** on all nodes. The local launcher starts one supervisor per assigned GPU and CPU workers. Fit claims use filesystem locks and atomic receipts. Workers dynamically take ready tasks; do not statically give one giant model sweep to one GPU. Across nodes, a common queue spreads ready fits over the chosen GPUs.

The destination Codex must wire this readable entry point into site-specific Slurm allocation/launching after the two owner answers. This is a Python command template for an allocated node, not a Slurm job:

    python -m ablation.campaign.launch --plan ablation/campaign/plans/owner_20261112.json --root ablation/campaign/runs/production_owner_20261112 --gpus auto --jobs-per-gpu 4 --gpu-memory-check required

Replace concurrency with the owner's answer. Configure memory and CPU budgets for the actual assigned hardware. Run --dry-run first. The default 70 GiB budget assumes an 80GB-class GPU and is unsuitable for a 40GB device. Node contrastive tasks reserve 64 GiB conservatively, so many run alone; ordinary fits may share. Timing/profiling tasks run exclusively. Do not lower reservations blindly to force four fits.

Before launch, run ablation.campaign.qualify_filesystem concurrently from multiple allocated hosts against one qualification root. It must pass process locking, atomic replacement, contention and participant checks. Use its --help for arguments. A single-host stress test cannot certify cross-host NFS/parallel-filesystem behavior.

On preemption send SIGTERM to the launcher and allow the configured grace period; children save progress. Restart the identical plan/root to resume. Do not delete claims/state or change source under a running campaign. A new scientific protocol needs a new campaign root. Preserve failed/numerical/capped outcomes. Exhausted caps are reported as unresolved, not convergence. Avoid manual simultaneous launches of the same fit outside the manager.

## Outputs and engineering receipts

Per task: tasks/<id>/output/, plus attempt invocations/stdout/stderr, config, checkpoints, validation curves, best predictions, timings and continuation receipts. Retain ENS's four best checkpoints and GNNM's full checkpoint for diagnostics. TEST is scored by explicit post-selection scoring tasks; pilots never read it.

Main report tasks: report/roman, report/selected_diversity, report/node_contrast, report/transfer and report/final. The final report writes training_steps.csv with actual stop and best-checkpoint counts per physical fit, plus depth_scores.csv retaining individual depths/splits. ENS compute sums its four native member counts; do not call the best checkpoint the total training horizon. The diversity and node-contrast reports write CSV/JSON and separate per-dataset PNGs; no pooled incompatible metrics or causal diversity claims. The final report distinguishes terminal queue execution from scientific completeness and retains all failures/unresolved cases.

Preparation evidence: ablation/campaign/evidence/owner_20261112/ and the reference audit. The engineering smoke plan is owner_smoke_20261112.json; scientific_results=false and heldout_scored=false. Fits are bounded at two node updates or two transfer batches, with full official validation. Never use these scores as scientific results. Regenerate and run a small subset after relocation, not the production plan, to validate destination CUDA/data access.

Do not transfer further or launch the production campaign until the owner gives the next instruction. No production scientific training was launched during preparation.
