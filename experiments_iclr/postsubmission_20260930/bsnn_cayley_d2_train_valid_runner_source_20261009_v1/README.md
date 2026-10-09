# Inactive original-author BSNN Cayley d2 TRAIN/VALID wrapper

This separate source packet implements one fixed full-Tolokers starting configuration and the three paired seeds **1103, 2207, 3301**. Execution is disabled. No model import, role-array read, numerical qualification, fit, outcome access or research-host action was performed while preparing it.

## Source and scope

The model is the unmodified `models.bayes_disc_models.BayesBundleSheafDiffusion` from the original paper-linked [author repository](https://github.com/patrick-gillespie/bsnn), commit `73b15dd1e7bb737d54dfe222b098a7618619cbb9`, tree `97ceb00fedc3c58f8f802cacd516251155d53335`, under Apache-2.0. `SOURCE_BINDINGS.json` pins the retained original author files and license, plus the actual `private_sheaf_train_valid_runner_20261009_v2` helpers and source seal. The updated author repository is not used.

The configuration is our declared starting point, not an author winning Tolokers configuration: d2, hidden_channels32 (width64), two layers, no LP/HP, input_dropout0, dropout0.3, second_linear false, Householder mean-map parameterization, learned edge weights, both left and right weights. There is one parameter bank and one configuration; no architecture search or independent learned-member ensemble is implemented here.

## Roles, TRAIN and selection

The exact V2 `common.read_roles` accepts only `x`, `edge_index`, `train_index`, `train_y`, `valid_index`, `valid_y`. It validates the official float32 Tolokers features [11758,10], complete canonical row-sorted unique nonloop reverse-paired directed support, actual support counts, official split0 of ten, disjoint complete TRAIN/VALID rows and both binary classes. No full y or TEST truth is accepted. Ordinary Tolokers remains an original-paper-benchmark exploratory experiment; no fresh-dataset or unused-split claim is made.

Every epoch uses one full native call and mean NLL over all TRAIN indices plus the returned native KL. The original zero-based 40-epoch schedule is preserved as `sigmoid(((epoch-1)%40)/2-10)` for this wrapper's one-based epoch log. The d2 native KL is an incidence mean computed once per full call, with no extra layer multiplier. Original `grouped_parameters()` supplies the two Adam groups; lr0.01 and both decays0.0005, max500 epochs and patience200. No clipping, scheduler or objective rewrite is added.

VALID and serving each make exactly four fresh full native calls, average their probabilities and score the log of that mean with the unchanged V2 `metrics`. Each call samples a new independent sheaf at each native layer even in eval mode. Four calls are four draws of the same model. The exact V2 checkpoint selector is VALID AUROC, then lower VALID NLL, then earlier epoch. TRAIN metrics are diagnostic only; TEST is absent.

## Placement, streams and selected state

The original constructor and static graph index preprocessing run on CPU with original float32 initialization, followed by explicit transfer of model/builder plain tensors and `EdgeWeightLearner.full_left_right_idx`. Parameter/buffer transfer uses ordinary `model.to`. No forward, Cayley map, orthogonal map, normalization, sparse propagation or KL implementation is replaced. The topology is static and the unused native graph-update path is never called. Root must qualify index and numerical/gradient behavior of this placement on the full input before release.

Each fit owns Python, NumPy, Torch CPU, all visible CUDA streams and the SciPy frozen SO sampler's `random_state`. Construction continues the initialized Torch stream into TRAIN. Evaluation snapshots those streams, seeds `base_seed+2000003+epoch`, makes four calls and restores the saved training streams exactly. NumPy RandomState and Generator sampler states are handled explicitly; the actual SciPy API/provider and RNG ownership remain unexecuted qualification requirements.

The selected checkpoint is created only in that fresh attempt. It stores exact model and optimizer state, all training streams, selected TRAIN/VALID outputs and metrics, evaluation seed/draw count and identity. It is loaded with `weights_only=True` into a fresh native constructor and optimizer. Exact model/optimizer and stream restoration are checked. A fresh four-draw selected-state replay supplies reported metrics. Log-probability/probability differences, prediction-change counts and signed/absolute metric differences are recorded; no floating output bitwise requirement or drift threshold is imposed, and no 1e-6 or repeat campaign is performed.

## Records and external custody

All three scheduled seeds must complete before any comparison can open. A failure cannot be replaced or omitted. Per-fit RESULT, partial HISTORY, selected checkpoints, failure records, ALL_FITS and COMPLETE preserve completed work and failures. Costs include construction, every TRAIN/VALID/reconstruction call, checkpoint work, wall time, CPU time, cumulative process RSS and CUDA allocation/reservation peaks. Sample counts are inferred from completed full calls times native layers; they are not hook-observed counts of failed partial calls. CUDA cost or final cleanup failures prevent completion. Host role arrays remain resident, so RSS is a cumulative process high-water value.

This wrapper has bounded epochs/patience but does not implement a subprocess supervisor or process/resource limits. Ordinary process custody, runtime/provider qualification, interruption/termination handling and resource limits remain root-owned. A hard process kill cannot run Python finalization; the external custodian must retain the last partial records and terminal/resource evidence.

## Release and verification

`RELEASE_TEMPLATE_DISABLED.json` is disabled and its bindings are placeholders. Root release requires the exact `SEAL.json` file hash, the committed execution-source identity, exact role archive/ROLE metadata hashes, a fresh normal-phase output directory, one visible cuda:0, all eight dependency versions, an explicit deterministic policy, runtime qualification, full native BSNN work qualification and baseline configuration review. The existing Torch2.1.2+cu118/NumPy1.26.4 runtime is required. Runtime metadata versions do not by themselves qualify an imported provider.

`static_verify.py` parses source and checks hashes only. Its report verifies the selector AST against actual V2, the author annealing expression with the one-based offset, fixed seeds/config, four-draw probability pooling, deferred numerical imports, source/license pins and preservation of the three earlier immutable packets. It does not import `runner.py`, `support.py`, the author model, numerical libraries or data.

Preparation stops at this sealed inactive source. Root qualification and configuration review remain release blockers; this packet makes no learning, accuracy, cost-performance or restored-output parity claim.
