# Inactive deterministic bundle fairness control

This separate source packet prepares the required three-seed deterministic bundle control: one fixed d2/f32/L2 starting configuration, seeds1103/2207/3301, with the same declared BSNN input/dropout/Householder choices. It is inactive. No model/provider import, data/array/outcome access, numerical/server action or installation was performed while preparing it. The running general-map15 screen and all earlier source packets are untouched.

## Original model and fixed choices

The class is the unmodified original author's `models.disc_models.DiscreteBundleSheafDiffusion`, from commit `73b15dd1e7bb737d54dfe222b098a7618619cbb9`, tree `97ceb00fedc3c58f8f802cacd516251155d53335`, under Apache-2.0. The native constructor, per-layer sheaf/edge-weight learners, intermediate-feature map path, normalization, Householder transform, sparse propagation and forward are retained. No deterministic rewrite of the Bayesian model is used.

Choices match the sealed BSNN start: d2, hidden_channels32 (width64), two layers, no LP/HP, input_dropout0, dropout0.3, second_linear false, Householder maps, learned edge weights, both left/right weights, nonlinear transport and the other declared native arguments. Adam uses original `grouped_parameters()`, lr0.01, regular/sheaf decay0.0005, max500 epochs, patience200, no clipping/scheduler. These are our fixed prospective Tolokers choices, not an author winning configuration. There is no alternative configuration, search or architecture selector.

The original deterministic model returns log-softmax only. Its TRAIN objective is mean NLL over all official TRAIN rows, without KL or annealing. The unchanged V2 `fit_one` performs TRAIN, validation and checkpointing. Deterministic eval disables dropout and uses one full native call for VALID and serving; BSNN uses four sampled calls. That real work difference remains visible.

## Exact V2 role/fit/report reuse

The exact actual V2 `common.read_roles` accepts only `x`, `edge_index`, `train_index`, `train_y`, `valid_index`, `valid_y`; official finite float32 Tolokers [11758,10], complete row-sorted unique nonloop reverse-paired support, actual metadata counts, official split0 of ten, disjoint complete TRAIN/VALID rows and both binary classes are checked. No full y or TEST truth is accepted. Exposure remains `original_paper_benchmark_exploratory`.

The actual sealed V2 `baseline_runner.fit_one` is called directly for each seed. It uses the unchanged V2 metrics and selector: largest VALID AUROC, lower VALID NLL on an exact tie, then earlier epoch. Its seed-owned Python/NumPy/Torch CPU/CUDA training stream continues from construction; evaluation snapshots/restores that stream around `base_seed+2000003+epoch`. There is no native SciPy sampler in this deterministic model.

Selected checkpoints contain model parameters/buffers, optimizer and training streams, TRAIN/VALID selected logp/metrics and identity. V2's report releases the trained model, builds a fresh original constructor, strictly loads the selected model state, checks exact parameters/buffers and recomputes TRAIN/VALID metrics with the recorded evaluation seed. Its practical gross gates are maximum role logp difference0.001 and absolute role AUROC difference0.001; role prediction-change counts are recorded. It does not require bitwise floating outputs or microscopic parity campaigns. This packet uses V2 reporting exactly: optimizer/training streams are saved but not reloaded for report-only reconstruction; optimizer resume is not implemented or claimed.

## Placement and cost extension

V2's original `make_native_placed_factory` performs CPU constructor/index preprocessing, original CPU/float32 parameter checks, explicit model/builder tensor transfer and the final plain-tensor/device and graph-identity guards. The adapter adds only the missing transfer of each original `weight_learners[layer].full_left_right_idx`, after the unchanged constructor and before V2 completes placement. The source forward and index values are unmodified. The original separate per-layer learner counts are checked. The graph remains static; no source graph-update path is invoked.

The private helper module's topology cost counter adds unique per-layer edge-weight index storage to the unchanged V2 model/builder count. Parameters, constructor cost, native saved-transport payload, all forward counts, checkpoint cost, CUDA peaks, complete attempt time and cumulative process RSS come from V2; per-returned-fit and full-panel CPU time are also recorded. Host role arrays remain resident, so RSS is cumulative. This extension changes accounting only. The helper module is private to this execution process; source files and the running general-map screen are unchanged.

Every one of the three scheduled attempts is retained, including failures, partial histories/checkpoints and interruption placeholders. Missing CUDA cost or cleanup evidence blocks completion. A mean selected reconstructed-score summary is emitted only if all three seeds return complete records and the panel has no exception. No survivor mean, retry, hidden replacement, architecture freeze or comparative opening is authorized. Root reviews learning curves and references to interpret competence.

Ordinary subprocess/resource custody is reused from root; no supervisor, framework, training system or server operation is introduced. Hard kills require root to preserve last partial records and external terminal/resource evidence.

## Fairness interpretation

This is the closest authentic deterministic orthogonal/bundle family control. **It is not a pure sampling/KL ablation.** BSNN has a separate `lin_maps` path, one distribution/edge-weight learner computed once per full call, fresh sampled maps at each layer, a concentration head and KL. The original deterministic bundle has separate per-layer learners, derives maps from intermediate prediction features and applies layer-dependent map-feature dropout during TRAIN. Parameter and transport/storage costs differ and are measured rather than forced equal.

The chosen BSNN's Cayley distribution surrounds Householder mean maps. This control keeps `orth="householder"`; using the option `orth="cayley"` would change the mean-map parameterization. General-map NSD and MLP remain broad anchors, while this separate control helps assess the orthogonal/bundle family. It does not replace or amend their ongoing screen.

## Release and static evidence

Root release must bind this exact source seal, committed execution-source identity, same official role archive/metadata hashes, fresh normal-phase output, existing eight provider versions, one visible cuda:0, TF32 disabled and an explicit deterministic policy. Runtime qualification, exposure audit and review of this original class/placement scope are required. Root reported actual SciPy1.14.1 with the existing Torch2.1.2+cu118/NumPy1.26.4 and qualified Householder overlay; that version is reflected in the disabled template without a new probe here.

`static_verify.py` parses source and verifies bytes only: single exact config/three seeds, direct V2 fit/metrics/selector/placement reuse, original author source/Apache license, no copied or adapted forward, and preservation of earlier sealed packets. Preparation stops at the inactive source seal.
