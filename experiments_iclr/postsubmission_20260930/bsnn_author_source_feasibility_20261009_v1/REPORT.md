# BSNN author-source feasibility for complete Tolokers roles

## Practical decision

**An authentic author-source BSNN baseline is feasible to prepare from the retained Apache-2.0 code. The first unmodified family to qualify is `BayesBundleSheafDiffusion` with stalk size 2.** It has the native Cayley sampler and closed-form KL. Its input is the full graph feature tensor; no author dataset loader or full label vector is needed by the model.

A small separate TRAIN/VALID runner is still required. Reuse the existing official Tolokers role contract, RNG/checkpoint machinery and ordinary process custody, then call this original class directly. Keep its sampling, KL reduction, annealing, sparse normalization, feature maps and classifier unchanged. Set serving to four fresh full-model draws and average probabilities. This is a declared Tolokers protocol around author model code, not replication of the authors' WebKB/Cora benchmark procedure. No runner adaptation, import, fit or qualification was performed here.

**Do not silently call the retained Gaussian/general branch an internally consistent standard ELBO implementation.** Both author repositories sample with `std=exp(a)` while their normal KL treats `a` as log variance. The sampler therefore has variance `exp(2a)` but the KL formula uses `exp(a)`. The issue is source-visible, not a numerical failure observed here. A corrected Gaussian branch would be a separately declared adaptation. The Cayley baseline avoids this particular inconsistency; it is not substituted for a Gaussian/general-map comparison without naming the family.

## 1. Actual paper link, license and versions

Appendix D of [Bayesian Sheaf Neural Networks, 2410.09590v1](https://arxiv.org/html/2410.09590v1) links [patrick-gillespie/bsnn](https://github.com/patrick-gillespie/bsnn). Retained identity:

- Commit: `73b15dd1e7bb737d54dfe222b098a7618619cbb9`.
- Git tree: `97ceb00fedc3c58f8f802cacd516251155d53335`.
- Actual `LICENSE`: Apache License 2.0; original copyright/modification headers preserved.
- Every retained file matches the pinned Git tree's blob SHA1 and its recorded SHA256. No repository split NPZ, data, checkpoint, figure or outcome payload was fetched.

The current original README explicitly redirects to [Layal-Bou-Hamdan/BSNN](https://github.com/Layal-Bou-Hamdan/BSNN). A bounded check retained its relevant core, driver, configuration and license files at commit `d2f8cb2f9577b675c13216c6f7d2786d774ddbaf`. It also has Apache-2.0 licensing. Exact same blobs were reused where possible. Its Gaussian sampler/KL inconsistency is unchanged. Its driver adds terminal-state saving; it still does not save and restore the validation-selected state. Its fold loop uses fixed split seed 0. We recommend the paper-linked source as the reference identity; the updated fork is preserved as a distinct inspected version, not conflated with it.

The preceding closest-prior packet and its manifest remain unchanged. This report adds source-level qualification to that paper-method assessment.

## 2. Exact network and sheaf ownership

The author classes are `BayesDiagSheafDiffusion`, `BayesBundleSheafDiffusion` and `BayesGeneralSheafDiffusion` in `models/bayes_disc_models.py`. One constructed model owns one feature lift, per-layer feature/stalk matrices, epsilon vectors, a classifier and one sheaf distribution learner. Repeated predictive draws reuse those parameter objects; they are sampled geometry paths, not separately optimized persistent model banks.

The nonsparse learner contains two bias-free Linear objects, `lin_mean` and `lin_var`, applied to concatenated ordered endpoint features. The sparse variant sums stalk blocks before those maps. Distribution parameters are computed once per full forward and reused to draw one independent Laplacian for each layer before propagation. They are not recomputed from the changing post-propagation states at each layer.

- **General/diagonal:** `lin1` and optional `lin12` produce the initial hidden state used by both the distribution learner and the prediction path. Native input/layer dropout can affect those distribution parameters during training. There is no separate map-feature MLP here.
- **Cayley/bundle:** a separate `lin_maps` plus ELU produces the map features from the full input. Optional degree concatenation and a learned scalar edge-weight path exist. The prediction path has its own `lin1`/optional `lin12`. There is one common sheaf learner across layers, not member-private learners.
- **All three:** every sampled Laplacian acts through the original sparse propagation and classifier. `model.eval()` disables dropout but does not disable sheaf sampling. `linear=True` is not a switch to a deterministic mean sheaf in these Bayesian forwards.

`grouped_parameters()` places parameter names containing `sheaf_learner` in the sheaf-decay group and all other parameters in the other group. Thus `lin_maps`, the edge-weight learner, feature matrices, epsilon and classifier are in the other group. Use the actual grouping; do not rename all geometry-related parameters into the sheaf group.

## 3. Distributions, KL and training schedule

| Family | Native sampling and objective behavior | Practical boundary |
| --- | --- | --- |
| General / diagonal | Means are tanh-bounded; `a=-ELU(-raw)`; code passes `exp(a)` as standard deviation to `mean + std * torch.normal(0,1,...)`. Normal KL is `mean_incidence sum_coordinates .5*(mean²+exp(a)-1-a)`. | The sampler/KL variance semantics disagree. No correction is made in this packet. |
| Cayley bundle | Mean parameters are tanh-bounded; `gamma=.95*sigmoid(raw)+.025`. SciPy `special_ortho_group.rvs` supplies CPU uniform SO(d) draws, converted to float32 on the target device. Nested native Cayley transforms and the native mean orthogonal map produce each incidence map. | NumPy/SciPy randomness and CPU sampling/transfers must be retained and charged. Stalk2 is a practical initial scope. |
| Cayley KL | `kappa=(1-gamma)/(1+gamma)`; at d2 the code returns `mean[-log(1-kappa²)]`. d3 has another closed form; larger d estimates KL through additional SciPy draws. | Start with d2 to avoid the higher-dimensional MC KL path. This is an explicit family choice, not evidence of competence yet. |

The native training function performs one stochastic full-model forward per epoch. `F.nll_loss` averages over TRAIN labels; if `use_kl=True`, the loss is `NLL + beta(epoch)*KL`. The exact author schedule is `beta=sigmoid((epoch % 40)/2 - 10)` for the zero-based epoch. KL is computed once per model call, averaged over incidences, and is not multiplied by the number of layers. Do not change its reduction, divide it by TRAIN size, or silently add a layer factor.

Adam has separate sheaf/other weight-decay groups with one common learning rate. Parser defaults are lr .01, regular decay .0005, sheaf decay inherited from regular decay, 1500 epochs and patience 200. The retained WebKB Bayesian sweeps specify 500 epochs, lr .01, patience 200, `use_kl=True`, three predictive draws, and varying stalk/layer/channel/dropout settings. These are configuration provenance, not a winning Tolokers configuration.

The Cayley builder retains paired incidence assembly, scalar weighted degree normalization with augmentation, native Householder/Cayley maps and `torch_sparse.spmm`. General maps use native block-degree SVD normalization, training jitter and clamping. The primary feasibility baseline must not replace these with a scalar shortcut or a different diffusion operator.

## 4. Author driver versus the required role/selection contract

The model constructor requires only `edge_index` and an argument dictionary; forward requires only `x`. The author `utils.heterophilic.get_dataset()` rejects Tolokers. Its author driver is unsuitable for the current role contract:

1. It loads repository split files and can permute TRAIN/VALID/TEST roles. The retained Bayesian sweeps set `permute_masks=True`; the loader then trains on the old validation mask, validates on the old test mask and tests on the old train mask.
2. Every epoch scores TRAIN, VALID and TEST; continuation can depend on the selected TEST accuracy.
3. Selection tracks best validation accuracy or NLL, but the original driver never saves/restores the selected model. The updated driver optionally saves the terminal model state.
4. Neither driver supplies complete binary Tolokers AUROC selection and selected four-draw serving reconstruction.

Do not call either driver with fake masks or a fabricated full `y` vector. A separate runner can index the returned native log probabilities with the real TRAIN and VALID indices and their corresponding labels, so TEST truth never enters the process.

Reuse the existing sealed official role contract: exactly `x`, `edge_index`, `train_index`, `train_y`, `valid_index`, `valid_y`, with pickle disabled and archive/array/source/role metadata hashes. Full float32 official features have shape `[11758,10]`; support is the complete canonical paired graph, with actual counts bound from role metadata rather than guessed. Validate sorted unique nonloop support, exact reverses, disjoint full official role indices and the two binary classes. This is the existing raw/PyG-aligned support convention; no author split file or automatic dataset downloader is required. Feed the already exported features as supplied and declare that choice; do not silently inherit the WebKB `NormalizeFeatures()` transform.

Ordinary Tolokers was one of the original benchmark tasks. This is an exploratory baseline comparison, not a claim of unused roles or fresh dataset confirmation.

## 5. Four-draw validation and serving

Set the declared predictive draw count to **4**, rather than the parser default1 or published sweep3. Perform four fresh `model.eval()`/no-grad full-model forwards on the same parameter bank, exponentiate the returned log probabilities, and take their arithmetic mean. Use the positive-class pooled probability for full VALID AUROC and pooled log probability for VALID NLL. Do not average logits, posterior map parameters, or four independently trained weight banks.

Only the evaluation draw schedule changes; the initial baseline retains the native one-draw training estimator. Four-draw training would be a separately labeled MC-budget control. Each serving draw resamples every layer's incidence maps; `num_ensemble` controls repeated complete model calls, not one sheaf reused across the four draws.

Reuse the existing evaluation-RNG isolation pattern, extending/qualifying it for SciPy: save and restore Python, NumPy, Torch CPU and all owned CUDA RNG states, use a prospective seed/epoch/draw schedule, and keep evaluation consumption out of training. Freeze the selection rule to full VALID AUROC, then lower VALID NLL, then the earlier epoch for exact ties. Save selected state/identity and the evaluation replay state/schedule. Reconstruct a fresh original model, strictly load its selected state, verify parameter/buffer equality, and replay the same four draws for selected-serving equality. Fresh independent held-out draws may later assess MC variability, but cannot replace selected-serving reconstruction or add hidden inference draws.

This is an explicit four-draw Tolokers validation protocol. It is not the authors' original three-draw/accuracy-selected benchmark procedure, and no numeric replay has been established here.

## 6. Existing Torch2.1.2 runtime and minimum runner work

The saved allocation runtime metadata reports Torch `2.1.2+cu118`, NumPy `1.26.4`, PyG `2.7.0`, sparse `0.6.18+pt21cu118`, scatter `2.1.2+pt21cu118`, householder `1.0.1` and sklearn `1.5.2`. The real householder module is in the existing private-sheaf dependency overlay. This was a metadata read, not a new server probe or import. SciPy is additionally required by the BSNN core; its installed version/provider still needs an authorized software check. The author environment pins Python3.9.9/Torch1.11/PyG2.0.4/CUDA10.2, so numerical compatibility with the existing runtime remains unqualified. Do not reinstall or downgrade to that old environment for this source task.

The selected import chain is the native Bayesian class, its sheaf learner/base, Bayesian builder, KL, `orthogonal_v2`, and `lib.laplace`. It genuinely imports SciPy and householder even for some other branches. A separate model-only runner avoids the author driver dependencies on W&B, GitPython, tqdm and continuous ODE machinery. It needs no W&B account, dataset download, remote service or fake dependency shim.

Minimum work to implement next:

1. **Private fresh process/namespace:** load the pinned author tree with verified module origins and the existing root-owned runtime; avoid collisions with another native NSD `models`/`lib` namespace. Construct one original Cayley model, not four deepcopied models.
2. **Explicit arguments:** use the complete native argument contract saved in `MODEL_ARGUMENT_CONTRACT.json`; set graph_size11758/input_dim10/output_dim2 from the official task metadata. Keep left/right weights enabled: the Bayesian loop indexes both module lists before entering the helper, so disabling them is not a supported shortcut. No PE, mask permutation or graph mutation is needed.
3. **CPU topology placement:** construct original index preprocessing on CPU to avoid millions of per-edge CUDA `.item()` calls, then transfer actual plain static tensors before the first forward. Extend the existing closed placement pattern to `weight_learner.full_left_right_idx` as well as model/builder graph/index/degree/time tensors. Preserve exact caller ordering, model/builder graph identity, native arithmetic and stochastic sampling. Do not use the broken/unused `update_edge_index` path. Qualify index/numerical parity rather than silently assuming the existing general-NSD helper covers BSNN.
4. **TRAIN/VALID loop and snapshots:** preserve the native one-draw NLL+KL/Adam/40-epoch schedule; use exact official role labels only; add the declared full-VALID selector, four-draw replay, strict selected-state restore and retained failures/costs.
5. **One bounded numerical qualification before fitting:** real dependency/provider import, original constructor/static placement, full-input TRAIN forward/backward with finite map/KL/output/gradient checks, parameter partition coverage, stochastic sampling under eval, NumPy/SciPy/Torch replay and selected-state reconstruction. This is future authorized work; none occurred here.

`PROPOSED_STARTING_CONFIG.json` supplies one disabled width64 d2/L2 source-supported starting point and a small bounded competence-screen proposal. Its Tolokers choices are ours, not extracted author winning parameters. Use full validation for competence; weak defaults do not establish a competitive BSNN baseline. Preserve all failures and amend only prospectively if a competent baseline cannot be obtained within the declared budget.

Every training/evaluation path includes incidence sampling, CPU SciPy work and transfer, mean-map transformation, block assembly/normalization and sparse propagation. Four-draw serving repeats all that graph work. The forward constructs a list of L sampled sparse operators before propagation; include that peak memory. Charge qualification, constructor/index work, actual train and validation draws, backwards, optimizer state, snapshots, reconstruction and complete wall/CPU/RSS/CUDA costs. No cost advantage or full-task feasibility is inferred from parameter count or these static checks.

## Limits and evidence

The exact author code, licenses, commits, blob/SHA hashes, scoped source passages and a small source diff to the author-recommended fork are retained. AST parsing and hashes establish source custody; they do not establish distributions numerically, compiled-extension compatibility, full Tolokers memory fit, competence or selected-serving replay. The Gaussian issue and source-versus-runner distinctions remain explicit. No source adaptation, research host contact, numerical package/model import, dataset/array/checkpoint/outcome access or prior-gap edits occurred.
