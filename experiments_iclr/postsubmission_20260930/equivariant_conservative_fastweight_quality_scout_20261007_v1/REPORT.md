# Equivariant conservative fastweight committees: quality scout

Date: 2026-10-07. Literature and static source only. No data, split arrays, checkpoints, models, scientific servers, training, imports of author code, or GPU execution. Existing research packets and the literature index are unchanged.

## Decision

**Reject the proposed construction as a defensible new molecular learning method on the evidence available.** A jointly learned shared body with member-specific multiplicity-channel factors is a valid BatchEnsemble/diagonal-adapter construction on an equivariant energy model. The symmetry and conservation restrictions, jointly learned body, absence of pretrained teachers, and private nonlinear trajectories do not establish a new operation. Prediction-quality superiority is untested.

A narrowly stated empirical question remains: do restricted multiplicative adapters give a better prediction-error/resource frontier than shared energy heads and jointly trained additive adapters on a capable molecular potential? That is a parameterization and optimization comparison. This packet supplies a prospective benchmark recipe, not a recommendation to adopt, train, or claim novelty.

## What was reused

The saved round-05 atomistic report already inspected C-NNP, MACE multi-head committees, P-MLIP, and RACE. The round-09 equivariant report already inspected Bayesian NequIP, BLoB, e²IP and reused ELoRA/BLIPs. Their completed method scopes were not reopened. The saved factor-expressivity conclusion establishes exact diagonal-adapter equivalence. The shallow-SchNet and shallow-ensemble training reports are also reused as inexpensive shared-body controls.

There are **zero newly read ensemble methods** here. PaiNN and MACE have new, bounded **benchmark/setup scopes**; source configurations and loss/checkpoint routines were inspected statically. Retrieval or a metadata search hit is not a primary method read.

## Overlap and the exact remaining uncertainty

| Verified prior, reused from saved reports | Relevant overlap | Limit |
|---|---|---|
| BatchEnsemble / exact diagonal-adapter derivation | Shared trainable weights and private two-sided multiplicative maps; no independent-teacher requirement. | Applying the operation to legal molecular channels is a domain restriction. |
| ELoRA, ICML 2025 | Equivariant tensor-product/path and self-interaction additive adapters. | Sufficient-rank adapters subsume the permitted effective updates. Frozen-base adaptation is not the only relevant control: fit additive adapters jointly with the same shared base. |
| MACE multi-head committees, arXiv:2508.09907v2 | Common learned descriptor bank, private scalar-energy heads, gradient forces. | Internal factors allow private body trajectories, but cost more than heads. Prior foundation-model use does not make pretrained teachers necessary for the ordinary shared-head control. |
| P-MLIP, arXiv:2605.19939v2 | One jointly trained backbone with private nonlinear trajectories through learned functional perturbations. | It requires full sampled backbone passes. Saved scope does not establish conservation for every wrapped architecture. |
| C-NNP; shallow SchNet ensembles | Shared descriptors/body with multiple potential/energy heads and committee prediction quality. | Cannot infer quality on PaiNN/MACE from a different descriptor or task. |
| Bayesian NequIP, BLIPs, RACE, e²IP | Conservative mean predictions, posterior/ensemble or evidential alternatives. | Uncertainty success is not evidence of lower point-prediction error; distributional symmetry is not necessarily symmetry of every sampled potential. |

For a channel map of type `(l,p)`, the proposed map is

`W_m = diag(s_m) W diag(r_m)`.

Each diagonal acts on **multiplicity** and is repeated across all `2l+1` magnetic components. With `A_s=diag(s_m)-I` and `A_r=diag(r_m)-I`, this is exactly `(I+A_s) W (I+A_r)`: two diagonal residual activation adapters. The equivalence persists when W is learned jointly; matching initialization, regularization and optimizer coordinates matters for comparing training trajectories. The outer-product mask is rank one, but `W_m-W` need not be low rank. These facts were already established in saved work.

Every member must produce invariant scalar energy `E_m(R,Z)` and force `F_m=-grad_R E_m`. Fixed-weight averaging is conservative. Factors cannot distinguish magnetic components, add non-scalar biases, introduce external axes, or vary across exchangeable atoms. Geometry-dependent ensemble weights require their derivatives. The same member realization must be retained during differentiation. Parameter sharing does not remove private body activation storage or member force differentiation; a batched implementation is not one body's compute.

A bounded new search did **not verify an exact published PaiNN/NequIP/MACE multiplicity-factor ensemble**. That statement is limited: four DuckDuckGo requests returned challenges, four Bing RSS requests returned irrelevant results, Crossref was noisy, and arXiv exact BE intersections returned zero while broader searches returned mainly unrelated mathematics or existing/uncertainty leads. Those failures and truncated broader result pages do not prove absence. No global novelty clearance is issued.

## Primary quality evidence

The [PaiNN paper](https://arxiv.org/html/2102.03150v2), §V.1 and Appendix B, uses QM9 110,000 training molecules, 10,000 validation molecules and the remaining characterized data for test, averaging three random splits. Its reported U0 MAE is 5.85 meV. The paper uses width 128, squared loss, batch 100, LR 5e-4, 5 Å cutoff, plateau decay0.5 with patience 5 and stopping patience 30; validation loss smoothing is 0.9. This is a competent scalar-property baseline, not a current state-of-the-art certificate. QM9 U0 has no force labels in this setup and cannot verify conservative-force performance.

The [MACE paper](https://arxiv.org/html/2206.07697v2), §5.3.1 and Appendices A.2.1/A.5, uses rMD17 official train/test pairs: 950 training, 50 validation and 1,000 test configurations per molecule. Five pairs exist. Reported MACE/PaiNN energy and force MAEs illustrate the relevant quality bar:

| Molecule | MACE E / F | Cited PaiNN E / F |
|---|---:|---:|
| Ethanol | 0.4 meV / 2.1 meV Å⁻¹ | 2.7 meV / 10.0 meV Å⁻¹ |
| Aspirin | 2.2 meV / 6.6 meV Å⁻¹ | 6.9 meV / 16.1 meV Å⁻¹ |

These are author-reported reference numbers, not reproduced outcomes or guaranteed thresholds for a new implementation. A factor ensemble improving a weak PaiNN implementation while remaining worse than a competent MACE single would not demonstrate practical superiority.

## Prospective small quality screen

Use **rMD17 ethanol and aspirin**, one model per molecule. Fix official pair 01 for an initial bounded screen, reserving 950/50 within its 1,000 known configurations and keeping the official 1,000 test configurations. The SchNetPack loader names this `split_id=0`; IDs 0–4 select files 01–05. It consumes CSV values without subtracting one; the actual index arrays were not acquired or checked. Freeze the validation subdivision and three initialization seeds before any fitting. One official split is only a screen, not evidence of robust superiority across split variation.

### Fixed baseline recipes and explicit adaptations

| Item | PaiNN screening recipe | Capable MACE quality reference |
|---|---|---|
| Source | SchNetPack `ab4314d6a328d5f2b4d7a72d5097e3a2e0fef61f`, stored ethanol config | MACE paper A.5.1 plus historical source `d569918828465361fa0e07f089f9f25b070eab32` |
| Architecture | 128 channels,3 interactions; distinct layer parameters/filters;20 Gaussian RBF; cosine cutoff5 Å; sum atomic energy; gradient forces | 2 interactions;256 multiplicity channels with hidden L≤2; spherical harmonics ℓ≤3; correlation 3;8 Bessel functions; cutoff envelope p5; cutoff5 Å; radial MLP [64,64,64,1024]; last readout 16 scalar units |
| Training units/loss | kcal/mol and kcal/mol/Å; `0.01 MSE(E)+0.99 MSE(F)` | eV and eV/Å; total-energy MSE weight 1, force-component MSE weight 1000, exactly declaring the paper's Eq.(15) reductions |
| Optimizer | AdamW, LR 1e-3, weight decay 0; batch 10 | AMSGrad Adam, LR 0.01, batch 5, betas 0.9/0.999, epsilon 1e-8; source parameter-group decay 5e-7 |
| Schedule | cap 1,000 epochs; plateau factor 0.5/patience 75/cooldown 10; early stopping 200; EMA 0.995; best validation loss each epoch | cap 2,048 epochs from historical CLI; plateau factor 0.8/patience 50; EMA 0.99; no SWA; validate each epoch and select best EMA validation loss; gradient clip 10 from source |
| Offsets/scales | training mean energy removal/restoration, center-of-mass translation; float32 body | training-only per-atom energy offset and force-component RMS scale; average-neighbor normalization from TRAIN; float64 |

The PaiNN stored example actually uses **900/100, `split_id=null`, random splitting**. The proposed 950/50 official-split version is an explicit adaptation. It is not the saved example's experiment or a verified reproduction. The 1,000-epoch cap is source-backed, but convergence on aspirin is unqualified.

The MACE paper does not supply its complete stopping horizon. The 2,048 cap, clipping and source parameter groups are explicit source-backed prospective choices, not certified paper settings. The historical CLI defaults to per-atom **weighted** energy MSE, which differs from the paper's displayed total-energy objective; its `loss=ef` route matches the latter. Its scheduler is stepped every epoch, so the prospective validation-every-epoch choice avoids counting stale validation values. Its source stores improved EMA checkpoints, then loads the latest saved improvement. A future port must bind the displayed loss, scale/offset policy, precision, tensor-product paths, and selected checkpoint. This packet does not certify runtime or exact paper-code correspondence.

### Quality controls and fair interpretation

On the same chosen backbone, compare a capable single; four jointly learned energy heads; four jointly trained diagonal-factor members; four jointly trained additive/path-adapter members with the same trainable body; and four independently fitted full models. Independent models are **quality references**, not teachers or a pretraining dependency. Head-only and factor models must start from scratch under the same data and member-loss contract. Add a larger single only when making a matched-resource claim.

Use fixed uniform energy means and their gradient forces. Fit memberwise energy/force MSE with declared reductions; pooling-only supervision is a different method. Select one synchronized family checkpoint by validation loss of its deployed mean and apply the same selection rule to independent families. Report ensemble and per-member E/F MAE/RMSE, seed variation and complete train/inference time, parameter/optimizer/activation memory. Uncertainty metrics are secondary. Compare measured frontiers; factor sharing chiefly saves parameter storage and does not justify a promised 4× serving speedup.

Keep the initial comparison fixed, with zero additional hyperparameter configurations per family beyond the chosen competent recipes. A concrete convergence failure can justify a separately recorded, equally allocated repair budget; test outcomes must not trigger recipe changes. Before claiming generality, evaluate remaining official split pairs and more molecules under a prospectively frozen continuation. No such continuation is authorized by this note.

An optional independent scalar check is full characterized QM9 U0 with the paper's 110k/10k/rest split, three fixed split seeds, native atom-reference/mean offsets, and paper learning-rate/stopping recipe. Do not call a small subsample of QM9 a competent published benchmark or reuse MD17 results as rMD17 evidence.

## Scientific boundary

This packet identifies ancestry and recipe traps. It does not establish factor benefit, independence of members, calibrated uncertainty, molecular dynamics stability, historical novelty, data correctness, implementation compatibility, or an admitted successor. The broad method route is rejected; only a plainly labeled empirical quality/resource hypothesis survives.

`READ_SCOPES.json`, `REUSED_CONCLUSIONS.json`, repository bindings, retrieval receipts and `MANIFEST.json` preserve the new scope and custody.
