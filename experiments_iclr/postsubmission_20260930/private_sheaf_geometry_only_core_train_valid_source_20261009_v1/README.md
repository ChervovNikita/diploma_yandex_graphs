# Inactive geometry-only shared NSD core and independent controls

This source prepares the regularizer-off core for **one architecture frozen by root after the complete native15 screen**. It implements three paired seeds1103/2207/3301, a shared geometry-only M4 bank and four genuinely independent native references per seed. It does not choose or freeze an architecture. No model/provider import, numerical/data/outcome/server action or installation was performed during preparation. No prior packet, running screen, root pointer or ledger is changed.

## Existing machinery inspected and reused

The original licensed NSD is `DiscreteGeneralSheafDiffusion`, Twitter repository commit `11e21b561d884713ab1a18a521a7dc2fb26b9361`. `private_sheaf_native_source_design_20261009_v1` was inspected first. Its existing `SharedNativeBank` attaches factors to every Linear, which exceeds this new geometry-only scope, so that class is not instantiated. The new bank reuses its `_shared_copy_memo`, `_module_at`, `_replace` and `SharedBELinear`, with replacement paths restricted to `sheaf_learners[layer].linear1`.

The actual V2 role loader, native placement/factory, metrics, selector expression, RNG helpers and independent `fit_one` are reused. Original propagation, reverse-edge pairing, full block-degree/SVD normalization, training jitter, augmentation/clamping, activations, sparse multiplication and residual epsilon updates remain in the author source. There is no scalar-degree substitute, extra readout, graph rewrite or context-objective implementation.

## Parameter identity and private state paths

One fresh original native prototype is constructed under the base seed. Three complete model/builder modules are deep-copied using the existing memo, which keeps every original learned parameter and immutable topology tensor shared by identity. Slow weights/biases of the stem, optional second input map, feature maps, classifier, residual epsilons and each layer's incidence map are shared across all four members. Slow parameters remain layer-specific as in native NSD; sharing is across members.

Only each ordered-incidence bias-free `Linear(2c,d²)` is wrapped with private input `r` and output `s`. Each member/layer uses `(W_l (r_ml * concat(x_row,x_col))) * s_ml`, followed by the original tanh and matrix reshape. The original native incidence bias is absent and checked. Existing biased body/head affine maps stay untouched and their native biases are shared; no private feature/head/epsilon factors are added.

Every r/s starts at one, with separate objects/storage and no RNG draw. Shared W/b/epsilon retain exact original initialization. At construction and reconstruction, runtime assertions verify all original slow parameter identities, disjoint factor storage, exact parameter coverage, four separate complete modules/builders, immutable common topology and no extra private heads/features. Active parameters are the original native count plus `4*layers*(2c+d²)` fast factors, checked against actual tensors. All four native states/operators/diagnostic caches are recomputed privately. No common hidden activation or map cache substitutes for a member forward.

Unit factors preserve native affine algebra but do not establish numerical parity or useful diversity. Native dropout and CPU SVD jitter consume different draws in fixed member order0,1,2,3. Identical or weak effective geometry remains a possible measured result. No learned-diversity claim is made here.

## Regularizer-off TRAIN and RNG

Shared TRAIN is precisely `F = (NLL_0+NLL_1+NLL_2+NLL_3)/4`, each over every official TRAIN row. The bank optimizer is zeroed once. A complete member forward and `NLL_m/4` backward are streamed for each member; its graph is released before the next. All parameter versions must remain unchanged through all four backwards. Finite accumulated gradients are checked, then **one** deduplicated Adam step occurs. No parameter update is interleaved between members, no pooled NLL replaces own NLL, and no unfinished context/diversity term is called.

“Regularizer off” means the additional context/diversity objective is exactly zero. Original frozen Adam lr, regular decay and sheaf decay remain. Slow incidence W and private r/s are in the original sheaf group, with its frozen sheaf decay; all other shared native parameters use frozen regular decay. There is no fast-factor LR multiplier, extra factor penalty, scheduler or clipping change.

One bank-owned Python/NumPy/Torch CPU/all visible CUDA stream continues from native initialization into TRAIN. Traversal is fixed. Validation snapshots all those streams, seeds base+2000003+epoch, makes one complete call per member, then restores training streams exactly. Serving uses the selected bank and mean class probabilities from the four complete private native paths. Selection is V2 full VALID AUROC, lower VALID NLL on exact ties, then earlier epoch. Individual member TRAIN/VALID metrics accompany pooled scores.

The owned selected checkpoint stores bank/optimizer state, training streams, pooled/member role outputs and scores, epoch and identity. Fresh reconstruction rebuilds the original bank, strictly loads exact parameters/buffers and optimizer, restores streams, rechecks ownership and recomputes four-path serving. V2 practical maximum role logp drift0.001 and absolute AUROC drift0.001 apply; probability/metric differences and prediction changes are recorded. Floating bitwise equality, 1e-6 campaigns and repeat updates are absent. Shared state dictionaries have repeated alias keys; actual checkpoint bytes/cost are recorded, with no compact-checkpoint claim.

## Genuine independent references

Each reference owns a complete fresh original encoder, feature path, incidence weights, epsilon, native classifier and separate Adam optimizer. Each trains its unscaled own NLL with the original V2 `fit_one`; its own full VALID selector gives its own best checkpoint. It is not a private-head control, shared-body model or pooled one-optimizer shortcut. Fresh member seed is `base+1000003*member`; each fit's initialization and training stream are isolated by the existing helpers.

Root may explicitly reuse the **exact original native member0** already fitted under the base seed, but only after the frozen receipt passes. Reuse verifies complete native family/config/seed, source and role identities, runtime/deterministic/device policy, original result/checkpoint/history/runtime file hashes and selected checkpoint identity. Original training/constructor/validation/checkpoint/report costs and histories remain referenced and charged; reused training is not free. Shared-bank initialization never loads a trained native checkpoint.

Members1,2,3 always train fresh full bodies under the offset seeds. With reuse off, member0 is also fresh. After all four individual fits complete, each own best state is rebuilt separately for selected serving. Inference streams **one full independent body at a time**, charges four original constructors and four full native calls, verifies exact parameter state and practical replay diagnostics, then pools probabilities. Each body uses its own selected epoch/evaluation seed; no new ensemble checkpoint selection is performed. The complete independent parameter total and actual streamed residency/cost are reported. Saved optimizer/streams remain part of the original individual checkpoints; this report-only pool does not claim optimizer resume.

## Root freeze, roles and reporting

The disabled freeze template requires root's native15 complete/competence/curve review, the exact SCREEN_SUMMARY hash and selected config, exact V2 source seal, optimizer/selector, original six-key role archive/metadata, runtime and deterministic policy. The actual summary must contain12 complete native fits,3 complete MLP fits, no failures and the provisional native competence gates. Only the single winner is used; no local grid or architecture selector exists. Any later amendment needs a newly declared source/freeze scope.

The unchanged V2 loader accepts only x/edge_index/train_index/train_y/valid_index/valid_y; complete official Tolokers graph/features and all official split0 TRAIN/VALID rows are validated. Actual canonical support counts come from metadata. Full y and TEST truth are rejected. Ordinary Tolokers remains original-paper-benchmark exploratory; the stale unused-task/count language in the earlier design is not adopted.

All18 logical records are required:3 shared fits,12 independent-member records (including any3 reused member0 records),3 independent pools. Failures, interruptions, partial histories/checkpoints, borrowed origins and costs are retained; no survivor summary, replacement or retry exists. Reconstructed pooled/member quality is reported only for a complete panel. Competence of all fresh controls and learning-curve/reference interpretation still require root review; comparative opening is false.

Costs include constructor/index preprocessing, clone work, actual unique parameter/topology storage, native detached transport-cache payload based on actual support counts, every native forward/backward/Adam attempt, checkpoint/restore, CUDA peaks, CPU/wall time and cumulative RSS. Streamed gradients do not remove four persistent native map caches or four-path compute. Host role arrays remain resident. Scoped original reused costs and new execution costs are separate; one cannot substitute for the other.

Root owns existing ordinary subprocess/resource custody and hard-kill partial-record retention. No supervisor, framework, server operation or installation is introduced. The source is unexecuted; actual full-input sharing/gradient/RNG/restore behavior remains for root qualification/review before activation.

## Static evidence

`static_verify.py` only parses source and verifies hashes: incidence-only factors/helper reuse, mean-own-NLL backward placement before Adam, original fit_one protocol contract, selector, required freeze/reuse receipts, source/licenses and all earlier sealed packets. It imports no runner, bank, author model, numerical provider, data or outcome. The public CLI defaults to inactive/unfrozen.
