# Original boundary GNNM: source locator and three-role contract

12 October 2026. Source/metadata only, grouped with the PT proposal. No imports of numerical/model modules, arrays, fitting, rescoring, remote access or current partial outcomes. Existing source/evidence remains immutable. This is an implementation contract for root, not execution admission.

**A fresh three-bank original-GNNM refit on PT blocks 10103/10427/10949 is the coherent paired anchor.** Re-scoring the old ten original checkpoints cannot supply it: they were trained/selected on different role masks. The original arm is `original_boundary_shared4`, with native boundary R/S/B and the LN/GELU/residual SAGE architecture. The meanSAGE S4 control is a separate procedure.

## Clean callable locators

| Purpose | Saved source and entry | Preserved receipt |
| --- | --- | --- |
| Completed Physics original acquisition template | `Physics_original_boundary_GNNM_anchor_source_20261011_v1/acquire.py`: `full_input_readiness(spec,root_guard)` and `acquire(spec,root_guard)` | SHA256 `5819fb4a4b816d04f191cb2e896e7c1a7088bc227c51eb2902626d316a24cde9` |
| Exact original constructor/forward | `common_wrapper_native_source_20261010_v1/models.py`: `TABMModel` and `model(data,data.x,tabm_seed=m)` | SHA256 `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303` |
| Native seed setup | `common_wrapper_native_source_20261010_v1/run_common.py`: `set_seed(block)` | SHA256 `e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc` |
| Exact multiclass optimizer step | `native_SAGE_Penn94_confirmation_integrated_source_20261011_v1/native_tabm_training.py`: `train_step_tabm(model,data,optimizer,False)`; caller injects `F=torch.nn.functional`, `TABM_INITS=4` | SHA256 `f70894304850660d611c6bbb873b47ee0a86840019eb3c5d52cacd71f7c19d78` |
| Custody/global-RNG/strict checkpoint helpers | `WikiCS_original_boundary_GNNM_comparator_source_20261011_v1/comparator.py`: `checked`, `global_rng`, `restore_rng`, `fingerprint`, `file_receipt`, `write` | SHA256 `a46b1d66ac22843314ba79e0ef785383d5c6d47187bef94069d89a2b63cc81c8` |
| Source manifest and exact dependencies | `Physics_original_boundary_GNNM_anchor_source_20261011_v1/MANIFEST.json` and `SOURCE_BINDINGS.json` | Manifest SHA256 `3085ae26d633595ca00495ade66ada155ec55a1db6985c2a1db50cdaaa9b4541` |

All six original-package manifest entries and available local runtime sources were checked against their recorded byte counts/hashes. The saved runtime binding also pins native PyG SAGEConv (`db5a4998387eb0ae2ee1cb3d957699938e6c0fd1b0a1c8b6135c691dfe5af7c7`) and MessagePassing (`66e4ef4afa1d1b2d46c805b8987b6ea0c52ec5c95a9beee8a62758358e481e6f`). Those are remote-path receipt authorities, not newly checked remote files.

The existing callable is hardcoded to ten old blocks, the old TRAIN-with-graph schema and ten-bank closure assertions. Create a sealed sibling driver using its model/training/selector/restore loop, changing only the explicit three-role roster, safe input plumbing and export/reader contract. Do not invoke it with a monkeypatched `BLOCKS` value or mutate its preserved bindings.

## Data and role compatibility

Use the already closed peer/GRU role authority: `private_continuation_PT_closed_source_metadata_root_20261012_v1/CLOSED_STAGE1_METADATA.json` binds its CUSTODY receipt, SHA256 `331680470f36e99b27d58160f271237d1ce5e22090feeac6fd77975f46af1cb4`. It is the same role custody used by the PT ordinary source quartets. The actual CUSTODY file/individual receipts remain on the allocation; root must supply that exact existing receipt map to the sibling contract.

The shared graph file is target-free `graph.npz` with dense float32 raw binary `x` of shape `(34493,8415)` and int64 `edge_index` of shape `(2,530417)`, including exactly one self entry per node. Use this dense X and original PyG edge orientation directly; add no loops, row normalization, transposed PT adjacency, incidence transform or H0 cache. Feature/edge digests are respectively `9df20be681bfe193a8565f272f2031f31d73871f09d1f56f24bd11d4d683e945` and `1b57360470b64c50a463ef8335f03cf4377b4d31a3c7af5a9494d103a47a8801`.

Current TRAIN/SELECT files are `block_{block}/train.npz` and `select.npz`, containing only `ids,y`, unlike the old driver's TRAIN-with-graph NPZ. Each block has 100 TRAIN (20/class), 150 SELECT (30/class), 150 CAL and 34,093 DEV rows. Load only TRAIN targets into model-facing `data.y/train_mask`; SELECT targets remain separate. Reuse existing label-free `select_ids.npy`, `cal_ids.npy`, `dev_ids.npy` and their custody hashes. Do not regenerate roles from seeds in the fitting driver. The saved role algorithm is `RandomState(block+5000011+223000019)`, class-wise permutation, but existing exact receipts are the authority. CAL truth stays in the scalar-calibration stage and DEV truth in the whole-family scorer.

## Frozen original acquisition law

- One fresh full model per block; call native `set_seed(block)` once immediately before constructing `TABMModel('SAGE',2,8415,128,5,1,8,'LayerNorm',.2,tabm_inits=4,device=cuda:0).to(device)`. Retain the constructor draw order, first R Rademacher, later R/S ones, B zeros, Xavier W and no W bias. Use original global sequential Python/NumPy/CPU/CUDA RNG; no PT private stream or source-quartet initialization.
- Keep full original model training, AdamW learning rate .001/weight decay0. Each step performs four native route forwards and mean TRAIN CE divided by4, four sequential backwards and one optimizer step. Retain float32/highest matmul, TF32 disabled and native deterministic seed setup. This is not the frozen meanSAGE/PT continuation objective.
- After every update evaluate all four routes with dropout off on SELECT. Select `argmax(mean_m raw_logits_m)` correct/150. Preserve the exact saved-best rule: `improving = accuracy > best_saved`; reset stale on any such improvement, including off-cadence; save and update best_saved/selected_step only when improvement occurs at a multiple of10. Stop at stale300 or update5000. No step0 selector, NLL substitution or patience50 rule. Restore the selected complete model, optimizer and Python/NumPy/CPU/CUDA RNG with strict custody/fingerprint equality.
- Whole acquisition closure is three selected banks, three optimizer fits and twelve selected native routes. The theoretical ceiling is15,000 optimizer updates and60,000 TRAIN routes/backwards plus60,000 SELECT routes. Record actual work and complete/failed/unstarted rows. Root supplies one exact-source finite-owner/readiness binding for the adapted interface; the earlier ten-block owner/admission is not the new admission.

## Minimal export and scoring law

At each restored selected state export complete label-free CAL/DEV IDs and four native FP32 route logits. Archive the operational FP32 `raw_logit_mean` and its argmax, plus member decisions and source/input/role/selector/selected-state receipts. The acquisition process consumes no CAL/DEV truth. Complete all three original fits/exports, the full PT panel and any frozen CAL endpoints before a single whole DEV comparison; never use original partial SELECT/DEV scores to choose PT continuation.

**Raw original probabilities are `softmax(mean_m raw_logits_m)`.** Raw accuracy uses its archived original FP32 raw-logit-mean argmax. Do not pass the original arm through PT's mean-of-member-probabilities assembler.

For a prospective common positive scalar CAL temperature, preserve original fusion: calibrate the archived pooled raw logits as `softmax(raw_logit_mean/T)`. Retain the same frozen scalar optimizer/budget/ties/clamp and CAL truth as the PT panel if root binds that operator; the original prior ten-block authority has no inherited CAL result. This is a separately declared new CAL endpoint. Preserve the original raw argmax/accuracy separately; positive pooled-logit temperature leaves the mathematical argmax unchanged. Do not apply temperature inside each member then probability-average, which would change its serving recipe. Root must freeze these calibration/protection rules before fitting.

Read all paired raw/calibrated accuracy, NLL/Brier, mean/worst member competence, coverage/pooling/aggregation-only accounting, class repairs/harms and full acquisition/serving cost on the identical three role populations. The original differs from PT in width, normalization, objective, optimizer and stopping; interpret this as a same-information whole-procedure benchmark, not isolated factor-placement or matched-capacity causality. The old ten-block original SELECT-only results remain acquisition context, not substitutes for these endpoints.
