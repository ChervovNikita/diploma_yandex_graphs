# Executable BUDDY shared-cache family — source v4

## Status

Integration code and fixed experiment contract are prepared. **Local source checks passed; numerical CPU parity, complete-data cache construction and resource qualification are pending.** This Mac has no Torch runtime. `test_cpu.py` supplies seven dataset-free CPU checks and emits the certificate required by training. No dataset, model, checkpoint, GPU or remote host was accessed here.

V4 retains the v2 guard repairs and v3 module-registration repair, and reads the official split name from inspected OGB1.3.6 `meta_info['split']`; see `REPAIR_REPORT.md`. The sealed v1/v2/v3 packets remain unchanged. The fixed recipe, all vendor snapshots, parameter counts, arms, seeds, objectives and cache contents are unchanged. V4 requires fresh source-bound certificates/cache/run identities.

This is a BUDDY/BatchEnsemble-style architecture comparison, not a new graph learner. The saved feasibility packet and structural-correction no-go were consulted first. No residual-guided, edge-routing or diversity-penalty arm is added. The useful question is whether compact factorized predictors retain ensemble benefit after strong independent predictors receive identical deterministic cache reuse.

## Fixed development task and arms

Use complete **ogbl-collab**, supplied 128-dimensional features and official temporal train/validation/test pairs. All preprocessing uses the complete **training-only weighted topology**, including at test time. Preserve all 235,868 nodes; no year filtering or largest-component reduction. The width-1024 anchor retains native predictor structure and the released collab learning recipe, but the full-node/year-0/training-only-test policy is a disclosed protocol adaptation, not a reproduction of published scores.

| Arm | Learned predictor | Trainable parameters, source formula |
|---|---|---:|
| `native1024` | Exact native BUDDY class, width 1024 | 1,185,091 |
| `single256` | Exact native class, width 256 | 99,907 |
| `factorized4` | Four nonlinear native branches; shared linear weights/bias, private R/S and BatchNorm | 106,349 |
| `independent4` | Four separate native width-256 predictors/optimizers, one common cache | 399,628 |
| `matched_single` | Exact native class, analytically matched width 266 | 106,457 |

Parameter figures include native unused `bn_RA` parameters and await instantiated CPU confirmation. Width 266 differs from the factorized count by 0.102%; its choice uses no labels. Three paired optimizer seeds **0, 1, 2** give **15 arm/seed cells and 24 optimizer fits**, counting four independent fits per ensemble cell. The native-width anchor is the competence reference; the other four arms isolate the compact predictor comparison.

One fixed recipe: 100 complete epochs; Adam lr 0.02, decay 0, native betas/epsilon; training batches 1024; feature dropout 0.05; structural-label dropout 0.1; normalized structure appended; hash hops 2, HLL p=8, MinHash 128/seed 1; `sign_k=0`; no learned node embeddings or RA. Native `sign_k=0` preprocessing still performs **one normalized feature propagation**, then the predictor has no trainable graph pass. Keep native `floor_sf=False` and `use_zero_one=False` (positions 4 and 5 set to zero). Extraction chunks 65,536 and evaluation batches 65,536 are fixed disclosed memory choices.

Generate one fixed training-negative set with seed 20261002, one negative per training positive, using the native PyG sampler against training edges and self-pairs only. Do not reject future positives. All arms/seeds use identical ordered pairs/cache. Retain the unchanged official 100,000-negative evaluation pools and strict Hits@50 threshold tie rule. This sampled BCE objective does not establish probability calibration.

## Native entry points and implemented glue

Pinned author commit: `melifluos/subgraph-sketching@3562d94a07d1166faa0949030824bf75ad9bb2c4` (2023-12-05). Exact snapshots are vendored; no upstream clone or fetch is needed for this package's selected path.

- Native `src/models/elph.py:BUDDY.__init__/feature_forward/forward` is extracted unchanged from the saved class. Only `sign_k=0`, feature-enabled, no-embedding/no-RA execution is supported. This avoids importing unrelated ELPH/GNN code.
- Native `src/datasets/elph.py:HashDataset._generate_sign_features` is extracted unchanged for feature preprocessing. Native `src/hashing.py:ElphHashes.build_hash_tables/get_subgraph_features` builds one sketch bank and query counts.
- `cache_builder.py` performs native collab weighted coalescing/degrees, builds common features/sketches once, and binds graph, weights, features, pair order, negative seed, source/settings and dependency versions. Its offline OGB loader refuses downloads. It reads official `train.pt` and `valid.pt` separately, using the installed loader's checked per-file convention and native NumPy-to-Torch conversion; the combined `get_edge_split()` accessor is never invoked. Test pairs/features are read only after family lock. An aggregate-only `split_dict.pt` release is unsupported until authentic per-split files are separately staged and qualified; the builder fails without opening it.
- `models.py` preserves endpoint projections/products, structure normalization, nonlinearities and scalar logits. The factorized port modulates every native learned linear map with `((x*r) W^T)*s+b`; R starts as random signs, S as one. BatchNorm affine/running states are private. This is a declared port, not exact GNNM source reproduction. Private dense/nonlinear work remains.
- `run.py` gives every arm **100 validation selections**, maximizing official pooled validation Hits@50 with earliest-epoch ties. Independent predictors retain separate own-member BCE/Adam fits and one synchronized ensemble checkpoint. All multi-member arms deploy by mean raw logits. Test arrays are not loaded by training/resource functions; final scoring requires all 15 completed locked cells.
- `guards.py` binds all executable Python files, actual vendor bytes and config/pins, validates exactly 100 consecutive finite/ranged ledger records, derives the earliest maximum, and binds identity/ledger/selection/checkpoint digests. `checkpoint_io.py` validates the selected envelope and strict CPU model state keys/shapes/dtypes/finiteness before either production lock creation or lock consumption permits test access. The saved loader source digest is checked before test deserialization. These checks establish consistency of recorded evidence; they do not independently authenticate the training chronology or official data provenance.
- Existing finite training-logit and prediction checks are retained. BCE losses, gradients, parameters/BatchNorm state, Adam state, accumulated BCE, validation records and reused final scores receive separate finite checks. Their execution cost belongs in resource qualification.
- `launch_family.py` queues local jobs over two GPU IDs. It has no network, SSH or Git code and prints a dry-run schedule by default. Root handles the authorized Git sync/host selection and runtime admission.

Batch order is paired across arms. Independent initialization uses member seeds `seed + 10000*m`. Dropout is reproducible but its native-call and stacked-member layouts are not exact paired masks; this limitation is disclosed rather than claiming identical stochastic trajectories.

## Run order in an authorized checkout/runtime

The saved README's historical CPU stack is Python >=3.9, Torch 1.13.1, PyG 2.2 and torch-sparse 0.6.17; also require NumPy, SciPy, pandas, datasketch, tqdm and OGB. Use an existing compatible runtime and record versions. No dependency installation was performed on this Mac.

Root's metadata-only runtime check reports Torch 2.1.2+cu118, PyG 2.7.0, NumPy 1.26.4, SciPy 1.14.1, pandas 2.2.3 and OGB 1.3.6. The proposed isolated qualification additions are datasketch 1.6.5, torch-sparse 0.6.18 and torch-scatter 2.1.2 if absent, using sparse wheels built for pt21cu118 and the runtime's Python ABI. Put them in a separate repo-local target and add that target only to the new process's Python path. Compatibility is unverified until all seven CPU tests pass; none may be skipped. The active-study environment remains root's responsibility.

```sh
python check_source.py
python test_guards.py
python test_cpu.py --output CPU_QUALIFICATION.json
python cache_builder.py build --dataset-root /absolute/existing/ogb-root --output cache
```

CPU checks cover native/factorized identity logits, shared-weight and private-BN gradients, private BN state, actual parameter counts, singleton-tail batch preservation, exact official metric ties, dense weighted feature-propagation oracle and sketch extraction chunk invariance. Failures block fits. The script creates no skip-based pass.

All seven CPU tests remain mandatory with the original 1e-12/1e-10 native parity tolerances. V2's six successes/one hash-loader error and v3's subsequent seven-test success remain preserved in their separate root receipts. V4 leaves `test_cpu.py` byte-identical to v3 and requires a fresh seven-test certificate. Its twelve stdlib guard checks include installed split metadata/basename validation and loader lifecycle/ledger guards. The native feature cast is preserved exactly even in the float64 parity path.

Before the family, execute one complete resource epoch plus validation-forward-only for **each arm** on seed 0, in distinct resource folders. These do not select checkpoints or score validation/test quality:

```sh
python run.py resource --cache cache --arm factorized4 --seed 0 --device cuda:0 --qualification CPU_QUALIFICATION.json --output resource/factorized4
```

Repeat for the other four arms. Charge cold cache construction/read/hashing, all 24 fits, 1,500 validation forwards, checkpoint writes, one final forward per cell, peaks and any profiling work. Forecast total wall/GPU time from these actual passes; obtain root's resource admission before starting the complete family. Do not claim a few-hour budget or a speedup from parameter counts. Fix any memory-driven batching change across all arms **before fitting**, version the config/cache identity and repeat CPU/resource qualification.

Then use both local GPU slots from the authorized checkout:

```sh
python launch_family.py --cache cache --runs runs --qualification CPU_QUALIFICATION.json --gpus 0,1
python launch_family.py --cache cache --runs runs --qualification CPU_QUALIFICATION.json --gpus 0,1 --execute
python run.py lock --cache cache --runs runs --output FAMILY_LOCK.json
python cache_builder.py finalize --dataset-root /absolute/existing/ogb-root --output cache --family-lock FAMILY_LOCK.json
python run.py test --cache cache --family-lock FAMILY_LOCK.json --qualification CPU_QUALIFICATION.json --device cuda:0
```

The builder never invokes native `get_ogb_data`, whose collab branch adds validation edges unconditionally, nor native `runners/run.py`, which scores test every epoch. Their stable predictor/preprocessing operations are used under the explicit contract above. A failed/missing cell leaves the family incomplete. Incomplete training folders require explicit replay review; final scoring can resume by reusing an existing result only when its locked artifacts match exactly. Existing outputs are not silently overwritten.

Report all three paired-seed contrasts: factorized versus single256, independent4 and matched_single; retain native1024 absolute performance. Use official Hits@50, every seed's selected epoch, total work/cost, cache-inclusive cold and warm prediction timing, peak memory, and predictor storage. Three seeds describe optimizer variability on one fixed development split; they do not establish graph generalization, state of the art or methodological novelty. NCN/NCNC/PENCIL and a separate untouched task are later confirmation controls if a broader LP claim is sought.
