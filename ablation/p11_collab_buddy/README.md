# Native BUDDY and original GNNM on Collab

The reference implementation is `melifluos/subgraph-sketching`, detached at
`3562d94a07d1166faa0949030824bf75ad9bb2c4` under `external_sources/buddy`.
Its README identifies the ICLR 2023 paper *Graph Neural Networks for Link
Prediction with Subgraph Sketching* and gives the Collab BUDDY command.
`upstream_manifest.json` freezes every reference Python source, README and
license by SHA-256. Each job also binds the adapter, original `models.py`, shared
worker/storage sources, dependency versions, raw data and official split files.

- [Published implementation](https://github.com/melifluos/subgraph-sketching/tree/3562d94a07d1166faa0949030824bf75ad9bb2c4)
- [Paper](https://openreview.net/forum?id=m1oqEOAozQU)
- [Official Collab rules and evaluation](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab)

## Frozen recipe

The published Collab command overrides learning rate to 0.02, feature dropout
to 0.05, label dropout to 0.1, degree-normalized structural features on, year
threshold to 2007 and metric to Hits@50. Remaining reference defaults give
hidden width 1024, training batch size 1024, Adam with zero weight decay, 100
epochs, validation every epoch, BCE-with-logits loss, two sketch hops, 128
MinHash permutations, HyperLogLog precision 8, and one negative per positive.
There is no scheduler or early stopping. The adapter retains all 100 validation
curve entries. The earliest strictly best validation epoch is selected.

Structural features use the reference MinHash/HLL estimators and exact native
feature formulas, including the disabled (0,1)/(1,0) features. Chunked min/max
neighborhood reductions give exact native sketch tables while bounding working
memory. The isolated-node reduction behavior also matches the reference.
Features use one weighted GCN-normalized propagation step with self-loops,
which is native `sign_k=0`. Weighted degrees feed normalized sketch features.
There is no resource-allocation branch and no learned node embedding.

Training supervision and message edges retain official training edges from
2007 onward. Negative edges are sampled once with PyG `negative_sampling`
against the filtered training graph and its self-loops. The preprocessing seed
is 0 and the resulting negatives are shared across paired fits, consistent
with the reference's persistent negative cache. Every epoch uniformly shuffles
the full mixed positive/negative pool. The adapter stores that permutation for
mid-epoch continuation.

The upstream cache establishes this shared-pool behavior explicitly.
`external_sources/buddy/src/runners/run.py` lines 56–59 call `set_seed(rep)`
before data loading on every repetition. In `src/data.py` lines 152–163,
`get_ogb_data` uses `dataset/ogbl-collab/negative_samples.pt` for one negative
per positive. Its path has no repetition or seed component. An existing file
is loaded, and generation followed by saving occurs only when the file is
absent. Consequently a fresh published multi-repetition run generates its
negative pool in repetition 0 and reuses it for subsequent repetitions.
The declared preprocessing seed 0 reproduces that common-pool policy regardless
of which queued fit is scheduled first.

Sketch realization is also shared by construction. `src/hashing.py` line 61
sets `minhash_seed=1`, line 110 creates a local NumPy `RandomState` from that
constant, and lines 121 and 128 deterministically hash node IDs for MinHash
and HLL. `src/datasets/elph.py` lines 154–173 name cached structural features
by split, year, hop count and negative count. These cache names also omit the
model repetition and seed. The adapter resets each model to its declared fit
seed after shared preprocessing, making model initialization independent of
whether preprocessing was read from cache or freshly generated.

Validation uses every official validation positive and negative edge with the
filtered TRAIN message graph. After model selection, TEST uses every official
TEST positive and negative edge. The reference Collab path unconditionally
adds validation edges to the TEST graph, including their weights. The frozen
recipe makes this choice explicit as `use_valedges_as_input=true`. Validation
edges enter message passing and sketch construction only in the TEST cache.
The official OGB evaluator computes Hits@50 on the complete negative pool.

## Paired fits and original routes

Each of five cohorts contains four independent native BUDDY fits, seeded
`cohort * 1000 + member`, and one four-route GNNM fit seeded `cohort * 1000`.
Native member 0 supplies BASE. The collector derives ENS4 by averaging raw
native logit scores from each independent member's own validation-selected
checkpoint. GNNM4 averages its four raw route scores and selects one shared
checkpoint using that averaged validation score. Training averages the four
route BCE losses in one optimizer update.

The GNNM adapter reuses the original repository `BEBlock` at BUDDY's parallel
input projections `label_lin_layer` and `lin_feat`, and final output projection
`lin`. The feature interaction, shared `lin_out`, BatchNorm buffers and all
intermediate transformations follow native BUDDY. Input R factors use the
original random signs, output R and all S factors start at one. Native affine
weights and biases are copied into shared W and the initial route B values.
Setting R and S to one exactly recovers the initialized native model for every
route. This preserves native bias initialization while using the original
`S * W(R * x) + B` law.

Historical source was inspected read-only at branch
`codex/postsubmission-research-20260930`, commit
`48329b19941ea869c439e5a220a3df0e30aa1237`. Its
`reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port/callbacks.py`
declares these same three BUDDY boundaries. The historical
`link_expansion_ports_BUDDY_production_v2/session.py` and
`buddy_native_provider_v2/provider.py` were also inspected under
`experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1`.
The present implementation uses the current repository's original BEBlock and
shared normalization state.

## Run and resume

Run from the repository root after sourcing the allocation environment.
All paths must resolve inside the repository.

```sh
source .transfer_setup_20261110/scripts/environment.sh
"$GNNM_ENV/bin/python" -B -m ablation.p11_collab_buddy.run \
  --root ablation/p11_collab_buddy/runs --data_dir data --device cuda:0
"$GNNM_ENV/bin/python" -B -m ablation.p11_collab_buddy.collect \
  --root ablation/p11_collab_buddy/runs
```

The training queue contains 25 fits. The collector derives the five ensemble
artifacts after checking each expected native directory, cohort, member, seed,
config.json and completed prediction identity. GNNM receives the same paired
identity checks. Copied native-member artifacts are rejected. Prediction edge
ordering and shape must agree across paired fits. All source, protocol, data
and dependency fields must agree across cohorts before a summary is emitted. It emits means, sample standard deviations and paired differences. Smoke
results are excluded by default. `--include-smoke --cohorts 0` explicitly
includes a complete engineering cohort for VALID aggregation checks. Missing
fit names remain explicit for every requested cohort, and incomplete collections
return a nonzero exit status.

An optional `--out DIRECTORY` writes `results.json` and the derived
`cohortN/ensemble` artifacts beneath that collection directory. Fit artifacts
continue to be read from `--root`. With `--out` omitted, derived ensemble
artifacts use their existing `--root/cohortN/ensemble` locations and the summary
is printed to standard output. Every supplied output path resolves inside the
repository.
Each fit uses the existing `JobStore` and `run_worker` APIs and saves model,
Adam, explicit scheduler state `None`, Python/NumPy/Torch/CUDA RNG, epoch,
permutation, minibatch cursor, best model, validation curve and patience counter.
Checkpoint writes and prediction outputs are atomic. Data caches use kernel
advisory locks, source/data/dependency identities and atomic writes.

```sh
"$GNNM_ENV/bin/python" -B -m ablation.p11_collab_buddy.run \
  --root ablation/p11_collab_buddy/smoke --data_dir data --device cuda:0 \
  --smoke --cohorts 0 --only native --member 0
"$GNNM_ENV/bin/python" -B -m ablation.p11_collab_buddy.run \
  --root ablation/p11_collab_buddy/smoke --data_dir data --device cuda:0 \
  --smoke --cohorts 0 --only gnnm
```

Smoke mode defaults to two optimizer updates from the full native shuffled
supervised pool and a complete official validation evaluation. TEST processing
raises an error in smoke mode. Explicit `--smoke_epochs`,
`--smoke_max_batches` and `--smoke_train_edges` flags control diagnostic caps.
The default train-edge cap is 0, meaning the full training pool. A positive
train-edge cap affects supervision only, with the full filtered TRAIN message
graph and complete official VALID retained. Smoke results carry
`role=smoke_train_valid` and have independent job identities. Full defaults
remain 100 epochs over the complete native training pool.

Operational `--eval_batch_size` and `--preprocess_batch_size` default to 8192.
They bound memory while traversing all required edges. `--status`, `--cohorts`,
`--only`, `--member`, checkpoint interval and failure/claim controls are exposed
by the runner. A new source, data, dependency or protocol identity requires a
new result root, including for already completed jobs.

## CPU qualification

```sh
"$GNNM_ENV/bin/python" -B -m unittest ablation.p11_collab_buddy.test_p11 -v
```

Seven focused regression tests cover exact native identity-factor output parity,
exact bounded/native sketch and structural-feature parity, exact model and
prediction continuation after an interrupted GNNM update, and correct
own-selected four-model score aggregation through the official OGB evaluator,
copied-member rejection, cross-cohort binding consistency and collector output
directory routing through the CLI.
`cpu_qualification.json` and `cpu_tests.log` preserve the initial qualification.
`cpu_qualification_collector_identity.json` and `cpu_tests_collector_identity.log`
record the refreshed collector qualification and exact dependency versions. GPU qualification and full runs are owned by the parent
experiment launcher.
