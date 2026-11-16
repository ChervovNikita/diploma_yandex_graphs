# NCN on ogbl-collab

The official GraphPKU NeuralCommonNeighbor repository is pinned to
`11d597013750da17ce7468e344bec756a7af39a4` in
`external_sources/ncn`. Its README identifies this as the paper's official
implementation. The Collab command with `--predictor cn1` defines this protocol.
Checksums and the upstream attribution are in `source_receipt.json`.
The pinned revision provides no LICENSE file.

## Fixed source recipe

The native model is the upstream `GCN` encoder with one GCN layer and width 64,
followed by the upstream `CNLinkPredictor`. This imports the original model and
common-neighbor sparse operations from the pinned checkout.

Training uses 100 epochs and Adam with the original encoder learning rate
0.0082 and predictor learning rate 0.0037. The source has no learning-rate
scheduler or patience termination. Training batches contain 65,536 positives
and an equal number of sampled negatives. Evaluation batches contain 131,072
query edges. The incomplete final training batch is dropped, matching
`PermIterator(training=True)`. Every evaluation edge is retained.

The active source flags are `--ln --lnnn --res --maskinput --use_xlin --tailact
--use_valedges_as_input`. The dropout settings are input 0.25, GNN 0.1,
GNN adjacency 0.25, predictor 0.3 and predictor adjacency 0.0. The source
command's tail dropout 0.05 is retained, although the GCN branch does not use
that parameter. Completion-only flags in the shared README command have no
effect on the NCN `cn1` predictor. All effective settings are recorded in
`protocol.py` and in every job configuration.

The official OGB train/valid/test splits are retained. Native negative sampling
runs once per epoch against the original training `edge_index`, before the
training permutation is drawn. Each batch constructs its symmetric adjacency
from the remaining training positive edges, matching source target-link removal.
Edge weights are omitted exactly as in the source loader. Python's RNG is
explicitly seeded alongside NumPy and Torch so the PyG negative sampler is
reproducible.

Validation uses the training graph. A strictly improved validation Hits@50
selects a checkpoint, with the first epoch winning ties. Full runs evaluate the
selected checkpoint on TEST at completion. Only that final TEST evaluation
adds the validation positive edges to the support graph. The OGB
`Evaluator(name="ogbl-collab")` with `K=50` computes all Hits@50 metrics.

## Three settings and five paired cohorts

Each cohort trains four independent native fits and one shared GNNM fit.
Seeds are `cohort * 1000 + member` for the native members and
`cohort * 1000` for GNNM. Cohorts are 0 through 4.

- BASE uses `native0`.
- independent4 averages the raw prediction-score arrays of each native
  member's own validation-selected checkpoint.
- GNNM4 uses four routes in one shared encoder and predictor and averages their
  raw scores for validation selection and inference.

This requires 25 fits. The five independent4 results are derived from those
fits. The collector computes paired differences and means with sample standard
deviations.

GNNM modifies two existing native affine boundaries. The first boundary is the
first GCN convolution, with input factors before its input projection and output
factors plus route bias after its native graph propagation. The second boundary
is the final predictor affine. The formula is
`W(x * R_m) * S_m + B_m`, with the first convolution additionally retaining its
native sparse propagation between the shared projection and output factors.
All intervening native modules, normalization, activations, common-neighbor
aggregation and dropout remain in the source forward path.

The first input R bank starts with independent Rademacher signs. Other R and S
banks start at one. Shared native weight initialization is retained, and B
starts from the corresponding native bias for each route. Unit factors recover
the complete native model exactly. The original repository BEBlock and the
historical postsubmission link boundary adapter informed the two-boundary
placement. Native weight/bias initialization is retained here to satisfy the
native-equivalence contract. One native two-group Adam optimizer updates shared
weights and route factors from the average of four member losses. Routes use
sequential forwards and backwards on the same minibatch and sampled negatives.

## Commands

Run from `/home/jovyan/gnnm_rebuttal_cluster` with the prepared environment.

```sh
python -B -m ablation.p12_collab_ncn.run \
  --root ablation/p12_collab_ncn/runs \
  --data_dir /home/jovyan/gnnm_rebuttal_cluster/data \
  --device cuda:0
```

Inspect the fixed 25-fit queue.

```sh
python -B -m ablation.p12_collab_ncn.run \
  --root ablation/p12_collab_ncn/runs --status
```

Bounded qualification uses a separate result root and only limits optimizer
updates. It retains the complete source validation set and its original
negative edges. TEST stays closed and all outputs carry `result_role=smoke_only`.

```sh
python -B -m ablation.p12_collab_ncn.run \
  --root ablation/p12_collab_ncn/smoke_two_updates \
  --data_dir /home/jovyan/gnnm_rebuttal_cluster/data \
  --device cuda:0 --cohorts 0 --methods native0 gnnm \
  --smoke-updates 2 --no_wait
```

The CPU contract suite uses synthetic data inside this module's directory.

```sh
python -B -m unittest ablation.p12_collab_ncn.test_contracts -v
```

Collect selected results and independent4 predictions.

```sh
python -B -m ablation.p12_collab_ncn.collect \
  --root ablation/p12_collab_ncn/runs \
  --out ablation/p12_collab_ncn/collected
```

The collector writes `cohorts.csv`, `summary.json` and the selected ensemble
prediction arrays. It reports missing cohorts and excludes smoke results by
default. Use `--cohorts 0 --include-smoke` for an explicitly tagged engineering
aggregation after all four native members and GNNM complete their smoke fits.
Duplicate member identities, conflicting seeds and conflicting configurations
are rejected. All requested cohorts must share protocol, source, data and
runtime identities before pooled summaries are produced. Full TEST artifacts
record a fingerprint of the complete positive and negative edge order, which
must agree across ensemble members and requested cohorts.

The collector exits with status 1 when any requested cohort is incomplete or
the requested collection is empty. It still writes available rows and
summaries, including explicit missing fits and settings, for inspection.

## Recovery and artifacts

The existing `JobStore` and `run_worker` APIs own claims, heartbeat, retries,
atomic state files and completion markers. Checkpoints include model,
optimizer, explicit scheduler absence, Python/NumPy/Torch/device RNG, the active
epoch's negative draw and permutation, next minibatch, selected model and
prediction arrays, selected epoch/update, training time and resume count.
Resume verifies configuration, code/source hashes, training/validation data
identity and dependency versions. A completed root also refuses a changed
configuration when restarted. Keep a new root for source or protocol changes.

Every completed fit retains `selected_model.pt` and
`selected_predictions.pt`. Native selected arrays are retained in official OGB
edge order for independent ensemble aggregation. Full runs include final TEST
arrays. Smoke fits include TRAIN and VALID arrays.

The upstream tested environment was Torch 1.13 / PyG 2.2 / OGB 1.3.5. This
extension's prepared environment uses Torch 2.1.2+cu118, PyG 2.7, OGB 1.3.6,
torch-scatter 2.1.2+pt21cu118 and torch-sparse 0.6.18+pt21cu118.
Exact installed versions are recorded with each checkpoint and result.
