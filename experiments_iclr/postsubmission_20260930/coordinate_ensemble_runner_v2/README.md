# Coordinate ensemble coordinator — source draft v2

This is source for a **new** LastFMAsia / Amazon Photo study. No data acquisition,
model imports, model execution, tests, SSH, score recalculation, or paper edits were
performed while preparing it. Root must review, freeze, and authorize each phase.
The protocol template is deliberately incomplete and `frozen: false`.

## V2 interface additions

V1 is preserved. V2 adds `profiling.selected_checkpoint_cold_deployment` with
fixed `warmups`/`repeats`, and `profiling.operation_breakdown` with an explicit
`enabled` flag, fixed counts and the operation list in the template. The latter
may be disabled during qualification. Both scopes and their measured overhead
must enter the independent budget forecast; learning recipes, selection gates,
and HPO scope are unchanged.

Validation now reports macro-F1 for probability pooling, the same-checkpoint
mean-logit sensitivity and each member. It is the unweighted mean of
`2TP/(2TP+FP+FN)` over all C output classes; a zero denominator contributes zero.
Member disagreement is the fraction of development nodes with different argmax
classes for each unordered member pair, averaged equally across pairs. Argmax
ties use the first class index. A one-member model reports null disagreement and
an empty pair list. These are diagnostics; validation NLL still selects checkpoints.

Selected deployment profiling uses a model-only checkpoint and repeated **fresh
resident setups within the existing process**. Each repetition reloads that CPU
checkpoint, constructs a fresh CPU model, applies its state, transfers the model,
graph/features and development gather indices to the device, and produces pooled
probabilities. No labels, optimizer states or target packs enter that function.
The old fitted GPU model/graph and parameter-list aliases are released first.
CUDA synchronization surrounds transfers/forward; each repetition records a direct
end-to-end duration plus components, with median/p95 computed over the measured
repetitions. Component medians are not added to make an end-to-end number.

This scope excludes a new process, imports, CUDA context initialization, OS cache
clearing, output host transfers, and allocator-clear/cleanup from the individual
end-to-end interval. The CUDA allocator cache is cleared before fresh setups.
Per-repeat baseline/peak allocated and peak reserved memory are retained. Total
profiling time includes warmups and cleanup. Qualification profiles its final
qualification state on training-node indices; fit profiles the selected checkpoint
on validation-node indices. Initial untrained first-forward timing remains a
separate field and is not used as selected-checkpoint deployment timing.

The optional operation breakdown measures actual selected/final full-graph member
forward, development-row gather and probability pooling in separate fixed loops.
It reports resident fixture bytes and synchronized median/p95. Fixture creation
and per-operation synchronization differ from end-to-end execution, so these
measurements are not an additive decomposition or a speedup claim. Warm complete
inference and complete training-update costs remain separately retained.

Per-step finite-gradient, nonzero-gradient-tensor count and post-update finite
parameter audits are unchanged, using batched reductions with one host result
per category to avoid a synchronization for each tensor.

## Commands and admission boundary

`runner.py` supports `acquire`, `qualify` (`preflight` is an alias), and `fit`.
Every invocation requires the exact SHA256 of the supplied frozen protocol JSON,
the exact SHA256 of the source-manifest JSON, and matching hashes for every source
file listed in that manifest. Qualify and fit additionally require one explicitly
listed dataset/split/seed/arm/recipe cell and the exact acquisition manifest digest.
There is no sweep, automatic calibration selection, resume, test-scoring command,
or automatic experiment admission. Existing output directories are rejected.

These commands are **interface examples**, not authorization to run them:

```sh
python /ABS/coordinate_ensemble_runner_v2/runner.py acquire \
  --protocol /ABS/acquisition_protocol.json --protocol-sha256 PROTOCOL_SHA \
  --sources /ABS/frozen_sources.json --sources-sha256 SOURCES_SHA \
  --dataset LastFMAsia

python /ABS/coordinate_ensemble_runner_v2/runner.py qualify \
  --protocol /ABS/execution_protocol.json --protocol-sha256 PROTOCOL_SHA \
  --sources /ABS/frozen_sources.json --sources-sha256 SOURCES_SHA \
  --dataset LastFMAsia --split core0 --seed 17 --arm coordinate --recipe main_lr_low \
  --data-manifest /ABS/acquire/public/LastFMAsia/data_manifest.json \
  --data-manifest-sha256 DATA_MANIFEST_SHA

python /ABS/coordinate_ensemble_runner_v2/runner.py fit \
  --protocol /ABS/execution_protocol.json --protocol-sha256 PROTOCOL_SHA \
  --sources /ABS/frozen_sources.json --sources-sha256 SOURCES_SHA \
  --dataset AmazonPhoto --split core0 --seed 17 --arm original --recipe main_lr_low \
  --data-manifest /ABS/acquire/public/AmazonPhoto/data_manifest.json \
  --data-manifest-sha256 DATA_MANIFEST_SHA
```

All framework caches are redirected to the supplied phase cache directory before
torch/PyG imports. Raw and processed PyG data live under the supplied acquisition
phase path. Output, cache, public, sealed, and PyG directories must be disjoint
subdirectories of their phase root. The configured CUDA lock is inside the
acquisition root and must be common to **all** jobs/protocols in the study. A second
coordinator GPU job fails immediately if it cannot obtain that lock. External
GPU jobs are outside this lock's control; root must keep the device exclusive.

## Source manifest

The JSON file has this structure:

```json
{
  "schema_version": 1,
  "model_entry": "/ABS/coordinate_ensemble_source_v1/models.py",
  "files": [
    {"path": "/ABS/coordinate_ensemble_runner_v2/runner.py", "sha256": "HEX", "role": "coordinator"},
    {"path": "/ABS/coordinate_ensemble_source_v1/models.py", "sha256": "HEX", "role": "model_entry"},
    {"path": "/ABS/propagation_cost_impl_v1/_pinned/frozen_models.py", "sha256": "HEX", "role": "author_model_dependency"},
    {"path": "/ABS/audited_pyg_lastfm_asia.py", "sha256": "HEX", "role": "author_dataset_loader"},
    {"path": "/ABS/audited_pyg_amazon.py", "sha256": "HEX", "role": "author_dataset_loader"}
  ]
}
```

Include every local helper that the model entry imports. All listed files are
verified and copied into each output receipt. The runner verifies sources again
immediately before importing the reviewed model entry. This uses ordinary hash
checks and explicit assertions; it does not sandbox reviewed Python source.
Exact installed torch/PyG versions must match the protocol. Acquire and qualify
copy/hash the actual installed dataset loader into their phase output before
model work, compare it against `pyg_loader_sha256`, and stop for a separate root
source review if bytes/versions differ. Reference snapshots labeled 2.7.0 do not
establish the installed provider's identity. A revised source manifest should
include the phase-retained actual loader snapshot. Normal fit reads that frozen
provenance and does not reread the installed provider source file.
Package/dependency versions and source review remain root's admission decision.

## Frozen protocol fields

`PROTOCOL_TEMPLATE.json` enumerates the interface. Null fields and placeholder
paths must be filled by root; it is not an adopted recipe.

- `schema_version: 1`, `frozen: true`, `study_kind: "new_study"`.
- `source_manifest_sha256`: the exact source-manifest JSON digest.
- `datasets`: keys `LastFMAsia` and/or `AmazonPhoto`. Each has the audited PyG
  loader digest, `rounding: "floor_test_and_validation_per_class_train_remainder"`, `test_holdout: {seed, fraction}`,
  and prospective `splits: [{id, seed, train_fraction_of_development}]`.
- `preprocessing`: `features: "none"` or `"row_sum"`; `edges: "as_pyg"`.
  Row-sum normalization requires nonnegative features and leaves zero rows zero.
  PyG preprocessing/coalescing is retained; the coordinator does not change edges
  or add self-loops. Self-loop/propagation behavior belongs to hashed model source.
- `recipes`: explicitly supply `num_layers`, `width`, `members`, `dropout`,
  `min_epochs`, `max_epochs`, `patience`, `eval_every`,
  `patience_unit: "epochs_since_improvement"`,
  `checkpoint_metric: "validation_nll"`, `checkpoint_ties: "earliest"`,
  `scheduler: "constant"`, and optimizer `name` (`Adam`/`AdamW`), `lr`,
  `weight_decay`, `betas`, `eps`. No learning recipe defaults are used.
- `model_kwargs`: explicitly supply `hidden_dim_multiplier`,
  `normalization: "layer"`, `num_heads`; arm-specific applicability is defined by
  the model source. GT-sep is a separate reference arm, not the same SAGE topology.
- `cells`: every permitted combination is an object with `dataset`, string
  `split`, integer `seed`, `arm`, independent integer `permutation_seed`, `recipe`.
  Recipe IDs allow a prospectively specified single-control calibration grid.
  A fit invocation selects exactly one cell and does no comparison/selection
  between recipes. Calibration aggregation and adoption are external root work.
  `--recipe` is mandatory in qualify/fit and participates in cell matching and
  the output directory name. This permits two LR calibration cells with the same
  dataset/split/seed/arm in one frozen protocol. Recipe IDs must use letters,
  digits, underscores or hyphens. Separate frozen protocols and disjoint output
  roots are also valid; existing outputs are never overwritten.
- Each cell also requires `storage_gate: {expected_model_tensor_bytes,
  expected_index_buffer_bytes, equal_byte_reference}`. The first two fields are
  exact prospective integers. The last is null for arms without byte matching,
  or `{bytes, relative_tolerance}` for a fixed complete-model-byte reference
  (tolerance at most 0.01). Actual storage and FP32/int64 dtype checks happen
  before complete updates. `storage_policy` is explicitly
  `{parameter_dtype: "torch.float32", index_dtype: "torch.int64"}`.
- `pooling`: primary `arithmetic_mean_softmax` and
  `same_checkpoint_mean_logit_sensitivity: true`.
- `qualification.complete_updates`: full-graph optimizer updates, not a synthetic
  forward or a reduced surrogate. The selected cell's ordinary architecture,
  dropout and optimizer are used; only training nodes enter the objective.
- `profiling`: `cold_calls: 1`, fixed integer `warmups`, `repeats`, and
  `scope: "full_graph_forward_probability_pool_and_development_gather"`.
  Also freeze `selected_checkpoint_cold_deployment: {warmups, repeats}` and
  `operation_breakdown: {enabled, warmups, repeats, operations}`. Counts for an
  enabled breakdown must be valid positive repeats/nonnegative warmups.
- `runtime`: explicit `device`, `torch_version`, `torch_geometric_version`,
  `deterministic_algorithms`, `allow_tf32`, positive `cpu_threads`,
  `cublas_workspace_config`, and common absolute `gpu_lock_file`.
- `phases`: acquire/qualify/fit each supply `root`, `cache_root`, `output_root`;
  acquire also supplies `pyg_root`, `public_root`, `sealed_root`.
- `test_policy: "sealed_independent_confirmation_only"`.
- `data_bindings` is unused during acquire and required during qualify/fit.
  For each dataset, bind the acquisition protocol digest, acquisition source
  manifest digest, and data-manifest digest. A later frozen recipe can reuse
  acquisition, but its dataset/split/preprocessing definition must be identical.

All absolute paths must name the authorized phase locations. Models/datasets are
not imported until frozen JSON/source checks have passed and a new output exists.

## Acquisition and the target lock

Acquire calls `LastFMAsia(root=...)` or `Amazon(root=..., name="Photo")`. It may
download only in this explicitly authorized mode. It ignores official masks and
creates the prospectively declared partitions with Python's local seeded random
generators. Class IDs are visited in sorted order; class node IDs are initially
in ascending order and shuffled. The test count is floored per class; the
validation count in the remaining pool is floored, with training taking the
remainder. Empty classwise
train, validation, or test partitions are rejected. The requested fractions are
nominal; the receipt records exact counts after rounding.

One class-stratified test holdout is made per dataset. Every calibration/core
split only repartitions its complement, so no cell trains on a held-out target.
All labels may be accessed during creation for stratification and creation-only
label counts. No scientific scoring occurs in acquisition.

Public input files contain:

- `graph.pt`: **only** `x` and `edge_index`, with no `y` or masks.
- `test_indices.pt`: node indices only.
- `<split>/train.pt`, `<split>/validation.pt`: their node indices and targets only.

The full-graph fit input never contains held-out targets. Target arrays must be
nonnegative and in class range; a `-1` sentinel is rejected for every train/val
target. Mask index digests, disjointness, node coverage, and file hashes are
checked before model execution. Test node features/structure remain visible as
normal transductive inputs.

`sealed_root/<dataset>/test_labels.pt` separately retains test indices/targets,
with its digest in the acquisition receipt. Its directory/file permissions are
0700/0600. This is a checkable file separation and runner access policy, not a
cryptographic or operating-system isolation guarantee against the same account.
The scientific reader never opens this file or the full-label PyG raw/processed
objects. Original author raw files, which may contain full labels, are copied
and hashed **inside the sealed acquisition area**, with digests and location
receipts in the public manifest. No raw snapshot is placed in public fit inputs.
Processed PyG files stay in the acquisition-only PyG path and are
also hashed for acquisition provenance. Root should retain the raw/processed and
sealed files outside the working inputs used for confirmation-free development.
Acquisition reports signed-feature entries, feature range/zero rows, duplicate
directed edge entries, self-loops, and entries lacking a reverse edge. These
audits do not trigger an automatic graph/feature transformation. Root reviews
actual metadata before admitting scientific execution.

## Learning, selection, and outputs

The model entry supplies `build_model(arm, input_dim, hidden_dim, output_dim,
seed, permutation_seed, members, num_layers, dropout, hidden_dim_multiplier,
normalization, num_heads)` and `storage_report(model)`. A graph exposes
`edge_index[2,E]`; `x[N,F]` is separate. Forward must return member logits
`[M,N,C]`; `single` and `gt_sep_single` return `M=1`.

Every member receives its own training CE. Shared ensembles optimize the mean
over all member/training-node pairs. The untied arm uses one optimizer per
disjoint member and backpropagates the sum of the individual node-mean CEs, so
each member receives its ordinary CE gradient without the ensemble averaging
factor's interaction with Adam epsilon. Its curve still logs mean member CE.
The runner checks disjoint parameter ownership and optimizer coverage; all
untied optimizers step in synchronized epochs. Training uses complete full-graph
forward/backward updates; no validation or test labels enter any objective.

Validation is evaluated every declared `eval_every` and at `max_epochs`. The
primary pool is evaluated stably as
`logsumexp(log_softmax(member_logits), member_axis) - log(M)`; there is no clipping.
Lower pooled validation NLL selects the checkpoint; exact ties preserve the
earliest checkpoint. Patience is counted in update epochs since the best epoch
and cannot stop before `min_epochs`. Stop decisions happen at scheduled validation
checks, so a nondivisible patience threshold can be exceeded by an evaluation
interval. Mean-logit pooling is reported on exactly the same selected/final
checkpoints and never participates in checkpoint selection or extra tuning.

Fit retains initial, selected and final model states (including permutation/index
buffers), selected/final optimizer state, the full training curve, and selected/
final **validation-only** member logits plus corresponding validation node IDs.
V2 also retains a model-only selected deployment checkpoint (qualification keeps
a model-only final qualification checkpoint) for the fresh setup measurements.
No full-node logits or test predictions/metrics are saved. Reports include member
and primary/sensitivity validation NLL/accuracy, stopping diagnostics, hashes,
runtime versions, full-update and validation time, checkpoint IO, profiling time,
peak allocated/reserved GPU memory, model storage, and serialized checkpoint size.
The report describes stopping; it does not certify numerical convergence.

Qualify retains initial/final states and complete-update curves, asserts finite
loss/gradients/parameters, and verifies at least one parameter tensor changed.
It produces no validation metrics/logits. At the declared `eval_every` and final
update it also performs a real no-grad full forward, train-node gather, stable
pooling and train-only evaluation; per-evaluation times are retained for the
budget forecast. Its useful latency output gathers train nodes; fit profiling
gathers validation nodes from the selected checkpoint.

Latency includes resident full-graph model forward, development-node gathering,
stable probability pooling and probability materialization. CUDA synchronization
surrounds each warmup/measured call. Warm latency excludes host/device input
transfers, output transfers, checkpoint loads, metrics and labels; report fields
state these limits. Setup separately reports device initialization, CPU data
loading/hash checks, CPU source/model construction, model transfer, graph/feature/
development-target transfer, initial checkpoint IO, and one first label-free
evaluation of the untrained model. Setup components are components of the setup
total, not additional durations to sum. Training updates, validation (or train-only
qualification evaluation), selected/final checkpoint IO, selected checkpoint
reload, and profiling are separate phase costs. The invocation wall-clock receipt
also includes source verification, snapshots and framework imports.
Warmup count, repeat count and individual measured times are retained, with mean,
median and p95 (nearest-rank percentile). The model stays
in eval mode. Optimizer/target GPU tensors are dropped before inference profiling.
Training/validation and inference peaks are reported separately. Inference peak
includes the resident model/graph and output/gather tensors. No FLOP-derived
speedup claim is made.

Deployed tensor bytes include all model parameters/buffers (including permutation
indices), and separately full graph edges/features and gather indices. Optimizer
state and targets are excluded. This is distinct from actual `.pt` archive size,
allocator peak/reserved memory, and non-tensor Python overhead.

Each output directory contains frozen-input/source snapshots, an invocation
receipt, ordinary artifact hashes, and a final report or failure receipt. Source
preparation has not established runtime compatibility or performance. The drafted
`draft_checks.py` is not run; root must review and authorize any checks separately.

## Timing and budget decision

Qualification measures actual complete updates with the same audit overhead as
fit. A budget forecast must use those measurements, all prospectively scheduled
maximum-duration fits, validation frequency, startup/checkpoint/profile overhead,
the design's contingency and reserve. The runner does not admit comparative fits,
shorten a competent recipe, or claim that a theoretical operation count meets the
budget. Root owns that independent gate and any later independent test confirmation.
