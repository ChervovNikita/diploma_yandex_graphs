# Native full-epoch runner, separate v2 source generation

This new runtime drives the existing frozen node/OGB sessions through complete
native TRAIN epochs. This directory does not replace the running `runtime/`
generation or its immutable pilot. It changes no frozen smoke file, native architecture,
native loss/optimizer or backbone hyperparameter. Scientific extensions enter
through a separately pinned session transform. Only the dedicated one-GPU
allocation is admitted for execution; root owns its maximum-two-process
scheduling. The giant 2,480-fit campaign is not enabled by these commands.

## Frozen single-fit job

```json
{
  "schema": "native-full-epoch-fit-v1",
  "id": "frozen-pilot/split-0/seed-101/native-body-0",
  "cohort_id": "frozen-pilot/split-0/seed-101",
  "family": "ogb",
  "recipe": {"id": "COPY_THE_COMPLETE_ACTUAL_SMOKE_RECIPE"},
  "callback_module": "PROJECT_PATH_TO_FROZEN_ogb_port/session.py",
  "data_root": "PROJECT_PATH_TO_AUTHENTIC_DATA",
  "procedure": "native_single",
  "base_procedure": "native_single",
  "seed": 101,
  "split": 0,
  "body": 0,
  "device": "cuda:0",
  "TEST_access": false,
  "policy": {
    "track": "matched_pilot",
    "maximum_epochs": 1000,
    "minimum_epochs": 100,
    "patience": 50,
    "metric": "rocauc",
    "direction": "max",
    "ties": "first",
    "evaluate_TRAIN_each_epoch": false
  }
}
```

The recipe illustration is deliberately abbreviated and cannot construct a
model. Copy the complete source-bound recipe from the actual smoke manifest,
including native configuration, architecture, redesign root and split bindings.
Node uses `family:"node"` and metric `accuracy`; its declared cohort split is
bound explicitly rather than always using smoke split0.

```sh
python -B runtime_v2/run_fit.py --project-root REPOSITORY \
  --job FROZEN_JOB.json --output FRESH_FIT_OUTPUT
```

Optional `callback_sha256` pins the callback in addition to the automatically
recorded native/runtime receipts. Each fit has one own optimizer/selector; I4
is not constructed as four bodies with a pooled epoch selector.

## Representative three-trial comparison

`run_pilot.py` consumes a `frozen-native-comparison-pilot-v1` object with `id`,
`family`, complete `recipe`, `callback_module`, `data_root`, `TEST_access:false`,
the same `policy`, explicit `trials:[{seed,split},...]`, four
`native_seed_offsets` (first0), and a `methods` list. At least three distinct
trial seeds are required. The base `gnnm_boundary_native` control remains in
every matched comparison.

Each method is `{id,base_procedure?,session_transform?,method_settings?}`.
Three trials with four own native bodies plus four GNNM arms produce exactly
24 fits. M1 aliases body0; I4 adds no fits. Source-reference plans use one native
M1 per trial and `methods:[]`.

```sh
python -B runtime_v2/run_pilot.py --project-root REPOSITORY \
  --spec FROZEN_PILOT.json --output FRESH_PILOT_OUTPUT --prepare-only

python -B runtime_v2/run_pilot.py --project-root REPOSITORY \
  --spec FROZEN_PILOT.json --output FRESH_PILOT_OUTPUT
```

Use different fresh outputs for preparation and immediate execution, or add
`--resume` to execute the exact already-prepared output. Preparation imports no
model or dataset. Execution uses fresh child processes sequentially. Root may
instead run the emitted individual jobs in its explicitly admitted slots;
this runner does not infer permission for extra simultaneous workers.

After four native fits complete, `aggregate_selected.py --members` consumes
their four `selected_VALID.pt` files. It verifies cohort, recipe, policy,
independent seeds/job authorities, exact IDs/order/labels and aligned raw
logits. It averages only individually selected outputs, with body0 as the exact
M1 authority. It never selects a pooled I4 epoch or executes a model.

## Extension API

Set a method's `session_transform` to
`{path:"project/module.py",function:"extend_session",sha256:"..."}`. Aliases
`extension_factory` and `session_extension`, and `path.py:function` spelling,
are supported. `base_procedure` defaults to `gnnm_boundary_native` when an
extension exists; the new scientific `procedure`/method ID is retained in all
records and is not rejected by a method-name allowlist.

The callable receives `(session, job)` after the actual native/base session is
constructed and before the driver captures its model/optimizers/checkpoints.
It may mutate the session or return a compatible session. Preserve original
native parameter objects, authentic role/data interfaces and a complete
state/load API. Register any new private parameters with the model and its
declared optimizer groups. A custom molecular `train_update()` must obtain its
batch from `self._next_train_batch()`; the full-epoch adapter binds that method
after extension installation. Return finite `loss` and `work` counts. Keep
`validate()` returning whole-VALID source logits, graph IDs and source metrics.
Extensions are forbidden in the source-reference track.

## Epochs, source reference and stopping

- A node callback is one complete full-TRAIN node epoch.
- Molecular training uses the native PyG DataLoader, batch32, native skipped
  singleton conditions and the original session's loss/Adam step. Exhaustion
  ends the current epoch even if the tail was skipped. Every TRAIN position is
  verified exactly once; the next epoch cannot leak into the current call.
- Matched pilots use a paired persistent training-loader generator and
  validation-only selection. The native optimizer settings remain constant.
- OGB source-reference uses global native DataLoader RNG and evaluates the
  complete shuffled TRAIN loader before VALID at each epoch, retaining those
  genuine RNG effects. Its native horizon is100 epochs with no early stop.
  Node source-reference retains cap2,000 and native patience250.
- TEST stays closed in every pilot. By default its usual diagnostic and loader
  iterator RNG effect remain omitted and disclosed. Source-reference native
  OGB M1 may explicitly opt into the source-bound iterator emulation below.
  No TEST prediction or metric is substituted. This does not establish exact
  whole-paper replication or resolve the published winning dropout/seed choice.
- Matched minimum/cap/patience are frozen before execution. Patience misses
  before the minimum do not accumulate. First ties remain first. Curves,
  selected/completed epochs and reasons are retained. Cap exhaustion is not
  declared convergence.

## Checkpoint, resume and evidence

Every completed epoch writes the full recovery checkpoint and, on improvement,
the complete selected checkpoint. They include native model/buffers, optimizer,
source state, adapter loader generators, global/private RNGs, selector and
completed-epoch clocks. Resume requires the exact immutable job and pinned
runtime/native source bindings. Restore does not replace source Parameters.

```sh
python -B runtime_v2/run_fit.py --project-root REPOSITORY --job FROZEN_JOB.json \
  --output EXISTING_FIT_OUTPUT --resume EXISTING_FIT_OUTPUT/recovery.pt
```

Recovery granularity is a **completed epoch boundary**, not an uncheckpointed
live DataLoader. A partial epoch lost to interruption is not counted as
complete; its terminal/partial records remain. Explicit resume may replay such
work, and unobserved costs after abrupt termination are marked unknown.
Live owner locks are never stolen. A terminated lock is archived only on an
explicit resume after its recorded PID is absent. There are no automatic
scientific retries. Prior terminal results are archived before resumed work.

On entry, the restored selector's nonmutating `stopping_status()` is checked.
If patience or the cap already stopped the trajectory, v2 reloads and exports
the selected state with **zero additional TRAIN epochs/optimizer calls**.
`completed_epochs`, selected epoch and stopping reason remain the restored
values; `training_epochs_this_invocation` and `finalization_only_resume` make
this visible. A selected checkpoint must also match the current runtime binding.
V1 checkpoints do **not** silently migrate to v2: changed runtime hashes are
rejected, even though the v1 stopping defect is fixed for new v2 trajectories.

Per-fit outputs: `JOB.json`, `CURVE.jsonl`, `PROGRESS.json`, `EVENTS.jsonl`,
`recovery.pt`, `selected.pt`, `selected_VALID.pt`, `RESULT.json`. The selected
checkpoint is actually loaded and whole VALID rerun before export. The pilot
manifest records each own fit and `COSTS.json` sums **per-invocation** observed
optimizer work, including failed/resumed work without double-counting the
reused checkpoint prefix. Native source work, wall times and own CUDA/RSS peaks
are recorded. Worker-time sums are not elapsed pilot time.

## Optional source OGB iterator RNG emulation

Native `main_pyg.py` evaluates shuffled TRAIN, nonshuffled VALID, then
nonshuffled TEST each epoch. For its unchanged zero-worker, map-style loaders,
the inspected Torch implementation draws one global CPU int64 base seed when
the TEST iterator is created. Native GIN/GCN evaluation uses `model.eval()`;
dropout is disabled, BatchNorm reads its running state, full categorical data
has no runtime transform, and mean pooling/embedding/linear/message passing
have no explicit RNG draw. These are the narrow assumptions of this emulation.

First root may run this finite **bookkeeping audit** on the allocation:

```sh
python -B runtime_v2/audit_source_rng.py --project-root REPOSITORY \
  --redesign-root PROJECT_PATH/reference_wrapper_redesign_20261013_v1 \
  --output FRESH_PROJECT_PATH/OGB_LOADER_RNG_AUDIT.json
```

It binds the actual installed Torch/PyG/OGB versions and implementation files,
the pinned native `main_pyg.py`/`gnn.py`/`conv.py`, and exact CPU RNG continuation
after serial integer/TensorDataset loaders. It constructs no model and accesses
no task dataset, TEST provider, split or label. This command has not yet been
qualified on the allocation by this source preparation work.

After reviewing that receipt, freeze this explicit job/spec field:

```json
{"source_rng_emulation": {
  "mode": "omitted_test_iterator_base_seed_v1",
  "audit": "PROJECT_PATH/OGB_LOADER_RNG_AUDIT.json",
  "audit_sha256": "SHA256_OF_THAT_AUDIT"
}}
```

Only native OGB `source_reference` M1 is admitted, with the original callback,
global loader generators, full features, mean pooling and no forward hooks.
Changed installed/source bytes or loader flags are rejected. After actual
TRAIN and VALID diagnostics, exactly one CPU seed draw is performed before
the epoch's RNG checkpoint. Its count is checkpointed and checked on restore.
Metadata says the **omitted iterator RNG effect** was emulated; TEST inference
and metrics remain omitted. The audit is not a claim about arbitrary callbacks,
stochastic evaluation extensions, whole-paper numerical matching or test quality.

## Explicit link dispatch

`family:"link"` never falls back to the OGB policy. Declare
`link_policy_module:"PROJECT_PATH/link_policy_v1/policy.py"` and its explicit
`link_policy_sha256`. The separately frozen module authenticates the native
roster/config/source controller and constructs its task-specific selector.
Pin `callback_module`/`callback_sha256` to
`link_native_integration_v1/session.py`, with explicit repository-contained
`recipe.data_files` containing only `graph`, `train`, `valid`. Generic session
transforms are rejected here; use that integration's finalized candidate IDs.

Its policy explicitly includes `family:"link"`, metric `hits@50`, max/first,
`evaluate_TRAIN_each_epoch:false`, and
`evaluation_mode:"VALID_only_disclosed_adaptation"`. Source reference is
100/100/no patience; matched extended is1000/100/50. Native TRAIN+TEST diagnostic
order/RNG is **not** reproduced, and native numerical/runtime qualification
remains pending. Full VALID requires60,084 positive and100,000 negative queries.
Exports and I4 pooling verify exact query endpoints, role counts and labels.

The source BUDDY selector starts at0 and updates only on strict improvement.
If its entire source-horizon curve stays0, terminal status is
`no_selected_state`, no selected checkpoint/export is invented, and source
pilot costs count the outcome separately. This is not an execution failure or
a fallback to final weights. I4 still uses four separately selected native fits.

The finite24-fit supervisor retains its specific eight-fits-per-trial layout;
it is not a general source-reference, variable-arm or massive campaign launcher.
All v2 model/epoch/link paths remain unqualified until root's separately
authorized real allocation checks. No server/model run occurred in v2 preparation.

## Preparation checks

`test_bookkeeping.py` checks pure selectors, stopped-resume decisions, explicit
link planning, native singleton-tail accounting and exact24-fit/M1 reuse.
It imports no tensor/model/data library. `test_source_rng.py` checks finite
CPU DataLoader RNG semantics and rejects changed worker/shuffle/generator
settings before iteration. Its local Torch2.5.1 checks are not allocation
PyG qualification. None of these checks supplies predictive-quality evidence.
