# Executable native full-epoch pilot runner

This new runtime drives the existing frozen node/OGB sessions through complete
native TRAIN epochs. It changes no frozen smoke file, native architecture,
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
python -B runtime/run_fit.py --project-root REPOSITORY \
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
python -B runtime/run_pilot.py --project-root REPOSITORY \
  --spec FROZEN_PILOT.json --output FRESH_PILOT_OUTPUT --prepare-only

python -B runtime/run_pilot.py --project-root REPOSITORY \
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
- TEST stays closed in every pilot. The source's usual TEST diagnostic and its
  loader iterator RNG effect are omitted and disclosed. This source-horizon
  control is not claimed as exact whole-paper RNG reproduction. An eventual
  authorized full-source diagnostic mode needs its actual guarded TEST path;
  no fake RNG consumption or TEST metrics are substituted here.
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
python -B runtime/run_fit.py --project-root REPOSITORY --job FROZEN_JOB.json \
  --output EXISTING_FIT_OUTPUT --resume EXISTING_FIT_OUTPUT/recovery.pt
```

Recovery granularity is a **completed epoch boundary**, not an uncheckpointed
live DataLoader. A partial epoch lost to interruption is not counted as
complete; its terminal/partial records remain. Explicit resume may replay such
work, and unobserved costs after abrupt termination are marked unknown.
Live owner locks are never stolen. A terminated lock is archived only on an
explicit resume after its recorded PID is absent. There are no automatic
scientific retries. Prior terminal results are archived before resumed work.

Per-fit outputs: `JOB.json`, `CURVE.jsonl`, `PROGRESS.json`, `EVENTS.jsonl`,
`recovery.pt`, `selected.pt`, `selected_VALID.pt`, `RESULT.json`. The selected
checkpoint is actually loaded and whole VALID rerun before export. The pilot
manifest records each own fit and `COSTS.json` sums **per-invocation** observed
optimizer work, including failed/resumed work without double-counting the
reused checkpoint prefix. Native source work, wall times and own CUDA/RSS peaks
are recorded. Worker-time sums are not elapsed pilot time.

`test_bookkeeping.py` checks only selectors, source caps, native singleton-tail
accounting and exact24-fit/M1 reuse metadata. It imports no tensor/model/data
library. No runtime execution or predictive-quality verification is inferred
from those tests; root stages and executes the authentic allocation pilots.
