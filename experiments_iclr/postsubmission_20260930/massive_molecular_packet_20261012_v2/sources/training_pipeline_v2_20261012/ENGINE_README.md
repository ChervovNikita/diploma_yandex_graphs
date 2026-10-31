# Training engine contract

`engine.py` imports only the Python standard library. The caller injects Torch
and constructs one adapter per independent trajectory. These CPU tests verify
engineering behavior; they provide no scientific accuracy evidence.

## Prospective engine specification

```python
{
    "fit_id": "fresh-paid-fit-id",
    "budget_steps": 1000,
    "budget_unit": "optimizer_updates",  # or "epochs"
    "validation_interval": 1,
    "validate_initial": True,
    "prefixes": [
        {"prefix_id": "legacy", "cap_step": 1000,
         "close_on_selector_stop": True},
    ],
    "inherited_fit_ids": ["previous-paid-fit-id"],
    "adapter_spec": {
        # Declare native loss scope/reduction, optimizer procedure, validation
        # fusion, selector/ties/cadence, RNG streams and nonpersistent inputs.
    },
    "convergence_window": 10,  # optional descriptive trace summary
    "recovery_interval": 100,  # optional; completed adapter units per recovery save
}
```

`budget_steps` counts adapter calls in the declared unit. An epoch call may
perform several optimizer updates. Reported total updates must equal the
declared budget when its unit is `optimizer_updates`. Each update-budget call
therefore normally performs one update. The initial and final validation calls
and planned prefix caps are explicit; other validation calls follow the interval.

## Adapter interface

- `model` is a Torch module and `optimizer` owns every trainable model parameter
  exactly once. Frozen parameters are excluded. `inspect_independence(adapters)`
  rejects shared parameter storage (including distinct parameter objects and
  registered frozen parameters), optimizer objects or adapter objects across
  independent bodies; each adapter must also own its selector state. Shared
  read-only ancestry belongs to declared external input custody.
- `train_step(step)` owns train modes, forward calls, backward calls and updates.
  It returns numeric `objective`, `ce`, `regularization` and `operation_counts`.
  Every successful call reports a positive integer `optimizer_updates`; all
  operation counts are nonnegative integers. Minibatched epochs report
  sample-weighted means over their declared training scope, together with
  sample counts. A last-minibatch loss is not an epoch metric.
- `validate(step)` uses validation authority and the declared dropout-off
  inference/fusion procedure. It returns `objective`, `ce`, `regularization`
  and its selection metric. The engine does not choose a fusion or loss rule.
- `observe_validation(step, metrics)` sees only validation metrics and returns
  `save_selected` and `stop_requested` booleans, plus optional plain selector
  details. The adapter owns min/max, ties, patience and save eligibility. For
  example, original GNNM can reset staleness on off-cadence improvements while
  saving/updating its saved best only on eligible multiples of ten.
- `capture_state()` returns a plain tensor/container tree containing the entire
  selector, every used global/private RNG stream and necessary nonpersistent
  input custody. `restore_state(state)` restores it. Neither method may draw
  randomness. Convert NumPy RNG arrays to tensors/lists before returning them.

Original GNNM raw-logit mean and PT probability mean remain distinct adapter
procedures. Independent I4 bodies use separate adapters, fits and selectors.
No backbone, dataset shape or universal hyperparameter recipe is built in.

## Running and restoring

```python
result = run_trajectory(torch, adapter, engine_spec, empty_output_directory,
                        resource_check=optional_callback)
restore_checkpoint(torch, checkpoint_path, adapter,
                   expected_fit_id=engine_spec["fit_id"])
```

The callback has signature `(phase, step, current_result_copy)`; it may return a
plain JSON resource observation or raise to stop the run. The copy includes
current status, counters, costs, prefix state and `resource_observation_count`;
growing observation/attempt/admission histories are omitted to avoid quadratic
copying during long runs. Boundaries include before
training, before/after each step, before validation and before/after restoration.
The caller owns hardware admission, construction costs, dispatch and registry
transactions. Engine wall timings measure its calls; an adapter must synchronize
accelerator work if it needs completed-device timing.

Training always consumes the full budget. A selector stop closes only eligible
virtual prefixes. Each prefix follows the same validation selector until its
declared stop or cap, then its separate `prefixes/<id>/selected.pt` bytes freeze.
The full selector continues; the engine never resets from a prefix checkpoint
to extend a trajectory. Each closed prefix gets a certificate with its closure
reason/step and selected step. `selected.pt` holds the full-budget selected
state; `final_training.pt` retains the final trajectory state. The engine restores
the full selected state before returning.

`latest_trajectory.pt` periodically retains the current trajectory model,
optimizer, modes, RNG, selector, logical counters, trace summaries and exact
selected/prefix file bytes. It is saved initially, at the recovery interval,
when a prefix closes, at prefix caps and at the final unit. The default interval
is 100 completed units. A controlled interruption raised by a before/after step
resource callback also saves the current complete unit; a mid-training or
mid-validation interruption retains the previous safe recovery state. Saving each unit
minimizes replay; a larger declared interval reduces checkpoint I/O. These
recovery files can be larger than an ordinary model checkpoint because they
include selected states too. Include this I/O in prospective resource budgets.

Checkpoints retain model/optimizer tensors, all named module mode flags,
`requires_grad`, adapter-owned RNG/selector state and validation selection
details. Restoration clears transient gradients and checks exact saved state.
Mixed module modes are restored directly, since recursive `.train()` changes
children. Checkpoints are ordinary trusted Torch files written atomically.

`spec.json`, `trace.jsonl` and `result.json` retain the prospective decisions,
objective/CE/regularization traces, operation counts, costs, diagnostics and any
failure. Failure records preserve earlier selected states and propagate the
original error. Raise `AdapterStepError(message, operation_counts=...)` after a
partially failed call to report operations already performed; ordinary training
exceptions explicitly mark partial operation counts unknown. Never overwrite a
previous attempt by rerunning in its output directory.

## Explicit interrupted-fit recovery

```python
result = resume_trajectory(
    torch, new_adapter, unchanged_engine_spec, previous_output_directory,
    admission={"previous_owner_terminal": True,
               "previous_owner": verified_owner_receipt},
    resource_check=optional_callback,
)
```

The registry/launcher must prove that the former owner has ended before giving
this receipt. A polling timeout or an old-looking lock is insufficient. The
engine does not auto-resume. It requires the same fit ID and exact prospective
specification; adapter restoration also verifies its bound external custody.

Recovery starts from the latest complete trajectory state, with its full
selector and prefix closure records. It does not start from the selected best.
The prior attempt's result and checkpoint files are copied under `attempts/<n>`;
the old trace remains and a new attempt appends records. Closed prefix bytes are
preserved exactly. Full selected and prefix snapshots are recovered from the
saved capsule even if later work advanced the ordinary endpoint files.

`operation_counts` includes all reported paid operations across attempts.
`trajectory_operation_counts` describes the recoverable logical trajectory and
is used to check its declared update budget. When a larger recovery interval
requires replay, previously paid updates remain charged and repeated updates
are charged again. `replayed_completed_steps` reports this difference. A killed
owner with no terminal result leaves explicit unknown partial operations and
incomplete actual costs; its known lower bound is retained. Recovery cannot
make unknown work disappear from the accounting.

The recovery API and interruption tests require allocation CPU verification
before being treated as verified. Production admission remains with the user
and the study launcher.

The convergence summary is descriptive: first/last/min/max values, recent mean
and recent change per step. It does not infer scientific adequacy or use DEV to
choose a budget. Inherited fit IDs are references; their paid costs belong to
the caller's registry rather than being charged again here.

## Engineering tests

From this package directory, with the existing approved Torch runtime:

```sh
python -m unittest discover -s tests -p 'test_engine.py' -v
```

Tests cover full-budget continuation beyond a stop, frozen prefix bytes,
initial/cap/last validation cadence, exact optimizer/mixed-mode/global/private
RNG restoration, parameter ownership, sample-weighted epochs with multiple
actual updates, retained intentional failures and nonfinite-loss failures,
parameter-storage aliases, interrupted-fit recovery from the latest state,
replay accounting and explicit incomplete accounting after a killed owner.
When Torch is unavailable, these tests skip; do not install a large runtime just
for the local engineering check.
