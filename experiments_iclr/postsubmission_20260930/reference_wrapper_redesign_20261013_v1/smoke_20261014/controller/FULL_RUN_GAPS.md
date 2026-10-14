# Remaining node and common full-run requirements

The **167-case short-test roster** checks authentic initial TRAIN/VALID updates,
parameter ownership, serialization/reload and replay. Individual outcomes remain
in their receipts. It does not implement the **2,480-fit campaign**: 270 cold
source-horizon M1 fits and 2,210 cold main-track fits. No full scientific GO has
been received. Original-paper datasets remain excluded; boundary GNNM is a new
baseline on the adopted new datasets.

## Concrete gaps

| Area | Current short-test implementation | Required before full-run admission |
|---|---|---|
| Contract-to-session binding | `create_session(recipe, procedure, data_root, device)` consumes smoke recipe fields, including `_seed` and `native_fixed_config`. Compiler jobs instead declare `factory`, `native_config`, `initialization_seed`, `role_bindings`, `training` and a contract hash. The corresponding full native factories/worker loop are not present in the redesigned runtime. | Implement one explicit mapping from each frozen job contract to its actual native session. Bind source/dependency/data receipts, paired split and seed, procedure, factor initializer and track. Reject changed contracts and original-five datasets. Keep the new native-preserving boundary baseline distinct from the historical generic scaffold. |
| Native completed epochs | A PolyFormer callback is one complete full-TRAIN forward/loss/update, so one successful callback is one node epoch. The smoke controller requests callbacks 1, 2, 3 and replayed 3, not a complete horizon. Minibatch tasks have their own source epoch boundaries. | Persist the native completed-epoch clock and actual optimizer maps separately. Run validation at the declared completed-epoch cadence, never treat every molecular/link minibatch map as an epoch. Record failed/replayed maps, source reductions and activation-checkpoint recomputation costs honestly. Link-specific epoch requirements are in the separate link document. |
| Node cohorts and validation selector | `_prepare_data` currently fixes `official_split_bindings['0']`. `NodeSession.validate()` returns whole-VALID logits and diagnostic CE, but empty selection metrics. `updates` is the only node trajectory counter. | Bind the actual compiler cohort split instead of hardcoding split0, retaining the prescribed ten split/seed pairs. Compute native whole-VALID accuracy for selection; preserve native maximum-accuracy/first-tie/checkpoint eligibility and source patience semantics. Keep TRAIN/VALID/TEST roles separate and do not create a mask/seed Cartesian product. |
| Source versus longer controller | The smoke session has no horizon, patience, best checkpoint or stop reason. The separate `LongerTrackSelector` is a state helper, not an attached training loop. | For PolyFormer source M1, retain cap2,000 and native patience250. For the main track, cap10,000, minimum2,000 and patience100 validation checks. Molecular/link use their own frozen contracts. Source and longer tracks are distinct cold fits; the main schedule is disclosed as extended training. Persist selection, stale counters, completed/selected epochs, curves, gates and stop reason. Hitting a cap is not convergence evidence. |
| Independent I4 selection and M1 reuse | `IndependentBank.train_update()` advances all four bodies together; its `validate()` returns empty metrics and a mean of member validation losses. Short checkpoint replay restores the combined smoke bank. It has no independent best-state or stopping authorities. | Execute the four compiled native bodies as four independent fit records, each with its own optimizer/RNG, validation selector, stopping and selected state. Pool only their final selected raw logits. M1 must reuse exactly body0's fit authority/checkpoint/predictions; do not train another M1 or choose one pooled I4 epoch. The smoke bank is not a substitute for this full lifecycle. |
| Durable state and resume | Short replay restores into an already constructed process/session, including an immediate exact loaded-state check in the amended controller. It does not qualify a fresh worker, epoch rollover, selector state or interrupted export. | Save complete recovery and selected-state checkpoints atomically: native model/buffers, optimizer, any scheduler/phase, module modes/flags, private/global RNGs, sampler cursor, epoch progress, selectors and costs. Reconstruct a fresh source session, verify source/contract identity and resume without resetting clocks or borrowing another fit's state. Retain every old replay failure and repair receipt. |
| Final TEST serving and endpoints | Smoke ports deliberately operate only on TRAIN/VALID. Their tensor receipts and checkpoint are not final selected prediction banks. The compiler's statistical endpoints are metadata declarations. | After selection is frozen, restore the fit's own selected state and export authenticated final predictions under the official native evaluator/query/topology law. Align sample/class axes and preserve native M1 interpretation; GNNM/I4 share the declared raw-logit mean. Export work belongs to fit finalization and is charged there. Statistical endpoints consume these banks without constructing models, selecting configurations or running heldout-guided retries. |
| Three-GPU dispatch and resources | Sequential smoke dispatch and a physical two-job allocation probe are engineering tools. `scheduler_policy.py` is pure admission logic; it launches no process and creates no registry lease. The documented per-GPU one-command interface is not yet an enabled supervisor. | Integrate an immutable campaign registry, atomic claims, owned process identity, recovery/terminal states and dependency checks with a detached supervisor. Bind allocation/GPU0 and authorized77/GPU0/GPU1; allow at most two owned jobs per GPU only when reservations fit measured availability. Sum VRAM per GPU and RAM/disk over both GPUs on77. Authenticate 77's current route/runtime before deployment, preserve other users' occupancy and never access the restricted allocation for science. |

## Resource and launch qualifications

A short pair probe measures its own two processes on one GPU. It is not an
upper bound for mature training, different batch sizes, caches, optimizer state,
dataset/preset combinations or the 77 runtime. Populate per-fit reservations
from actual receipts plus a documented headroom policy; record uncovered
combinations instead of allowing an unmeasured second slot. Sampling alone can
miss a transient peak. Fit counts and relative route/epoch weights are not
measured GPU-hours or a completion ETA.

The final one-command entry points must verify the active campaign/contract
hashes, explicit human GO, authorized hostname/GPU/runtime/source bindings and
the retired-v29 guard before claiming work. A second invocation must neither
duplicate a supervisor nor steal a live lease. Keep entire frozen cohorts on
their assigned GPU as declared; merge results by exact campaign and fit identity.
No scientific fit or TEST evaluation is authorized by this gap document.

## Admission evidence still needed

The future readiness receipt must identify the implemented full-worker source,
all27 preset bindings, own-I4 selection/member0 reuse, complete native epoch and
fresh-process recovery paths, selected-state serving, atomic multiworker claims,
measured resource reservations and the three dedicated launch commands. Keep
unresolved source/seed/publication fidelity explicit. Do not infer these from a
passed short matrix, metadata QA, physical pair probe or proposed manifest.

Source-specific link work is documented in `../link_port/FULL_RUN_GAPS.md`.
Molecular full-epoch/source requirements belong to its port owner; this document
does not duplicate those implementation details.

## Sources inspected

- `../node_port.py`: `_prepare_data`, `train_update`, `validate`, state/load.
- `../independent_bank.py`: own RNG streams and current short bank lifecycle.
- `smoke_case.py`, `dispatch_smokes.py`, `pair_memory.py`: engineering scope.
- `../../engineering/campaign_compiler.py`, `convergence_policy.py`,
  `scheduler_policy.py`, `DESIGN.md`: disabled campaign contracts/helpers.
- `../../README.md`: adopted count, horizons, roles and reporting rules.

This is read-only source analysis plus this documentation file. No source code,
plan, manifest, setting, model, dataset or numerical result was changed; no
server execution or paper review was performed.
