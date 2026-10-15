# Link policies bound to the frozen 27-cell roster

This package defines policy and selector code for the three existing link cells:
`collab-ncn`, `collab-ncnc` and `collab-buddy`. `BINDINGS.json` pins the frozen
27-cell `ROSTER.json`, saved native main programs, native snapshot manifest,
source command evidence, loader/RNG evidence and the frozen integration manifest.
It copies the relevant roster contracts; it does not alter or expand the roster.

Status: pure policy/bookkeeping checks pass. Actual native gradients, complete
epochs, checkpoint/replay behavior, resources and integration with runtime_v2
still require qualification. No model or dataset was imported, no saved native
program was executed, and no server, optimizer, VALID or TEST operation ran here.
`link_native_integration_v1` remains frozen.

## Explicit policies

| Policy field | `source_reference` | `matched_extended` |
| --- | --- | --- |
| Frozen roster track | `native_replica` | `long_convergence` |
| Family / selector | `link`, `hits@50`, max, first ties | same |
| Maximum completed native epochs | 100 | 1000 |
| Minimum before staleness counts | 100; no early stopping | 100 |
| Patience | disabled | 50 full VALID checks |
| VALID interval | every completed native epoch | same |
| Checkpoint zero | excluded | excluded |
| Warmup / source schedule | zero / constant native optimizer groups | same groups and LR/WD |

Source caps come from the saved author Collab commands/defaults and the frozen
roster, including NCN's command override of its generic parser default. The
larger cap, minimum and patience come from the roster's already declared longer
controller. This package does not choose new caps or translate a node/OGB
fallback into a link policy. It rejects `matched_pilot`: a shorter pilot needs a
separately frozen policy specification.

Declare this complete dictionary, choosing the values for the desired track:

```python
{
    "family": "link",
    "track": "source_reference",
    "maximum_epochs": 100,
    "minimum_epochs": 100,
    "patience": None,
    "metric": "hits@50",
    "direction": "max",
    "ties": "first",
    "evaluate_TRAIN_each_epoch": False,
    "evaluation_mode": "VALID_only_disclosed_adaptation",
}
```

The explicit evaluation mode is mandatory. The frozen link integration exposes
full VALID and has no native TRAIN diagnostic method or TEST path. A request for
source-equivalent evaluation, TRAIN diagnostics, last-tie selection, a different
metric/cap/cadence or an unsupported native phase fails instead of inheriting a
default. Policy construction authenticates the pinned source bytes without
importing or executing them.

## Selector behavior

The common runtime metric key is lowercase `hits@50`, in raw 0..1 units. NCN's
native main records `Hits@50` among its displayed metric names. Both saved mains
use strict validation improvement; equal scores retain the earlier winner.
Selection uses a positive completed native epoch and the entire official
VALID population, 60,084 positive and 100,000 negative query rows.

There is a source-specific initialization detail:

- NCN/NCNC initialize `bestscore` from the first completed-epoch evaluation,
  including a first score of zero.
- BUDDY initializes `val_res=0` and only updates on `tmp_val_res > val_res`.
  Source-reference BUDDY therefore has no winner if every score is zero. The
  selector preserves `selected_epoch=None`; the runner must retain a
  `no_selected_state` outcome and must not substitute an arbitrary epoch.
- The declared extended controller initializes best from its first eligible
  positive epoch, as the saved longer-track selector does. Thus extended BUDDY
  can select a first score of zero. This is an explicit controller difference.

All epochs in this API are completed-epoch counts starting at one. BUDDY's
native `best_epoch` is a zero-based loop index; a native winning index maps to
this API's count by adding one. Persisting full selected model/buffer/RNG state
is new runner bookkeeping; it is not a claim that the original BUDDY logger or
NCN's optional save flags used the same checkpoint-writing behavior.

For the extended track, pre-minimum misses reset staleness to zero. A miss at
epoch100 counts as the first stale check, consistent with the saved
`unit >= minimum_units` gate. With a best score at epoch1 and no later
improvement, patience50 stops after epoch149. Strict improvement resets the
counter. The source has no warmup or schedule transition under the pinned
Collab settings, so its named phase gate is resolved from those settings;
this says nothing about numerical/runtime qualification.

`stopping_status()` is read-only. A runner must check it before TRAIN,
including after resume, so a cap or patience-complete checkpoint does not
receive another optimizer map. A cap reached without a patience plateau is
reported as `cap_without_patience_plateau`, not as evidence of convergence.

## APIs for runtime_v2

```python
policy = LinkPolicy.from_dict(value, native_family="NCN", root=repository_root)
policy = create_policy(job, root=repository_root)
selector = create_selector(policy)
selector.stopping_status()
selector.observe_validation(completed_epoch, raw_full_VALID_result)
saved = selector.state()
selector.load_state(saved)
```

`LinkPolicy.__dict__` and selector state are JSON-safe and bind the source,
evaluation adaptation and stopping rules for strict resume. `create_policy`
requires exact roster `native_fixed_config`, dataset `ogbl-collab` and split0.
The source role admits unextended native M1 only, repetition seeds0..9 and
member zero. Four independent models require four own fit jobs, selected
checkpoints and stopping counters. The policy rejects a pooled `independent4`
fit job.

`LinkSelector` exposes `completed_epoch`, `selected_epoch`, `best_metric` and
`stale`, plus `observe(epoch, metrics)`, `state()`, `load_state()` and the
nonmutating `stopping_status()`. Prefer `observe_validation`, which authenticates
the VALID role/completeness/count envelope. A runtime calling `observe` directly
must validate that envelope before passing metrics. Resume must restore the
selector together with the corresponding complete-epoch native session and
global RNG checkpoint.

The frozen native integration factory remains
`create_adapter(torch, job, policy, root)`. It recognizes `policy.family="link"`
and the fixed metric/direction/tie fields. runtime_v2 loads this policy module
using a pinned project path, delegates `create_policy` and `create_selector`,
and provides any runner-specific receipt bridge outside the frozen integration.
Static policy tests do not authorize or qualify native dispatch.

## Source role and evaluation/RNG limitations

The source role means native M1 at the source settings and 100-epoch horizon,
with the evaluation adaptation declared above. It is not an exact reproduction
of the whole author main program.

Both native mains evaluate TRAIN, VALID and TEST during training. The current
integration evaluates VALID alone. VALID remains a selection split; it never
stands in for TEST. Native auxiliary metric logging, source TEST topology and
TEST evaluation are absent. A final selected-state TEST evaluator would require
separate native graph/query bindings and authorization; none is provided here.

For BUDDY Collab, `train_eval_loader` reuses the shuffled TRAIN loader, and its
source VALID and TEST loaders also shuffle. Ordered VALID in the frozen port
and the omitted evaluation calls change loader/RNG consumption. No skipped TEST
iterator or guessed RNG compensation is added. NCN source runs TRAIN prediction
before VALID and may rebuild TRAIN+VALID topology before TEST; omitted calls and
construction order are not assumed RNG-neutral. The NCN source seeds Torch
CPU/CUDA and NumPy; the common controller also binds Python random.

Source repetition seeds0..9, official temporal split and independent member
offsets remain explicit separate axes. Source/global/provider RNG equivalence
is false/unverified in the policy. Native state replay within the adapted
runner must still be qualified. BUDDY's original persistent feature cache also
differs from the frozen port's on-demand native feature hydration; storage and
timing fidelity remain disclosed limitations.

Run the pure check from the phase directory:

```sh
python3 -B benchmark_expansion_20261015_v1/link_policy_v1/check_policy.py
```

See `SOURCE_EVIDENCE.md`, `LOCAL_CHECKS.json` and `MANIFEST.json` for source
locations, checks, preserved preparation failure and exact source receipts.
