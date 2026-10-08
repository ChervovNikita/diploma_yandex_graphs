## v2 serialization-only successor

The v1 first scientific attempt stopped when the JSON trace writer received
three detached native scalar tensors: `loss`, `own_mean`, and `auxiliary`.
The sole program change calls `.item()` for each existing `TRAIN.native` value
when forming the trace entry. Every field and scalar value is retained; no
prediction is converted or dropped. The raw `train_step` return/API, bank code,
updates, evaluations, strict-first selectors, masks, seeds and gates are unchanged.

`SERIALIZER_ONLY_PATCH.json` binds v1 and the unchanged native return source.
The v1 CUDA engineering receipt covers the unchanged numerical code; this patch
has static verification only and requires root source adoption before a fresh
complete-family launch. Failed v1 artifacts and every sealed predecessor remain
intact. No qualification run, dataset, checkpoint, partial score or server was
opened during preparation.

# Disabled single-trajectory four-bank screen

This new callable source reuses the sealed native integration, candidate and
controls unchanged. No numerical imports, datasets/checkpoints, server calls,
qualification, scientific launches or canonical edits occurred in preparation.
All runtime/competence/resource/selected-state gates remain separate.

## Frozen operation

One native WikiCS single per seed6101/6203/6307 runs the original1100epochs,
100local then1000global, two own-CE stochastic forwards and one native Adam.
The original public `Session.train_step` performs one backward call through its
two-view mean loss. Four independently owned correction banks receive **the same
second existing old-state H/base capture**: C4, S_joint4head, U4_sharedB and
secondary S_one_path. There is no native replay, extra TRAIN forward, both-view
correction update, learned sharing across banks or correction feedback into B.

Every bank issues its own pending token from the same explicit private mask seed;
ordered positions/IDs/draw numbers/value scales must match before any bank update.
All Q labels are excluded from all literal contexts. Mask draws precede native
features. Native RNG/parameter custody guards wrap construction, all bank updates,
serving and restoration. Native gradient copying is optional engineering
observation (`verify_native_gradients=True`), not a training intervention.

`SEEDS.json` prospectively fixes candidate/one-path/control route0 initializer
to the native seed and other full control heads to native_seed+1009/+2018/+3027.
Mask seed is native_seed+1900001. This deliberately bypasses the predecessor's
`_new_corrector/_core_seeds` rule (+1700001), as root resolved in favor of the
frozen screen. The predecessor remains unchanged; native reset/dropout streams
are unchanged and every constructor is isolated.

Per epoch, correction work is13attention branches,10own-CE backward calls and
7Adam calls. Joint single's one backward differentiates all four heads. C4/U4
use CE/4; the singles use one own CE. All the source maps/label widths,
all-neighbor denominators, conditional value scale and probability serving remain
the sealed operation. At fixed parameters and mask-independent H/scores, the
fixed **linear** joint readout preserves the restricted message/residual-logit
first-moment identity. It does not make probabilities, loss or gradients unbiased.

The one original complete5274-node native evaluation supplies a shared full
capture. Each bank then serves all development IDs with all580TRAINlabels,
scale1. The native own-local strict-first correctcount selector controls the
stage transition. At epoch101 the native model/Adam is restored once, and every
bank's full learned parameters/Adam are restored from that same native epoch.
All end-local native/mask streams, physical steps and counters stay live.
Parameter-origin epoch and logical executed-step count are recorded separately.

Each arm has its own strict-first full-development correctcount maximum across
all1100epochs and saves a coherent same-epoch native mode/model/Adam plus that
arm's corrector/Adam. Selected epochs may differ across arms. Secondary one-path
cannot replace either co-primary. The driver computes no contrasts or whole-family
promotion verdict. Private per-epoch readouts are retained for required selection,
not authorization to open partial comparative quality. All12correction records
and all3native blocks must close under the unchanged protocol before root's
aggregate opening. Source whole-family gate/owner/collector are not supplied here.

Root supplied structural reach52.37%Dev and39.405%expected TRAIN-query visibility,
original byte-identical21MBpayload custody and80GBVRAM/~3.8GBused. These are
reported context, not measurements by this packet. They change no population,
operator, threshold or subset-scoring rule; runtime/data-copy verification is
still required.

## Public callable interface

`run_complete(train=..., valid=..., output=..., polynormer=..., seed=6101,
device="cpu", later_execution_authorized=False)` completes exactly one frozen
native seed/all four arms in a fresh output. Default refusal precedes dependency
imports or data access. Partial failures retain counters/costs; no retry/resume
or campaign launcher exists. Complete records declare one native trajectory and
four correction records; three separately admitted blocks would reduce12native
replays to3. This is a source design, not measured speed or runtime parity.

`make_session(train_data=..., seed=..., device=..., polynormer=...,
purpose="engineering_qualification", later_execution_authorized=False)` exposes
the executor used by that full driver. Its session has `native`, `banks`,
`capture`, `engine/common/data/driver` and:

- `draw_common_masks()` and `train_step(batch, labels, epoch=...,
  masks=None, verify_native_gradients=False)`.
- `update_banks(H, base, masks, native_capture=...,
  verify_native_gradients=False)` reuses the same bank update path.
- `serve_banks(H, base, heldout_ids, native_capture=...)` and
  `serve_ids(heldout_ids)`; one native serving capture, no heldout truths.
- `snapshot(epoch=..., kind="engineering_snapshot", native_metric=None,
  stats=None, arm=None)` and `restore_native_own_local(state)`.
- `close()` removes native read-only hooks.

Engineering purpose refuses the full evaluation method before VALID truth access.
Its discarded snapshot explicitly records `selector_performed=False`,
`snapshot_purpose="engineering_qualification"`, top statsNone and native
metric/member valuesNone. A predetermined local snapshot at epoch1 can restore
native/Adam and all banks once after two local updates; engineering epochs101/102
can then have live logical steps3/4. This does not reproduce the scientific
1100/100 schedule or select an epoch from outcomes.

`reconstruct_for_serving(state, train_data=..., polynormer=..., device="cpu",
later_execution_authorized=False)` accepts trusted coherent final-arm or discarded
engineering state. It exposes native/banks, restores all saved learned/Adam state,
checks actual TRAIN/context/source/recipe identities and supplies `serve_ids`/close.
Training/update/mask generation and evaluation are refused in this serving-only
session. It reads no VALID truth and offers no exact training resume.

Actual full-data CUDA construction, isolated RNG/gradient behavior, nonvacuous
mask/zero-context diagnostics, local/global restoration and serving roundtrip
remain engineering qualification work. Complete label-aware single competence,
finite owner/release/runtime/resource custody and the later independent4,
label-aware/live-gradient/label-free/unused-confirmation references remain pending.
