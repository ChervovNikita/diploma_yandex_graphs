# Composable normalization v2 — selected-state lifecycle repair

This immutable source successor preserves `../composable_normalization_v1`.
It retains the same policy, candidate algorithms, 2,890 proposed contract
migrations, 3,570 unchanged fit definitions and 6,460-fit total. No roster,
production job, callback or native checkpoint was changed or regenerated.

## Two source blockers reproduced and repaired

The pinned `LinkSameCallMeanAdapter.load_state`, inherited by
`LinkDeclaredExtensionAdapter`, loads its nested parent and immediately calls
that parent's `state()` for exact verification. V1 held the normalization
owner in `load` mode throughout, so its native export guard rejected the read.
The previous flat-parent scalar fixture did not exercise this real lifecycle.

V2 identifies only that exact pinned parent method and temporarily owns its
exact nested `load_state`/`state` target. The first load must return successfully
before one verification read is admitted. During only that read, the owner
switches from `load` to `export`, then restores `load`. Native state guards
still require the exact operation role. An early/repeated read or a changed
lifecycle refuses; original target methods are restored in `finally`. Plain
native export during load remains forbidden. Guards are never disabled.

The parent selected-checkpoint binder also expects its own parent state,
whereas a genuine composed checkpoint contains the complete new envelope.
V1 delegated the complete envelope directly, so the parent identity check
necessarily rejected it. V2 restores and verifies the full composed envelope,
then authenticates that exact envelope, job/epoch/checkpoint/runtime view and
live private-BN bank state at selected binding. It passes an explicit shallow
payload view containing only `parent_adapter_state` to the parent's unchanged
identity check. The caller's complete payload is not mutated or discarded.

The outer owner retains the full-envelope and bank identity. Selected mean
export rechecks that authority, delegates the parent's existing zero-forward
getter, and appends its composed selected-state proof. Changed payload banks
or live banks refuse. A new TRAIN call or restore invalidates selection;
rejected nested TRAIN calls do not clear a valid proof. No new validation,
prediction, metric, checkpoint selection or optimizer work is introduced.

The checkpoint envelope layout and underlying BN law remain unchanged; exact
new producer/source identity still prevents V1 states from being relabeled as
V2 states. The original inference API and all original role/state guards remain.

## Focused evidence and limits

`check_source.py` retains the three source/scalar route cases and adds one
selected-state regression executing **actual AST-extracted parent method
bodies** for restore, bind and mean export. It reproduces both V1 errors, then
passes the V2 path, verifies scoped export permission, restoration of original
methods, altered-bank rejection and zero-forward export. Tensor identities,
runtime checks and native sidecar data use explicit scalar fixtures; these are
not native model, real runtime, role-population or numerical qualification.
No Torch import, native forward/update, TEST payload, remote call or Git work
occurred. `LOCAL_CHECKS.json` records the actual result; `DELTA.diff.txt` shows
the narrow source changes against preserved V1.

Existing native/input/population/resource evidence remains reusable within
its original scope. The minimal new native checks remain the molecular and
nonzero-seed BUDDY owner/state representatives from V1, now including the
actual selected-checkpoint bind and same-call mean export. Inspect finite
shared/private/gate gradients and per-route BN counters. A two-update molecular
harness must own the parent's real epoch iterator; a direct session update on
an inactive iterator is not supported. Do not change the production epoch
driver to accommodate a fixture or add a full method/preset test matrix.

Genuine producer/runtime/serving and portable-path-schema integration remain
pending. In particular, the parent's `_check_runtime` still authenticates its
original helper runtime; this repair does not substitute new producer hashes
or weaken that check. A future producer must bind outer and parent helper
source identities honestly. Current frozen runtime entrypoints do not call this
new constructor. The scalar fixture explicitly stubs runtime qualification.
No campaign readiness, accuracy, novelty or GO is asserted here.
