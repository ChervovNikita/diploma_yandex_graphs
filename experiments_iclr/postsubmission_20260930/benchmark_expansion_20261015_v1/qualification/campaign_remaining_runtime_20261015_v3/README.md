# Remaining runtime witnesses V2 — reviewed bug repairs

Source preparation only. No remote/model/tensor/array execution occurred while
building this package. Both root templates remain **TEMPLATE_NOT_ADMITTED**.
Publication, admission and remote execution remain with root. Full campaign GO
is still closed.

## Frozen predecessor and narrow repairs

V1 remains byte-for-byte frozen at manifest SHA
`a6b267210c900c61c6852fe0b1df07580b273c0102d8a33857fb0285b31334df`.
Its source review found two deterministic blockers before any runtime launch:

- The actual bundled host descriptor has `worker`, but the profile projector
  requires `host`. V2 derives `host=allocation` from the verified exact worker,
  preserving every original config field and rejecting conflicting identities.
- The unchanged native epoch driver calls `train_update` until its own
  `EpochComplete` exception. After 38 successful updates it needs one final
  callback to discover loader exhaustion. V2 records **39 callback calls,
  38 successful updates, and one exhaustion-only probe** separately. Native
  forward, reverse-mode and optimizer guards remain closed during that final
  probe; early exhaustion, an extra return, a second probe and callbacks outside
  the epoch are rejected. The original native loop handles its own exception.

`runtime_guards.py` contains the exact functions used by the child and tests.
`regression_checks.py` uses the actual pinned allocation JSON, extracts the
unchanged native `EpochComplete` class and epoch loop AST, and executes only
that scalar control loop against metadata callbacks. It executes no native
model, tensor, array, optimizer or data payload. The source test sends the
adapted real host config to the actual frozen profile projector; local byte
closure is checked separately from the remote destination root. The port,
model/helper sources, training limit and BUDDY replay remain unchanged.

## Bounded checks

| Case | Actual work admitted | Reservations | Deadline |
| --- | --- | --- | --- |
| `bound_observer_molbace` | One complete new-producer native MolBACE epoch (**38 new updates**), own selected restore/export, then stopped finalization with zero additional updates. Two constructors, three restores, three whole VALID calls, 212 route forwards total, two same-call means and two genuine bound selected-export sidecars | 4 GiB GPU; 4 GiB host RSS; 1 GiB disk; 2 CPU / 1 interop threads | Child 180 s; supervisor 190 s; TERM then KILL after 10 s |
| `buddy_selected_VALID` | Frozen repaired replay of existing paid state: one constructor, one restore, one whole VALID (160,084 queries), four BUDDY forwards, **zero TRAIN/backward/optimizer/reselection/TEST** | 40 GiB GPU; 16 GiB host RSS; 8 GiB disk; 2 CPU / 1 interop threads | Child 600 s; supervisor 610 s; TERM then KILL after 10 s |

Each case is a separate process and fresh output. No per-seed grid, automatic
retry, queue wait or fit fallback is provided. Do not invoke
`run_remaining_runtime_check_31.py --case buddy_seed1`: it repeats the 1,585-update
paid qualification. The BUDDY branch here calls the unchanged frozen
`composed_BUDDY_selected_VALID_v1/replay.py` with repaired scorer V3.

CPU cannot discharge the original BUDDY witness: its authenticated original job,
producer and CUDA RNG require `cuda:0`. The new producer witness is also admitted
on the allocation GPU. Children use physical mask
`GPU-44039938-fd82-41d2-fefd-de71514e2fac`. The supervisor reuses the helper's
`device='cpu'` host-metadata preflight branch only, then separately checks the
physical GPU and peer growth; this is not CPU native execution.

## Why the new producer needs 38 updates

The earlier unpublished draft could have replayed old MolBACE stopped state
with zero updates, but that would exercise the old producer. The actual
successor is `server51_campaign_port_v1/MANIFEST.json`, SHA256
`5107ebd694b1333629ec306808f64eb3e63c2b9fc24e4f4e3cbab7f0649b9209`.
Its new composed manifest is
`30f41e1f37d467ea68537408fc90da0d07ce19f6f05a8506aac234cdaff853f5` and checkpoint
binding schema is `server51-composed-normalization-genuine-runtime-binding-v1`.
The old checkpoint has the original producer manifest and schema. Reusing or
relabeling it would fail genuine producer matching. The source tests explicitly
verify rejection by the successor authority.

The native selected checkpoint appears only after a complete epoch. Therefore
this check honestly admits and charges 38 new updates, retaining the earlier
38-update qualification as separate historical cost. It uses the authenticated
original one-epoch engineering job on the allocation with its native source,
settings, data and policy unchanged. It trains a new own trajectory under the
actual successor producer, then resumes only that new own recovery checkpoint.
No old tensor checkpoint or old selected export is deserialized.

## Actual reporting path

The child uses the successor's actual `reporting_projection.project_profile`
to project a singleton engineering profile. Canonical engineering identities
and nonempty binding fields remain unchanged; runtime descriptor, source and
new-producer authority are bound separately. This profile is not a campaign row
or production admission. It binds native source, data-role/source metadata,
controller, evaluator and normalization to the retained authentic engineering
inputs. All observed source receipts are verified before and after execution.

Both native invocations pass through the successor's genuine
`ObservedArgvTransform` and unchanged `reporting/runtime_entry.main`, which
loads its own genuine runtime and uses the genuine normalization-aware observer
around original `run_fit.run` and original `atomic_torch`. The guard wrapper
installs native counters only after that entry imports its runtime. Generic
`host_binding`, `common`, and `adapters` modules are not preloaded ahead of the
entry. A scalar projection helper named `descriptor_binding` is removed before
native imports to keep the fresh producer module invariant.

The single epoch has 38 native optimizer updates and 152 TRAIN route forwards.
Its 39th session callback is the separately counted native exhaustion-only probe;
it performs no native forward, reverse-mode or optimizer transition.
The epoch VALID pass and two selected finalizations each have 151 graphs in
five batches / 20 native route forwards: 212 forwards total. Only the first
stage may train or execute reverse mode. Selected/recovery checkpoint bytes
must remain unchanged during the second, stopped invocation. The mean exporter
must use the same predictions and add zero native forwards. Validation checks
original graph IDs/order and unchanged model/private BN/optimizer state.

Pass requires two actual selected sidecars and two invocation records, genuine
new-producer witnesses, exact attached engineering bindings with no unresolved
reasons, matching selected export/checkpoint receipts, and zero extra native,
RNG or tensor-deserialization work added by the observer. Each first-stage
RESULT/export is preserved before stopped finalization writes its second result.

This qualifies the actual successor's composed producer + profile projection +
argv bridge + reporting entry + observer + own checkpoint/selected export on the
allocation. It does not certify all production profiles, destination GPU
runtimes, lexical-launcher exec behavior, full backend/ledger custody, pair
capacity, or the final campaign release. Those remain separate explicit gates.
The frozen BUDDY replay qualifies its repaired serving source separately; its
custom saved schema cannot substitute for this genuine selected-export witness.

## Shared slot, peers and live HEAD

The allocation PubMed owner holds
`analysis_outputs/selected_VALID_collection_gpu_slot.lock` through all eight
fits + four collections + export. Between-fit gaps do not free it. Invoke only
after its terminal and lock release. The supervisor takes that lock nonblocking
and holds it through child cleanup and terminal evidence writing. A busy slot
preserves a failed preflight without a native child. There is no retry or wait.
The checks may run while server51 pilots remain, subject to fresh resources.

The only peer allowed is the existing GRU PID 640428 / start ticks 6055039165,
or no peer after it exits. Its growth envelopes remain 16 GiB host / 32 GiB GPU
and its separate seven-day limit is unchanged. CPU capacity, cache-aware cgroup
host headroom, disk, physical GPU identity, GPU processes and peer growth are
checked before launch; whole child process-group RSS/GPU/disk are sampled.
Never add a third GPU model worker. This does not qualify arbitrary concurrent
campaign jobs.

Preserve live HEAD `48329b19941ea869c439e5a220a3df0e30aa1237` and physical map
SHA `31cdc3346597c1d622a90ef02e48d21360a0ca0bd4cc43369c4437b40a4f5898`
while PubMed owners verify them before each child. Publish through a separate
checkout and stage this new directory without changing existing source, maps,
descriptors or live HEAD.

## Exact root commands and detached option

Use the reviewed literal allocation route with hostname/sole-UUID guard and
qualified lexical interpreter. The following commands perform no SSH or Git
actions. Set these task-specific variables after source review/publication:

```sh
remaining_root=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
remaining_expansion=$remaining_root/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1
remaining_package=$remaining_expansion/qualification/campaign_remaining_runtime_20261015_v2
remaining_python=$remaining_root/.venv/bin/python
remaining_source_sha=ROOT_REVIEWED_MANIFEST_SHA256
```

Choose `--gru-peer present` only while the exact original GRU process exists;
use `absent` after confirming exit. The supervisor rechecks it and refuses stale
admission. These observer commands first materialize explicit root admission,
then detach one supervisor (the command returns PID/start ticks and log path):

```sh
"$remaining_python" -B "$remaining_package/root_admit.py" --case bound_observer_molbace --reviewed-source-sha256 "$remaining_source_sha" --gru-peer present --output "$remaining_expansion/qualification/remaining_bound_observer_molbace_v2_attempt01" --admission "$remaining_expansion/qualification/remaining_bound_observer_molbace_v2_admission01.json"
"$remaining_python" -B "$remaining_package/launch_detached.py" --admission "$remaining_expansion/qualification/remaining_bound_observer_molbace_v2_admission01.json"
```

After reviewing that terminal and releasing its lock, BUDDY is a separate call:

```sh
"$remaining_python" -B "$remaining_package/root_admit.py" --case buddy_selected_VALID --reviewed-source-sha256 "$remaining_source_sha" --gru-peer present --output "$remaining_expansion/qualification/remaining_buddy_selected_VALID_v2_attempt01" --admission "$remaining_expansion/qualification/remaining_buddy_selected_VALID_v2_admission01.json"
"$remaining_python" -B "$remaining_package/launch_detached.py" --admission "$remaining_expansion/qualification/remaining_buddy_selected_VALID_v2_admission01.json"
```

For an attached supervisor instead, call `supervise.py --admission EXACT_JSON`
with the same qualified interpreter. Do not invoke both launch forms for one
admission. Detached launch writes an exclusive `*.LAUNCH.json` claim and fresh
`*.supervisor.log`; neither is overwritten by another launch. The supervisor
still enforces source, resources, slot and one fresh output before native work.

Each attempt preserves SUPERVISOR, RESOURCES, CHILD_ADMISSION, child log, COST
and available native RESULT. Observer evidence additionally contains the
producer decision, engineering parent/projected profiles and index, own fit
checkpoints/exports, per-invocation bridge commands and preserved exports, plus
actual reporting INDEX/INVOCATION/SELECTED_EXPORT sidecars. Preserve every
failure and actual cost. No tolerance relaxation, device fallback or unrecorded
repeat is allowed.

## Preparation verification and remaining gates

The local stdlib source check verifies 139 source receipts, the successor's
actual scalar profile projection, old-producer rejection, actual argv bridge,
stale-resume rejection, peer growth/identity, third-worker and overbudget
rejection, stopped-selector behavior and closed root templates. No native
constructor, array, tensor, data-payload or fabricated prediction export is
executed. The earlier normalization reporter's unchanged scalar checks were
also run. One initial test assertion used a macOS temporary-directory symlink
instead of its resolved receipt path; that local-only assertion was corrected.

`SOURCE_CHECKS.json` records source evidence. `READINESS.json` distinguishes
completed reusable witnesses from pending actual successor runtime, BUDDY
serving, destination/release applicability, resource pair admission and user GO.

## Selected replay receipt successor v3 (16 October 2026)

This successor changes only the BUDDY child package path to
`composed_BUDDY_selected_VALID_v2`, adds its exact source pin while retaining
the predecessor pin, and refreshes source checks. The supervisor, admission,
resource, slot, peer and zero-update controls otherwise remain unchanged.
Use one fresh attempt02 output in tmux. The MolBACE case has already passed
and is not part of this launch. Prior failures and paid checkpoints stay fixed.
