# All-fit smoke and training execution

This package implements the user's 16 October conditional GO. It does not change
scientific descriptors, presets, fixed ownership, sources, HEADs, or native
training policy. Root stages and runs it on the verified allocation and server51.
No command below is a remote login command.

## Workflow

1. Prepare each host's exact smoke inventory with
   `qualification/all_fit_smoke_20261016_v1/prepare.py`. One server51 inventory
   includes both server51 workers; allocation has its own inventory.
2. Prepare one execution plan per worker using the commands below.
3. Start three tmux owners. Server51 starts smokes immediately. Allocation waits
   its existing pilot leases and `planning/all_fit_launch_20261016_v1/ALLOCATION_PREREQUISITES.json`
   with `status: PASS`. A preserved allocation GRU (PID640428/start6055039165)
   consumes a slot and reserves remaining growth to32GiB VRAM/16GiB RAM while alive.
4. Every assigned fit runs its actual two-update smoke. First failed smoke stops
   new children on that worker; already-running children finish and other workers
   continue. Inspect retained logs/results, fix the real blocker, and explicitly
   restart that worker with `--retry-failed` after updating any affected pinned
   inventory/plan as required. A restart alone does not silently rerun failures.
5. Each worker writes immutable `SMOKE_SUMMARY.json` and waits. Root retrieves the
   two inventory files and three summaries, creates the bundle described below,
   and runs the smoke validator. After exact6460-fit PASS, root copies the same
   `GLOBAL_SMOKE_PASS.json` to this package directory on BOTH hosts.
6. Owners automatically start full training. Full runs use the actual pinned
   `reporting/runtime_entry.py` with original presets and fresh seeds. A smoke
   checkpoint is never a full-training warm start. All270 I4 exports consume only
   their assigned four selected VALID exports after the four fits finish.

## Local-on-host preparation and launch

Set shell variables to the verified deployment (use literal values returned by
preparation; no source or Git change is required). `QUEUE` is this package's
absolute deployed directory; `PYTHON` is `PROJECT/.venv/bin/python`.

```sh
"$PYTHON" -B "$QUEUE/prepare.py" \
  --worker server51/gpu0 \
  --inventory "$INVENTORY" --inventory-sha256 "$INVENTORY_SHA" \
  --authorization "$AUTHORIZATION" --authorization-sha256 "$AUTHORIZATION_SHA" \
  --output "$QUEUE/PLAN-server51_gpu0.json"
```

Use the returned plan SHA and its absolute path:

```sh
"$PYTHON" -B "$QUEUE/launch.py" start \
  --worker server51/gpu0 --plan "$QUEUE/PLAN-server51_gpu0.json" \
  --plan-sha256 "$PLAN_SHA"
```

Repeat with `server51/gpu1` and `allocation/gpu0`, their matching plans/inventories.
Default tmux sessions are `gnnm-allfit-server51-gpu0-20261016`,
`gnnm-allfit-server51-gpu1-20261016`, and `gnnm-allfit-allocation-gpu0-20261016`.
`remain-on-exit` is enabled. Launch records expose exact plan/source/inventory and
status/log paths. A live pane is never killed. Restarting a dead pane preserves
all state and starts a new owner log. Add `--retry-failed` only for a repaired
failure; otherwise completed work is reused, and a crashed unfinished attempt
is retained and reported for an explicit retry.

```sh
"$PYTHON" -B "$QUEUE/launch.py" status --worker server51/gpu0 \
  --plan "$QUEUE/PLAN-server51_gpu0.json" --plan-sha256 "$PLAN_SHA"
"$PYTHON" -B "$QUEUE/launch.py" stop --worker server51/gpu0 \
  --plan "$QUEUE/PLAN-server51_gpu0.json" --plan-sha256 "$PLAN_SHA"
```

`stop` finishes current children and launches no new ones. `start` acknowledges
that STOP file. Abrupt supervisor loss leaves its children running with inherited
GPU leases; a new owner waits until those children exit, then reads durable
attempt terminals. It never signals unrelated processes.

## Global bundle

On the validator host, write:

```json
{
  "workers": [
    {"worker":"allocation/gpu0", "inventory":{"path":"LOCAL_COPY","bytes":0,"sha256":"ACTUAL_SHA"}, "summary":{"path":"LOCAL_COPY","bytes":0,"sha256":"ACTUAL_SHA"}},
    {"worker":"server51/gpu0", "inventory":{"path":"LOCAL_COPY","bytes":0,"sha256":"ACTUAL_SHA"}, "summary":{"path":"LOCAL_COPY","bytes":0,"sha256":"ACTUAL_SHA"}},
    {"worker":"server51/gpu1", "inventory":{"path":"LOCAL_COPY","bytes":0,"sha256":"ACTUAL_SHA"}, "summary":{"path":"LOCAL_COPY","bytes":0,"sha256":"ACTUAL_SHA"}}
  ]
}
```

Replace every path/byte count/SHA with actual transferred-file receipts. The
server51 inventory receipt is shared by its two rows. Original embedded paths
remain unchanged; the validator verifies local copy bytes and embedded child
RESULT hashes. Run:

```sh
"$PYTHON" -B "$SMOKE_PACKAGE/validate.py" --bundle "$BUNDLE" \
  --output "$FRESH_GLOBAL_SMOKE_PASS"
```

Workers check the explicit sorted6460 IDs and coverage, all three worker hash
bindings, their exact inventory/summary hashes, and the local smoke validator
source SHA. The global source receipt's original path can be on another host;
only its bytes/SHA bind the locally verified equivalent source.

## Ownership, resources and recovery

- Each worker holds the old slot lock plus canonical physical-GPU and worker
  leases. Guardian AND numerical child inherit all three plus an attempt lease.
  Startup reservations are written under a host-wide flock before spawning.
  A supervisor-to-child recording gap cannot release a live task's slot.
- A host-wide reservation ledger coordinates server51's two GPUs. At most two
  compatible owned/explicitly-preserved jobs share a GPU; the supplied resource
  API enforces pair envelopes, including serial69GiB Physics tasks. Unrelated
  observed usage reduces actual free capacity; unrelated jobs are never exempted
  from memory accounting or controlled.
- Admission subtracts remaining owned peak growth from free GPU/RAM, covers
  host-wide disk/CPU reservations, and discovers effective cgroup-v2 ancestors
  inside the actual tmux owner. Every finite memory and CPU limit applies.
  Missing non-controller files at the cgroup root are skipped. Admission leaves
  1GiB GPU and8GiB host RAM/disk margins. It scans compatible later tasks when
  an earlier heavy task is temporarily blocked, keeping fixed worker ownership.
- Each attempt retains invocation, admission snapshot, stdout/stderr, process
  identities, terminal exit/signal/wall/CPU accounting, and archived native result.
  Missing custody/terminal records preserve uncertainty and halt for repair.
- Full retry uses only that exact fit's own full-output `recovery.pt`; native
  checkpoint/job validation remains authoritative. Without a recovery file, a
  retry uses a new output directory and retains the failed attempt. Native
  no-selected-state outcomes are preserved, and dependent I4 exports are marked
  blocked instead of synthesizing selections.
- `runs/WORKER/STATUS.json` provides phase and counts; `CAPACITY_WAIT.json` gives
  the last live resource denial; `smokes/` and `full/` remain separate.

## Engineering verification

`python3 -m unittest discover -s "$QUEUE" -p test_execution.py -v` runs only
stdlib scalar/process-stub checks: exact global barrier, inherited lease custody,
host-wide remaining growth/preserved peer constraints, failed/forced-killed
attempt accounting, and prerequisite-process lock ordering. It imports no torch, accesses no dataset, and starts no
native training. Real all-fit smokes remain the required execution gate.

## Version 2 startup repair

The first server51 v1 queue owners stopped before any numerical child because
`/sys/fs/cgroup` and `/run/cgroups/cpu` are two views of the same cgroup-v2
filesystem (device0:30, root `/`). Version2 preserves those v1 source files,
owner logs, and failure records. It resolves the real membership through every
accessible hierarchy and chooses its widest view, preferring the canonical
`/sys/fs/cgroup` path among equivalent aliases. Every visible ancestor limit
still applies; duplicate views are not counted twice. No mounts, settings,
datasets, descriptors, optimizer behavior, or scientific schedules change.

Prepare fresh v2 execution plans and start the exited server51 tmux sessions
with their v2 plan paths and hashes. V2 uses its own state directory; both v1
server51 owners launched zero numerical children. Physical-GPU leases remain
shared across versions. Keep both server51 workers on v2 so their host-wide
reservation ledger is shared. Allocation v1 may remain waiting/running when its
single canonical cgroup view resolves normally; its host is independent.

## Portable BUDDY deployment repair

This successor binds server51_campaign_port_v5 (manifest 3fa47ca1ec206990fd25e37aa2f273e97aae8ec9e37c47800e9cfeda75654d13).
Its training settings, exact fit ownership and two-update smoke requirements
are unchanged. Previous sources and failed attempts are preserved.
