# Portable dispatcher v2 — source handoff, execution unadmitted

This successor implements the three-worker dispatcher path: allocation/gpu0,
server77/gpu0 and server77/gpu1. Both77 configurations remain disabled and
reject before hostname, GPU, project-authority or resource access. No native
job, tensor read, hardware probe, remote access or77 connection was performed
for this package. There is no campaign GO.

The native code comes from runtime_v4, runtime_link_v9 and runtime_v2's I4
aggregator. `RUNTIME_DELTA.json` records 16 module comparisons; only portable
guards and CLI path handling differ. Native training, selection, checkpoint,
evaluator and export functions retain their ASTs. Compiler4 readiness and full
index pins are in `PARENT_PINS.json`; v8/compiler3 are historical only.

## What is implemented

- `prepare_projection.py`: finite metadata compiler, exact6730 tasks/6460 fits,
  270 I4 exports, one owner per ID, whole canonical cohorts, and I4 co-location.
  Default placement is allocation only. An exact reviewed `group_to_worker`
  map can prepare future77 projections without enabling that host. Canonical
  descriptor hashes/science/provenance remain separate from explicitly bound
  runtime path hashes. No automatic fit/checkpoint relocation exists.
- `ownership.py`: one immutable global assignment plus a permanent owner
  record and permanent task claims on each host. A new namespace cannot
  bypass prior claims. There is no timeout, online lease, heartbeat or owner
  reassignment after a disconnect.
- `host_ledger.py`: one host-wide flock and durable checksummed reservation
  state, shared by both77 GPUs. GPU slots/VRAM are per physical UUID; CPU,
  RAM and disk reservations cover both worker processes. RAM availability is
  capped by host MemAvailable and every qualified finite cgroup-v2 limit,
  using hard headroom plus max(inactive_file - shmem, 0). Raw accounting is
  retained; cache reclamation is never forced. Startup counts.
  The lock covers probe, reservation, Popen and PID identity commit. CPU
  slots are a reviewed envelope; native thread count remains2. Observed
  free resources include other users' consumption; their processes are
  never signaled, killed, adopted or modified.
- `backend.py`: the sole production Popen. A post-Popen identity/commit
  failure returns the actual child to the engine for draining and retains
  uncertain custody. Only reap plus a durable terminal releases a normal
  reservation. Missing/reused/orphan PIDs never cause optimistic release.
  Explicit ledger reconciliation preserves the original reservation and
  retained costs; root-reviewed evidence is sufficient for routine dead-owner
  disposition. It adds no user-permission gate and cannot authorize an
  automatic fit retry.
- `engine.py`: ordinary fit failures retain all costs, are skipped on
  restart unless explicitly reviewed for same-fit resume, and allow other
  admissible tasks to proceed under the pinned policy. Their dependent I4
  exports become terminal `blocked_dependency_failed`. Corruption, uncertain
  orphan custody and explicit STOP halt admissions for our host workers;
  each worker drains only its own children. A worker waits while its peer
  can release host resources. Existing same-fit resume and native terminal
  reconciliation checks are reused.
- `stop_transport.py`: local immutable outbox/inbox files, expected-byte
  pins, monotonic sequence, idempotent identical redelivery, conflicting
  replay rejection and per-host received acknowledgments. Delivery writes
  a host-wide STOP under the same admission lock before ACK. ACK means
  **received**, not all children drained. Worker terminal receipts establish
  drain completion. STOP is never cleared; continuation must review every
  retained STOP receipt.

## One command per physical GPU, after qualification and explicit GO

`run_gpu.py --worker WORKER --run-spec ABSOLUTE_JSON --run-spec-sha256 SHA256`
is the implemented interface. Run it with the actual qualified lexical venv
Python, not its resolved base binary. `RUN_SPEC_TEMPLATE.json` is deliberately
incomplete; its null authorities cannot launch.

On each authorized host, from its canonical repository, the eventual shell
form is one command per GPU (replace the uppercase placeholders with reviewed
actual values):

```sh
nohup .venv/bin/python -B -u ABSOLUTE_PACKAGE/run_gpu.py --worker WORKER --run-spec ABSOLUTE_RUN_SPEC --run-spec-sha256 REVIEWED_SHA256 >> ABSOLUTE_WORKER_LOG 2>&1 < /dev/null &
```

Use distinct host-local logs for each worker. `nohup`, redirected local logs,
stdin closure and background execution keep the supervisor independent of
the SSH/Mac relay terminal. Its admitted children use separate sessions.
The real host must qualify survival of the authorized route's disconnect;
site session cleanup policies cannot be inferred by these local source checks.
There is no Mac coordinator dependency once the command has been admitted.

The source-only metadata preparer is also guarded and was **not executed**:

```sh
.venv/bin/python -B ABSOLUTE_PACKAGE/prepare_projection.py
```

It reads exact compiler4 JSON/source inputs, creates `generated_v2` once, and
produces unadmitted host releases/candidates. Optional reviewed placement uses
`--placement ABSOLUTE_JSON --placement-sha256 REVIEWED_SHA256`. It cannot create
resource evidence, readiness or GO. Stage the same immutable ownership bytes,
source/metadata and exact local projection on each subsequently authorized
host. Verify destination source/data bytes there; path relocation is not data
qualification. No prior claim or selected/checkpoint state is moved.

## STOP and disconnect behavior

Issue a STOP file on the controlling machine with exact ownership and issuer
authority pins:

```sh
python3 -B ABSOLUTE_PACKAGE/stop_transport.py issue --ownership OWNERSHIP_JSON --ownership-sha256 SHA256 --outbox LOCAL_OUTBOX --issuer reviewed_campaign_operator --target allocation --target server77 --reason REASON --authority AUTHORITY_JSON --authority-sha256 AUTHORITY_SHA256
```

Transfer the returned immutable file through an authorized available route,
then on **each** host use:

```sh
.venv/bin/python -B ABSOLUTE_PACKAGE/stop_transport.py deliver --ownership OWNERSHIP_JSON --ownership-sha256 SHA256 --command STOP_JSON --command-sha256 STOP_SHA256 --worker WORKER
```

Delivery on either77 worker covers both GPUs via the shared host directory.
Allocation delivery covers its sole worker. No remote command is implemented
or performed by this transport. An undelivered STOP **cannot stop a disconnected
host**. Training may continue there until delivery; report missing ACKs explicitly.
Immediate local STOP on a reachable host halts its new admissions and drains
our children. A STOP does not kill in-progress fits or another user's work.

## Reporting integration

An optional reviewed `reporting` binding contains exact `bridge`, `entrypoint`,
`registry`, and `manifest` pins and is bound identically in the host release
and run spec. The complete reporting manifest closure must be in the source
closure. `ObservedArgvTransform` runs inside the reservation-owning backend;
it cannot replace `start` or Popen. The portable backend retains
`qualified_python_entry.py` around the observer, preserving lexical venv
validation in the child. This observer remains allocation-only. Existing
reporting registries must be rebound to the exact portable native entrypoints
and descriptor receipts before use; no old registry is silently accepted.

## Verification and exact remaining gates

`CHECKS.json` records30 new local metadata/source checks. Two real
multiprocessing races exercise flock admission across workers; the rest cover
shared RAM/disk, startup crash custody, permanent claims, STOP delivery/replay,
failure/dependency policy and disabled77 CLI behavior. They run no scientific
worker. Nine unchanged durable-engine methods reuse the predecessor's recorded
38 checks; the broad fake-science suite was not rerun. `ENGINE_DELTA.diff.txt`
is the focused engine review surface. `SOURCE_REVIEW.json` retains the first
local check failure and repair.

Before actual use, root must review this source successor, execute the guarded
metadata compiler against the complete compiler4 inventory, then supply:

1. Explicit user77 reauthorization before any77 route/probe; actual hostname,
   ordered two-GPU UUID table, canonical repo/phase/ledger paths, lexical venv
   environment, UID and Linux process identity receipts for each worker.
2. Exact source/data/provider receipts at destination and native family/scope
   evidence for the v9/compiler4 descriptor lineage. Reuse applicable evidence;
   do not rerun obsolete v8/BUDDY checks. NCN/BUDDY family lifecycle gaps and
   source-only observer qualification remain external gates.
3. Mature target reservations including optimizer state, peak GPU/RAM/disk,
   descendants/provider CPU envelope, unchanged two-thread runtime setting,
   shared host CPU capacity and headroom, plus all effective cgroup ancestor
   paths and limits. One child is valid; two require
   measured reciprocal compatible-profile evidence on that physical GPU.
4. Real Linux/filesystem flock, durable rename/fsync, worker restart/orphan,
   host STOP/continuation, cost custody and disconnect-survival qualification.
   The Mac metadata races do not establish these host properties.
5. A reviewed owner map, full local source/registry closure, per-task bindings,
   exact readiness and explicit campaign GO for each admitted physical worker.

No production launch,77 enablement, destination resource admission or campaign
readiness is asserted by this package.
