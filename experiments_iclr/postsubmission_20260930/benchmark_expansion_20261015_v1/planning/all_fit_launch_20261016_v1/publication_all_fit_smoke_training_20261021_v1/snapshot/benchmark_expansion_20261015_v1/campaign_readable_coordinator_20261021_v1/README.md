# Readable global smoke coordinator

This additive package replaces the compressed executable transport in
`planning/all_fit_launch_20261016_v1/collect_global_smoke_barrier_port6.py`.
Historical source, owners, smoke records and proofs remain evidence.

The default operation is `verify`: read current remote metadata, fetch raw JSON,
and run the unchanged frozen smoke validator locally. It creates a local
`CANDIDATE_GLOBAL_SMOKE_PASS.json`, never a remote gate. A separate explicit
`release` operation and the candidate's exact SHA256 are required to publish the
canonical execution7 marker. Publishing that marker can allow existing waiting
owners to continue full training. The dispatcher must apply the owner's current
launch decision when selecting this operation; verification never selects it.

## Frozen scientific inputs

`BINDINGS.json` is a byte-identical copy of `GLOBAL_SMOKE_BINDINGS_PORT6.json`:
1,688 bytes, SHA256
`b605b36e90fe86b7b58cbb366918ea7c7efdba331a33fd897962696d4f4eda96`.
It pins the existing port6, execution7 and smoke6 manifests and all three large
worker plans. Each probe checks package members, plans, owner bindings,
inventories and PASS summaries using the existing coordinator checks.

The unchanged `qualification/all_fit_smoke_20261016_v6/validate.py` checks all
6,460 distinct fit IDs and embedded result receipts, two real optimizer updates,
finite TRAIN/VALID, checkpoint reload, original inventory/source identities and
scope restrictions. A count alone cannot produce the local candidate. Release
also calls the unchanged execution7 `common.global_barrier` for each local
worker and repeats the current metadata probe before the atomic marker write.

## Source and transport

- `remote_backend.py`: standalone plaintext remote metadata program. Default
  `probe` and `read-json` are read only; `release` is the sole marker write.
- `readable_transport.py`: literal authorized allocation SSH and normal SSH to
  server51 through `maclink.py run --` argument vectors. It never selects the
  interactive MacLink shell or its file-upload command.
- `coordinator.py`: local collection and validation, explicit source staging,
  and separate explicit release.

Allocation staging uses ordinary SCP after hostname and complete GPU UUID
verification. Server51 staging sends visible, bounded plaintext heredocs as
SSH command arguments through MacLink `run`. Staged files have source-addressed
paths; checksums and exact plaintext readback are required before invocation.
Configuration and later release data are separate JSON files. Remote result
files return as raw JSON stdout. MacLink's standard output framing carries data;
its `run` command is sent as ordinary readable JSON arguments.

The locally inspected MacLink implementation forwards `message.argv` to
`asyncio.create_subprocess_exec`; `operate` sends those arguments unchanged.
`ROUTE_REVIEW.json` records the inspected local source identity. This static
review does not attest the installed agent source on the paired Mac; live route
inspection and authentication remain the dispatcher's responsibility. No
connector restart, source modification, new route or job operation is included.

Every call verifies the literal hostname and full ordered GPU UUID list before
project operations. Every exact non-secret command is saved before dispatch,
with output identity and result afterward. Timeouts are recorded as uncertain
delivery and are never automatically retried. Evidence uses no-overwrite writes.
Source staging uses fresh incoming files and refuses different existing files.

## Commands

Run from this package directory. Set `COORDINATOR_SOURCE_SHA` to the reviewed
`MANIFEST.json` SHA256 from the freeze receipt. The required source hash is
checked against every listed file before any transport call.

```sh
python3 -B coordinator.py stage \
  --source-sha256 "$COORDINATOR_SOURCE_SHA" \
  --bindings-sha256 b605b36e90fe86b7b58cbb366918ea7c7efdba331a33fd897962696d4f4eda96

python3 -B coordinator.py verify \
  --source-sha256 "$COORDINATOR_SOURCE_SHA" \
  --bindings-sha256 b605b36e90fe86b7b58cbb366918ea7c7efdba331a33fd897962696d4f4eda96
```

`stage` copies and verifies only the named backend and frozen configuration.
`verify` requires that staging to have completed and makes only remote reads.
`probe` is an optional shorter source/plan/summary check. Omitting the operation
selects `verify`.

Only under an explicit release decision, set `CANDIDATE_SHA` to the exact
candidate hash reported by successful verification and use:

```sh
python3 -B coordinator.py release \
  --source-sha256 "$COORDINATOR_SOURCE_SHA" \
  --bindings-sha256 b605b36e90fe86b7b58cbb366918ea7c7efdba331a33fd897962696d4f4eda96 \
  --confirm-release-sha256 "$CANDIDATE_SHA"
```

Release rechecks all three workers, stages separate candidate/request JSON,
and installs the same bytes at each host's existing
`campaign_execution_20261016_v7/GLOBAL_SMOKE_PASS.json`. Cross-host delivery is
sequential, not atomic. If one delivery succeeds and another fails, preserve the
receipts and inspect both hosts before another explicit release attempt. A
different existing marker is rejected; an identical marker is idempotent.
This package does not start, stop, replace or restart owners or numerical jobs.

Local proof and attempt files live in `proofs/<bindings-sha256>/`. Candidate,
binding, inventory, summary, bundle and validation receipts are immutable under
this tool. No prior proof tree is overwritten. A changed source or summary
requires investigation and a separately reviewed new proof location/version.

## Local verification

```sh
python3 -B -m unittest -v test_coordinator.py
```

Seven tests pass, including the unchanged frozen validator on synthetic metadata
for all 6,460 fits, rejection of a rehashed one-update receipt, stale proof and
mixed inventory rejection, read-only defaults, no implicit release, explicit
release through execution7's checker, and unchanged marker preservation.
Fixtures use temporary directories, no native fits and no remote calls. The
fixture coverage proves metadata handling; it is not live campaign evidence.
