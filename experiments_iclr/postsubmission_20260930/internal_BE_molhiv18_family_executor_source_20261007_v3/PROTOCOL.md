# Frozen MolHIV 18-fit family executor

V3 narrowly adds `row.reason` for noncomplete closure rows, using retained
specific error/unavailable text and existing exit/failure custody. Sealed v1/v2
are preserved; v2 signal/cap guards, dispatcher, adopted work, limits, and
dependencies are unchanged.

The launch declaration also replaces `allocation_exclusive_for_family` with
`GPU_memory_admission_confirmed` for normal shared-host execution.

This source-only packet adds a serial family owner and an ordinary outer parent
around the adopted one-cell dispatcher. The disabled release template does not
authorize a scientific process. Root separately admits the family after source
review, same-host GPU readiness, and a freshly bound observation of at least
8 GiB physical GPU memory free. This is admission at launch on the normal shared
host. Other jobs are left running; this packet does not reserve host resources
or add a resource scheduler.

## Fixed work

The roster is seeds `7101`, `7203`, `7307`, each in adopted order `single`,
`independent4`, `O`, `I`, `P`, `G`. O/I/P/G use `be_init` and fixed lambda `0.5`.
Every fit calls the exact sealed `run_cell.sh` with the frozen TRAIN/development
role paths and a fresh flat output directory. The existing drivers retain their
100 epochs, 25,800 updates, full-development strict-first selector, and ordinary4
own-selected bank. No training/model/selector loop is copied here. I remains the
adopted candidate. There is no retry, resume, tuning, next task, or TEST access.

## Existing safety limits

`SOURCE_BINDINGS.json` and the disabled root template bind the original
`learnable_internal_be_contrastive_multitask_suite_20261007_v4/configs/molhiv.json`.
Only its existing per-cell caps are reused: **32,390 seconds active + 10 seconds
cleanup = 32,400 seconds hard**. Its older roster fields do not expand this family.
The finite whole-family hard bound is **18 × 32,400 = 583,200 seconds (162 hours)**.
These are safety limits, not an ETA or an optimization budget.

The outer parent publishes one monotonic deadline that the family owner shares.
The last 10 seconds are reserved for outer cleanup/reap. A cell is admitted only
when its full hard cap fits before that reserve. An active timeout is retained as
failed, never as a shortened complete fit. Each child has its own process group
and actual PID/start-tick custody; owned cleanup uses TERM followed by KILL within
the remaining cap. Every TERM/KILL freshly checks the saved leader PID, start
ticks, group, and session. If the leader disappears while a numeric group appears
live, no signal is sent to that group; unavailable custody is retained and blocks
terminal readiness. The same rule applies to emergency cleanup. Failed fits are retained and the finite roster continues;
missing custody or invalid artifacts stops admission and retains unlaunched rows.

## Receipts and closure

Per-cell receipts contain the actual command, fit owner identity, exit code,
reap/group-terminal evidence, elapsed/resource accounting, and output path.
`LEDGER.json` always accounts for the 18 frozen cells. A complete row requires
the actual full-horizon RUN/COMPLETE/PROGRESS identities, original source/data
hashes, required policy counters, and hash/byte custody of `selected.pt`,
`VALID_TRACE.json`, and metadata. Ordinary4 also binds its original bank and four
own checkpoints. This wrapper hashes checkpoint and trace bytes; it never
deserializes a checkpoint, selects an epoch, or reads trace prediction/quality
values. COMPLETE engineering fields are checked without copying its objectives.

At both ordinary and fallback closure, every failed/invalid/unlaunched row retains
its specific `reason`, `error`, or `unavailable_reason` text as the readout's
`row.reason`. Existing nonzero exit codes and FAILURE path/hash/bytes are included
in that reason. Missing row exit metadata may be copied only from its existing
owner handle; a present FAILURE receipt is hash-bound if not already retained.
Missing exit/reap or failure evidence is never invented, and a specific reason
is never replaced with a generic one.

`FAMILY_CLOSURE.json` uses the existing MolHIV readout's whole-family closure
schema. `TERMINAL_EVIDENCE.json` is written by the outer parent after actually
waiting/reaping the family owner. Its handles include the family owner and all
attempted fit owners. If the owner fails before closure, the parent preserves the
last ledger or the unlaunched roster as an explicit closure. Emergency orphan
cleanup never invents exit/reap evidence; unavailable custody keeps terminal
readiness false. Outer terminal evidence records elapsed seconds, the unchanged
active/hard bounds, active timeout, and whether the hard cap was exceeded; a hard
cap excess also blocks terminal readiness. A closed family can therefore contain failed, invalid, or
unlaunched cells. The readout needs its own separate exact root release.

## Invocation after separate root admission

Stage this packet unchanged under the server phase. Save a separate root release
outside this source directory, fill its source-manifest hash and fresh
phase-relative execution directory, bind the existing adoption/budget entries,
and set its four admission declarations plus `enabled` to true. Preserve
`TEST_access=false` and `automatic_retry=false`.

From `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs`:

```text
/usr/bin/python3 -I -S -B experiments_iclr/postsubmission_20260930/internal_BE_molhiv18_family_executor_source_20261007_v3/executor.py --release <absolute-root-release.json> --release-sha256 <exact-root-release-sha256>
```

The wrapper needs Linux `/proc`, Python 3.9+, `/bin/bash`, and
`/usr/bin/nvidia-smi`. It accepts only hostname `anogena-2-0` and sole GPU
`GPU-44039938-fd82-41d2-fefd-de71514e2fac`. The unchanged dispatcher uses
`native_ncn_runtime_20261005_v1/.venv/bin/python`, the existing dependency overlay,
and the repository's existing Python 3.11 site-packages. Preserve the virtualenv
executable path when spawning it. The four adopted source packages, handoff
dispatcher, budget file, and frozen numeric roles must be staged unchanged.

Preparation performs only AST/JSON/seal checks and local CLI help. No server,
GPU, numerical/model import, numeric payload read, outcome inspection, or fit
has occurred for this packet. Runtime execution and readout remain unverified.
