# Acquire/qualification stage bridge — source only

`stage_entry.py --request ABS_REQUEST_JSON --supervisor ABS_INNER_SUPERVISOR`
is entered only as the child of the existing chain:
`bounded_run_v1.py → run_authorized_v2.py → run_logged.py → stage_entry.py`.
Root supplies the execution JSON and independently admits the stage. This packet
was neither imported, compiled, tested nor executed during source preparation.

## Request schema

Required fields:

- `schema: "coordinate-ensemble-stage-entry-request-v1"`, `root_admitted: true`.
- `mode: "acquire"` or `"qualify"`; no alias, fit or test mode is accepted.
- `phase_root`: exact remote phase; `output`: a new receipt directory under
  `coordinate_ensemble_execution_root_v1`, outside runner phase roots.
- `gpu_uuid: "GPU-44039938-fd82-41d2-fefd-de71514e2fac"`.
- `cuda_visible_devices: "0"` (or that exact UUID), matching the parent wrapper.
- `whole_cap_seconds`, matching the immutable outer START receipt.
- Optional `per_operation_timeout_seconds`: positive, at most1200; defaults1200.
- `allocation_evidence: {path, sha256}`: retained successful one-GPU authorization
  receipt independently binding the exact UUID and corrected SSH destination.
- `protocol`, `sources`, `runner`: each `{path, sha256}`. Acquire permits the
  independently frozen `coordinate_ensemble_runner_v1/runner.py`; a separately
  admitted future qualify request requires `coordinate_ensemble_runner_v2/runner.py`.
- `protected_files: [{path, sha256}, ...]`: bind this entry, all three supervisors,
  `protocols/repo_env.sh` and `AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json`.
- `operations`: exact ordered operations. An acquire operation has only
  `{dataset, argv}`. A qualify operation has only `{dataset, argv, cell,
  data_manifest}`, with `cell: {split, seed, arm, recipe}` and the canonical public
  acquisition `data_manifest: {path, sha256}`.

`argv` is an exact JSON string array, compared against the bridge's reconstructed
runner command. For acquire it is:

```text
[REPO/.venv/bin/python, PHASE/coordinate_ensemble_runner_v1/runner.py, acquire,
 --protocol, PROTOCOL_PATH, --protocol-sha256, PROTOCOL_SHA,
 --sources, SOURCES_PATH, --sources-sha256, SOURCES_SHA,
 --dataset, LastFMAsia]
```

Qualify uses the v2 runner, mode `qualify`, then additionally `--split`, `--seed`,
`--arm`, `--recipe`, `--data-manifest` and `--data-manifest-sha256`, in that order.
No `-B`, shell string, extra argument, implicit sweep or resume is inserted.
The existing environment wrapper disables bytecode. Root's acquisition example
has two ordered dataset operations, timeout1200 each and whole cap2450 seconds.

## Runtime guards

The admission boolean is one check. The bridge also verifies exact remote cwd,
repository venv lexical entry point, corrected route, wrapper/cache confinement,
inner recorded argv/environment/source hashes, outer recorded request/hash/argv,
whole cap and supervisor hashes, prior allocation evidence and a fresh physical
GPU query before every child. The physical allocation must contain only the
pinned idle A100-SXM4-80GB. Parent CUDA exposure must match exactly. Acquire's
child environment forcibly sets `CUDA_VISIBLE_DEVICES=""` and requires a CPU
protocol. Qualify exposes only the pinned single device as `cuda:0`.

Every request/protocol/source/data-manifest path is checked for phase confinement
and symlinks; all manifest sources are hash checked. Runner acquire/qualify/fit
roots must be disjoint children of the new coordinate execution namespace;
data/cache/output/sealed/PyG subdirectories are disjoint. Bridge receipt and
runner outputs must be new. Each requested qualifier must occur exactly once in
the frozen protocol and prescribe100 complete optimizer updates with eval_every10
under the reviewed train-only v2 qualification implementation. Source/protocol
hashes are checked again before and after every child.

Children use `subprocess.Popen(argv, shell=False)` in the existing inherited
process group, with explicit per-child timeouts. No child creates a new session
that could escape the outer group's TERM/KILL bound. The bridge requires enough
remaining outer runway, retains partial child logs, and records operation/stage
START/TERMINAL receipts. Timeout/interrupt cleanup terminates the immediate child;
the existing whole-process supervisor remains the final descendant-group owner.
A forced SIGKILL can prevent a bridge terminal write; the outer TERMINAL record
is authoritative in that case. Partial stages do not count as completion.

These are ordinary hash, path and reviewed-source guards, not an OS sandbox or a
permission grant. No model, dataset, SSH, acquisition, qualification, fit, test
score or source check execution was performed here. Root owns request creation,
any runtime checks, and actual admission.
