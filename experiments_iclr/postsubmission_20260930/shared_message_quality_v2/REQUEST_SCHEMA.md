# Root engineering request schema

V2 schema is `gnnm-shared-message-root-engineering-request-v2`. Each operation
must run through the sealed `protocols/bounded_run_v1.py` outer wrapper, then
`run_authorized_v2.py --require-idle-gpu`, then this exact argument sequence:
`<repo/.venv/bin/python> <packet/qualify.py> <operation> --request <absolute-request> --supervisor <absolute-inner-directory>`.
The outer wrapper's `--request` is the same exact request and
`--cap-seconds` equals `wall_cap_seconds`; `GNNM_BOUND_START_JSON` is supplied
by the wrapper. The outer and inner output directories must be new and
separate. The RESULT remains provisional until both inner completion and a
successful completed outer TERMINAL exist.

Real admission adds exact hashed references for synthetic request, RESULT and
outer TERMINAL. Their request/source/CLI/output/START/artifact/inner/outer
chain is audited, and runtime policy, tolerance and GPU equal this request.
Synthetic RESULT must contain bounded peak allocated/reserved memory and
the exact verified import tree/module hashes. See README for timeout and
memory-enforcement scope. Engineering selection is train+validation/5.

This describes required root-owned evidence; it is not an authorization file.
No filled launch request is supplied. `ref` is exactly `{path, sha256}` with a
canonical path relative to the parent phase. Request/output files are inside
`shared_message_quality_v2`, after the immutable source generation is sealed.

Both operations require `gnnm-shared-message-root-engineering-request-v2`:

- `operation`: `synthetic` or `real`, matching CLI operation.
- `source_manifest_sha256`: exact sealed `SOURCE_MANIFEST_v1.json` digest.
- `root_authorized_train_only_engineering: true`;
  `scientific_training_authorized: false`, `quality_scoring_authorized: false`,
  `heldout_labels_authorized: false`.
- `recipe`: exact `counts.RECIPE`; `seed: 17`. No scientific epoch count.
- `gpu_uuid`: actual authorized one-A100 UUID.
- `wall_cap_seconds`: positive, at most1800;
  `prior_allocation_seconds`: all actual prior charged allocation work,
  nonnegative, with prior+cap at most28800. Root must reconcile failures and
  other concurrent/allocation work before each request.
- `memory_cap_bytes`: positive and at most60GiB;
  `disk_cap_bytes`: positive and at most64MiB for metadata only.
- `output_relative_to_phase`: a new exclusive directory inside this packet.
- `runtime_policy`: `device: cuda:0`, `TF32: false`, `AMP: false`, positive
  integer `CPU_threads`, boolean `deterministic_algorithms`, exact
  `CUBLAS_WORKSPACE_CONFIG`, actual `package_versions` including `torch`,
  `numpy`, `torch-geometric`. Environment must set CUBLAS exactly and
  `PYTHONDONTWRITEBYTECODE=1` before third-party imports. Deterministic CUDA,
  if requested, needs `:4096:8` or `:16:8`; false does not claim determinism.
- `equivalence_tolerance`: exactly actual root-frozen finite nonnegative
  `atol`/`rtol`, each at most.001. Choose from actual qualification evidence,
  before quality outcomes. These are numerical tolerances, not quality margins.

`real` additionally requires:

- `tasks`: exactly `[Reddit2, ogbn-products]`;
  `engineering_phases`: both actual complete admitted engineering phase refs.
  Each must match its task and actual GPU; sealed loaders replay exact complete
  train-only acquisition/prepared arrays, not provider objects.
- `synthetic_request: ref`: the exact synthetic root request consumed by its completed CLI.
- `synthetic_outer_terminal: ref`: the successfully completed external `TERMINAL.json`.
- `synthetic_gate: ref`: this source generation's actual passed synthetic CUDA
  `RESULT.json`; `synthetic_supervisor_directory`: its completed corrected
  supervisor. The entrypoint must be this sealed `qualify.py synthetic` on the
  same actual GPU. Every declared synthetic check must have passed.
- `warmups`: integer1..5; `samples`: integer3..10. These control bounded short
  engineering updates and do not define a scientific training budget. A
  reasonable initial request is2 warmups+3 samples, all12 variants per task.

CLI templates, not launch instructions:

```text
<repo-python> <phase>/shared_message_quality_v2/qualify.py synthetic
  --request <root-synthetic-request> --supervisor <this-supervisor-directory>

<repo-python> <phase>/shared_message_quality_v2/qualify.py real
  --request <root-real-engineering-request> --supervisor <this-supervisor-directory>
```

Root must later audit exact completed supervision and every metadata artifact.
A source seal, passed short engineering result or speed-selected path alone
does not admit scientific fits or quality claims. Forecast complete updates,
real validation metrics, serialization/hashing/replays, preparation, remaining
allocation, CPU and disk separately under the adopted quality protocol.
