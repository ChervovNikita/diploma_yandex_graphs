# Root-owned actual admission fields

This is a schema guide, not an approval or an evidence receipt. No qualified
scientific input or launch artifact is created by this source packet.
`ref` below means exactly `{path: <canonical phase-relative name>, sha256:
<actual SHA256>}`. All root files belong inside this source packet.

## Method admission

`gnnm-propagation-fixed30-scientific-admission-v1` requires:

- `scientific_training_authorized: true` only after root actually qualifies the
  full study; exact `recipe: contract.RECIPE` and `cells: contract.cells()`.
- `test_target_access_before_complete_family_lock: false`.
- `source_bindings_sha256`, exact amendment `ref`, and `root_approval: ref`.
- `gpu_uuid`: the actual authorized A100 UUID.
- `qualified_gates`: exactly `synthetic_cuda`, `complete_engineering_Reddit2`,
  `complete_engineering_ogbn-products`, `complete30_resources`, `source_audit`,
  `runtime_freeze`, `execution_equivalence`. Each has `report: ref`,
  `report_sha256` and a root-qualified boolean. CUDA/engineering/equivalence
  rows also require completed corrected `supervisor_directory` evidence on the
  same actual GPU. Root source/resource/runtime reports need no invented GPU
  supervisor.
- `resource_ledger: ref`: its actual report has `qualified`, `fit_count: 30`,
  `all_artifact_replay_component_timing_costs_included`,
  `total_GPU_seconds_forecast <= 23400`, `total_CPU_seconds_forecast <= 57600`,
  `projected_additional_disk_bytes <= 214748364800`,
  `peak_reserved_bytes <= 64424509440`. These fields must reflect real expanded
  forecasts, not simply declare them true.
- `tasks`: exactly both task names. Each has `engineering_phase` (the actual
  complete phase used by the sealed loader, with matching `task`),
  `prepared_manifest: ref`, `acquisition: ref`,
  `acquisition_completed_audit: ref`, `preparation_charged_seconds` (all five
  arms' attributable once-only charges), `preparation_cost_accounting: ref`,
  `validation_label_request_payload_sha256`, `test_label_request_template`.
- `execution_policy`: both task names then all five arms. Each has
  `training_validation` and `deployment`, each an exact object containing
  `arm`, `execution`, `naive_path`, `checkpoint_blocks`, `native_source`.
  Candidate and detached must be packed CSR. Freeze both paths from qualified
  outcome-blind engineering; no later score-based variant selection.
- `runtime_policy`: `device: cuda:0`, `TF32: false`, `AMP: false`,
  `package_versions` (actual installed distribution names and versions,
  including torch/numpy/torch-geometric), `CPU_threads`,
  `deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`. The environment must
  match CUBLAS before any Torch import; deterministic CUDA must use `:4096:8`
  or `:16:8`. Set `PYTHONDONTWRITEBYTECODE=1` through the repository wrapper.
- `execution_equivalence_policy`: actual root-qualified `atol`, `rtol` for
  selected/epoch100 train-path versus deployment validation logits. Initial
  source forward equivalence does not claim all training gradients equivalent.
- `output_root`: a new canonical phase-relative directory inside this packet.
- `remaining_resource_policy`: `prior_GPU_seconds` (all previous charged
  allocation work), `prior_additional_disk_bytes` (prior phase bytes, excluding
  this output root), `prior_CPU_preparation_audit_wall_seconds`. Own completed
  scientific child wall times are added conservatively to both allocation and
  CPU preparation/audit wall accounting. This is wall-time accounting, not a
  claim that multithreaded process CPU core-seconds equal elapsed wall time.
- `timing_policy`: `warmups: 10`, `timed_forwards: 50`,
  `separate_component_warmups: 10`, `separate_component_forwards: 50`,
  `paired_order_seed_offset: 50000`, `component_order_seed_offset: 60000`,
  `resident_family_memory_admission`.
  The resident admission has `qualified`, `report: ref`,
  `maximum_reserved_bytes` for both tasks and `method`, either
  `conservative_sum_of_selected_individual_peaks` or
  `measured_complete_resident_family`. The latter requires its actual completed
  `supervisor_directory`. Either method must bound each task at 60GiB.

The separate `gnnm-propagation-fixed30-root-approval-v1` receipt hashes the
canonical method payload **excluding `root_approval`** and has `qualified`,
`all_complete_inputs_and_gates_qualified`,
`validation_label_admission_authorized` true, and
`test_before_family_lock_authorized` false.

## Split requests and actual target freezes

A `gnnm-propagation-root-split-admission-request-v1` has:

- `root_authorized`, `task`, `split` (`valid` or `test`), `binding: ref`.
- `provider_storage_sha256`: exact audited `PROVIDER_STORAGE.json` digest.
- `processed_public_file: ref`: the exact named public processed file in that
  inventory; the canonical path contains `/processed/`.
- `provider_storage_inventory_field`: `processed` for the ordinary provider
  export or `original_provider_files` for products v4 read-only recovery.
- `expected_processed_tuple_length`: 2 for PyG Data/slices, 3 for
  PyG dict/slices/class. This must match the actual public provider format.
- `output_relative_to_phase`: a new target export directory inside this packet.

Validation request hash in method admission excludes only `binding`, avoiding
a self-reference cycle. The test template in method admission is the same
request payload **without** `binding`, `family_lock_payload_sha256` or
`family_lock`. Finalization inserts actual locked choices hash and binding,
hashes that payload, and writes the request with the lock reference appended.

After the two actual validation exporters complete, root creates
`gnnm-propagation-root-validation-target-freeze-v1` with `root_qualified`,
`method_binding: ref`, `test_targets_loaded_or_referenced: false`, and
`validation_exports` mapping both task names to exact `bundle: ref`,
`receipt: ref`, `supervisor_directory`.

The finalization request is exactly
`gnnm-propagation-fixed30-root-finalization-request-v1` plus
`binding: ref`, `root_authorized: true`, `validation_target_freeze: ref`.
It cannot nominate alternative fit directories or omit cells.

After finalization and both post-lock test exporters complete, root creates
`gnnm-propagation-root-test-target-freeze-v1` with `root_qualified`,
`method_binding: ref`, `family_lock: ref`, and `test_exports` mapping both tasks
to their actual `bundle: ref`, `receipt: ref`, `supervisor_directory`.

Every label receipt is checked against its completed supervisor and method
binding. Each target bundle contains one split only. Test receipts additionally
bind the exact completed all 30 pretest lock. No test indexing, metric or
member/execution/checkpoint selection is permitted before that lock.
