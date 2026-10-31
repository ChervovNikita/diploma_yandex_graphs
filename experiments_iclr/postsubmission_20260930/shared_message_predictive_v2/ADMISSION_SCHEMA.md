# Actual root bindings required

All references are exact canonical phase-relative `{path, sha256}` objects.
Paths and output roots remain inside this package; source/input dependencies
may refer to other preserved project packets. No example approval is issued.

## Scientific binding

`schema=gnnm-shared-message-fixed30-admission-v2` requires:

- `scientific_training_authorized=true`,
  `validation_target_export_authorized=true`,
  `test_target_export_or_evaluation_authorized=false`.
- Exact `design`, `rng_member_order_policy`, `fit_order`, complete `cells`,
  `model_arm_adapters`, `source_manifest_sha256`, `source_bindings_sha256`, root v2 `recipe`,
  `validation_policy`, `quality_screens`, `epochs_by_task` (300 each).
- `root_approval` with schema
  `gnnm-shared-message-fixed30-root-approval-v2`, qualified actual gates,
  validation permission, no test permission, and
  `approved_payload_sha256` over binding excluding that reference.
- `gpu_uuid`, actual `runtime_policy` (CUDA0, package versions, CPU threads,
  deterministic flag, CUBLAS workspace, false AMP/TF32), exact timing policy
  from `contract.TIMING`, and equivalence policy `{atol:1e-5, rtol:1e-5}`.
- `tasks` keyed by both task names, with hashed `engineering_phase` JSON
  reference accepted by the sealed quality loader (not an inline phase),
  `acquisition`, `prepared_manifest`, `input_completed_audit`,
  `parameter_control_report`, `provider_storage` and
  `validation_request_payload_sha256`. The parameter receipt must contain
  `qualified=true` and `source_plan_sha256=digest(counts.plan(task))`.
- `execution_policy[task][root_arm]` with exact fields `training`, `validation`,
  `deployment` (packed/serial) and boolean `checkpoint_blocks`. Shared and
  detached are packed; singles use their one packed interface.
- `qualified_gates`: source_audit, protocol_audit, synthetic_numerical,
  runtime_freeze, execution_equivalence, complete_training_Reddit2,
  complete_training_ogbn-products, complete30_resources,
  validation_export_source_audit. Each has qualified=true, report reference
  and matching report_sha256; numerical/training gates also require completed
  same-GPU supervisor_directory, exact quality-v3 request reference and successful
  outer_terminal reference. The quality source/result/runtime/tolerance/artifact
  chain is audited; source seals alone do not qualify these gates.
- `resource_ledger` reference with qualified=true, 30 fits, 9,000 updates,
  full forecast/memory/disk bounds and exactly the 12 explicit cost components
  checked in contract.py. No omitted metrics/transfers/replay/hash term.
- `prior_accounting` (root-v1 resource schema) and `remaining_resource_policy`
  with prior_GPU_wall_seconds and prior_additional_disk_bytes at least that
  receipt's full successful/failed child wall and conservative phase bytes.
- `whole_process_policy` has exactly `bounded_source_sha256` (the pinned
  b096966c generation) and finite positive `per_child_wall_cap_seconds` at most
  21,600. Each actual whole-process envelope cap must also fit the remaining
  family budget.
- New exclusive package-relative `output_root`.

## Separate validation request and freeze

Request schema `gnnm-shared-message-root-validation-request-v2` contains
root_authorized=true, binding reference, matching source_manifest_sha256,
task, split=valid,
test_target_access_authorized=false, provider_storage reference,
provider_storage_inventory_field, processed_public_file reference,
expected_processed_tuple_length (2 or3), and prescribed output_root.
Its digest excluding binding is frozen in the scientific task binding.

Completed receipts are root-frozen in
`gnnm-shared-message-validation-freeze-v2` with binding, root_qualified=true,
test_targets_loaded_or_referenced=false and both validation_exports. Each
export has bundle/receipt references, completed supervisor_directory and the
exact outer_start reference. Receipt live/import evidence and successful outer
completion are checked before loading its validation targets.

## Separate bounded engineering request

`gnnm-shared-message-cost-preflight-request-v2` requires root_authorized=true,
scientific_training_authorized=false,
validation_target_export_or_access_authorized=false,
test_target_access_authorized=false, retained_weights_or_scores=false,
warmup_updates=1, measured_updates=3, actual runtime_policy/gpu_uuid,
source_manifest_sha256, source_bindings_sha256, pinned bounded_source_sha256,
both tasks with all loader/input/parameter receipt
references, the complete execution_policy, new package output_root and caps
at most 900 seconds per child, 1 GiB transient file and 60 GiB peak reserved.
The separately hashed approval schema is
`gnnm-shared-message-cost-preflight-root-approval-v2`; it binds the payload
excluding root_approval and has qualified=true. This request has no power to
export validation or admit a scientific update.

Resource arithmetic consumes all ten exact preflight report refs with task,
arm, request ref, execution policy, same actual gpu_uuid and completed
supervisor_directory and exact outer_start reference. Four checkpoint/logit hash
passes are charged for creation, replay audit, family lock and the separate
family-lock audit. Required fixed_costs keys are in resource_forecast.py.
It returns an estimate; only a new independently qualified root resource
ledger can open scientific admission.


## Complete-process launch and receipts

Invoke the exact repository interpreter through `protocols/bounded_run_v1.py`,
then `run_authorized_v2.py --require-idle-gpu`, then the sealed entrypoint. The
same actual root request/binding must occur once under --request (preferred
when present) or --binding in the child argv, and --supervisor occurs exactly
once. GNNM_BOUND_START_JSON is the outer START path. The outer deadline begins
before entrypoint bootstrap and applies to the full process group, including
its reserved termination grace.

Every operation retains live.outer_start. Successful outer TERMINAL evidence
must include the exact authorization.json, completion.json, command.json,
environment.json and stdout_stderr.log bytes from the inner supervisor. The
terminal's whole_supervised_seconds is authoritative; inner seconds alone do
not establish eligibility or complete cost. Import receipts include the exact
merged runtime_import_tree, imported_phase_modules,
generated_module_policy_sha256 and installed_generated_code_sources.
