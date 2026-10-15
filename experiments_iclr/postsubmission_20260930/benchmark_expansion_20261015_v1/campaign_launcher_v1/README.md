# Finite campaign launcher v1

**Source preparation only. No scientific launches or remote actions.**
`PREPARATION.json` and the three saved dry runs are **BLOCKED_NOT_COMPILED**.
The retained redesign plan has 2,480 historical fits, assigned 793/785/902;
none has a compiled runtime job/resource binding here. These are not a final
expanded roster, a measured load balance, an ETA, or campaign authorization.

## Commands: one supervisor per declared worker

Run from the project root `postsubmission_research_20260930`:

```sh
python3 -B benchmark_expansion_20261015_v1/campaign_launcher_v1/supervisor.py run --project-root . --manifest benchmark_expansion_20261015_v1/campaign_launcher_v1/PREPARATION.json --worker allocation/gpu0 --dry-run
python3 -B benchmark_expansion_20261015_v1/campaign_launcher_v1/supervisor.py run --project-root . --manifest benchmark_expansion_20261015_v1/campaign_launcher_v1/PREPARATION.json --worker server77/gpu0 --dry-run
python3 -B benchmark_expansion_20261015_v1/campaign_launcher_v1/supervisor.py run --project-root . --manifest benchmark_expansion_20261015_v1/campaign_launcher_v1/PREPARATION.json --worker server77/gpu1 --dry-run
```

After a separately completed bridge, compile exact reviewed bindings:

```sh
python3 -B benchmark_expansion_20261015_v1/campaign_launcher_v1/supervisor.py compile --project-root . --source-plan reference_wrapper_redesign_20261013_v1/PLAN.json --bindings reviewed/EXECUTION_BINDINGS.json --release-id REVIEWED_ID --output-root campaign_outputs/REVIEWED_ID --output reviewed/COMPILED.json
```

The same three worker commands accept `--manifest reviewed/COMPILED.json`.
Actual allocation execution additionally replaces `--dry-run` with
`--run --go reviewed/GO.json --readiness reviewed/READINESS.json` and uses the
qualified repository interpreter. That is one supervisor process; it starts
at most two children. **Server77 actual execution always returns BLOCKED before
project metadata reads**, including GPU1. No SSH, transport, device remapping
or route-guard bypass is supplied. Current authorization remains allocation-only.

## Exact execution contracts

`contracts.py` accepts the current `reference-wrapper-campaign-v2` source
schema. It never calls retired v29 commands or resolves abstract factory names.
`compile` requires a `finite-campaign-execution-bindings-v1` object containing:

- `source_plan`: exact `{path,sha256}` of the source plan;
- `source_sha256`: reviewed relative-path→SHA256 closure of executable source,
  native/provider/evaluator dependencies and any explicit transform/RNG audit;
  required runtime-v2 and supervisor files are checked;
- `python_executable_sha256_by_worker`: qualified interpreter bytes for each
  declared worker, verified for the actual executing worker;
- `fits`: exactly every source fit ID, mapped to `{job,resources,measured_cost}`.
  `job` is an exact path/SHA256 receipt for a complete immutable
  `native-full-epoch-fit-v1` JSON descriptor, with `TEST_access:false`.
  Recipe, native settings, seed, split, body, procedure, track and horizon must
  match its source contract. Callback, transform and link policy are pinned;
- optional `VALID_analyses`: explicit `{bank_id,resources,measured_cost}` entries
  for declared native-I4 banks. They consume four own selected VALID exports
  after their fit dependencies complete. Cross-worker analysis is rejected.

Resources carry positive integer `vram_bytes`, `ram_bytes`, `disk_bytes`,
`measured:true`, `mature_optimizer:true`, a source `evidence` receipt, a
`profile`, `max_concurrent_owned_children` (1 or 2), `pair_qualified`, and
`qualified_peer_profiles`. Serial-only profiles may have pair qualification
false. Two-child overlap requires reciprocal qualified profiles and measured
headroom. Full process/context/mature optimizer and variable batch/cache peaks
must be covered; a PyTorch allocator peak alone is insufficient.

Measured cost is `{measured:true,observed_wall_seconds,observed_optimizer_calls,
scope,evidence}`. It is an observation with a declared scope, never a fabricated
full-fit cost, balancing weight or ETA. The compiler carries it unchanged.

`finite-campaign-human-GO-v1` must contain `user_explicit_GO:true`, exact raw
`manifest_sha256`, `source_bundle_sha256`, exact `readiness` receipt,
`allowed_workers`, and `review_evidence` citing the actual human instruction.
No production GO receipt is created here. Receipt custody must come from the
reviewing human/root workflow; a JSON boolean cannot authenticate human intent.

`finite-campaign-readiness-v1` binds the same manifest/source hashes. Each worker
needs evidence-pinned PASS records under `workers[worker].gates` for:

```text
reviewed_finite_roster             source_and_data_binding
native_runtime_and_controller      checkpoint_resume_and_rng
validation_only_selection          resource_mature_and_two_slot
complete_failure_and_cost_reporting  campaign_scope_bridge
```

Its `budget` declares nonnegative `gpu_headroom_bytes`,
`host_ram_headroom_bytes`, and `disk_headroom_bytes`. Missing, non-PASS or
changed evidence blocks execution. Manifest, GO, readiness, source, evidence
and the selected job are rechecked before every admission.

## Supervisor behavior

- Fixed child handlers are runtime-v2 `run_fit.py` and `aggregate_selected.py`.
  There is no arbitrary command, shell, entrypoint, retry, resume or TEST option.
  Statistical `evaluate_roster` endpoints are counted but not executed here.
- Every owned training/analysis child consumes a slot from launch through reap,
  including initialization before CUDA appears. Unknown own GPU PIDs block
  admissions. Other users' occupancy is retained through physical free VRAM.
- Admission includes physical free VRAM, `/proc/meminfo` available RAM, disk free
  space, explicit headroom and live reservations not yet physically allocated.
  RAM/disk reservation logic includes both same-host GPU workers. Current
  production routing admits only the allocation's one GPU.
- A repository-wide worker lease prevents duplicate local supervisors. Exclusive
  permanent fit claims and fresh output checks prevent overwrites and repeats.
  Worker output is one-shot too; stale claims/outputs are never stolen or cleared.
- Child/launch/source/claim/admission failure writes shared `STOP.json`, stops
  admissions and lets live peers finish. A STOP from another sharing worker is
  honored. No peer is killed. Abrupt supervisor/host death retains claims; manual
  root investigation and a separately reviewed disposition are required.
- `START.json`, claim receipts, per-child logs, `PROGRESS.json` and worker
  `RESULT.json` retain terminal outcomes, failures, native result hashes,
  per-invocation optimizer work/resources/wall time, and observed launch-to-reap
  time. Missing native costs are unknown, never zero. `no_selected_state` remains
  a native scientific outcome; an analysis requiring that state is blocked.

## Outstanding bridge and gates

1. Freeze the final finite expanded roster and create complete runtime-v2 jobs
   for every fit. Current source abstract factories/provider ports are not this
   bridge. Bind all data/split/evaluator/source dependencies and explicit
   repository-root fields required by private transforms. Qualify every native
   controller, complete epoch, VALID export, RNG and checkpoint/resume path.
2. Obtain mature single/pair resource profiles and measured-cost receipts on
   the target environment. Existing small MolBACE observations and CPU semantic
   checks cannot fill these fields. Qualify external occupancy and headroom.
3. Review campaign scope: runtime-v2 child receipts still say
   `full_campaign_GO:false` and describe bounded validation-only pilots. This
   launcher preserves that source and records its separate reviewed GO. The
   `campaign_scope_bridge` gate must explicitly resolve this provenance, the
   source-horizon/no-TEST diagnostic adaptations, final reporting/statistics and
   any unsupported source clock; this source does not declare it qualified.
4. Server77 needs explicit reauthorization, verified host/GPU/environment
   identity, a separately qualified route-aware driver/device mapping, shared
   host resource ledger, cross-host claim/STOP transport and failure-cost custody.
   Editing a CLI worker name cannot supply this bridge. Existing runtime-v2's
   allocation-only guard remains unchanged.
5. Commit source, review all readiness evidence and issue an exact matching GO.
   `PREPARATION.json` cannot be used as a compiled execution manifest.

`checks.py` has **18 passing stdlib synthetic checks**, with Popen and system
probes mocked; no Torch/model/dataset imports or actual scheduler children.
Run it with `python3 -B benchmark_expansion_20261015_v1/campaign_launcher_v1/checks.py`.
`MANIFEST.json` records source/read hashes and the saved blocked dry runs.
