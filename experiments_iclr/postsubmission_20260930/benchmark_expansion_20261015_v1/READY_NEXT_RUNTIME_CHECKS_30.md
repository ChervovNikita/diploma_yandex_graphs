# Two remaining runtime checks

Prepared 2026-10-15 from retained local source and metadata. Root owns publication,
admission and execution. No remote status, model/tensor execution or Git operation
was performed for this note. Paths below are relative to this expansion directory
unless explicitly absolute.

Root reports release30 published/pushed at commit
`564f55ed34734488e73ef55d7111dd7657f331c1`. This note does not independently
verify publication or confer runtime admission.

## Readiness and exact remaining work

| Check | Available executable / admission | Remaining paid work |
| --- | --- | --- |
| NCNC selected CUDA replay | `qualification/final_selected_node_NCNC_VALID_v3/replay.py`; inactive `ROOT_ADMISSION_TEMPLATE_NCNC_C4.json` in that package | 1 constructor, 1 selected restore, 1 whole VALID pass; 4 encoder forwards, 8 top-level predictor calls; **0 TRAIN updates** |
| Composed BUDDY | `composable_normalization_runtime_v1/qualify_native.py`; inactive `analysis/composed_buddy_qualification_handoff_v1/ROOT_ADMISSION_TEMPLATE.json` | 1 complete TRAIN epoch, **1585 optimizer transitions**; 3 constructors, 4 whole VALID passes, 2 same-call mean exports; stopped-finalization resume adds 0 TRAIN |

The inspected evidence contains templates, not an active admission for either
check. `NODE_SCORER_REPAIR_REVIEW_26.json` explicitly leaves
`NCNC_execution_admitted=false`. The corrected node replay already passed; its
admission does not transfer to NCNC.

No additional source repair is indicated for these two checks. NCNC v3 pins the
repaired `final_selected_scoring_v5`. The older composed qualifier uses its
composed owner's `state_signature()` and unchanged composed serving functions;
the inherited tensor-only extra-state defect does not affect that branch.

### Source and input bindings

| Binding | SHA256 |
| --- | --- |
| `qualification/final_selected_node_NCNC_VALID_v3/MANIFEST.json` (1804 bytes) | `8fc619d48cc82b1106c1ca6e3534376e661a1dd1c9987f6b335d477fc94f0f97` |
| Same package `INPUTS.json` (1932 bytes) | `47bf7c3a817d4a8cea511e9317c387c76f9f3cb785e5be1c26853d7450dacd67` |
| `final_selected_scoring_v5/MANIFEST.json` (2835 bytes) | `e0b86279593c9f20e49b615c57fce41360babc40201b3172ae76db6eab6f9659` |
| `final_selected_VALID_qualification_plan_v2/MAPPING.json` (56166 bytes) | `8c4cbdbcc6576d39a45b187f2cfd6f755a06221402423beade0990aac4d4c6a0` |
| `composable_normalization_runtime_v1/MANIFEST.json` (4779 bytes) | `518784a9e83f4ba92c4f7bff55705999a787c7a67ff69c0cc9c09b5530550d38` |

NCNC keeps the genuine `runtime_link_v6` producer, job
`runtime-link-v6-NCNC-conditionalJ-two-epoch-CUDA-lifecycle-v1`, job hash
`25be2dd5b81b4d9352d0c5a14c712121b549a3464caa7e7c23dcde7cc8a095ac`,
and epoch-1 selection. `INPUTS.json` and MAPPING pin its original JOB, lifecycle
metadata and exports. The original checkpoint is
`results/runtime_link_v6_native_lifecycle_attempt01/qualification/own_fit/selected.pt`
(47870570 bytes; SHA256
`d306a40c16c96ec3edbdf45a863bc64ca3b9a07ab98e11dfe4a25b44c35e8253`).
Its sibling `selected_VALID.pt` is 11613944 bytes, SHA256
`6c6d5d47652fcad3284cd5df0ac633f18a360904276e17c95870cd8dbfa0c778`.
Do not substitute a current portable runtime for this producer.

BUDDY uses case `buddy_seed1` and the authentic input receipts in
`NORMALIZATION_QUALIFIER_INPUTS_20.json` and its admission template:

- Parent `portable_dispatcher_v3/generated_v3/allocation/jobs/117db28ae163add5eda2807ad96da485f29f5cce753f81a2f8d17866ed5ec832.json`:
  28590 bytes, SHA256 `f45efba0aee3f5fa42b389eea095ef4b93aef545fe1475cc900bd46ed511b3ee`.
- Contract `final_role_population_preparation_v1/allocation_population_v3/contracts/collab-buddy.json`:
  4313 bytes, SHA256 `01dbc8adf427adf1e478b876a796fb76ff0470e4b97a9b32885ae83461e7835a`.

The parent declares seed 1, `interaction_input_conditionalJ`, and closed TEST.
The contract's schema name contains TEST, but the qualifier calls literal VALID
only. Keep both original files unchanged and authenticate their actual allocation
bytes before admission. The earlier handoff verified 28 runtime members and 71
dependency pins; root must confirm the same closure at execution.

## Resources: reservations versus observations

| Check | Proposed physical reservation / limits | Retained measurements from the earlier lifecycle |
| --- | --- | --- |
| NCNC replay | 32 GiB GPU; 8 GiB host RSS; 1 GiB disk; 2 CPU threads, 1 interop thread; internal 600 s, outer 610 s plus 10 s TERM-to-KILL | Prior 34-update lifecycle: supervisor 181.319 s, child 180.20 s; RSS 1847004 KiB; CUDA allocated 12673890304 bytes, reserved 26698842112 bytes |
| Composed BUDDY | 40 GiB GPU; 16 GiB host RSS; 8 GiB disk; 2 CPU threads; internal 3600 s, outer 3610 s plus 10 s TERM-to-KILL | Original 3170-update BUDDY lifecycle: child 1066.694 s; sampled RSS 4749701120 bytes, sampled GPU 4922015744 bytes; CUDA allocated 2726136320 bytes, reserved 4389339136 bytes |

NCNC measurements come from `qualification/NCNC_RUNTIME_V6_QUALIFICATION_07.json`
and the retained lifecycle referenced by the replay README; BUDDY measurements
come from `BUDDY_LIFECYCLE_SUMMARY_22.json` and `BUDDY_LIFECYCLE_TERMINAL_22.json`.
They are historical costs, not measurements or timing guarantees for the two
missing checks. Enforce physical RSS/GPU/disk bounds with root's existing
supervisor; do not add a virtual-address-space cap.

## Root's minimal execution sequence

1. Let the existing structural selected collection/complete-analysis queue
   (`qualification/structural_complete_collection_queue_run_v1/QUEUE.json`)
   release its shared slot after its terminal gates. Root's current priority is
   **NCNC, then composed BUDDY, before admitting long PubMed24 if resources fit**.
   Retain the six structural-fit and seven PubMed-engineering outcomes under
   their existing gates; do not repeat completed prerequisites. Honor actual
   failures and current owners. This note supplies no fresh terminal/resource
   claim and creates no new queue.
2. Use the literal authorized allocation route from `../AGENTS.md`. Before
   project reads, verify host `anogena-2-0` and sole physical GPU
   `GPU-44039938-fd82-41d2-fefd-de71514e2fac`. Verify published source closures,
   inputs and the chosen interpreter. Reuse the already qualified interpreter
   review when its exact identity still matches.
3. Acquire `analysis_outputs/selected_VALID_collection_gpu_slot.lock`
   nonblockingly and hold it through each supervised child. If occupied, defer
   to its owner. Refresh host/GPU/disk and authenticate every peer while holding
   the slot. Two model workers is a ceiling, not automatic permission. The old
   GRU's PID 640428/start ticks 6055039165 and 16 GiB host/32 GiB GPU growth
   envelopes apply only if that same identity is still present. No third worker.
   Require the new reservation after accounting for retained peer growth.
4. Admit NCNC first if the prerequisites hold, then composed BUDDY, serially.
   Make separate fresh copies of the templates; preserve the templates. For
   NCNC fill manifest/INPUTS hashes, fresh output, exact interpreter and review
   receipt, fresh peer/resource receipt, and set `status=ADMITTED`. For BUDDY set
   both root-review booleans true only after review, fill
   `root_resource_evidence`, and retain the 1585-update/3600-second bounds and
   exact input receipts. Its proposed output is
   `qualification/composable_normalization_buddy_seed1_v1_attempt01`; verify
   output and log are fresh. An existing attempt must be retained and reviewed.
5. Run each command below once under the existing supervisor, which must retain
   exit status, actual costs, timeout/resource violations and cleanup of its own
   process group. Do not launch either command unsupervised. Keep the full
   reservation through child exit, release the shared slot, and evaluate its
   terminal result before proceeding. No automatic retries, including when a
   terminal RESULT is absent.

### Allocation-local child commands

These commands consume root-created admissions and run inside the held slot.
They are not admissions themselves. Set `REPLAY_PYTHON`, `NCNC_ADMISSION` and
`BUDDY_ADMISSION` to the exact paths root reviewed.

```sh
ROOT=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
EXPANSION="$ROOT/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1"

env PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES=GPU-44039938-fd82-41d2-fefd-de71514e2fac \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  timeout --signal=TERM --kill-after=10s 610s \
  "$REPLAY_PYTHON" \
  "$EXPANSION/qualification/final_selected_node_NCNC_VALID_v3/replay.py" \
  --root "$ROOT" --admission "$NCNC_ADMISSION" --case NCNC_C4

env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  timeout --signal=TERM --kill-after=10s 3610s \
  "$ROOT/.venv/bin/python" -B -u \
  "$EXPANSION/composable_normalization_runtime_v1/qualify_native.py" \
  --project-root "$ROOT" --admission "$BUDDY_ADMISSION"
```

## Required outcomes and scope of closure

**NCNC:** Preserve `START.json`, `EVENTS.jsonl`, `RESULT.json`, `VALID.pt` when
emitted, and supervisor logs. Require exit 0, `status=passed`, the exact operation
counts above and all 160084 VALID queries (60084 positive, 100000 negative).
Recursive work must remain 8 task dispatches/16 recursive dispatches, with only
8 task output corrections. Verify learned readout and same-bank mean at fixed
`rtol=1e-4, atol=1e-5`, exact ordered IDs/labels/endpoints, restored saved RNG,
unchanged model parameters/buffers/configuration and optimizer state, and zero
TRAIN/backward/steps/TEST/reselection. Preserve numerical failure exports.

**BUDDY:** Preserve `RESULT.json`, `PARENT_ENGINEERING_JOB.json`, `JOB.json`,
`own_fit/{RESULT.json,selected.pt,recovery.pt,selected_VALID.pt,...}`, per-invocation
copies under `invocations/one_complete_epoch` and
`invocations/stopped_finalization_resume`, `serving_cache`, and supervisor
evidence. Require exit 0, `status=passed`, exact operation counts, first-two-update
finite/nonzero shared/private/correction/gate gradients, equally advancing active
BN route counters and zero unused counters. Require genuine selected producer
binding, restore/resume, fresh full-VALID serving, unchanged model/private BN banks,
exact IDs/labels/endpoints, and logits within `rtol=1e-4, atol=1e-5`.

**Blockers are distinct:** source preparation for these checks is available;
fresh admission, scheduler/resource availability and the actual runtime witnesses
remain. A composed BUDDY pass does not qualify the new reporting observer/profile
integration, which this older qualifier does not attach. An NCNC pass covers its
retained producer/current serving route and does not migrate its checkpoint to a
current portable runtime. Root-reported normalization INDEX SHA256
`cd2bc64047e630452eb7b4ce13a0829b205327876c5183d7e28853f6d4445db6`
is generated metadata, not either runtime witness. These two passes do not establish
full-campaign resource readiness, scientific claims, release readiness, TEST
authorization or full-campaign GO.

**Already paid and retained:** original BUDDY 3170 updates; repaired NCN 34
updates; composed MolBACE 38 updates/3 constructors
(`NORMALIZATION_MOLBACE_OUTCOME_21.json`); corrected node selected replay
(`NODE_SELECTED_VALID_REPLAY_CORRECTED_26.json`, 1 constructor/1 restore/4 forwards,
1 VALID of 665 nodes, 0 TRAIN/0 CUDA). Do not repeat them to close these gaps.
Preserve the failed node v2 evidence in `NODE_SELECTED_VALID_REPLAY_26.json` too.
