# Three-GPU campaign readiness — server51 target

15 October 2026. Local metadata/source inspection and a standalone dormant-queue builder. No remote access, full campaign regeneration, model execution or Git operation. This concerns the full **6,460-fit campaign**, not the separately prepared 24-fit PubMed pilot. Latest steering replaces server77 with **server51**, now verified by root as hostname **olig**, two NVIDIA A100 80GB PCIe GPUs and the repository `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git`. Device0 is `GPU-9c6c20fa-a620-479d-f72f-163ac74deef2`; device1 is `GPU-30283973-140d-9e7b-e780-0fcc487dec54`. `TARGETS_AND_WEIGHTS.json` pins that saved inventory. Resource/launch admission remains separate.

## Prepared builder

`build_dormant_queues.py` reads the exact existing canonical JSON (SHA256 `be7bfe7c3beb5e5c9e639b53404455ee8b92b20f4dc038b05224fe141782ff64`, 27,035,931 bytes) and emits three compact `QUEUE_*.json` files, `SUMMARY.json` and an output manifest. Root will execute it on the allocation metadata; actual campaign queue counts are pending that execution. The builder imports only the Python standard library, does not open descriptors/datasets/checkpoints, and cannot dispatch workers.

It groups **(cell_id, split, cohort_seed)**, merging source-reference and matched-track canonical groups. I4 export keys are inferred from their four ordered native dependencies because exports need not contain split/seed fields. Each cell/profile is processed in three-cohort rounds, with each worker used once per round. Among eligible workers, the next highest-weight cohort goes to the lowest current total. Cells are processed by descending total weight. Equal-load ties use cohort count, then declared worker order. This is deterministic and invariant to input task order.

Declared relative weights are **native fit 1; shared fit 4; export .05**, implemented as integer ticks (20 ticks/unit). They are route-fit scheduling proxies, not measured seconds, epochs, MACs or complete-fit costs. All devices have an explicit equal-speed planning assumption. Per-cell stratification spreads dataset/architecture/stopping-law uncertainty; it does not prove time balance. No GPU-hour estimate is emitted.

Compact rows preserve canonical IDs, runtime/science/source hashes, dependency order, original group and worker, output identity, and a digest of the entire original task. The planned target exists only in the queue envelope. Every task is checked exactly once, dependencies must be earlier on the same worker, and canonical tasks remain unchanged. Native body0 remains M1; no new fit is added. Output directories must be fresh and input/source hashes must match before output creation. Queues explicitly carry `launch_admitted=false` and `dispatch_supported=false`.

Seven source tests cover a synthetic full 6,730-task inventory, export inference/track merging, field/digest preservation, stratification, input-order determinism, missing/duplicate tasks, cross-cohort and cyclic dependencies, I4/M1 violations, input/source tampering, existing-output refusal and the CLI output hashes. All pass. The first test run exposed a temporary-directory path-normalization mismatch in the source checker; resolving the package path fixed it. This is metadata evidence, not actual campaign generation or destination qualification.

## Exact current assignment

The saved actual normalization preparation receipt and allocation-only source invariants agree:

| Existing worker | Fits | Export-only tasks | Total tasks | Admitted children |
| --- | ---: | ---: | ---: | ---: |
| allocation/gpu0 | 6,460 | 270 | 6,730 | 0 |
| server77/gpu0, legacy disabled | 0 | 0 | 0 | 0 |
| server77/gpu1, legacy disabled | 0 | 0 | 0 | 0 |

The fits comprise 2,890 normalized replacements and 3,570 unchanged candidates. The 270 exports preserve native I4 dependencies and body-0/M1 reuse. The inherited grouping has 540 canonical groups. **The production dispatcher has no server51 queue.** The new builder prepares planning views for allocation GPU0 plus the two verified server51 GPUs. Their exact new assignment counts remain unset until root runs the builder; the production ownership above is unchanged.

Count authority: `normalization_portable_dispatch_v1/ACTUAL_PREPARATION.json`, with its computed SHA256 in the accompanying `SUMMARY.json` source inventory. This is a saved generation receipt, not a fresh server census. The generated canonical plan, ownership and release are on the allocation; they were not fetched or regenerated here.

## Cost balance

**No supported campaign wall-time estimate or time balance is currently recorded.** All current tasks are on one GPU. Empty legacy queues have zero assigned work; allocation time is unknown. Source metadata leaves `estimated_seconds` and campaign ETA null.

The retained catalogue has MolBACE GIN/dropout .5 completed-fit priors of 662.73 s for native single, 1,495.46 s for native boundary GNNM and 1,718.96 s for task-geometry alignment. They cover only 60 precursor targets (0.93% of 6,460), from 18 matching-case observations within a 24-fit pilot. Runtime/reporting generations and seeds differ; contention was present. These are not measurements for the normalized generation, other datasets, server51 or concurrent pairs. Six source-mismatched observations deliberately received no numeric target weights. The old 793/785/902 allocation belongs to an earlier 2,480-fit design and must not be reused.

The prepared placement uses the declared proxy above because host-specific timing ranges are unavailable. Future cost-based revisions can replace this heuristic after source/destination applicability is established; preserve whole comparison cohorts and label known-cost coverage and uncertainty. A count or proxy-unit split is not a GPU-hour balance. Do not divide elapsed time by two merely because the child ceiling is two.

## Concrete source barriers

1. `portable_dispatcher_v3/host_binding.py` and ownership validate only allocation/server77 worker and host names. Both77 configurations are disabled with absent identities. A **new server51 configuration/host-schema successor** is required; server77 names must not impersonate server51.
2. Normalization `prepare_projection.py:126–138,173` preserves allocation-only placement. `binding_rules.py:49–62` supports only same-allocation composed execution. Moving normalized jobs requires typed rebinding of their parent execution descriptors and full path/source closure; scientific identities and settings stay fixed.
3. Normalization `run_gpu.py:57–63` requires reporting and then explicitly requires `allocation/gpu0`. Reporter/scorer/profile paths and index identity also bind the allocation generation. A host-local reporting projection and corresponding launcher successor are required for server51.
4. Existing ownership requires one owner per task, whole cohorts, co-located I4 dependencies, and one permanent output/claim namespace. Audit existing owner/claim/checkpoint custody before assigning a new map; metadata preparation creating zero claims does not prove the remote namespace is still empty.

These are source restrictions, not a request for another user permission round. The user has asked to prepare server51 queues. Binding that instruction and observed host evidence into a reviewed successor is the work needed to make the source accept that target.

## Smallest preparation for three reviewable dormant queues

1. Host/device identity is now supplied and pinned. Python/environment pins, effective CPU/RAM/cgroup/disk envelopes and runtime/source applicability remain production-release work. Leave77 excluded.
2. Run the prepared builder once on the existing normalized canonical plan, verifying its exact byte/hash pin. It extracts compact task ID/kind/cohort/dependency/science-hash/output rows. Reuse the 6,460 descriptors; do not rerun the compiler, normalization migration or scientific fits.
3. Review its three ordered task views and count/proxy-work summary. It preserves all 6,730 IDs exactly once and all dependency/cohort constraints. The plan remains provisional in time terms. This compact packet is sufficient for dormant queue review before deployment. The existing ownership/claim namespace still needs a fresh custody audit before any production reassignment.
4. Prepare a narrow dispatcher/host/reporter rebinding successor and, subsequently, two host-local release projections from that same ownership map. Keep `human_GO=false`, `all_tasks_qualified=false`, bindings unadmitted, and no claim/worker launch. Produce three pinned run-spec drafts and three command strings. Source or reporting relocation cannot be achieved with an assignment JSON alone.

Activation gates remain per-task source context, provider/split/evaluator, native epoch controller, selected export/resume RNG, mature target resources, reporting/statistics/access, source-reference-or-matched-controller policy, and campaign scope bridge. Worker identity/environment, effective cgroup limits, shared CPU/RAM/disk budget, claim/STOP/failure/resume custody, exact source closure, readiness and GO pins also remain. Existing component evidence may satisfy portions after applicability review; the inherited `remaining_blocks` list is not a fresh claim that all components are unfinished. Pair resource evidence is necessary only before admitting two overlapping children; a single child per GPU avoids assuming pair capacity.

## One-command interfaces

From the allocation, after staging this frozen package, the metadata-only command is:

```sh
python3 -B /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/three_gpu_campaign_readiness_20261015_v1/build_dormant_queues.py --source-manifest-sha256 REVIEWED_MANIFEST_SHA --output /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/three_gpu_campaign_readiness_20261015_v1/generated_v1
```

The canonical input path is already pinned in the config. Optional `--canonical-plan PATH` accepts an exact local copy with the same bytes/hash; it cannot switch campaigns. This command creates no ownership, claims, run specs, release, worker processes or host probes. Its only writes are the fresh dormant output directory.

The existing normalized launcher accepts only `--worker`, `--run-spec`, and `--run-spec-sha256`; it has **no dry-run switch**. Do not run it as a queue preview. After a server51 successor and actual admission, the intended per-GPU form is:

```text
QUALIFIED_PYTHON -B -u HOST_LOCAL_SUCCESSOR/run_gpu.py --worker WORKER --run-spec HOST_LOCAL_RUN_SPEC.json --run-spec-sha256 REVIEWED_SHA256
```

Three concrete launch strings still need the production successor/run-spec pins and admission. The old `prepare_projection.py` command creates another allocation-only production projection; use the new dormant builder for queue review.

**Outcome:** the standalone builder, pinned target/weight config and source tests are prepared and frozen. Root's actual generation will establish the three proposed queue counts and proxy-work balance. Production ownership, synchronization/qualification and launch remain separate; no measured time balance is claimed.
