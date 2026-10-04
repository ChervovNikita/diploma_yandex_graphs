# Scientific v1 to v2 repair

## Evidence used

Only the root's `WORKER_MONITOR_FAILURE.json` and `FAILURE_METADATA.json` were read from scientific v1 execution. They are copied exactly under `evidence/V1_…`. No scores, checkpoint payload or per-epoch quality table was opened.

The worker's monitor recorded `RuntimeError: CUDA reserved cap` at 425.569 s, with stage `native_full_epoch`. Root reported epoch index 1 and 106 optimizer updates. This is a reserved-cap failure; no OOM or metric issue has been identified. The supervisor recorded a live-session closure exception and missing live RSS observation while owned processes exited. Final metadata confirms child exit 2, physical session closure and reap, with unresolved cleanup false. No seed completed; seeds 1 and 2 were not attempted. Failed v1 outputs remain unadopted and preserved.

## Exact runtime changes

| Policy | Scientific v1 | Scientific v2 |
| --- | --- | --- |
| CUDA reserved ceiling | 75 GiB = 80,530,636,800 bytes | 80 GiB = 85,899,345,920 bytes |
| CUDA allocated ceiling | 70 GiB | 70 GiB, unchanged |
| Wall / output ceilings per fit | 14,400 s / 2 GiB | Unchanged |
| Sampled owned-session RSS / shutdown budget | 64 GiB / 10 s | Unchanged |
| Allocator environment | Existing default profile | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

The standard expandable-segments policy is a disclosed runtime allocator adaptation. `PROPOSED_COMMAND.json` supplies it before either supervisor or worker imports Torch. The stdlib gate requires that exact environment, binds the allocator policy in the external root release, and still requires Torch absent from `sys.modules` during admission. CUDA allocations remain constrained by the pinned device's actual capacity; the 80 GiB policy ceiling does not change that capacity. No successful resource outcome, fragmentation diagnosis, numerical parity or performance benefit is claimed.

The supervisor changes only `members_of_session`. After reading `/proc/<pid>/status`, it rereads process identity. A disappeared process, replaced birth identity, or observation outside the originally observed session is omitted. For the same birth/session, current state and group replace the stale first observation. The same live process must still expose exactly one `VmRSS` field; a confirmed zombie may lack RSS. A changed owned process group remains visible to the unchanged strict group checks. This addresses the live-to-zombie observation race without accepting missing RSS from a still-live owned task.

All other supervisor functions are AST-identical to v1. The direct-child birth check, held `waitid/WNOWAIT` identity, exact owned-group SIGKILL, strict live-descendant closure, bounded cleanup, spent-attempt/active-fit locks and no automatic retry remain unchanged. The sampler is locally checked with eight isolated stdlib mock cases: valid live RSS, live-to-zombie, missing/duplicate live RSS rejection, disappearance, PID reuse, changed session, and changed owned group. These checks access no real `/proc`, process or signal and do not prove OS execution behavior.

## Scientific source and fresh-fit custody

`worker.py` is byte-identical to scientific v1. All 20 native files, scientific sampling/candidate/loss/evaluation code, model/optimizer initialization, dependency inventory, data/runtime authorities, precision policy, score export, selected model/optimizer/RNG state policy and strict first-tie VALID Hits@50 selection are unchanged. Seeds 0/1/2 each run 20 epochs from scratch. The v2 execution directory is `pencil_collab_paired_predictive_execution_root_20261004_v2`; it loads no failed-v1 or resource state and supplies no donor/continuation.

The earlier one-epoch v3 resource evidence remains historical input, under its original allocation profile. Scientific v1 failed and v2 has not run. This repair establishes no further resource feasibility. No resource-only pilot, numerical/native import, server action, install, fit, launch or execution client is produced by this preparation. Templates remain disabled for root's exact technical source review and fresh-fit release.

`V1_TO_V2.diff` includes every changed v1 payload and new repair metadata/documentation. `V1_TO_V2_PROVENANCE.json` binds v1 source/seal, exact failure metadata, caps and source identities. `INPUT_BINDINGS.json`, local source checks, `MANIFEST.json` and `SEAL.json` preserve the closure. V1 source and execution evidence remain unchanged.
