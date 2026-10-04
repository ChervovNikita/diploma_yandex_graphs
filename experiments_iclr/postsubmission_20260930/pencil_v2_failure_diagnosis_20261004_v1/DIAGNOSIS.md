# PENCIL v2 failure diagnosis

**The recorded initiating failure is the supervisor's RSS observation assertion.** Its own cleanup then sent SIGKILL, producing exit−9. No wall, sampled RSS or output-cap violation is recorded. This diagnosis proposes a specific observer repair; it neither authorizes nor performs a retry.

Exact v2 source manifest: `c937462c776f48ef3ce84d428ede3f3d5ccc4837c4874d54bd8961638a74bbc4`. Seal: `6fcf6d2992f51c638b57e7d1ec462cc6b2256ca941d9f212ec2016fd323029c4`. Supervisor: `ae5f8e80afdb3005b39a1cc97e8c65a36decaf8dd128ee032fb1def27f823e5e`. External release: `9fe8f9dbac65aeb5ced77d83b3934dd7ee831a1563ddb24decfcef4749ee1e5f`. The scoped remote hashes match the exact local source.

The physical receipt records `RuntimeError: Owned live RSS observation missing`, stop `SUPERVISOR_EXCEPTION`, and SIGKILL of the held group3258600 at15:23:40.930UTC. It subsequently confirms physical session closure, direct-child reap and unresolved_cleanup=false. The last successful STATUS at3068.746s showed the direct worker in stateR with twelve owned DataLoader descendants; failure/cleanup completed at3069.299s. Root's progress record reports epoch_index7,800 optimizer updates,6,344 TRAIN batches and1,256 VALID batches—eight complete traversals counted, with the worker still at native_full_VALID. No scientific fit completed; seeds1/2 were unattempted.

| Recorded resource | Observed | Cap |
| --- | ---: | ---: |
| Peak sampled owned-session RSS | 30.022 GiB | 64 GiB |
| Physical parent wall | 3069.299 s | 14,400 s |
| Final published output bytes | 330,336,998 | 2,147,483,648 |

The worker resource-monitor failure receipt is absent. No CUDA peak is present in the permitted failure metadata, so this memo does not establish an instantaneous CUDA bound or prove full-fit feasibility. The available evidence supplies no recorded worker cap failure.

## Why the v2 fix remains incomplete

The exact fault is `supervise.py:50–59`, inside members_of_session. It first reads `/proc/PID/status`, then rereads `/proc/PID/stat` and revalidates birth/session. It accepts missing VmRSS only if the second state isZ. This fixes a stale first-state snapshot but still assumes memory release and zombie-state visibility happen together.

Linux can release a task's mm during exit before its task state appears asZ. The status read can then omit VmRSS while the later stat still reportsR/S. The current sampler treats that legitimate transition as fatal, and main's exception handler kills the entire otherwise active owned fit. DataLoader worker teardown after full VALID is consistent with the observed counters and twelve descendants.

The exact failing PID/state was not recorded. The evidence cannot prove which descendant/direct worker caused the assertion or exclude an unlogged generic worker exception beginning during teardown. The direct child had not reached an already-final natural exit: its recorded−9 follows the supervisor's SIGKILL. The assertion's branch follows successful birth/session revalidation; changed PID/session observations are skipped before it, so the recorded error is not that mismatch branch. No general worker failure receipt exists; the code writes only resource MONITOR_FAILURE. Stdout/stderr and quality logs were not opened.

This differs from v1. Its worker explicitly recorded `CUDA reserved cap` at425.569s and exited2; observer/closure errors accompanied shutdown. In v2 the resource receipt is absent and the RSS assertion is the initiating recorded cause.

## Minimum repair

In a new source/execution version, replace only the RSS observation helper with a strict single `/proc/PID/stat` snapshot containing state, start_ticks, session, group and RSS pages. Field24 is RSS in pages, index21 after the comm field; multiply by `os.sysconf('SC_PAGE_SIZE')`. A valid R/S task with RSS0 is accepted as sampled zero memory, without guessing that a missing reading means zero. Reject malformed/negative values. Preserve exact group/session/birth checks, held waitid/WNOWAIT identity, owned signaling, physical closure/reap and spent-attempt/active-fit locks. Retain all existing cap values and disclose that RSS is still sampled approximate accounting.

`MINIMUM_REPAIR_PROPOSAL.py` is a source fragment for that correction. It is not installed or used against real processes. `TARGETED_SAMPLER_REPRO.py` extracts the actual v2 function without importing its gate/numerical code. One synthetic same-birth/sessionR task with absent VmRSS reproduces the actual assertion text. The proposed stat parser accepts R/RSS0, S/positiveRSS and Z/RSS0; negative RSS, truncated stat and mismatched PID fail. No live /proc, process creation or signaling is exercised. This is a focused observer regression, not scientific resource evidence.

Root's DDI receipt/runtime agent was informed directly and has confirmed it will replace this sampler only in the new DDI execution root while retaining the held identity/waitid/owned-kill primitives. The sealed PENCIL v2 source was left untouched.

All v1/v2 failures, original output directories and spent locks remain preserved and unadopted. Any consciously admitted restart should be fresh in a new directory with the same20epochs, recipe, native sampling, workers, model, optimizer, precision, allocator policy, cap values and first-tie selection. No stored scientific state is read, donated or resumed. No scientific qualification ladder, new resource-only fit, seed replacement or success-only mean is proposed. Root controls prospective admission and scheduling; this diagnosis performs no launch, signal or automatic retry.

The one serialized relay call read only physical/error/resource/start/queue receipts and source hashes. It accessed no scores, selected checkpoints, datasets, other jobs or success/quality logs. `FAILURE_RESOURCE_RECEIPTS.json` retains their hashes and exact permitted metadata.
