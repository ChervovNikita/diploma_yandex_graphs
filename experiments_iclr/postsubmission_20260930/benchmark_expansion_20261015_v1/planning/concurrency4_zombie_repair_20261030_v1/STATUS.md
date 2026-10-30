# Allocation handover repair — 30 October 2026

## Incident

The allocation's two existing fits completed successfully and its old queue
owner stopped at 2026-10-29 07:16:58 UTC (10:16:58 Moscow). The process remained
in Linux state `Z`. The handover watcher incorrectly used an unchanged PID
identity as proof that the process was still running, and therefore never
activated the replacement scheduler. This was a defect in the 29 October
handover implementation. The allocation remained idle for approximately
18.81 hours before repair dispatch at 2026-10-30 02:05:46 UTC (05:05:46 Moscow).
The replacement launched four pending fits at approximately 05:06:26 Moscow.

Server51 completed both handovers without this failure. GPU0 has four active
fits. GPU1 is occupied by a workload outside this campaign (about 74.6 GiB GPU
memory at the observation); its queue correctly waits for resource capacity.

## Repair and verification

`finish_handover.py` distinguishes process identity from liveness. A matching
zombie is exited. Before completing the original request, it verifies the
stopped queue, empty active IDs, no current failures, exact dead tmux pane,
original task plan, smoke release, recovery authority, and its own STOP marker.
It acquires all four original queue locks, then terminates only the exact stuck
watcher using a Linux PID handle. The prior STOP, queue and watcher records are
preserved. No native training process was signalled, and no completed result
was rerun or changed. It launches the already-tested four-slot scheduler from
the original request, preserving its exact scientific invocation and manifest.

All 11 local regression tests passed, including zombie/missing/reused process
identities, unreadable process state, queue and pane gates, and ensuring the
signal targets only the exact watcher. Live allocation inspection confirmed
state `Z`, a stopped queue with 48 completed tasks and no running/failed tasks,
and the matching sleeping watcher before the repair.

At 05:08 Moscow on 30 October:

| Worker | Completed tasks | Running | Pending |
|---|---:|---:|---:|
| Allocation GPU0 | 48 | 4 | 2,192 |
| Server51 GPU0 | 172 | 4 | 2,067 |
| Server51 GPU1 | 62 | 0 | 2,181 |
| Total | 282 | 8 | 6,440 |

This is 83 additional completions since the previous 199-task check. All three
schedulers are alive in full-training phase with the four-slot limit. The
allocation's molecular fit had reached epoch 73; its three new BUDDY processes
were active but had not yet published a completed epoch in that snapshot.
Server51 GPU0's epoch counters advanced between observations. No current failed
tasks or scheduler errors were present. These operational observations are not
a scientific comparison or a claim that all GPUs are exclusively available.

Local health and capacity evidence is in `planning/progress_20261030_v1/`.
Remote repair custody is in this directory's `allocation_gpu0/` subdirectory.
The original handover package remains unchanged as historical source. Its
identity-only wait must not be reused for a new handover.
