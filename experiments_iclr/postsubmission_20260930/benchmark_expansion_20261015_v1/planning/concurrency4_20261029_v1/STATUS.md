# Four-job queue change — 29 October 2026

The user requested four concurrent models on each of the three campaign GPUs.
The new scheduler and its 10 focused tests passed on both authorized hosts.
Exact frozen task plans and scientific commands are unchanged.

The source manifest is
`82b056580848c7074d6d9e2a3aa9a13f298c2123256d0d9f07c9139ba6bbe912`.

At deployment (approximately 05:15 Moscow time), all three independent tmux
handover watchers were verified alive. Each recorded `draining_existing_fits`
with target concurrency four. All six original training fits remained alive;
none was signalled or restarted. Counts were 46, 93 and 60 completed tasks for
allocation GPU0, server51 GPU0 and server51 GPU1, respectively.

The old schedulers contain a fixed two-job limit. Each will finish its current
pair, then its watcher will replace the stopped owner in the original tmux pane.
The four-slot source is deployed and scheduled; four simultaneous jobs on each
GPU have not yet been observed. This distinction must remain in status reports.

The STOP files are deliberate graceful-drain requests owned by these watchers.
Do not remove them or start competing supervisors. Inspect the corresponding
`planning/concurrency4_20261029_v1/<worker>/STATUS.json` and `WATCHER.json` first.
Any failed fit encountered during the drain remains preserved and blocks the
handover pending inspection.

The frozen resource catalogue retains its historical two-child metadata; the
new scheduler records the explicit four-slot override. It preserves aggregate
VRAM growth, host RAM, CPU and disk checks. Four large fits may exceed capacity;
the queue scans for smaller compatible pending tasks. A four-job ceiling is not
a promise of four jobs in every resource combination or a measured speedup.

Evidence: local `*-preflight.json`, `*-scheduled.json`, `*-handover-status.json`,
source staging receipts and readable command/result records; remote worker
journals also retain the exact prior owners, queue status and launch requests.
