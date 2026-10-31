# Training queue recovery and audit — 24 October 2026

## Verified operational findings

The authorized one-GPU allocation and server51 were checked through their pinned
hostnames and GPU UUIDs. Server51's expired SSH control session was restored.
No existing training process was killed and no other user's work was changed.

The allocation had stopped on two context-affinity BUDDY runs after 149 and 168
epochs. Server51 GPU0 had stopped on two aggregation BUDDY runs after 149 epochs;
GPU1 had one analogous failure and another original BUDDY run still progressing.
These failures occurred in result reporting after training and selected VALID
export. The reporting code rejected legitimate documentation dependencies by
filename extension. The previous one-README exception did not cover later methods.

The additive `campaign_reporting_repair_20261024_v2` verifies every declared
dependency as a canonical, regular file inside the project with its exact recorded
SHA256. Reading a dependency for provenance does not execute it. The scientific
sources, descriptors, optimizer settings, splits and checkpoints remain unchanged.
The source package passed 15 focused CPU tests on both hosts.

Both allocation failures were recovered with **zero new epochs or optimizer
updates**. The selected epochs, selected and recovery checkpoint bytes, node/edge
IDs, labels, endpoints and VALID logits were unchanged. The queue then started two
new fits. The original failed attempts and their costs remain recorded.

Server51 GPU0's saved checkpoints passed the same preflight. Its restarted queue
waits for sufficient memory for the larger recovery tasks and can run compatible
smaller pending fits. GPU1's original run is left intact. A separate tmux watcher
waits for its natural exit, checks the exact expected reporting failures, audits
the saved checkpoints and dispatches the same recovery. Unexpected states stop
the watcher for investigation. The watcher passed six focused gate tests.

All workers retain the process title `AI Auto Research Pipeline` and the maximum
of two compatible owned jobs per GPU. The GPU1 recovery watcher and queue owners
survive a MacLink disconnect.

## Remaining queue audit

All three exact frozen plans were compared with the complete smoke inventory:

| Worker | Training fits | I4 export tasks |
|---|---:|---:|
| Allocation GPU0 | 2,154 | 90 |
| Server51 GPU0 | 2,153 | 90 |
| Server51 GPU1 | 2,153 | 90 |
| Total | 6,460 | 270 |

There are no duplicate assignments, missing or cross-worker dependencies, cycles,
smoke identity mismatches or reservation mismatches. All 1,080 I4 dependency edges
refer to the four native bodies in the same cohort on the same worker.

A second reporting defect was found: the old collector used canonical output
locations, whereas the actual queue retains outputs inside per-task attempt
folders. The additive queue collector reads those verified terminal records and
preserves original, v1-repaired and v2-repaired provenance separately. It does not
move results or perform model forward passes. See the audit package for tests and
live collection receipts.

At the initial audit, free storage was approximately 457 GiB on the allocation
and 568 GiB on server51. Current campaign queue artifacts occupied approximately
1.7 and 2.5 GiB respectively. The large total filesystem usage is not a measure of
this campaign's retained output. The launch policy keeps a 20 GiB floor plus
candidate and active reservations. It does not reserve the whole campaign's
eventual retained storage.

## Evidence and limits

- `planning/allocation_context_recovery_20261024_v1/FINALIZATION_AUDIT.json`
  verifies the completed zero-update allocation recoveries.
- `planning/server51_recovery_20261024_v1/` records server51 audits, activation,
  the pinned GPU1 watcher and its state.
- `planning/future_queue_audit_20261024_v1/QUEUE_AUDIT.json` records complete plan
  coverage and the reporting defects; source and tests are in the same folder.
- `planning/both_hosts_audit_20261024_v1/queue_health.py` reads current counts,
  owned process identities, progress, tmux state and available storage.

Two-update smoke PASS establishes startup and checkpoint reload, not every
full-training finalization path. The known reporting failures are repaired; later
full fits still need operational monitoring. These engineering checks establish
no new prediction-quality claim and do not change the original paper's scores.
