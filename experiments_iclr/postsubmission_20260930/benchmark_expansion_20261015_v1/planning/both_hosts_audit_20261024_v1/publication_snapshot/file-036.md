# Server51 queue audit and reporting recovery — 24 October 2026 Moscow

## Confirmed failure

The three failed wrapper fits on server51 stopped after training, while the
reporting observer was verifying dependency files for the selected VALID export.
The frozen verifier incorrectly excluded a pinned BUDDY README because it was
not a Python or JSON file. Each failed fit had completed 149 epochs and 236,165
optimizer transitions. No TEST scoring was performed.

- GPU0, seed 1, conditional aggregation: selected epoch 4.
- GPU0, seed 1, global aggregation: selected epoch 6.
- GPU1, seed 2, conditional aggregation: selected epoch 9.

GPU1's global aggregation fit was still training, so it was not interrupted or
modified. Its existing owner identity and pane are pinned in `WAIT_CONFIG.json`.

## Applied recovery

The additive package `campaign_reporting_repair_20261024_v2` verifies all
canonical project dependency files by their observed SHA256, independently of
extension. It preserves frozen runtime sources and scientific settings. Its
explicit finalization recovery can add zero optimizer transitions and cannot
change the selected epoch. Original failed attempts and exports are retained.

The server51 GPU0 CPU checkpoint audit passed. Both failed fits' actual runtime
bindings passed the repaired reporting checks (41 dependencies each). The
existing dead tmux pane `%111` was restarted with the repaired queue owner.

GPU0 still had other users' large workload. Its resource-aware scheduler selected
a smaller compatible pending job while the larger BUDDY finalizations waited
for capacity. No other user's process was signalled or altered.

A separate tmux session, `gnnm-reporting-drain-server51-gpu1-20261024`, runs the
verified one-use waiter. It waits for the original GPU1 job to finish naturally,
checks that the exact two expected reporting failures and original owner remain,
and requires the existing queue pane to be dead and its supervisor lock free.
It then performs the same CPU checkpoint audit and activation once. Unexpected
states fail closed. It neither kills training nor retries unknown failures.

## Verification

- 15 reporting/recovery CPU tests passed on server51.
- 6 natural-drain gate tests passed on server51.
- All 14 transferred text files were read back and checked byte for byte.
- Server51's `.venv/bin/python` matches the frozen worker's interpreter path;
  its resolved binary is inside the project and provides Torch 2.1.2+cu118.
- `ACTIVE_PROGRESS_AFTER_RECOVERY.json` confirms real training after activation:
  GPU0 PolyFormer CiteSeer Cheb native fit reached epoch 221; GPU1's existing
  BUDDY aggregation fit reached epoch 108 / 171,180 optimizer transitions.
- Both training queue panes and the waiter were alive; WAIT_STATUS reported
  `waiting_for_owned_training_to_finish`.

At this snapshot the two GPU0 reporting recoveries had not yet completed because
of capacity constraints. GPU1's reporting recovery had not yet begun. Those are
pending actions, not completed fits. Frozen assignments and all completed results
remain unchanged.

## Evidence

- `FAILED_AND_ACTIVE.jsonl`: initial failures and active-job audit.
- `REMOTE_CPU_TEST_LOG.txt`: all 21 focused remote test outcomes.
- `RECOVERY_CONFIG-server51_gpu0.json`: exact checkpoint recovery authority.
- `GPU0_ACTIVATION.txt`: recovery owner dispatch receipt.
- `WAIT_CONFIG.json`: pinned GPU1 natural-drain authority.
- `GPU1_WAITER_DISPATCH.txt`: waiter tmux dispatch receipt.
- `STAGING_RECEIPTS.json`: plaintext staging and readback hashes.
- `commands/`: exact plaintext commands and transport outcomes, including the
  preserved initial waiter-launch syntax error that occurred before dispatch.

Source pins:

- Reporting repair manifest:
  `235cda9eed0838cb4596d9a2f30837ffae3ee18f98c6f956bb00c58f3ccda7fe`
- Natural-drain waiter manifest:
  `ab3ae52a170af8aaf4e38358745aca7c529b830fa8a141eadcd77e9ac2820fdd`
- GPU0 recovery authority:
  `1e50c52b1c2083fc9338278c80c02a662c9fa2fb75c2ca3914192e57dcf5e83d`
- GPU1 waiter authority:
  `7716b653430ff982a7a79b036f4046eee90674f5adbc00a4823af540cff506f9`
