# 18.77 server status and project mistakes

Checked at 2026-10-03 08:41:29 UTC (11:41:29 Moscow).

The server is reachable. Uptime is 198 days. Both authorized GPUs reported 96% utilization. Both BUDDY fit workers are alive, with factorized seed 1 at epoch 31 and independent seed 1 at epoch 15. Approximately 105 GB of host memory is available. This observation establishes current reachability and training progress, not a complete host integrity audit.

## What our work changed

- Created and synchronized the authorized project checkout.
- Installed missing Python dependencies into the project's `.gnnm_runtime/buddy_extra_v1` directory. The setup receipt records no modification of the existing base environment.
- Created project scripts, datasets/caches, experiment outputs and dependency qualification files.
- Paused and terminated only our four recorded BUDDY family processes on October 2, then restarted our family with ordinary host execution at 20:02 UTC.
- Tried isolation around our child processes to enforce repo-only runtime writes. This prevented CUDA initialization in those child processes. The records report no host mount or driver changes. Applying that isolation was unnecessary after the user's allowance for incidental runtime caches and cost training time.
- Fixed a separate CUDA initialization order error in our resource script.

No reviewed command or receipt indicates a host driver change, reboot, system configuration edit, or stopping another user's jobs. Do not describe these project script/runtime failures as evidence that 18.77 was broken.

The October 3 08:22 UTC SSH transfer returned connection reset. Its underlying cause has not been established. A later transfer succeeded at 08:34 UTC, and training continued. Do not infer a server outage from the failed connection.

## Evidence

- `commands/runtime_setup_status_v1/RECEIPT.json`
- `commands/repo_boundary_pause_v1/RECEIPT.json`
- `commands/boundary77_terminate_own_v1/RECEIPT.json`
- `commands/buddy77_normal_runtime_qualify_v1/RECEIPT.json`
- `commands/buddy77_normal_family_restart_v1/RECEIPT.json`
- `commands/ncnc77_external_inputs_transfer_v1/RECEIPT.json`
- `commands/ncnc77_external_inputs_transfer_v3/RECEIPT.json`
- `commands/gpu77_read_only_health_after_user_question_v2/RECEIPT.json`

## Fresh follow-up after the user's server question

At 2026-10-03 10:19:49 UTC (13:19:49 Moscow), the existing read-only observation command completed successfully. Uptime was still 198 days. Both authorized GPUs reported 98% utilization. The same BUDDY fit workers, PIDs 3111132 and 3112565, were alive and had advanced to epochs 63 and 37 respectively; the previous 09:32 UTC observation recorded epochs 47 and 26. Approximately 105 GB of host memory was available. No job or configuration mutation was performed by this check, and no outcome metrics were requested.

The new command and full receipt are preserved in `commands/gpu77_accountability_user_followup_20261003_v1/`. This verifies present reachability and continuing training. It is not a complete audit of host integrity. The execution/isolation mistakes described above are ours and must not be described as a demonstrated server failure.

## Current user question: fresh observation

At 2026-10-03 11:15:12 UTC (14:15:12 Moscow), the same read-only command completed successfully through the authorized MacLink forwarding route. Uptime remained 198 days. The two authorized GPUs reported 100% and 99% utilization. The same BUDDY workers, PIDs 3111132 and 3112565, advanced from epochs 63/37 at 10:19 UTC to epochs 79/49. Host memory available was 104,631 MiB. Git HEAD was `a7f23686e38df4a8d0d9297a2edb4e0dbeba373b`.

The exact command and receipt are preserved in `commands/gpu77_accountability_current_user_question_20261003_v1/`. This observation did not restart jobs, change configuration, or request outcome scores. The reviewed isolation capability command used Bubblewrap child user/mount namespaces; its private device remounts must not be described as host device remounts. The unnecessary isolation and our job restarts remain our execution mistakes.

## Recheck at 14:57 Moscow

At 2026-10-03 11:57:03 UTC (14:57:03 Moscow), the same read-only health command succeeded. Uptime remained 198 days. Both authorized GPUs reported 100% utilization, with 32,680 MiB and 32,212 MiB of 81,920 MiB used respectively. Available host memory was 102,455 MiB. The original BUDDY workers remained alive and advanced to epochs 89/57, compared with 79/49 at 11:15 UTC. The exact command and receipt are preserved in `commands/gpu77_accountability_user_recheck_20261003_v1/`.

The recorded changes include project checkout/data/dependencies and our training jobs on both GPUs. The original unnecessary isolation and termination/restart of our own four verified BUDDY processes were agent mistakes. The reviewed commands do not indicate a driver replacement, host mount/configuration change, reboot, or termination of another user's job. The cause of the earlier isolated SSH connection reset is still unestablished. Reachability and progress do not constitute a complete host integrity audit.

## Read-only recheck at 15:51 Moscow

At 2026-10-03 12:51:49 UTC (15:51:49 Moscow), the same read-only health command succeeded. Uptime was 198 days, 20:44. Both authorized GPUs reported 100% utilization and temperatures 54/56 C, with 32,630/36,920 MiB used out of 81,920 MiB each. Host memory available was 100,429 MiB. Original BUDDY workers 3111132/3112565 remained alive; epoch metadata was 95/59, compared with 89/57 at 11:57 UTC. The exact command and receipt are preserved in `commands/gpu77_accountability_fresh_check_20261003_v1/`. This was a read-only observation and did not alter jobs or configuration. These observations establish reachability and ongoing work, not a full integrity audit.

## Read-only recheck at 16:45 Moscow

At 2026-10-03 13:45:13 UTC (16:45:13 Moscow), the same read-only command succeeded. Uptime remained 198 days. GPU utilization was 98%/100%, temperatures 43/54 C, and memory use 19,793/29,281 MiB of 81,920 MiB per device. Available host memory was 112,252 MiB. The original BUDDY supervisors remained alive. Factorized seed 1 reached its 100-epoch trace, the queue advanced to matched-single seed 1 (11 epoch records), and independent seed 1 advanced to 63 epoch records. No scores were requested. Receipt: `commands/gpu77_user_server_accountability_20261003_v2/RECEIPT.json`. This check establishes present reachability and ongoing training; it does not establish a complete host integrity audit. The unnecessary child isolation and earlier restart of our own jobs remain our mistakes.

## Read-only recheck at 17:38 Moscow

At 2026-10-03 14:38:34 UTC (17:38:34 Moscow), the same read-only command succeeded. Uptime remained 198 days. Both BUDDY supervisors and the two current workers were alive. Independent seed 1 advanced to 78 epoch records and matched-single seed 1 to 79, compared with 63/11 at 13:45 UTC. Available host memory was 111,627 MiB. The GPU utilization snapshot was 18%/90%, with temperatures 39/41 C. Receipt: `commands/gpu77_user_server_accountability_20261003_v3/RECEIPT.json`. This check did not change jobs or configuration and did not request scientific scores. It establishes reachability and advancing jobs, not complete host integrity.

## Read-only recheck at 18:32 Moscow

At 2026-10-03 15:32:37 UTC (18:32:37 Moscow), the same read-only command succeeded. Uptime was 198 days, 23:25. GPU utilization was 100%/85%, temperatures 55/45 C, and available host memory was 112,146 MiB. The original BUDDY supervisors remained alive. Independent seed 1 advanced to epoch 96, matched-single seed 1 finished 100 epochs, and the queue advanced to native1024 seed 2 at epoch 32. Git HEAD remained `870bfae124e0606c01039bddccbeba015656fbf0`. Receipt: `commands/gpu77_user_server_accountability_20261003_v4/RECEIPT.json`. This observation changed no jobs or configuration and requested no outcome scores. The unnecessary isolation and restart of our four verified project processes were our execution mistakes; no cause has been established for the earlier isolated SSH reset.
