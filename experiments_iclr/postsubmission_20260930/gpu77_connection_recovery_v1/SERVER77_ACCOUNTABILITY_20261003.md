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
