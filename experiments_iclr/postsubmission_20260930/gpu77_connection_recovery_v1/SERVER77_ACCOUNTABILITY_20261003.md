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
