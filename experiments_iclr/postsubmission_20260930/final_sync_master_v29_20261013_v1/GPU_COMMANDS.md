# Final v29 GPU commands

These commands bind the frozen v29 master with canonical SHA256 `0a35daf95e7f1ad8e8df1037923a5a38638ed37e8c0a65c9dbcc3bf73f55488b`. No start command has been executed and no GO receipt has been created.

Set `GNNM_PHASE` to the synchronized research phase on that host, `GNNM_APPROVAL` to the existing exact explicit-GO receipt inside that phase, and `GNNM_WORKER_PYTHON` to its qualified worker interpreter. On the allocation the interpreter is `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python`. Server77 execution still requires the user's authorization for that route.

## One wrapper command per GPU

```bash
bash "$GNNM_PHASE/final_sync_master_v29_20261013_v1/run_gpu.sh" allocation 0 "$GNNM_APPROVAL" "$GNNM_WORKER_PYTHON"
bash "$GNNM_PHASE/final_sync_master_v29_20261013_v1/run_gpu.sh" server77 0 "$GNNM_APPROVAL" "$GNNM_WORKER_PYTHON"
bash "$GNNM_PHASE/final_sync_master_v29_20261013_v1/run_gpu.sh" server77 1 "$GNNM_APPROVAL" "$GNNM_WORKER_PYTHON"
```

Run the two server77 commands sequentially. They admit GPU0 and then GPU1 into the same host supervisor and rendezvous owner. The nonblocking host startup lock rejects simultaneous startup contention. Existing live owners are reused without restarting scientific children. The wrapper refuses an absent receipt and rechecks the exact PLAN, SUMMARY, INPUT_INDEX and controller manifest bytes before delegating to the frozen controller.

## Direct frozen-controller commands

```bash
"$GNNM_WORKER_PYTHON" -B "$GNNM_PHASE/master_controller_source_20261013_v29/runtime/gpu_command.py" start --project-root "$GNNM_PHASE" --plan "$GNNM_PHASE/massive_practical_master_20261013_v29/PLAN.json" --approval "$GNNM_APPROVAL" --host allocation --gpu-index 0 --python "$GNNM_WORKER_PYTHON"
"$GNNM_WORKER_PYTHON" -B "$GNNM_PHASE/master_controller_source_20261013_v29/runtime/gpu_command.py" start --project-root "$GNNM_PHASE" --plan "$GNNM_PHASE/massive_practical_master_20261013_v29/PLAN.json" --approval "$GNNM_APPROVAL" --host server77 --gpu-index 0 --python "$GNNM_WORKER_PYTHON"
"$GNNM_WORKER_PYTHON" -B "$GNNM_PHASE/master_controller_source_20261013_v29/runtime/gpu_command.py" start --project-root "$GNNM_PHASE" --plan "$GNNM_PHASE/massive_practical_master_20261013_v29/PLAN.json" --approval "$GNNM_APPROVAL" --host server77 --gpu-index 1 --python "$GNNM_WORKER_PYTHON"
```

For a GPU-specific preparation check, replace `start` with `preflight` and omit `--approval`. The preflight validates the frozen plan, assigned GPU and exact numerical runtime without creating Registry claims, approvals or scientific workers.

## Admission and phase behavior

Two slots per GPU is a ceiling. The worst registered pair on allocation/GPU0 and server77/GPU0 exceeds its declared VRAM capacity. The supervisor admits one large job or two jobs whose reservations fit current headroom. Assignments, dependencies and global phase gates remain those of the frozen master.

The source-pinned HPO policy permits at most two unfinished releasable cohorts per GPU. Ready metadata selectors take priority. Completed but unreleased and failed cohorts retain their reservation; reopening requires exact validated release tombstones. The 76 protected source/lifecycle fit states have a separate fixed retention roster. These are finite planning bounds. Throughput, efficient balance and ETA have not been measured from the placeholder timing weights.

The wrapper is outside the immutable controller and PLAN. Its receipt is bound separately in the final handoff.
