# Allocation queue expansion — 31 October 2026

The user authorized three additional devices: physical GPUs 2, 3 and 4 on
`anogena-0`. The allocation campaign can now use six GPUs: the sole GPU on
`anogena-2-0`, plus GPUs 1, 2, 3, 4 and 6 on `anogena-0`. Server51's assignments
and settings remain unchanged. GPUs 0 and 5 on `anogena-0` are excluded.

## What moves

This package redistributes the same 2,244 allocation tasks. It preserves every
task ID, scientific descriptor, smoke receipt, completed score and canonical
attempt folder. No new scientific configuration or repeated completed fit is
introduced. Datasets and environments stay in the existing shared repository.

Only groups with no prior attempt can move. A group containing any completed
or running task stays on its previous physical worker. The four independent
members and their aggregation task always remain together. Pending work is
balanced using observed per-setting durations where available and a disclosed
fallback elsewhere; these estimates are scheduling weights, not finish dates.

## Handover

Each prior queue receives a graceful STOP request and acknowledges that it
cannot admit more tasks. Running numerical processes continue. These checks
run on the correct host; a foreign process ID is never interpreted locally.
The new partition is published once under shared locks and must be an exact,
disjoint union of the original task assignment.

New queues on previously used devices wait for prior supervisors and inherited
physical leases. New devices can start their disjoint pending work immediately
after the global handover checks pass. Preserve all earlier STOP markers and
partitions. Do not restart superseded queue owners or remove their STOP files.

Each GPU admits up to four compatible jobs. Admission still checks available
GPU memory, host RAM, CPU capacity, shared disk space and the expected remaining
resource growth of active jobs from every queue generation. Existing external
jobs are preserved. GPU4 was partly occupied at preparation, so four simultaneous
jobs are a ceiling rather than a promise.

## Verification and monitoring

Operational checks cover authorized devices, retained prior ownership, closed
ensemble dependencies, partition uniqueness and handover boundaries. Live
preflights check the UUID mask, CUDA operation, native imports and invocation
assembly. Existing all-fit smoke qualification remains required. These
engineering checks do not add scientific evidence or access TEST data.

Once `PARTITION.json` and the launch receipts exist, run this from the repository
on the original allocation:

```sh
.venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/allocation_six_gpu_20261031_v1/status.py --worker allocation/gpu0
```

On `anogena-0`, use `--worker anogena/gpu2` (or another authorized device).
The command reads all six queues. Individual queue `STATUS.json` files show
whether a successor is running or waiting for prior ownership. Canonical task
states include the preserved running fits. Use live receipts and status records
to distinguish a dispatched queue from verified training progress.
