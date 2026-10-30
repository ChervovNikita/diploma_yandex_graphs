# Allocation queue sharing, 30 October 2026

The user authorized the original one-GPU allocation plus physical GPUs 1 and 6
on `anogena-0`. Server51 is unchanged. All deliberate files remain in the same
repository; datasets and environments are shared, not duplicated.

The frozen 2,244-task allocation assignment is partitioned once. Any attempted
group stays on the original device. Only wholly pending groups may move; four
independent fits and their ensemble export always stay together. Task IDs,
training settings, data, descriptors and canonical attempt folders are unchanged.
The actual execution device is explicit in host configs, invocation provenance,
physical_shard and actual_host_config fields. Canonical worker remains
allocation/gpu0 for the frozen evidence validator.

The old owner's STOP prevents further admission. Its current fits are preserved.
Partition publication requires either full drain or proof that the owner is
sleeping between iterations while its admission ledger is locked. Each start
checks that this STOP is unchanged. The original-device successor waits for all
old inherited leases. The new GPUs may proceed immediately on their disjoint
pending groups. No process is signalled. Previously excluded evidence remains
excluded.

Each GPU admits up to four tasks, subject to the existing VRAM, RAM, CPU and disk
rules. Host-local process accounting is separate; disk reservations count both
hosts, including old fits during drain. A real cross-host flock probe verifies
that shared-file custody locks exclude both hosts.

## Monitoring

Run from the repository on either allocation (choose an authorized physical
worker on the host used):

```sh
.venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/allocation_gpu1_gpu6_20261030_v1/status.py --worker allocation/gpu0
```

On the seven-GPU host use `--worker anogena/gpu1` or `--worker anogena/gpu6`.
The command reads all three canonical task shards without treating foreign PIDs
as local. The original queue STATUS becomes historical after its drain; never
remove its STOP or restart the old unpartitioned assignment.

## Checks

Seven regression tests cover exact device authorization, full inventory order,
owner disappearance, deterministic disjoint dependency groups, and the narrowly
changed reporting entry. Each actual host/device passes native source imports,
descriptor host validation, inherited invocation assembly and two bounded CUDA
engineering updates. Those updates are not scientific fits and access no TEST.
The exact original 6,460-fit smoke barrier remains required before training.
