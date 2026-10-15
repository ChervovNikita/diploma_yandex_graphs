# Resource API for the approved all-fit queue

Runtime dependencies: `resources.py` and `RESERVATIONS.json` only, Python
stdlib. The45KB catalogue holds27 presets,26 finite methods and81 resource
profiles derived from the independently reviewed mapping. No old per-task
qualification gates, reviewer-status switches, native model or tensor data are
loaded. The latest user GO authorizes the smoke/full sequence; this interface
does not ask for another approval.

```python
from resources import reserve_for, can_pair
budget = reserve_for(actual_release_task, actual_job_descriptor)
compatible = can_pair(first_budget, second_budget)
```

`reserve_for(task, job=None)` returns exactly:

```text
vram_bytes, ram_bytes, disk_bytes, cpu_slots, thread_count, profile
```

Byte/slot/thread values are positive integers. Task fields are `kind`,
`cell_id`, `scientific_method_id`, `family`; `worker` is optional and, when
present, must be one of allocation/gpu0, server51/gpu0, server51/gpu1. The
actual descriptor can supply `recipe_id`/`recipe.id`, canonical
`scientific_method_id`, and `family`. A frontend `procedure` alias is only a
fallback when no canonical method ID exists. Supplied native fixed settings
must equal the finite source preset; supplied `torch_threads` must be2. Export
tasks use kind `aggregate_I4_exports`, their cell ID and no new fit descriptor.

`can_pair(left, right)` is symmetric and checks unchanged known budget values.
It allows at most64GiB combined VRAM/48GiB RAM. The69GiB shared Physics-Mono
profile is always serial. This function does not choose physical GPUs, inspect
PIDs, promise speedup or start a job. The scheduler handles same-GPU placement,
at most2 children, aggregate host RAM/CPU/disk, current capacities, cache-aware
cgroups, custody/slot locks and output retention. Smoke initially uses one
child per GPU while retaining the full mature-state reservation.

The scheduler must account for **remaining growth of existing reservations**.
Current free GPU memory includes only what a process uses now. Allocation's
existing GRU has a32GiB reservation; its current roughly1GiB footprint is not
permission to consume its remaining headroom. Until that owner finishes,
69GiB Physics shared jobs cannot fit. Any47/48GiB job also needs its live
headroom checked. Scan ahead for feasible lower-memory jobs, release tentative
tokens while waiting, retain the large jobs and retry after capacity changes.
Never lower their envelopes or signal the GRU to force progress. No source or
scientific failure should be recorded merely because capacity is temporarily
unavailable.

## Deployment audit at2026-10-15 23:19UTC

Both actual host deployment receipts pass614 source files and all2,154 /
4,306 execution descriptors. Server51 core transfer has a terminal exact
destination-hash receipt for19/19 files (16,029,302,220bytes); earlier11/19
progress is stale. This removes missing transfer/source-path objections from
those recorded checks. It does not replace the newly authorized all-fit
two-update smoke, which is the functional coverage test.

Root's current server51 preflight has both old pilot owners exited,16 CPUs,
111,868,276,736 available RAM bytes and683,431,538,688 free disk bytes. GPU0
andGPU1 report80,673MiB and78,415MiB free. Both hosts have tmux. Peer usage
remains capacity pressure, with no PID whitelist or signal authority.

Allocation still has its prior pilot and the existing GRU. Its1.88TB host
MemAvailable is not its allocation budget; use the effective cgroup/cache
bound. Wait for the prior pilot's exact slot lock to release. This is a real
scheduling constraint, not a reason for new user approval. Source path checks
are positive; no new scientific or source blocker is identified by this audit.
Actual smoke failures must retain their task ID, source, seed/split, cost and
error and be fixed before promoting that task to a full fit.

Source receipts are pinned in `AUDIT.json`. The old historical deployment
records' `human_GO:false` fields describe their original time; they do not
override the user's later explicit instruction to run the smoke/full sequence.

Tests: finite namespace/config/thread rejection, exact69GiB heavy envelope,
pair reciprocity/tamper safety, frontend aliases/CPU I4, and equality against
all1,776 previously reviewed class reservations. No native model/update occurs.
