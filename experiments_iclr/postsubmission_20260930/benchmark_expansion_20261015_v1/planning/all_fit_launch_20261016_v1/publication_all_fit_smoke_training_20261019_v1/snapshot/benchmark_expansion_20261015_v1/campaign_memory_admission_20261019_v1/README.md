# Clean file-cache memory admission for execution7

This small scheduler wrapper imports the frozen `campaign_execution_20261016_v7`
Worker, pinned to manifest SHA256
`52a996e6effb77748339c57a716159a80a06272d88914d37beaa2dd680e7e890`.
It uses each existing execution7 plan, inventory, state root, receipt and lease.
The frozen parent package and campaign are unchanged. It does not clone plans,
create a new campaign, retry unchanged failures or re-run verified PASS fits.

The parent probe measured only inactive file cache as reclaimable. The allocation
diagnosis saved on 19 October shows 235,813,666,816 bytes of active file cache
and only 14,950,277,120 bytes of inactive file cache. The old estimate therefore
waited on the RAM guard while the GPU was idle.

For each accessible cgroup and ancestor, the new probe computes:

```text
reclaimable = max(0,
    min(memory.current, file - shmem, active_file + inactive_file)
    - file_dirty - file_writeback - unevictable)
headroom = max(0, min(memory.max, memory.max - memory.current + reclaimable))
available_ram = min(host MemAvailable, every finite cgroup headroom)
```

`file` and the file LRU counters describe overlapping pages, so they are bounded
by a minimum. Dirty/writeback overlap and unevictable pages already absent from
the file LRUs can be subtracted twice; this deliberately reduces the estimate.
Anonymous memory, swap and slab never add credit. Reclaim credit cannot exceed
the current charge. The subtraction preserves an existing over-limit deficit
before clamping at zero. Unlimited ancestors do not add capacity or bypass other
limits. Missing exclusion counters fail closed. The diagnostic omits the
`unevictable` counter; the recorded scalar test assumes zero, while the live probe
requires and uses the real value. Actual host MemAvailable always bounds admission.

The wrapper first calls the original probe and then replaces only RAM accounting
with fresh kernel counters. GPU identity/free memory, CPU quota/usage, process
usage and disk measurements remain original. The original capacity decision,
remaining-growth reservations, 8 GiB RAM/disk guards, 1 GiB VRAM guard, two-child
limit and compatibility checks remain intact. No cache drop, host setting,
mount change, process signal or system service is involved.

`AdmissionWorker` inherits `run`, `run_phase`, `start`, `load_states`, receipt
validation, smoke summary, guardians, leases and full training from the parent.
Its only scheduling change replaces the parent worker's probe callable in this
process. Constructor/report/invocation hooks record the actual wrapper manifest
alongside the unchanged parent manifest in OWNER, STATUS, each new INVOCATION and
a retained MEMORY_ADMISSION_OWNER record. While the original supervisor lease
is held, a narrow constructor metadata hook fsyncs the exact previous OWNER
bytes into an archive named by their SHA256 before the parent replaces OWNER.
Old attempts retain their original provenance.

All assigned smokes must pass before the unchanged parent creates its summary.
The unchanged exact-ID global 6,460-fit barrier is then required before the
ordinary full phase and aggregation can start. This wrapper does not create or
relax that global marker. It exposes no failed-fit retry switch.

Root owns route verification, package transfer, graceful STOP/drain, unchanged
plan verification and tmux restart. The original launch.py starts the original
worker, so root must invoke this wrapper in the replacement tmux session. After
the old supervisor has exited, preserve/acknowledge any existing STOP using the
established restart procedure. The wrapper itself refuses a live supervisor and
preserves physical leases held by existing children.

Allocation command inside the root-controlled tmux session, with the new
manifest hash supplied by root after verifying all manifest-listed files:

```sh
TASK_PROJECT=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
TASK_E="$TASK_PROJECT/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1"
"$TASK_PROJECT/.venv/bin/python" -B -u \
  "$TASK_E/campaign_memory_admission_20261019_v1/admission_worker.py" \
  --worker allocation/gpu0 \
  --plan "$TASK_E/campaign_execution_20261016_v7/PLAN-allocation_gpu0.json" \
  --plan-sha256 19fa1a8b96bcf3383965c266c4fa900e0ac09c62ca040553e7669acb659ffaad \
  --source-sha256 "$TASK_ADMISSION_MANIFEST_SHA"
```

The same wrapper can use the existing server51 plans if root elects to deploy
there; use its actual project root and corresponding unchanged plan hash:

| Worker | Existing plan SHA256 |
| --- | --- |
| server51/gpu0 | `b23301b8674faf5dfad2000efeeb067d0c1a2fb97524277c2185d174a307a345` |
| server51/gpu1 | `bc025b3209c0a7f5c2f10fd011fd54ca072c78269693b035ca3e53bbf243ad3c` |

Plan hashes above come from the existing GLOBAL_SMOKE_BINDINGS_PORT6.json.
Staging or importing this package starts no jobs. Preparation ran only the
12 focused local tests in `test_admission.py`; there was no remote access or
native training. Tests cover memory accounting, existing guards, exact PASS
receipt reuse, failed-fit preservation, global-barrier/full-phase sequencing,
the exact global ID gate and owner/provenance handling.

```sh
python3 -B test_admission.py
```
