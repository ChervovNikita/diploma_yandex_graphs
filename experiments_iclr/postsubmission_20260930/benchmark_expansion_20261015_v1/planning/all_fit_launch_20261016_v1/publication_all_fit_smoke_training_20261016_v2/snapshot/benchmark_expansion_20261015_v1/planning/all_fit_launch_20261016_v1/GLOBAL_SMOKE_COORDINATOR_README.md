# Collect the global smoke barrier

`collect_global_smoke_barrier.py` makes no remote call until explicitly invoked.
Root writes and freezes the binding file after port2, smoke2 and execution3 source
freezes and all three execution plans are final. Fill every receipt with actual
bytes/SHA; the placeholders below are intentionally unusable.

```json
{
  "schema": "all-fit-global-smoke-coordinator-bindings-v1",
  "sources": {
    "port": {"bytes": "ACTUAL", "sha256": "ACTUAL_PORT2_MANIFEST_SHA"},
    "smoke": {"bytes": "ACTUAL", "sha256": "ACTUAL_SMOKE2_MANIFEST_SHA"},
    "execution": {"bytes": "ACTUAL", "sha256": "ACTUAL_EXECUTION3_MANIFEST_SHA"}
  },
  "workers": [
    {"worker": "allocation/gpu0", "plan": {"path": "ALLOCATION_ROOT/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/campaign_execution_20261016_v3/PLAN-allocation_gpu0.json", "bytes": "ACTUAL", "sha256": "ACTUAL"}},
    {"worker": "server51/gpu0", "plan": {"path": "SERVER51_ROOT/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/campaign_execution_20261016_v3/PLAN-server51_gpu0.json", "bytes": "ACTUAL", "sha256": "ACTUAL"}},
    {"worker": "server51/gpu1", "plan": {"path": "SERVER51_ROOT/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/campaign_execution_20261016_v3/PLAN-server51_gpu1.json", "bytes": "ACTUAL", "sha256": "ACTUAL"}}
  ]
}
```

Run with the saved absolute binding-file path and its actual SHA256:

```sh
python3 -B collect_global_smoke_barrier.py \
  --bindings "$ABSOLUTE_BINDING_FILE" --bindings-sha256 "$ACTUAL_BINDING_SHA"
```

Before all three immutable summaries are PASS, output is `pending`; no inventory
or summary payload is transferred and no global marker is created. A source,
plan, owner or inventory mismatch is an error. The guarded existing transports
check each scientific host and complete GPU identity before project operations.

After all three PASS, the helper retains the exact two inventories and three
summaries under this folder's `global_smoke_barrier_proof_v1/BINDING_SHA`, runs
the pinned smoke2 validator, and delivers its identical marker to execution3 on
both hosts. Evidence downloads use compressed 24KiB chunks; marker uploads use
1KiB chunks. A hash-bound compressed copy of the metadata engine is staged once
per host using at most3000-byte payload chunks; each subsequent request is
compressed and the raw Python command is capped at5000 bytes before SSH quoting.
This avoids sending the full engine or a large upload through nested shell
quoting. Small compressed transport snapshots and this exact engine remain
inside the remote execution3 project folder. Frozen source files are unchanged. Prior local proof,
attempts and identical remote markers are preserved. Retry the same command after
a temporary transport failure; conflicting evidence or marker bytes are refused.

This helper does not start or restart jobs. Existing tmux owners consume the
validated marker through their existing global barrier.
