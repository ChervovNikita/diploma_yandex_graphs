# Allocation repository cleanup — 27 October 2026

## Scope and result

Cleanup was confined to `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs` on the authorized one-GPU allocation. Audit copies are in the corresponding local project workspace. No server51/server77 files or host settings were changed. Existing training jobs were preserved.

The repository measured **552.2 GB before** and **526.8 GB after** (490.6 GiB). The after measurement finished at 2026-10-27T13:51:59.546477+00:00. Training continues to create output, so this is a point-in-time measurement.

| Removed item | Allocated storage before removal | Reason |
| --- | ---: | --- |
| Repository uv package cache | 6.6 GB | Download/build cache; installed environment and its dependencies were retained and verified. |
| `V29_EXACT_SOURCE_TRANSFER.tar` | 14.7 GB | Superseded transfer archive; every historical member was preserved through existing identical files or the verified difference archive. |
| October 10 `untied_local_mul4/seed1/SELECTED_STATE.pt.tmp` | 4.1 GB | Incomplete atomic save from a documented interrupted engineering run. |
| **Total** | **About 25.4 GB** | Only these reviewed categories were removed. |

The net measured reduction is 25.42 GB. Small differences from candidate totals reflect retained audit files and ongoing training writes. Shared filesystem free-space changes cannot be attributed solely to this cleanup.

## Preserved research

Metrics, experiment ideas, decisions, unsuccessful-run records, cost logs, scientific datasets, finalized checkpoints, selected prediction exports, current campaign source/manifests, queue state, smoke-test evidence, the installed environment, managed Python, and Git history remain intact. Some directories named `cache` contain required datasets and were deliberately retained. The inventory found about 436.9 GB of `.pt` files; these were not treated as disposable merely because they are large.

## Transfer-history recovery

`TRANSFER_PRESERVATION.json` records all 6,922 original archive members. For each member it identifies either an identical retained repository file with its checksum or content/metadata preserved in `transfer_historical_differences.tar.gz`. The latter is 3,713 bytes with 97 entries. Its checksum was verified after preservation and after the local audit copy was fetched.

To retrieve historical content, consult the per-member index and read the recorded retained path or the matching difference-archive member. Restore into a separate directory inside the repository if needed; do not overwrite live files. The historical member contents are preserved, but exact reconstruction of the original TAR byte stream is not claimed.

The original transfer receipt and preservation report remain unchanged. The preservation report describes its pre-deletion state; `REDUNDANT_FILE_DELETION.jsonl` and `REDUNDANT_FILE_DELETION_RECEIPT.json` record the later removal and both deleted-file hashes.

## Verification

- Repository Python successfully imported PyTorch, PyTorch Geometric, torch-sparse, torch-scatter, OGB, NumPy, SciPy, and scikit-learn after cache cleanup.
- Both explicit redundant files were confirmed absent.
- At 2026-10-27T16:01:33.387365+00:00, the allocation queue reported 38 completed tasks, 2 running tasks, and no failed tasks. The tmux owner and both active process identities were valid. One previous fit completed and its replacement started after cleanup.
- No heldout test outputs were inspected and no scientific settings or training processes were changed.
- The original verification SSH client stalled after the remote check had completed. The durable remote verification receipt was retrieved separately; only that local SSH client was then closed. This transport issue did not terminate remote training.

## Evidence

`INVENTORY.json`, `CACHE_REMOVAL_RECEIPT.json`, `TEMP_REMOVAL_AUTHORITY.json`, `TRANSFER_PRESERVATION.json`, `REDUNDANT_FILE_DELETION_RECEIPT.json`, `POST_CLEANUP_VERIFICATION.json`, and `FINAL_QUEUE_HEALTH.json` document the decisions and outcome. Readable cleanup scripts, exact operational command receipts, and the preserved historical differences are in this audit directory.
