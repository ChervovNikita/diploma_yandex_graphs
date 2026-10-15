# Expanded launch preparation v1

A finite, guarded **source/metadata dry-run** for the 6,460-fit descriptor inventory. No campaign, model, numerical test, TEST scoring or server77 access is launched. The current `run` action always returns BLOCKED. Production adaptation and exact remaining measurement fields are documented in `ADAPTATION_PLAN.md`.

## One worker command per physical GPU

The authorized allocation command is:

```sh
python3 -B benchmark_expansion_20261015_v1/expanded_launch_preparation_v1/worker.py dry-run --manifest benchmark_expansion_20261015_v1/expanded_launch_preparation_v1/generated_v1/PLAN.json --worker allocation/gpu0
```

Run from the allocation phase directory `experiments_iclr/postsubmission_20260930`. A supervisor restart uses the same manifest/worker plus an optional immutable `--state` JSON journal. Completed physical fits are skipped after exact terminal identity checks. Interrupted, live/orphan and failed claims require reconciliation; a reviewed complete-epoch resume disposition preserves the same fit. No repeated fit or prediction job is created.

| Intended worker | Current assignments | Execution |
|---|---:|---|
| allocation/gpu0 | 6,460 fits + 270 I4 export tasks | Blocked pending qualification and GO |
| server77/gpu0 | 0 | Disabled, unprobed, unauthorized |
| server77/gpu1 | 0 | Disabled, unprobed, unauthorized |

All workers have a maximum of two owned children per physical GPU. Actual overlap requires reciprocal mature measured profiles and headroom, including startup/evaluation/export. No pair or capacity is presumed. Future verified profiles and a new 540-group placement file can distribute disjoint IDs across all three GPUs without changing scientific contracts or seeds.

## Fit and export ownership

The 270 separate source-reference fits retain their own native controllers. The 1,080 matched native body fits and 5,110 shared-method fits retain the matched controller. Named protocol queues make this distinction explicit; queue order never changes a horizon/selection policy.

Main M1 aliases exactly I4 body 0. Each of 270 native I4 export-only tasks waits for its four independent own selected VALID exports, with seed offsets `[0,100003,200003,300007]`. It averages existing logits once: zero extra fits and zero member prediction forwards. Shared actual-serving and applicable same-call fixed-mean exports are bookkeeping inside each fit's own selected finalization. They create zero scheduler scoring replicas. No new statistical decisions or repeated-fit scoring are scheduled.

The existing descriptor inventory has 6,370 source-bound jobs and 90 explicit BUDDY seed-context blocks. Source/provider/lifecycle/resource/reporting gates and human GO remain separate. Every task is blocked for execution in this preparation. No historical worker assignment is reused as resource evidence.

## Cost evidence

The authenticated 24-fit MolBACE ledger supplies descriptive measured case priors only for native bodies, original GNNM and exact molecular geometry on the same matched GIN/drop0.5 preset. These priors cover 60 target intentions, whose actual costs are unmeasured. Older residual/combined measurements are kept separately as source-mismatched observations. Unknown/source-reference cases have `null` weights. No count-based balancing or campaign ETA is supplied.

Full PLAN/placement/dependency manifests remain on allocation. Only source, documentation, compact summary, dry-run and cost catalogue receipts are collected locally. The narrow transport uses the reviewed expansion `remote_control.py`; no broad upload or Git operations are used.
