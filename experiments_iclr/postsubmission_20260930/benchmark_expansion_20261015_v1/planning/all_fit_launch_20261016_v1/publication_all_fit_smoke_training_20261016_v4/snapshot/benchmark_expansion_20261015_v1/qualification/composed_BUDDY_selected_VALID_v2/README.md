# Composed BUDDY: repair selected serving without another fit

Source preparation only. Root owns review, publication, fresh resource admission
and execution. This package does not launch, queue, train, resume training, select
a checkpoint, or access TEST. All earlier sources and outcomes remain unchanged.

## Retained work and narrow defect

`buddy_seed1_RUNTIME_CHECK_31.json` records 1585 completed/attempted TRAIN updates,
3 constructions, 2 same-call mean exports, and successful one-epoch fitting plus
stopped-finalization resume with zero additional updates. The subsequent fresh
serving attempt failed in `HashDataset` because its copied argument namespace
lacked `subgraph_feature_batch_size`. The authenticated native recipe declares
11000000; the lean training constructor does not put it in `native.args`.

That attempt cost 562.334 seconds of supervisor time, sampled RSS 7014760448
bytes and GPU process memory 4922015744 bytes. Those costs and the failure remain
in the record. No training or lifecycle step is repeated to repair serving.

The same helper defect is present in the frozen runtime's nested scorer,
`composable_normalization_final_scoring_v2`, and `final_selected_scoring_v5`.
The successors `composable_normalization_final_scoring_v3` and
`final_selected_scoring_v6` add the missing argument to the deep-copied role
namespace from the declared recipe. They require equality with the frozen author
configuration and reject a conflicting existing value. The only other executable
change is each successor's source schema. This affects BUDDY role-dataset
construction; node, molecular, NCN and NCNC take other branches. The historical
`final_test_serving_v1/link/backend.py::_buddy_dataset` remains unchanged and is
not called here; its `_buddy_predict` helper is reused unchanged.

## Exact replay

`INPUTS.json` pins the failed qualification, its final completed fit result,
epoch-1 selected checkpoint, final-resume VALID export, original genuine producer
manifest/entrypoint, fixed role contract and serving successor. The original
`JOB.json` must have its recorded canonical SHA256. The original producer and
full selected adapter state are reconstructed; this is no checkpoint migration.

Exactly 1 construction, 1 selected restore and 1 whole VALID pass are allowed.
The 160084 queries (60084 positive, 100000 negative) run in the original combined
inference batch, once per each of 4 routes: 4 native BUDDY forwards. No second
native forward is used for the same-bank mean. Actual readout and mean both use
the unchanged `rtol=1e-4`, `atol=1e-5`; ordered IDs, labels and endpoints are exact.
The replay checks full selected-state and RNG restoration, parameter/buffer
object identity, private BN banks, optimizer state and the TRAIN update clock.
The retained inference-only guard blocks TRAIN callbacks, backward, gradient
collection, optimizer steps and additional validation callbacks.

Physical reservations remain conservative: 40 GiB GPU, 16 GiB host RSS, 8 GiB
disk, 2 CPU threads, 1 interop thread, 600 seconds internal wall. Construction
in the retained fit/resume took 16.106/12.856 seconds; successful repaired fresh
role-dataset construction has not yet been measured. Use root's existing
supervisor to enforce RSS/GPU/disk, outer 610 seconds and 10-second TERM/KILL
cleanup, with no virtual-address cap.

## Root execution handoff

1. Authenticate the literal allocation host/GPU before project reads and verify
   the frozen source manifests and inputs. Reuse the exact original `.venv`
   interpreter review only after its identity matches.
2. Obtain the existing `analysis_outputs/selected_VALID_collection_gpu_slot.lock`
   nonblockingly and hold it through the child. Refresh actual peer identities
   and resource headroom; this replay cannot add a third model worker beside GRU
   and PubMed. No waiting scheduler is included.
3. Create a separate admitted copy of `ROOT_ADMISSION_TEMPLATE.json`, filling
   source/INPUTS hashes, fresh output, interpreter and review receipts. Set
   `status=ADMITTED` only under root's review. Do not modify the template or the
   failed attempt's output/cache. Use a fresh role cache in the new output.
4. Run once under that supervisor, inspect the full terminal result and retain
   every outcome. No automatic retry, tolerance change or training fallback.

Allocation-local child command after admission (variables supplied by root):

```sh
env PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES=GPU-44039938-fd82-41d2-fefd-de71514e2fac \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout --signal=TERM --kill-after=10s 610s \
  "$REPLAY_PYTHON" -B -u \
  "$EXPANSION/qualification/composed_BUDDY_selected_VALID_v1/replay.py" \
  --root "$ROOT" --admission "$REPLAY_ADMISSION"
```

Preserve `START.json`, `EVENTS.jsonl`, `RESULT.json`, `VALID.pt` when emitted,
fresh `serving_cache`, and supervisor logs/costs. Exit 0 and `status=passed`
require all state/work/population/parity checks. A missing terminal result leaves
an incomplete attempt with potentially unrecorded work. The original qualification
continues to say failed; a successful replay supplies the missing serving witness
alongside the paid original fit/resume evidence. It does not qualify the reporting
observer, authorize TEST or establish full-campaign readiness/GO.

## Source-only checks

Run `PYTHONDONTWRITEBYTECODE=1 python3 -B check_source.py` in this directory.
The check extracts only scalar argument setup from the actual frozen constructor
and author configuration, reproduces the missing attribute, and verifies all
17 argument reads in the unchanged `HashDataset` class are supplied. It checks
unchanged producer arguments, invalid/missing/conflicting settings, both identical
successor repairs, unchanged unrelated serving code and original sources, exact
retained input bindings, and a single literal VALID call with no fit/TRAIN call.
No model, tensor, array, dataset or checkpoint is executed or loaded.

## Receipt conversion successor v2 (16 October 2026)

The attempt01 all-fit prerequisite stopped before model construction because
`relative()` retained `path_relative_to_expansion` while adding `path`. The
genuine serving contract requires exactly `path`, `bytes`, and `sha256`. This
successor constructs that exact receipt, rejects malformed or escaping relative
paths, and leaves every other replay function and all paid inputs unchanged.
The predecessor and failed attempt remain evidence. No optimizer update is
authorized by this repair. `receipt_regression.py` exercises the actual helper
and strict serving contract without model/data execution.
