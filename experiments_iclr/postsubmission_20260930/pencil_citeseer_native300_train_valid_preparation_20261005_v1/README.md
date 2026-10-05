# Frozen native300 PENCIL Citeseer comparator preparation

This is substantive source for root review. No fit is authorized, staged or dispatched by this preparation.

The comparator, feature mode and seeds were fixed in `pencil_citeseer_heart_competitor_source_plan_20261005_v1`, before NCN comparative outcomes were opened. This packet preserves seeds 0, 1 and 2; all 300 native TRAIN epochs; the full-precision, batch-256, accumulation-8, eight-worker recipe; and complete VALID every two epochs. The author-supported early-fusion adaptation supplies the same authenticated raw features as NCN. It remains a declared feature adaptation rather than a reproduction of the official structure-only score or a SOTA claim.

## Execution and selection

The worker imports the pinned author model, sampler, `build_loaders`, `train_loop` and `evaluate_loop` unchanged. All original native optimizer-step sites execute normally. The native final partial accumulation window still divides by eight; no renormalization, truncation, negative redraw, scheduler, clipping or early stopping is added. Each seed constructs its own scratch model and fresh native AdamW.

The first **ordinary** AdamW step records actual moment allocation, shape, dtype, device and finite state for every parameter with an active gradient. Gradients are checked before the native step and model parameters are checked after it. This adds no extra update or separate resource fit. Resource limits remain enforced afterward; successful first-step allocation cannot guarantee all later epochs will fit.

The native condition `(epoch + 1) % 2 == 0` yields 150 complete VALID passes, at epochs 2 through 300. Each pass traverses all unique native query pairs and restores all 113,727 original positive/negative occurrences before native MRR evaluation. Selection preserves the native float32 score-broadcast roundtrip and strict improvement, so the first exact tie wins. This differs from NCN's four-decimal selector and is disclosed. TEST remains absent from every interface.

## Resource and cost basis

The completed resource-only seed-0 probe measured 22,693,889 trainable elements and peaks of 9.395 GiB CUDA allocated, 20.178 GiB CUDA reserved and 41.615 GiB summed probe/loader RSS. Adam moments were not allocated in that probe. The provisional fresh-free-memory rule is 22.516 GiB, and every future fit still requires a fresh exact-host/singleton-UUID gate and the in-fit first-step check.

Under existing GPU contention, epoch-0 native TRAIN cost was 17.85 seconds and complete VALID cost was 93.38 seconds. The actual frozen schedule gives:

| Schedule | Arithmetic cost before Adam, saves and variability |
| --- | ---: |
| 300 TRAIN epochs + 150 VALID passes | 5.378 hours per fit |
| Three serialized fresh fits | 16.135 hours |
| Counterfactual 300 TRAIN + 300 VALID passes | 9.269 hours per fit |

These are arithmetic extrapolations from one discarded epoch-0 workload, not completion-time guarantees or uncontended efficiency measurements. The earlier resource summary's 9.269-hour arithmetic remains preserved and is explicitly the 300-VALID counterfactual. A root-reviewed plan proposes ceilings of 10 hours per fit and 30 hours for the cohort, with the existing 24/28 GiB allocated/reserved CUDA ceilings and 128 GiB process-tree RSS ceiling. No pressure to shorten the native schedule follows from those caps.

## Retention and ownership

The proposed supervisor serializes the three seeds and owns a distinct child session for each. It reuses the exact previously completed resource supervisor's process-identity and cleanup primitives. Its gates and watchdogs never signal other jobs. A failed seed, failed memory gate or timeout stops the remaining queue and preserves evidence without retry or recipe changes.

Only the strict-best full model/AdamW/RNG state and complete restored VALID arrays are retained for each fit. Atomic replacement includes the simultaneous old/new temporary state in a proposed 1 GiB per-fit retention ceiling. Small epoch histories, actual query/update counts, stream hashes, costs and failures remain. Native per-epoch/latest checkpoints are not stored; the storage-policy change is disclosed. Checkpoints stay on the server and are never downloaded to the Mac.

The supervisor reads only coverage/progress and final artifact custody metadata. It leaves scientific quality inputs closed until all three fits complete and seals a cohort freeze. This packet supplies no scientific comparison or claim of predictive benefit. Root must review the exact source, cost and caps, create a unique explicit release and stage the source before any launch.
