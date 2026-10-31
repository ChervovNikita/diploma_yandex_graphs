# One-A100 staged maximum-duration forecast — v2

1 October 2026. Existing timing/hash/JSON/CSV inspection and pure arithmetic only. No model, graph data, source script or SSH execution. V1 and root observations are preserved.

The frozen Stage1 amendment retains all nine arms, both graphs and seeds17/29/43 on prospective core0:54 normal1000–2000-update fits. All four single calibrations are now complete and root audited. Photo SAGE LR is3e-5, CS SAGE LR3e-4, and GT-sep LR remains3e-5. Calibration is charged as prior elapsed work and is not forecast again.

## Conditional duration proposal

- Stage1,54 remaining fits: **9.67 hours**, including20% contingency and one0.5-hour reserve.
- Core1/core2 continuation,108 fits: **18.33 additional hours**, conditional on a Stage1 pass and separate admission. This includes20% contingency and does not charge the same reserve again.
- Both stages if admitted and reached: **28.00 hours** of future planning time.
- Mean update/evaluation rates with the same proposed IO assumptions give Stage1 **8.18 hours**. The primary estimate uses observed maxima.
- Retained prior qualification/calibration launchers total **465.448 seconds**. This includes the failed first Photo guard but excludes other source/acquisition/storage/earlier phase work.

These estimates are conditional on adopted timing/IO allowances. A maximum over100 qualification updates is not a guaranteed future bound. No user deadline or cap is inferred. The former six-hour proposal is unadopted. Actual elapsed work and unreclaimed conservative reservations stay in separate ledgers.

## Measurements and required assumptions

All18 table rows agree with retained report/curve hashes and the inspected CSV update, warm/fresh inference, memory, storage and invocation fields. Normal single calibration report/curve hashes also agree on both graphs. The forecast charges each arm's setup, source/import-inclusive residual, synchronized complete updates, aggregate warm/operation/fresh deployment profiling, final hashing/process-exit residual and loop bookkeeping.

Pure framework import duration is unavailable. Invocation-minus-scientific residual includes imports, source verification, snapshots and report work. Setup subcomponents and latency median/p95 are nested diagnostics and are not added again. Complete ensemble observations already include every member.

Direct single calibration validation and aggregate checkpoint timings are retained observations. They do not measure200-write maxima or multi-member fit-only IO. Root has stated that it will adopt the following conservative planning assumptions or use representative preflight evidence:

- At least50MiB/s effective serialized artifact service, plus0.25seconds per checkpoint/logit pair and selected reload.
- Use the observed model/optimizer checkpoint length plus validation FP32 member logits, int64 node IDs and1MiB metadata allowance for every selected/final pair.
- Add0.10seconds at every evaluation for fit validation host copying and scope differences. Four-member CS validation logits are879360bytes; Photo logits are195584bytes.
- Add2seconds bridge/supervisor work per fit and the full15-second idle-guard allowance.
- Transfer matching graph/architecture setup and aggregate profiling observations to fit. Scale the entire unattributed scientific residual to2000 updates; this conservatively repeats qualification-only audits with bookkeeping.

The numeric IO/copy/metadata/bridge bounds remain labeled proposals in the JSON until root freezes adoption. They are not measured physical storage guarantees. Multi-member fit-only copy/checkpoint memory and timing transfer from CS qualification LR3e-5 to frozen LR3e-4 also need adoption or representative evidence. Maximum observed qualification training allocation is 9337.46MiB, without a claim that it guarantees fit peak memory.

## Cost build

`T = fixed measured projection +2000×update rate +2000×bookkeeping residual rate +200×(evaluation reference+copy allowance) +201×checkpoint pair allowance +selected reload allowance +15-second guard +2-second bridge allowance`.

This charges200 evaluations, up to200 selected checkpoint/logit pairs and one final pair. Initial checkpoint/cold evaluation and warm/operation/fresh profiling are already in fixed work. No early-stopping saving is credited.

| Dataset | Arm | Mean update ms | Observed max update ms | Proposed pair write s | Max-rate fit projection min | Three fits with20% min | Required operation timeout s |
|---|---|---:|---:|---:|---:|---:|---:|
| AmazonPhoto | original | 57.61 | 95.70 | 0.546 | 5.94 | 21.40 | 408 |
| AmazonPhoto | bias_only | 57.38 | 99.07 | 0.548 | 6.04 | 21.73 | 415 |
| AmazonPhoto | factor | 66.25 | 104.36 | 0.555 | 6.21 | 22.37 | 427 |
| AmazonPhoto | coordinate | 74.11 | 117.75 | 0.559 | 6.68 | 24.04 | 461 |
| AmazonPhoto | permutation | 64.73 | 102.32 | 0.552 | 6.14 | 22.11 | 422 |
| AmazonPhoto | single | 19.58 | 62.00 | 0.564 | 4.78 | 17.19 | 324 |
| AmazonPhoto | untied | 43.93 | 87.05 | 0.567 | 5.66 | 20.37 | 388 |
| AmazonPhoto | heads | 23.18 | 65.91 | 0.567 | 4.91 | 17.66 | 333 |
| AmazonPhoto | gt_sep_single | 132.44 | 181.60 | 2.768 | 16.42 | 59.10 | 1162 |
| CoauthorCS | original | 124.33 | 158.43 | 0.921 | 9.37 | 33.75 | 655 |
| CoauthorCS | bias_only | 125.34 | 160.79 | 0.922 | 9.45 | 34.02 | 660 |
| CoauthorCS | factor | 143.12 | 191.76 | 0.929 | 10.52 | 37.87 | 737 |
| CoauthorCS | coordinate | 158.61 | 193.57 | 0.934 | 10.84 | 39.03 | 761 |
| CoauthorCS | permutation | 141.82 | 175.04 | 0.927 | 9.96 | 35.86 | 697 |
| CoauthorCS | single | 31.64 | 72.86 | 0.928 | 6.43 | 23.13 | 443 |
| CoauthorCS | untied | 53.88 | 91.16 | 0.947 | 7.11 | 25.59 | 492 |
| CoauthorCS | heads | 35.64 | 72.52 | 0.943 | 6.44 | 23.20 | 444 |
| CoauthorCS | gt_sep_single | 170.18 | 211.43 | 3.483 | 19.86 | 71.51 | 1410 |

## One-device schedule and bounded execution

Run sequentially on the single authorized A100. Freeze every Stage1 cell before the first comparative result. The JSON's transparent per-arm grouping has18 three-seed batches, alternating Photo/CS in fixed arm order. This avoids charging every cheap arm at the largest mixed-arm timeout. Root may instead queue its stated three immutable18-cell requests, each with1500seconds per child and28000seconds whole cap. All54 cells remain unchanged; the exact queue/request bindings still require review.

Under uniform1500-second children,18 cells reserve `18×(1500+15)+3=27273≤27995` normal child seconds inside a28000-second whole cap. Three such caps reserve84000seconds. This is a conservative reservation, not an actual GPU-time or9.67-hour forecast claim. Old reservations remain retained.

The primary proposal requires1410seconds for CS GT-sep's child, above bridgev3's1200-second ceiling and within prepared bridgev4's1800-second ceiling. V4 still retains a28800-second whole-cap ceiling, also enforced independently by bounded_run_v1. A single82000-second invocation is rejected by both. Fresh28000-second bounded batches resolve that confinement without shortening the competent horizon.

Stage1 uses the frozen effect/cost thresholds and PF/F wins≥2/3 per dataset. Only a pass permits core1/core2 continuation. Full original gates, including≥6/9 wins, must then pass before any confirmation. This forecast computes no comparative result or scientific verdict.

| Proposed IO MiB/s | Stage1 with20%+one reserve h | Largest operation timeout s | Fits v3 operation ceiling | Fits prepared v4 ceiling |
|---:|---:|---:|---|---|
| 50 | 9.67 | 1410 | no | yes |
| 100 | 8.29 | 1018 | yes | yes |
| 200 | 7.60 | 823 | yes | yes |

These sensitivity rows are unadopted proposals with the same observed maximum rates and other allowances. No fit is admitted by the forecast alone. Exact sources/hashes, all observations, assumptions, component arithmetic, schedule and pending admission gates are in the companion JSON.
