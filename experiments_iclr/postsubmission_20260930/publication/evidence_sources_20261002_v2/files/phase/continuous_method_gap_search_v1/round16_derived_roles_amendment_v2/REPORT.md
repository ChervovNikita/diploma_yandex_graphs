# Derived-role amendment v2

## Diagnosis

The first acquisition attempt failed in `prepare`, before label acquisition or fitting, because v1 requires every published Squirrel validation node to fit inside `floor(.50N)-floor(.20N)` source nodes. The pinned filtered release has N=2223, teacher count444 and reservoir667. Published splits0/1/2 have validation counts718/700/726, exceeding the reservoir by51/33/59. This is an infeasible mask assumption, not a training failure.

Parent supplied these mask-only counts from `protocols/PUBLICATION_METADATA_v1.json`; mask SHA256 is `e246809add63d65bf2228580d06f9f0ee2972415904a6d857fdb822908383da8`. The local `GRAPH_INPUT.json` and `PREPARATION.log` were inspected. The masks themselves are remote-only; child did not read any dataset values or labels. Source provenance is `yandex-research/heterophilous-graphs`, release `squirrel_filtered.npz`, Git blob `a59f6d2e0dc1f11b8e503c426e86f53124c21eb1`, raw SHA256 `f06370ffb3116c5de8b7a600005400b2293c025e0ce2f6916d8201a825505ac7`.

## Amendment

Use a label-blind uniform subset of published validation when it exceeds the fixed reservoir; move its surplus to final pool. Every published test node remains in final pool. Teacher20% and the total source≈50% budget remain fixed. The original feasible branch keeps its previous RNG stream; oversized masks consume one additional validation permutation before reservoir permutation.

The scientific guarantee changes: **v2 no longer keeps every published validation node in source roles**. For these three Squirrel splits, the reservoir comes entirely from a uniformly selected published-validation subset; all unused published train and validation nodes join published test in final pool. Neither official-split accuracy nor performance on only the original test population is claimed. The final calibration/test estimand is the newly frozen mixed pool.

| Split / seed | Published train | Published validation | Published test | Validation surplus to pool | v2 train / validation / A / B / D / pool |
|---|---:|---:|---:|---:|---|
| 0 /17 |1053|718|452|51|444 /222 /111 /222 /112 /1112|
| 1 /29 |1080|700|443|33|444 /222 /111 /222 /112 /1112|
| 2 /43 |1047|726|450|59|444 /222 /111 /222 /112 /1112|

Counts are metadata arithmetic, not an executed preparation. No labels, class balancing, outcome search, model import, fit, score, or silent retry occurred.

## Delivery and preservation

`prototype/correction_screen_driver.py` is a new source identity. Both companion adapters are copied byte-for-byte from sealed v1. `SOURCE_PATCH.diff` exposes the exact change. All three source files were AST parsed only. The sealed round5 packet and failed acquisition_run01 were not modified.

Root may separately acquire/run with a new output directory and newly frozen driver/admission hashes. All methods within a new cell must share the same v2 arrays. The failed v1 attempt remains a retained preparation failure; future v2 results must identify this amendment. Only root is authorized to execute.
