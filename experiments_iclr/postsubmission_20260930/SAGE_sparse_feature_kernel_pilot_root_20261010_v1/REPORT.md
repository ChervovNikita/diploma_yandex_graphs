# Complete sparse feature-message kernel comparison

All 18 fresh banks / 27 optimizer-body fits and 15 original banks / 33 selected-fit records are complete. Native acquisition ended at 21:14:39 UTC in 972.270261 s; the full reader closed successfully at 21:17:04 UTC, with all 90 new calibration fits and 75 exact reused endpoints terminal. No TEST, original score change, replay of old fits or calibration endpoints occurred.

| Arm | Raw accuracy % | Raw NLL | Calibrated accuracy % | Calibrated NLL |
|---|---:|---:|---:|---:|
| shared4_unchanged | 80.571356 | 1.001493 | 80.596638 | 0.682065 |
| shared4_private_kernel | 78.131715 | 1.521414 | 78.207559 | 0.807864 |
| shared4_common_kernel | 78.289723 | 1.788636 | 78.239161 | 0.820854 |
| shared4_fixed_feature_union | 78.346606 | 1.107697 | 78.390848 | 0.761459 |
| ordinary_M1_feature_kernel | 76.930856 | 0.868812 | 76.930856 | 0.788223 |
| factorized_M1_feature_kernel | 76.671723 | 1.536989 | 76.671723 | 0.860632 |
| genuine_factorized_I4_feature_kernel | 78.226520 | 0.923934 | 78.239161 | 0.755217 |

The declared private kernel fails every raw contrast criterion. It loses 2.439641 points against unchanged shared, 0.094805 against equipped genuine I4, 0.158008 against common learned kernels and 0.214891 against fixed uniform support. All three seeds lose to unchanged shared and fixed union; only one gains against common/equippedI4. Calibrated NLL harms against baseline/equippedI4/fixed are 0.125799/0.052647/0.046405. No joint or accuracy-only private clue is admitted. Both equipped singles remain weaker; that does not rescue losing to unchanged shared or richer I4.

Compared with unchanged shared, private mean-member accuracy falls 1.472633 points. Across the three paired readouts, correct-alternative coverage drops 604 occurrences (13740→13136), while averaging loses 218 fewer alternatives (993→775). Aggregation-only correctness stays 1. The exact pooled difference is therefore−604+218=−386 correct readouts. This rule weakens acquisition despite the smaller averaging loss. These are fitted/stopped/selected-procedure contrasts, not a same-state causal learning effect or independent-node replications.

Against fixed union, private mean-member accuracy improves 0.568828 points at all three seeds, yet coverage falls 239 and pooling loses 204 fewer alternatives, with one added aggregation-only success:−239+204+1=−34 pooled correct readouts. Better average members alone do not establish a useful ensemble. Common and fixed variants also substantially underperform the original support. The exact raw-feature top20 union/kernel recipe is closed without a rescue grid. These results do not exclude all graph weighting, conservative mixtures or alternative operators.

Next acquisition preparation changes context-positive target/support pairs while preserving the original graph, with equally equipped singles / I4 and mixture/full/shuffled controls. The source is independently reviewed; actual TRAIN qualification is still required. The separate 77 F reference comparison changes no current kernel decision. Original manuscript scores remain untouched.

Complete per-seed/class/member/error/strict-rival/cost tables, descriptive uncertainty and calibration failures are in the 16 bounded JSON partitions. The full report, checkpoints and prediction arrays remain on the authorized allocation. The initial compact collector failed because it admitted the full report to a less than 2 MB transfer class; v2 excludes only that named full report, retaining all compact partitions and the full-report hash/bytes in the summary. No reader or scientific fit was rerun.
