# Complete three-seed aggregation sensitivity

All 27 completed MolBACE fits are included: four independent native bodies and five shared methods per seed. Their saved selected VALID predictions supply 21 bank comparisons across seeds101/202/303. The two fixed rules use the identical selected checkpoints and all151 validation graphs. No model was constructed or run; no training or TEST scoring was performed.

| Bank | Mean-logit AUC ± SD (%) | Mean-probability AUC ± SD (%) | Mean-logit BCE | Mean-probability BCE |
| --- | ---: | ---: | ---: | ---: |
| M1 | 73.321 ± 1.172 | 73.321 ± 1.172 | 0.779 | 0.779 |
| I4 | 74.847 ± 0.500 | 74.896 ± 0.148 | 0.890 | 0.820 |
| G0 | 67.778 ± 4.390 | 69.011 ± 5.461 | 0.719 | 0.642 |
| geometry | 71.172 ± 2.657 | 70.256 ± 1.301 | 0.551 | 0.552 |
| correction | 71.758 ± 3.907 | 71.587 ± 3.027 | 0.941 | 0.638 |
| both | 69.280 ± 1.801 | 70.317 ± 3.571 | 0.882 | 0.707 |
| privateBN | 73.614 ± 1.057 | 73.468 ± 0.680 | 1.086 | 1.007 |

Probability averaging helps base GNNM mean AUC by1.233points and improves its AUC in all3seeds, but base remains5.885points below the independent ensemble under that same rule. The known private-BN control remains the highest mean-AUC shared method among these completed fits under either rule. Its I4 gap changes from1.233 to1.429points. Thus a simple change in averaging does not close the current ranking gap.

For private input corrections, mean BCE drops from0.941 to0.638 while mean AUC changes from71.758% to71.587%. The independent ensemble also benefits in BCE, from0.890 to0.820. Geometry has much lower BCE than privateBN under both rules. This supports keeping ranking and probability loss separate in decisions; an AUC improvement alone does not establish better probabilities.

The earlier seed101-only report is preserved. Its pending independent-ensemble comparison is now completed, and the conclusion is based on all declared banks rather than one seed. Neither rule was selected as a new winner, no original score or campaign serving rule changed, and both sets of per-seed values remain in RESULT.json.

These are posthoc sensitivity results on validation examples that already selected the checkpoints. All runs were selected under their original protocols, not independently reselected for probability averaging. Three seeds on one scaffold split give descriptive evidence only; the standard deviations are not confidence intervals or a superiority test.

Cost: 7.810 seconds including imports, CPU2, peakRSS499.69MiB. Zero new fits, updates, forwards, model constructions or GPU use. All27tensor files remain on the allocation. The source and frozen plan reproduce the arithmetic.
