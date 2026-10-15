# Ranking recovery comes with a class-specific probability failure

This diagnostic uses the unchanged saved predictions of all 11 completed MolBACE methods across seeds 101, 202 and 303. It adds no training, checkpoint selection, fitted calibration, new readout or TEST access. The primary MolBACE metric remains AUC. The original scores are unchanged.

## Main finding

The residual-only routes recover ranking quality, but their served probabilities favor the negative class too strongly on this VALID population. Relative to I4, mean loss on the 130 positive molecules increases in every seed, while mean loss on the 21 negatives decreases in every seed. At the fixed 0.5 probability threshold, false negatives also increase in every seed. This is a descriptive class-specific failure; it does not prove its training cause or estimate the true probability for any individual molecule.

| Seed | Residual − I4 BCE, all graphs | Difference on positives | Difference on negatives | False negatives, residual / I4 (of 130) | Top five share of net excess BCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | +2.410 | +2.853 | -0.332 | 105 / 67 | 12.3% |
| 202 | +0.615 | +0.759 | -0.280 | 95 / 46 | 17.7% |
| 303 | +0.161 | +0.190 | -0.020 | 78 / 60 | 38.2% |

The loss increase is spread across many molecules: 121, 116 and 91 of 151 graphs have higher loss than I4. The five largest positive contributions account for only 12.3%, 17.7% and 38.2% of the **net signed** excess loss. All beneficial contributions remain in that denominator. Removing those five contributions would leave positive excess in every seed; this arithmetic is an error diagnosis, not permission to remove observations.

Averaged across the three seeds, positive-class BCE is 2.235 for residual-only versus 0.968 for I4, while negative-class BCE is 0.199 versus 0.409. Overall Brier loss also worsens, 0.473 versus 0.289. The original mean AUC comparison remains 74.518% versus 74.847%; ranking and probability conclusions must be stated separately.

## Confidence and prevalence do not give a single causal explanation

The strongest confidence failure is seed 101. Its 106 wrong threshold predictions assign the true label mean probability 0.070, versus I4's 0.242 on its own 71 wrong predictions. Those error sets differ, so this is not a paired confidence comparison. Seed 303 makes more threshold errors, but mean true-label probability on its wrong set is slightly higher than I4 (0.236 versus 0.219). A blanket claim that every seed merely became more confident on wrong predictions would be inaccurate.

The observed positive prevalence is 480/1210 = 39.67% in TRAIN and 130/151 = 86.09% in VALID. Both reference and candidate face that same scaffold split. The counts do not establish pure label shift, explain the candidate's excess loss, or justify a correction fitted from VALID. No intercept, temperature, threshold or readout was fitted here.

The complete 33-row table retains the other methods. The geometry-only GNNM has lower mean BCE (0.551) and Brier (0.183) than the residual-only variant but lower AUC; it is not silently promoted to the successful method. The methods and selectors are unchanged.

## Consequences for the research

1. Describe the completed result as **ranking recovery**, not general recovery of prediction quality or calibrated route competence. Preserve the remaining AUC deficit, seed variation and probability harm.
2. The frozen structural study continues unchanged. Its final report must distinguish repairs to common ranking mistakes from loss shifted between classes; no partial outcome is used to change the recipe.
3. Keep PubMed/Cheb as the default next representative study after measured admission. Its native model has no BN, making it useful for testing whether this recipe helps outside the current molecular normalization setting. This does not identify BN as the cause.
4. Keep the pairwise objective as a distinct prepared hypothesis. A better ranking surrogate may increase score margins without repairing probability quality. Retain native BCE and same-auxiliary I4 controls; an AUC-only gain must be labeled accordingly.
5. Do not add an arbitrary probability penalty, calibration fit or new hyperparameter search on these same exposed molecules from this diagnostic alone. A new intervention needs a concrete hypothesis and a separately frozen comparison.

## Costs and reproducibility

One bounded CPU analysis passed: 3.779 seconds in the child, 4.506 seconds under its supervisor, with sampled peak RSS 374,833,152 bytes. It performed zero model constructions, forwards, optimizer transitions or checkpoint loads. All 39 saved exports and 27 member collections were checked against the completed report; all 33 pooled losses agree with the original values. No running structural output was opened.

The compact `SUMMARY.json` and full table are reviewable locally. The complete 1.73 MB scalar report and 9,966 graph-level contributions remain on the allocation, with hashes recorded in the summaries and execution receipt. Do not copy the per-example artifact into Git. Three optimizer seeds on this repeatedly examined scaffold split do not provide independent confirmation or a calibrated generalization interval.
