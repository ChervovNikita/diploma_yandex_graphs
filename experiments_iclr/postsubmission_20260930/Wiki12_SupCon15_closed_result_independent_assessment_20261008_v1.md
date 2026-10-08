# Independent assessment of the closed Wiki12 / SupCon15 result packet

Date: 2026-10-08. This is an aggregate scientific result assessment of fixed selected states, not inference reproduction or manuscript peer review.

## Scope and integrity

Only the supplied manifest and its 11 bound files were read for scientific evidence. The manifest SHA256 is `481df58795d5adf9817cabd61a608f4a49c81812bab608116a11944926b08c59`; all 11 lengths and hashes match. No author history, external source, dataset, checkpoint, raw prediction array, or training process was consulted. No remote access, new fit, or numerical fixture was run. Calculations below use the sealed aggregates.

The packet covers all five arms at all three paired optimizer seeds (6101, 6203, 6307): 15 complete selected states, four members each, on 5,274 WikiCS split0 development nodes. P = plain; A = alignment only; R = residual only; C = combined; S = the adapted canonical SupCon Eq2 control. Collection custody accounts for old10 + unused2 + SupCon3. PER_SEED duplicates agree exactly with PER_CELL; no arm, seed, or member was excluded. All confusion matrices have the same truth marginals, the correct population size, and diagonals matching served correctness.

The bound owner exit records actual `subprocess.Popen.wait/poll` exit code 0, a reaped child, no remaining child CUDA rows, no signals, and no exceeded cap or telemetry error. COST records 15 checkpoint deserializations, exactly 60 attempted and completed member forwards, zero optimizer constructions/backward calls/train updates, and a complete readout in 78.835 inclusive seconds. This establishes the documented clean collector finish; these costs do not constitute a comparison of arm training costs. Referenced remote checkpoints, raw arrays, and prior closure receipts were not independently rehashed by this review.

## Arithmetic and consistency

No material arithmetic, completeness, or internal count error was found. Recalculation checked all eight contrasts and all ten metrics per contrast from their stated coefficients and fixed seed order; all values, means, sample SDs, df = 2, and intervals agree. The intervals use mean ± t(0.975, 2) × SD / sqrt(3), with t ≈ 4.30265273. REPORT.md agrees with the JSON at its displayed precision. The 18,179 aggregate checks also covered all 105 paired cohort rows, all four member summaries and changes, rates, confusion counts, frozen cohort sizes, coverage and common-competitor flows, repairs and introduced errors, and collection totals; there were zero mismatches (numerical discrepancies no larger than 1.1e−10).

For every paired row, repairs − harms equals the served correct-count change; gained − lost coverage equals the any-member-correct change; common-competitor cleared/persisted/appeared counts reconcile with both margins. Per-member correctness acquisitions and introduced errors also reconcile. Baseline error cohorts overlap and must not be added. In a baseline-wrong-only cohort, zero introduced served errors is structurally expected and says nothing about harms elsewhere.

## Every selected state

Accuracy and member competence are percentages; NLL is nats; Brier is the sum over ten classes. Mean and worst accuracy use all four members; worst NLL/Brier in the packet are metric-specific maxima. Epochs refer to the original selected checkpoints.

| Seed | Arm | Epoch | Served accuracy (%) | NLL | Brier | Mean / worst member accuracy (%) |
|---|---|---:|---:|---:|---:|---:|
| 6101 | P | 130 | 81.64581 | 1.174707 | 0.328649 | 81.61263 / 81.58893 |
| 6101 | A | 68 | 81.28555 | 1.124514 | 0.323974 | 81.23815 / 81.20971 |
| 6101 | R | 132 | 81.56997 | 1.120731 | 0.322406 | 81.56523 / 81.51308 |
| 6101 | C | 58 | 81.13386 | 1.052188 | 0.322565 | 81.11016 / 81.05802 |
| 6101 | S | 58 | 81.19075 | 1.057734 | 0.318867 | 81.24763 / 81.17179 |
| 6203 | P | 127 | 81.58893 | 1.094227 | 0.322382 | 81.55100 / 81.49412 |
| 6203 | A | 60 | 80.94425 | 1.080856 | 0.326786 | 80.95848 / 80.90633 |
| 6203 | R | 148 | 81.22867 | 1.351522 | 0.333928 | 81.22393 / 81.19075 |
| 6203 | C | 131 | 81.03906 | 1.121374 | 0.330416 | 80.97270 / 80.94425 |
| 6203 | S | 1073 | 81.15283 | 2.795923 | 0.361977 | 81.00114 / 80.86841 |
| 6307 | P | 66 | 80.81153 | 1.139152 | 0.328954 | 80.80205 / 80.77361 |
| 6307 | A | 53 | 81.07698 | 0.990119 | 0.318972 | 81.08646 / 81.05802 |
| 6307 | R | 133 | 81.19075 | 1.198728 | 0.332665 | 81.14334 / 81.02010 |
| 6307 | C | 63 | 81.66477 | 1.107684 | 0.318512 | 81.70743 / 81.64581 |
| 6307 | S | 50 | 81.11490 | 1.011188 | 0.319342 | 81.07224 / 81.03906 |

## All planned contrasts

Accuracy differences are percentage points, with seed columns in the fixed order 6101 / 6203 / 6307. Positive accuracy favors the first arm; negative NLL/Brier favors it. Every accuracy interval contains zero; the served NLL and Brier intervals also all contain zero.

| Contrast | Accuracy differences by seed (pp) | Mean accuracy (pp) | Descriptive 95% df2 interval (pp) | Mean NLL difference | Mean Brier difference |
|---|---|---:|---|---:|---:|
| C-P | -0.51195 / -0.54987 / +0.85324 | -0.06952 | [-2.05525, 1.91621] | -0.042280 | -0.002831 |
| A-P | -0.36026 / -0.64467 / +0.26545 | -0.24649 | [-1.40312, 0.91014] | -0.070866 | -0.003418 |
| R-P | -0.07584 / -0.36026 / +0.37922 | -0.01896 | [-0.94556, 0.90764] | +0.087632 | +0.003005 |
| C-A | -0.15169 / +0.09480 / +0.58779 | +0.17697 | [-0.75837, 1.11230] | +0.028586 | +0.000587 |
| C-R | -0.43610 / -0.18961 / +0.47402 | -0.05056 | [-1.21991, 1.11878] | -0.129912 | -0.005836 |
| C-A-R+P | -0.07584 / +0.45506 / +0.20857 | +0.19593 | [-0.46405, 0.85591] | -0.059046 | -0.002418 |
| SupCon-A | -0.09480 / +0.20857 / +0.03792 | +0.05056 | [-0.32723, 0.42836] | +0.556452 | +0.010151 |
| SupCon-P | -0.45506 / -0.43610 / +0.30338 | -0.19593 | [-1.27036, 0.87850] | +0.485586 | +0.006734 |

## Repairs, harms, and coverage

Each seed entry below is served repairs / introduced served errors on the full population. Coverage entries are net any-member-correct changes for 6101 / 6203 / 6307. The factorial interaction C−A−R+P is a linear contrast, not a two-state transition.

| Contrast | 6101 repairs / harms | 6203 repairs / harms | 6307 repairs / harms | Net coverage by seed (nodes) |
|---|---:|---:|---:|---|
| C-P | 161 / 188 | 103 / 132 | 128 / 83 | -25 / -28 / +54 |
| A-P | 148 / 167 | 113 / 147 | 99 / 85 | -18 / -28 / +13 |
| R-P | 95 / 99 | 112 / 131 | 145 / 125 | -4 / -10 / +19 |
| C-A | 104 / 112 | 129 / 124 | 137 / 106 | -7 / +0 / +41 |
| C-R | 132 / 155 | 125 / 135 | 128 / 103 | -21 / -18 / +35 |
| SupCon-A | 97 / 102 | 230 / 219 | 63 / 61 | +1 / +28 / +2 |
| SupCon-P | 137 / 161 | 189 / 212 | 97 / 81 | -17 / +0 / +15 |

For C−P, the plain-frozen common-competitor cohorts contain 959 / 966 / 1,003 nodes. Coverage is acquired on 156 / 104 / 133 and served correctness is repaired on 156 / 100 / 124, while 803 / 862 / 870 remain wrong in every member. Across the full population, new common-competitor errors appear on 181 / 132 / 79 nodes. The favorable conditional repairs coexist with full-population harms of 188 / 132 / 83. Pooling rescues zero all-member-wrong nodes in every selected state, and each state loses 5–28 covered nodes through pooling. Any-member coverage and served accuracy are therefore distinct quantities.

## Scientific conclusions and limits

1. The fresh C−P comparison does not show an accuracy gain: mean −0.06952 pp, two losses and one gain, with a wide interval [−2.05525, 1.91621]. Average NLL and Brier improve, but both worsen in seed 6203. The observed common-error repairs support a descriptive account of changed errors; they do not establish a successful combined mechanism or overall superiority.

2. Component results are mixed. A−P lowers NLL in all three seeds, but loses mean accuracy (−0.24649 pp) and has a Brier harm in seed 6203. R−P loses mean accuracy and worsens mean NLL/Brier. C−A gains mean accuracy while worsening mean NLL/Brier; C−R lowers NLL in all seeds while losing mean accuracy. The positive mean interaction (+0.19593 pp) changes sign across seeds and has interval [−0.46405, 0.85591], so it cannot establish positive synergy.

3. The prospective SupCon−A contrast has a small mean accuracy gain (+0.05056 pp; interval [−0.32723, 0.42836]) and mean NLL/Brier harms (+0.556452 / +0.010151). Seed 6203 has +1.715067 NLL and +0.035191 Brier relative to A despite +0.20857 pp accuracy; its selected epoch is 1073. Seed 6101 improves both probability scores while losing accuracy; seed 6307 gains accuracy while slightly worsening both scores. These are retained outcomes, with no exclusion or alternate checkpoint. The contextual SupCon−P comparison loses mean accuracy and worsens mean NLL/Brier. This packet does not show a consistent canonical-loss improvement or evidence of specialization.

4. All readouts reuse the development population that selected the checkpoints. The three optimizer seeds are paired blocks on one graph, not independent graphs; nodes and members are not independent model replicates. The correctly computed t intervals are descriptive and do not establish nominal inferential coverage after selection, confirmation, novelty, generalization, or a graph-specific causal mechanism. Similar member accuracy and zero within-state pool rescues cannot prove useful diversity or its absence: aggregate competence alone does not describe prediction dependence. Cohort freeze timing and selected-state custody are documented assertions, not independently reconstructed from raw arrays. COST explicitly says the portable data interface did not verify official source provenance; official mask identity is consequently unverified here. No independent TEST evidence is supplied.

The sealed packet supports a complete, internally consistent selected-development comparison with real improvements and harms, and a clean collection exit. Its strongest scientific result is the absence of a stable accuracy advantage in the primary C−P comparison, alongside heterogeneous probability-score and error-transition effects that require independent evidence before broader claims.
