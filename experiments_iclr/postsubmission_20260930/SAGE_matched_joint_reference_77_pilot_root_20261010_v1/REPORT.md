# Complete interpretation of the matched SAGE F reference study

## Finding and exact frozen rule

Shared4 supplies a real accuracy benefit in this complete encountered-WikiCS F repeat. It beats all four tested references at every paired seed, with mean raw gains of 0.720516 percentage points over matched joint untied4 and 0.537227 points over genuine factorized I4. Both frozen raw gates pass: mean gain ≥ 0.2 points, all three differences nonnegative and at least two positive.

The complete accuracy-and-confidence conjunction fails exactly one condition. Calibrated NLL harm versus genuine I4 is **0.021448383545404**, above the fixed 0.02 mean cap by 0.001448383545404. Its seed harms are +0.007869, +0.032667 and +0.023809, all below 0.05. Against joint untied4, mean harm +0.013310 and maximum seed harm +0.021990 both pass. Thus the raw screen passes, every per-seed NLL cap passes, and the I4 mean-NLL cap fails. Calibrated accuracy, member quality and costs add no veto. The failed conjunction preserves the accuracy-only clue and its confidence tradeoff.

## Complete five-arm readout

All values are three-seed means. Accuracy is percent; pooling gain is pooled minus mean member accuracy in percentage points. “Worst” is the mean of each seed's worst member.

| Arm | Raw accuracy | Member mean / worst | Pool gain | Raw NLL | Calibrated accuracy | Calibrated NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared4 | 80.6219 | 78.5062 / 77.9737 | 2.1157 | 1.015767 | 80.6282 | 0.681339 |
| Joint untied4 | 79.9014 | 78.7274 / 78.3656 | 1.1740 | 0.717174 | 79.9330 | 0.668029 |
| Factorized M1 | 79.1303 | 79.1303 / 79.1303 | 0.0000 | 0.778711 | 79.1303 | 0.707176 |
| Genuine factorized I4 | 80.0847 | 79.2109 / 78.7574 | 0.8738 | 0.710204 | 80.0910 | 0.659891 |
| Ordinary M1 | 79.3199 | 79.3199 / 79.3199 | 0.0000 | 0.754213 | 79.3199 | 0.696492 |

| Arm | Raw accuracy, seed 7301 | Seed 7403 | Seed 7507 |
| --- | ---: | ---: | ---: |
| Shared4 | 80.4513 | 80.8684 | 80.5461 |
| Joint untied4 | 79.8256 | 79.9393 | 79.9393 |
| Factorized M1 | 79.2378 | 79.4084 | 78.7448 |
| Genuine factorized I4 | 80.0910 | 79.9583 | 80.2048 |
| Ordinary M1 | 79.2567 | 79.1430 | 79.5601 |

| Arm | Raw member NLL mean / worst | Calibrated member NLL mean / worst |
| --- | ---: | ---: |
| Shared4 | 1.386156 / 1.427088 | 0.776060 / 0.793967 |
| Joint untied4 | 0.797848 / 0.821994 | 0.713386 / 0.724029 |
| Factorized M1 | 0.778711 / 0.778711 | 0.707176 / 0.707176 |
| Genuine factorized I4 | 0.804459 / 0.869474 | 0.712186 / 0.738777 |
| Ordinary M1 | 0.754213 / 0.754213 | 0.696492 / 0.696492 |

Shared's raw confidence is much worse than either ensemble reference. Equal-policy calibration reduces its NLL from 1.015767 to 0.681339, while preserving its accuracy advantage. Its calibrated Brier is 0.292765, versus 0.299865 joint and 0.296169 I4; this favorable proper-score comparison does not change the frozen NLL gate. Calibration changes pooled correct counts by +1 shared, +5 joint and +1 I4 across the three readouts; single decisions and all member argmaxes remain unchanged.

### Paired Shared4 comparisons

Positive accuracy differences favor Shared4; positive NLL differences mean harm. Seed triples follow 7301 / 7403 / 7507.

| Reference | Raw accuracy seed differences, pp | Raw mean gain, pp | Calibrated mean gain, pp | Calibrated mean NLL difference |
| --- | --- | ---: | ---: | ---: |
| Joint untied4 | +0.6257 / +0.9291 / +0.6068 | +0.720516 | +0.695234 | +0.013310 |
| Factorized M1 | +1.2135 / +1.4600 / +1.8013 | +1.491594 | +1.497914 | -0.025837 |
| Genuine factorized I4 | +0.3603 / +0.9101 / +0.3413 | +0.537227 | +0.537227 | +0.021448 |
| Ordinary M1 | +1.1945 / +1.7254 / +0.9860 | +1.301985 | +1.308305 | -0.015153 |

Neither tested single is stronger in accuracy: Shared gains 1.491594 points over equipped factorized M1 and 1.301985 over ordinary M1, all seeds positive. After calibration Shared also has lower mean NLL than both singles. Genuine I4 remains the stronger calibrated-NLL reference, and shared acquisition/serving observations do not establish speed superiority. These distinctions limit a broad superiority claim.

## Pooling, coverage and sharing

Counts sum three dependent readouts of the same 5,274 development nodes, totaling 15,822 readouts. Coverage means at least one member is correct; losses are covered nodes the probability mean gets wrong. Pooled correct = coverage − losses + aggregation-only correct.

| Arm | Coverage | Pooling losses | Aggregation-only correct | Pooled correct |
| --- | ---: | ---: | ---: | ---: |
| Shared4 | 13,731 | 976 | 1 | 12,756 |
| Joint untied4 | 13,447 | 805 | 0 | 12,642 |
| Factorized M1 | 12,520 | 0 | 0 | 12,520 |
| Genuine factorized I4 | 13,506 | 835 | 0 | 12,671 |
| Ordinary M1 | 12,550 | 0 | 0 | 12,550 |

Shared members are weaker on mean accuracy than joint and I4, while the selected shared pool is stronger. Its pooling gain is 2.115725 points, versus 1.173998 joint and 0.873783 I4. Relative to joint, Shared obtains +284 coverage, incurs +171 pooling losses and adds one aggregation-only correction, net +114 correct. Relative to I4 the decomposition is +225 coverage −141 extra losses +1, net +85. This is useful complementarity in the served prediction, not a weaker-member veto. Raw cross-procedure exchanges are Joint untied4: 671 repairs / 557 harms; Factorized M1: 825 repairs / 589 harms; Genuine factorized I4: 645 repairs / 560 harms; Ordinary M1: 813 repairs / 607 harms.

The strongest architectural contrast is Shared versus full-width joint untied4: initial route functions, route factor/dropout starts, the mean-own-CE bank objective and coherent bank selection are matched. Shared retains a common backbone with rank-one route factors; joint untied has four disjoint complete bodies. The accuracy result favors the sharing procedure over merely acquiring/averaging the matched untied procedures in this study. Independent I4 additionally uses its own native starts, unscaled per-body CE, optimizers and selectors, and Shared still gains raw accuracy over that capable control.

Different parameter ownership, gradient aggregation, AdamW states, stopping and selected updates are part of these full procedures. The result does not isolate a same-state causal effect or prove that rank-one factorization alone is the mechanism. Compact outputs contain no covariance matrix or estimated error-correlation statistic; coverage and repairs cannot identify covariance by themselves. Their complementarity pattern is consistent with useful diversity under sharing, without a covariance-reduction guarantee.

For common-rival diagnosis, Shared corrects 289 of 2317 joint-reference strict-rival readouts and 241 of 2262 I4-reference readouts. Its own strict-rival cohort has 1961 readouts and zero own pooling corrections. These are fixed reference cohorts with a common wrong class strictly above truth in every reference member. Cross-procedure correction includes learning/stopping/selection; own-cohort zero is not zero repair of reference failures.

## Observed costs and closure

Times below sum the three banks' recorded scopes, not concurrent wall time. Serving includes selected native forward/metrics/CPU transfer and probability pooling. Parameters are the served model per seed.

| Arm | Parameters | Backwards/Adam steps | TRAIN route trajectories | Acquisition, s | Restore, s | Selected readout, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared4 | 216,674 | 1,558 | 6,232 | 71.676 | 0.0566 | 0.5404 |
| Joint untied4 | 832,256 | 1,016 | 4,064 | 50.025 | 0.2115 | 0.0716 |
| Factorized M1 | 208,064 | 1,011 | 1,011 | 14.286 | 0.0535 | 0.0223 |
| Genuine factorized I4 | 832,256 | 4,082 | 4,082 | 58.620 | 0.2054 | 0.0761 |
| Ordinary M1 | 205,194 | 1,006 | 1,006 | 12.209 | 0.0333 | 0.0214 |

Shared uses 216,674 parameters against 832,256 for joint/I4, but takes more recorded acquisition time and more updates/trajectories. All five-arm acquisition scopes sum 206.815 seconds; the two admitted GPU shards overlapped. Recorded shared CUDA allocated/reserved peaks are 1,164,760,064 / 1,447,034,880 bytes; joint is 1,173,879,296 / 1,457,520,640; I4 and factorized M1 are 428,319,232 / 763,363,328; ordinary is 354,326,016 / 534,773,760. RSS is a process lifetime peak; no isolated throughput or packed-execution claim follows. Total preparation and runtime-qualification costs are unavailable in these compact outputs; the recorded timing scopes do not give an end-to-end experiment wall time.

All 15 banks / 24 optimizer bundles / 33 native-body records closed before analysis. All 75 fixed five-fold scalar endpoints succeeded, 37,500 updates, 153.171 seconds summed fit work within 154.277 seconds calibration wall time, with no new native model/forward, retry or final refit. The v2 CPU watcher completed in 160.409 seconds. Its original v1 alias failure remains retained at 160.905 seconds; that waiting interval overlaps training and is not an additive fit cost. The repair caused zero scientific reruns and the first reader launch completed successfully.

The closure receipt pins the fetched summary; all 14 compact partitions match its byte/hash descriptors and row coverage. Raw archived float32 decisions remain authoritative. Reader diagnostics show zero FP64 raw-pool or raw-member argmax disagreements and only ≈1.1e−16 maximum stable-NLL difference from the reported original. No score, criterion, source or historical manuscript value was changed.

## Scope

These are three paired optimizer seeds on one encountered development graph, with the same VALID role used for native checkpoint selection. Cross-fitted temperature assessment follows that selection; it is not whole-pipeline unused confirmation. Raw gain descriptive 95% t intervals are [0.2712, 1.1698] points versus joint and [−0.2653, 1.3398] versus I4; both three-seed sign-flip p values are 0.25. The repeated-node totals are not independent sample sizes or significance evidence.

The closed result supports accuracy-oriented sharing under these capable controls and retains the narrow calibrated-NLL tradeoff. It supplies no primitive novelty, paper-acceptance or universal-transfer verdict. K remains null. Full arrays stayed server-side, no remote execution or native replay was performed for this interpretation, and no active continuous-affinity partial result was read. Exact 15-bank seed readouts, 42 route metrics, all 20 ordered comparisons for both readouts and complete cost fields are retained in the accompanying CSVs.
