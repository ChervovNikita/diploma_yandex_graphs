# Completed probability-failure diagnostics

All11 methods x3 seeds; original serving and report remain unchanged.

TRAIN positives:480/1210 (39.67%); VALID positives:130/151 (86.09%). This observed difference does not establish pure label shift or explain paired excess BCE.

| Seed | Bank | Pool BCE | Class0 BCE | Class1 BCE | Wrong graphs | Mean BCE on wrong | Mean true-label probability on wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | M1 | 0.92069 | 0.58613 | 0.97473 | 65 | 1.92250 | 0.22046 |
| 101 | I4 | 0.97641 | 0.41812 | 1.06659 | 71 | 1.84527 | 0.24245 |
| 101 | n0_g0_c0 | 0.49848 | 1.02501 | 0.41343 | 27 | 0.98379 | 0.38421 |
| 101 | n0_g0_c1 | 0.54666 | 0.82242 | 0.50212 | 40 | 1.03787 | 0.36944 |
| 101 | n0_g1_c0 | 0.55421 | 0.82440 | 0.51056 | 32 | 0.90365 | 0.41015 |
| 101 | n0_g1_c1 | 0.51682 | 0.91355 | 0.45273 | 30 | 0.90244 | 0.41475 |
| 101 | n1_g0_c0 | 0.56048 | 0.77090 | 0.52649 | 42 | 1.50478 | 0.27835 |
| 101 | n1_g0_c1 | 1.14440 | 0.30573 | 1.27988 | 67 | 2.30958 | 0.19653 |
| 101 | n1_g1_c0 | 0.82497 | 0.38864 | 0.89545 | 58 | 1.71016 | 0.23767 |
| 101 | n1_g1_c1 | 0.79944 | 0.44184 | 0.85721 | 53 | 1.76051 | 0.22826 |
| 202 | M1 | 0.66998 | 0.58095 | 0.68436 | 45 | 1.60968 | 0.25462 |
| 202 | I4 | 0.75403 | 0.46395 | 0.80088 | 53 | 1.59404 | 0.27056 |
| 202 | n0_g0_c0 | 0.87341 | 0.82547 | 0.88116 | 46 | 2.46353 | 0.18057 |
| 202 | n0_g0_c1 | 0.75337 | 1.27236 | 0.66954 | 42 | 2.43373 | 0.17866 |
| 202 | n0_g1_c0 | 0.44218 | 1.66380 | 0.24485 | 29 | 1.71239 | 0.23958 |
| 202 | n0_g1_c1 | 0.86475 | 0.41914 | 0.93674 | 87 | 1.21240 | 0.33047 |
| 202 | n1_g0_c0 | 0.84221 | 0.52178 | 0.89397 | 57 | 1.81603 | 0.25221 |
| 202 | n1_g0_c1 | 0.66639 | 0.86342 | 0.63456 | 43 | 1.81857 | 0.23811 |
| 202 | n1_g1_c0 | 0.59322 | 0.71848 | 0.57299 | 49 | 1.27065 | 0.31320 |
| 202 | n1_g1_c1 | 0.98716 | 0.66554 | 1.03911 | 75 | 1.82722 | 0.24162 |
| 303 | M1 | 0.74776 | 0.63517 | 0.76595 | 52 | 1.78239 | 0.22708 |
| 303 | I4 | 0.94007 | 0.34523 | 1.03616 | 64 | 1.82856 | 0.21891 |
| 303 | n0_g0_c0 | 0.78598 | 0.44429 | 0.84118 | 68 | 1.18477 | 0.32847 |
| 303 | n0_g0_c1 | 1.52233 | 0.13805 | 1.74595 | 85 | 2.42909 | 0.19133 |
| 303 | n0_g1_c0 | 0.65519 | 0.55130 | 0.67198 | 48 | 1.21399 | 0.32703 |
| 303 | n0_g1_c1 | 1.26367 | 0.21753 | 1.43266 | 99 | 1.76056 | 0.22244 |
| 303 | n1_g0_c0 | 1.85388 | 0.15289 | 2.12866 | 85 | 3.13050 | 0.14413 |
| 303 | n1_g0_c1 | 0.63688 | 0.90098 | 0.59422 | 44 | 1.73813 | 0.25512 |
| 303 | n1_g1_c0 | 1.94682 | 0.12655 | 2.24087 | 84 | 3.36723 | 0.15327 |
| 303 | n1_g1_c1 | 1.30281 | 0.26578 | 1.47033 | 80 | 2.24839 | 0.18993 |
| 101 | residual_only_private_BN | 3.38622 | 0.08614 | 3.91931 | 106 | 4.75316 | 0.06999 |
| 202 | residual_only_private_BN | 1.36874 | 0.18418 | 1.56009 | 96 | 1.97529 | 0.22116 |
| 303 | residual_only_private_BN | 1.10114 | 0.32527 | 1.22648 | 82 | 1.82715 | 0.23556 |

RESULT.json retains pooled/native-member raw-logit/true-margin quantiles, per-class BCE/Brier, correct/wrong splits and wrong-prediction true-label probabilities, signed paired-loss summaries and top1/top5 graph concentration including all negative offsets. All graph-level contributions remain in the separately receipted allocation-only PER_EXAMPLE.jsonl; do not sync or commit it. Fractions of net excess can exceed1; no net excess is reported as unsupported concentration.

No fitted calibrator, prior adjustment, bootstrap, significance test, ECE or new scoring rule is used. No model/checkpoint, current partial fit or TEST payload is read.
