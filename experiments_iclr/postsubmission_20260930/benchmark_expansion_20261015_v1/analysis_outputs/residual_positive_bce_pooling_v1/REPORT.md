# Current33 fixed pooling and class BCE

Both fixed readouts are shown for all11 banks x3 seeds; original scores and selection stay authoritative.

## Residual minus I4

| Seed | Positive BCE excess: logit | Positive BCE excess: probability | Fraction remaining | Residual AUC change (pp) | I4 AUC change (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 2.852718 | 2.937021 | 102.96% | -0.0366 | 0.4762 |
| 202 | 0.759208 | 0.790095 | 104.07% | -0.0366 | -0.3297 |
| 303 | 0.190316 | 0.318975 | 167.60% | 0.0000 | 0.0000 |

Mean positive-class excess changes from 1.267414 to 1.348697; 106.41% remains (ratio of seed-mean signed excesses).

## All banks, three-seed arithmetic means

| Bank | AUC %: logit / probability | BCE: logit / probability | Positive BCE: logit / probability |
| --- | ---: | ---: | ---: |
| M1 | 73.3211 / 73.3211 | 0.779476 / 0.779476 | 0.808347 / 0.808347 |
| I4 | 74.8474 / 74.8962 | 0.890168 / 0.820155 | 0.967880 / 0.884886 |
| n0_g0_c0 | 67.7778 / 69.0110 | 0.719292 / 0.641944 | 0.711920 / 0.626720 |
| n0_g0_c1 | 71.7582 / 71.5873 | 0.940789 / 0.638032 | 0.972533 / 0.624805 |
| n0_g1_c0 | 71.1722 / 70.2564 | 0.550529 / 0.552120 | 0.475795 / 0.497107 |
| n0_g1_c1 | 69.2796 / 70.3175 | 0.881745 / 0.707269 | 0.940707 / 0.722895 |
| n1_g0_c0 | 73.6142 / 73.4676 | 1.085523 / 1.006798 | 1.183039 / 1.091184 |
| n1_g0_c1 | 72.1856 / 71.8315 | 0.815891 / 0.789913 | 0.836220 / 0.808112 |
| n1_g1_c0 | 73.6508 / 73.5897 | 1.121673 / 0.995713 | 1.236438 / 1.087832 |
| n1_g1_c1 | 73.0281 / 73.2112 | 1.029805 / 0.972390 | 1.122219 / 1.054777 |
| residual_only_private_BN | 74.5177 / 74.4933 | 1.952035 / 1.950576 | 2.235294 / 2.233583 |

BCE uses stable log probabilities without clipping. Mixture AUC uses stable log odds, with exact numerical ties worth half credit; tie counts are in RESULT.json. Class-only AUC is undefined. RESULT.json also gives both class losses, Brier, fixed-threshold errors, every seed and signed paired differences.

Native saved pools are used even when collected member means have small reconstruction differences; those differences and retained collection sanity are disclosed. No alternative is promoted from VALID. No fit, calibration, prediction edit, model/checkpoint load, forward, update or TEST access occurred.
