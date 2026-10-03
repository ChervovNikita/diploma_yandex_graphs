# Complete BUDDY pilot: paired summary

All values below concern three optimizer seeds on the same graph and split.

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| native1024 | 54.053 | 53.124 | 53.144 | 53.440 ± 0.530 |
| single256 | 53.517 | 53.945 | 54.417 | 53.960 ± 0.450 |
| factorized4 | 52.827 | 53.705 | 53.057 | 53.196 ± 0.455 |
| independent4 | 53.068 | 53.474 | 53.217 | 53.253 ± 0.205 |
| matched_single | 52.863 | 53.057 | 53.653 | 53.191 ± 0.412 |

Hits@50 is reported in percent. Differences are percentage points.

| Factorized minus control | Paired differences | Mean | 95% t interval | Family 95% interval | Holm sign reference p |
|---|---|---:|---|---|---:|
| single256 | -0.691, -0.240, -1.360 | -0.763 | -2.164, 0.637 | -3.252, 1.726 | 0.750 |
| independent4 | -0.242, 0.231, -0.160 | -0.057 | -0.684, 0.571 | -1.172, 1.059 | 1.000 |
| matched_single | -0.037, 0.648, -0.596 | 0.005 | -1.542, 1.552 | -2.745, 2.755 | 1.000 |

The t intervals require independent, approximately normal seed differences. Three seeds cannot check that assumption. Family intervals use Bonferroni over the three fixed contrasts. The sign reference assumes equiprobable independent signs under a zero-median null. Its smallest two-sided p-value with three nonzero pairs is 0.25. These summaries do not establish graph generalization, state of the art or a new method.
