# Graph connectivity does not isolate the ensemble deficit

8 October 2026. This is a descriptive profile of complete saved Wiki24 predictions. It diagnoses previously selected models and performs no training. Original paper scores remain unchanged.

Each of four members predicts one of ten classes for every node. Mean member accuracy describes the individual predictors. Correct-member coverage counts nodes where at least one member predicts the true class. It is an oracle diagnostic, since the true answer is unavailable at inference. Served accuracy uses the actual saved mean of class probabilities and its chosen class.

The comparison uses all 5,274 development nodes for each of three fixed seeds. Degree counts unique incoming nonself neighbours in the complete 11,701-node graph. Its four boundaries were calculated from 580 TRAIN nodes before prediction values were loaded. The upper boundaries are 4, 11 and 44. The same nodes are paired across both model banks.

## Complete aggregate profile

All differences below are ordinary ensemble minus unit+contrast. Values are descriptive means over the same three seeds, in percentage points. The complete 231 cohort/class rows and all seed outcomes remain in PROFILE.json and ALL_FIXED_COHORTS.csv.

| Population | Nodes per seed | Mean member accuracy difference | Correct-member coverage difference | Served accuracy difference |
| --- | ---: | ---: | ---: | ---: |
| All development nodes | 5274 | -0.142 | +3.938 | +0.411 |
| Degree ≤ 4 | 1360 | -0.368 | +3.652 | +0.294 |
| Degree 5–11 | 1203 | -0.284 | +5.015 | +0.194 |
| Degree 12–44 | 1351 | +0.160 | +4.787 | +0.518 |
| Degree > 44 | 1360 | -0.092 | +2.426 | +0.613 |
| No incoming nonself neighbour | 147 | -2.154 | +4.762 | -0.680 |
| At least one incoming nonself neighbour | 5127 | -0.085 | +3.914 | +0.442 |

The ordinary ensemble has greater correct-member coverage in every degree bin for every seed. Its serving advantage is positive in every seed in the two upper degree bins. The lower bins have mixed serving differences. On the 147 isolated nodes the serving differences are 0, −5 and +2 correct nodes. The deficit is therefore not confined to nodes without neighbours.

Individual competence and coverage are different properties. The unit+contrast members have slightly higher mean accuracy overall, but their correct decisions overlap more. Replacing their serving rule with a convex reweighting cannot correct a node where the same wrong class is strictly above truth in every member. This is the existing fixed-prediction limitation, not a new theorem about retrained models.

## Per-seed serving repairs and harms

A repair means the ordinary pool is correct on a node where unit+contrast is wrong. A harm means the reverse. Both counts are retained.

| Population | Repairs by seed 6101 / 6203 / 6307 | Harms by seed 6101 / 6203 / 6307 |
| --- | --- | --- |
| All development nodes | 123 / 131 / 119 | 99 / 121 / 88 |
| Degree ≤ 4 | 27 / 31 / 30 | 25 / 33 / 18 |
| Degree 5–11 | 31 / 36 / 32 | 23 / 39 / 30 |
| Degree 12–44 | 37 / 38 / 37 | 30 / 33 / 28 |
| Degree > 44 | 28 / 26 / 20 | 21 / 16 / 12 |
| No incoming nonself neighbour | 6 / 2 / 3 | 6 / 7 / 1 |
| At least one incoming nonself neighbour | 117 / 129 / 116 | 93 / 114 / 87 |

## What follows for experiments

The observations support testing acquisition of useful different predictions across the graph. They do not identify degree as the cause of the errors. A specialization rule that allocates training emphasis by degree remains an untested hypothesis. A profile spread across all degree bins does not refute that rule, but it supplies no specific reason to prioritise it.

The registered Context9 and published-loss comparisons will test changes to learned representations. Their full-family gates and required capable references remain unchanged. No new degree weights, bins, model selection or additional fit was chosen using this profile.

## Limits

Both banks were selected using this development population. They are different complete learners, with different sharing, objectives and checkpoint selection. The result does not isolate an effect of sharing. Degree groups also have different class composition. All ten class tables are retained, including empty rows. One graph, one selected split and three correlated seed blocks support descriptive associations. No significance, causal effect, new quality improvement, independent confirmation or general superiority is established.

The analysis authenticated six complete prediction archives and all original role/closure metadata. It reconciled every degree, zero-neighbour and class partition with the full counts. Execution used 2.04 seconds and 62.8 MB peak RSS on the authorized allocation CPU, with zero model calls, optimizer updates or checkpoint loading.
