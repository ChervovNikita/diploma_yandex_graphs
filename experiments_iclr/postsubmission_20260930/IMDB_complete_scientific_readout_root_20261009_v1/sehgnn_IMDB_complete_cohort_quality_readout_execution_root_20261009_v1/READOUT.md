# Complete closed IMDB pilot readout

Practical advancement decision: **do not advance under the frozen joint rules**.

This is three paired repetitions on one graph, with VALID reused for selection. It is not a confirmatory, novelty, equivalence or acceptance verdict.

## Paired micro-F1 contrasts

| Comparator | Pair deltas (percentage points) | Mean | SD | Range | Leave-one-pair-out means |
|---|---|---:|---:|---|---|
| shared_own_only | +0.000, +0.000, +0.000 | +0.000 | 0.000 | +0.000 to +0.000 | +0.000, +0.000, +0.000 |
| native_pool_credit | +0.000, +0.000, +0.000 | +0.000 | 0.000 | +0.000 to +0.000 | +0.000, +0.000, +0.000 |
| source_view_supervision | +0.000, -0.062, +0.000 | -0.021 | 0.036 | -0.062 to +0.000 | -0.031, +0.000, -0.031 |
| uncoupled_source_contrast | +0.000, +0.000, +0.000 | +0.000 | 0.000 | +0.000 to +0.000 | +0.000, +0.000, +0.000 |
| COMMON_cycle | +0.000, +0.000, +0.000 | +0.000 | 0.000 | +0.000 to +0.000 | +0.000, +0.000, +0.000 |
| plain_native | -1.390, -1.598, -0.465 | -1.151 | 0.603 | -1.598 to -0.465 | -1.031, -0.928, -1.494 |
| untied_same_six_factors | -1.228, -1.851, -0.405 | -1.161 | 0.726 | -1.851 to -0.405 | -1.128, -0.816, -1.540 |
| matched_native_single | +0.654, -1.318, -0.648 | -0.437 | 1.003 | -1.318 to +0.654 | -0.983, +0.003, -0.332 |

## Frozen checks

- **fail** plain_native:mean_micro_gain
- **fail** plain_native:minimum_pair
- **fail** plain_native:positive_pairs
- **fail** untied_same_six_factors:mean_micro_gain
- **fail** untied_same_six_factors:minimum_pair
- **fail** untied_same_six_factors:positive_pairs
- **fail** matched_native_single:mean_micro_gain
- **fail** matched_native_single:minimum_pair
- **fail** matched_native_single:positive_pairs
- **fail** plain_native:mean_BCE
- **pass** plain_native:mean_macro
- **pass** plain_native:label0
- **pass** plain_native:label1
- **fail** plain_native:label2
- **fail** plain_native:label3
- **pass** plain_native:label4
- **fail** untied_same_six_factors:mean_BCE
- **fail** untied_same_six_factors:mean_macro
- **pass** untied_same_six_factors:label0
- **pass** untied_same_six_factors:label1
- **pass** untied_same_six_factors:label2
- **fail** untied_same_six_factors:label3
- **pass** untied_same_six_factors:label4
- **pass** all_actual_candidate_members_vs_matched_native
- **pass** candidate_member_mean_vs_own_only
- **pass** candidate_worst_mean_vs_own_only
- **pass** candidate_worst_minimum_pair_vs_own_only

## Primary supporting changes

| Independent reference | Mean BCE change (lower is better) | Mean macro-F1 change (pp) | Five mean label-F1 changes (pp) |
|---|---:|---:|---|
| plain_native | +0.007432 | -0.952 | +0.862, +1.433, -2.118, -4.652, -0.283 |
| untied_same_six_factors | +0.006753 | -1.014 | +0.577, +1.459, -1.899, -4.934, -0.272 |

## Every candidate member

| Pair | Member | Micro-F1 | Matched native floor | Meets frozen floor |
|---|---:|---:|---:|---|
| 1 | 0 | 0.68582 | 0.65051 | True |
| 1 | 1 | 0.68705 | 0.65051 | True |
| 1 | 2 | 0.68525 | 0.65051 | True |
| 1 | 3 | 0.68643 | 0.65051 | True |
| 2 | 0 | 0.66480 | 0.64736 | True |
| 2 | 1 | 0.66542 | 0.64736 | True |
| 2 | 2 | 0.66480 | 0.64736 | True |
| 2 | 3 | 0.66419 | 0.64736 | True |
| 3 | 0 | 0.68887 | 0.66596 | True |
| 3 | 1 | 0.68832 | 0.66596 | True |
| 3 | 2 | 0.68947 | 0.66596 | True |
| 3 | 3 | 0.69008 | 0.66596 | True |

Candidate-minus-own-only member mean deltas (pp): +0.000, +0.000, +0.000.
Candidate-minus-own-only worst-member deltas (pp): +0.000, +0.000, +0.000.

## All five source-native results

| Source seed | Fresh VALID micro-F1 | Macro-F1 | BCE | Selected epoch |
|---|---:|---:|---:|---:|
| 1 | 0.68051 | 0.65787 | 0.532078 | 21 |
| 2 | 0.67736 | 0.64521 | 0.542936 | 31 |
| 3 | 0.69596 | 0.67189 | 0.530620 | 21 |
| 4 | 0.69694 | 0.68096 | 0.544828 | 33 |
| 5 | 0.70793 | 0.68295 | 0.523295 | 21 |

Every arm, member, label, signed U/D cell and repair/harm stratum is in the accompanying tables. Supporting macro/per-label/BCE gates apply to both independent references. Native singles have the separate micro rule and member floor; fresh native per-label predictions are unavailable.

Costs are reported at their original inclusive scopes; nested entry/family/cell timings are not summed. All negative results are retained. Control contrasts describe unresolved mechanisms; a matched untied J counterpart and stronger comparators remain required.
