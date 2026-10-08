# Wiki12 union and canonical SupCon exploratory comparison

WikiCS split0 development: official validation union stopping masks;5274 nodes.

Original selected states; the selection population is reused for this analysis. No independent TEST evidence. Three optimizer seeds on one graph; nodes and members are not independent model replicates.

| Contrast | Accuracy mean (pp) | SD | Exploratory df2 95% interval | NLL mean | Brier mean |
|---|---:|---:|---|---:|---:|
| C-P | -0.0695234 | 0.799364 | [-2.05525, 1.91621] | -0.0422798 | -0.00283098 |
| A-P | -0.246492 | 0.465606 | [-1.40312, 0.910137] | -0.0708656 | -0.00341777 |
| R-P | -0.0189609 | 0.373006 | [-0.945558, 0.907636] | 0.0876317 | 0.00300479 |
| C-A | 0.176969 | 0.376523 | [-0.758366, 1.1123] | 0.0285858 | 0.00058679 |
| C-R | -0.0505625 | 0.470725 | [-1.21991, 1.11878] | -0.129912 | -0.00583577 |
| C-A-R+P | 0.19593 | 0.265679 | [-0.464053, 0.855912] | -0.0590459 | -0.00241801 |
| SupCon-A | 0.0505625 | 0.152082 | [-0.32723, 0.428355] | 0.556452 | 0.0101514 |
| SupCon-P | -0.19593 | 0.432515 | [-1.27036, 0.878496] | 0.485586 | 0.00673365 |

Assess fresh C-P before component attribution. These intervals are descriptive and do not establish novelty, confirmation or a graph mechanism. No seed/checkpoint rescue selection.

PER_CELL/PER_SEED contain accuracy, NLL, Brier and all four members plus mean/worst competence. PAIRED_ERRORS contains repairs, harms and common-wrong-competitor flows on plain-frozen Wiki12/SupCon-P cohorts and alignment-frozen SupCon-A cohorts. Raw logits, representations, IDs, truth and masks remain server-side.

SupCon-A is the prospective matched loss contrast; SupCon-P is contextual. Original old-owner failure and costs remain retained. A canonical-loss gain is not a sharing, diversity or novelty claim.
