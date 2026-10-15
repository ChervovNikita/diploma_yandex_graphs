# Complete route-specific BatchNorm result

The frozen three-fit follow-up is complete. All three seeds use the same native MolBACE/GIN configuration and controls as the parent pilot. Learned native affine parameters remain shared; each route stores its own running means, variances and counters. This is an established normalization-state control, not a new primitive.

| Model | Selected VALID AUC, mean ± sample SD (%) | Mean BCE |
| --- | ---: | ---: |
| Native single | 73.32 ± 1.17 | 0.779 |
| Independent four | 74.85 ± 0.50 | 0.890 |
| Base GNNM | 67.78 ± 4.39 | 0.719 |
| GNNM with private running statistics | 73.61 ± 1.06 | 1.086 |

AUC improves over base GNNM in all three paired seeds: +12.05, +3.48 and +1.98 points (mean +5.84). It remains below independent four on average by 1.23 points and is better in only one seed. Mean BCE worsens by 0.366 versus base; the third seed has BCE 1.854. This is a ranking improvement with a probability-loss tradeoff, not a general predictive winner.

The candidate completed 149, 149 and 166 epochs, selecting epochs 92, 54 and 116. In total that is 464 whole TRAIN epochs and 17,632 optimizer transitions. These are additional to the parent 24 fits. The validation stopping rule is unchanged. No run hit the 1,000-epoch cap. This does not establish optimization stationarity or cross-dataset robustness.

This pilot motivates preserving private normalization state as a separate controlled method in the broad benchmark. It also narrows the current ranking gap enough to justify measuring its interaction with the already defined private corrections in a future explicitly frozen study. No such combination was run or added to the 6,460-fit roster here. The adverse probability-loss result must remain visible.

Selection and analysis reuse the same 151 validation graphs. These three seeds establish a development observation, not significance, an untouched-test estimate or a novel theoretical contribution. Different selected epochs also prevent attributing the entire score change to a same-weight evaluation intervention. Original paper scores and parent pilot scores are unchanged. All seed rows and authorities are in RESULT.json.
