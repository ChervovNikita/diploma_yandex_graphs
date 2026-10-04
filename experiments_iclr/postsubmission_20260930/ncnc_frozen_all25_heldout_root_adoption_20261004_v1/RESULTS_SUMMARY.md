# Complete frozen NCNC heldout comparison

All 25 cells from the frozen 35-fit study completed one official ogbl-collab TEST evaluation. All 40 original scorer calls and 25 official metrics passed, with source/data/runtime/selected-state and loaded-graph/query custody intact. The owned child exited 0; inclusive wrapper time was 98.73 seconds. There were no new training updates, refits, reselections, calibration or retries.

| Model | TEST Hits@50 mean (%) | Seed SD (pp) |
| --- | ---: | ---: |
| Native single (64) | 66.4426 | 0.6233 |
| Independent ensemble (4) | 67.6298 | 0.2267 |
| GNNM, private completion (4) | 67.2909 | 0.7676 |
| GNNM, pooled completion (4) | 67.0673 | 0.2595 |
| Native capacity control (70) | 66.7642 | 0.7862 |

| Private completion minus control | Mean (pp) | Descriptive 95% seed interval (pp) | Positive / negative seeds | Exact two-sided sign-flip p |
| --- | ---: | ---: | ---: | ---: |
| Native single (64) (exploratory) | +0.8483 | [+0.3836, +1.3130] | 5 / 0 | 0.0625 |
| Independent ensemble (4) (exploratory) | -0.3389 | [-1.0838, +0.4061] | 1 / 4 | 0.3125 |
| GNNM, pooled completion (4) (frozen primary) | +0.2236 | [-0.7775, +1.2247] | 3 / 2 | 0.6875 |
| Native capacity control (70) (exploratory) | +0.5267 | [-0.9006, +1.9539] | 2 / 3 | 0.5000 |

The frozen private-versus-pooled contrast is +0.2236 percentage points, with three positive and two negative seeds. Its uncertainty includes zero; the earlier +0.8432-point validation difference is not robustly confirmed on TEST. This result does not establish the proposed architectural advantage.

Private completion exceeds native64 by +0.8483 points in all five seed blocks, an exploratory baseline comparison. The independent ensemble has a higher mean by 0.3389 points; native70 has mixed paired results. These observations support continued quality research but establish neither a novel mechanism nor superiority over independent ensembles or the field.

Intervals are paired t intervals over five training-seed blocks conditional on one graph/time split, assuming approximately normal independent seed effects. The exact sign-flip calculation enumerates all 32 sign assignments and assumes pairwise sign exchangeability under the null. With five seeds its smallest two-sided p-value is 0.0625. Baseline contrasts have no multiple-comparison adjustment and must remain exploratory. These are not confidence intervals over new graphs or splits.

All seed scores and failed-attempt history are preserved. TEST is now consumed for this frozen family. No successor can claim that these TEST values were unseen during its design. The separately prepared training-only pattern/count hypotheses remain hypotheses and require fresh paired fits and additional benchmark evidence. Original paper scores are unchanged.
