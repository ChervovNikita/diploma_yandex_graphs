# Completed MolBACE: convergence interpretation

**The fits met their validation stopping rule; they did not reach an observed TRAIN-loss plateau. The available evidence does not support a blanket increase in epochs.**

Root's completed metadata extraction covers the exact **36 unique fits and 5,517 epochs**, with all logged selector decisions replayed exactly. M1/body0 is counted once within the 12 native bodies; the other 24 fits are the shared/private-BN factorial. No partial runs, models, tensors or TEST data were read.

## Stopping and late trends

All 36 stopped by patience at **149–210 epochs**, with **zero 1,000-epoch cap hits**. Selected epochs were 2–160 (median 39). The frozen controller starts counting non-improvements at epoch 100 and stops after 50, so epoch 149 is its earliest patience stop. Thirty fits ended there.

There were **zero strict new VALID-AUC records in the final 50 epochs**. This is expected from the stopping rule; exact replay verifies its enforcement, rather than independently proving convergence. One fit tied its earlier best in that window. Every final-epoch AUC was below its selected checkpoint's AUC.

Comparing each fit with itself:

- All 36 had lower mean TRAIN update loss in the last 25 epochs than in the preceding 25: **7.73–16.08% lower**, median **10.78%**. All also had lower final-50 than preceding-50 TRAIN loss.
- Final-50 mean VALID AUC increased in **16/36** fits versus the preceding 50. Within the final 50, the latter 25 improved in **22/36**. Validation therefore was not uniformly flat or declining.
- Final-50 mean native VALID BCE was **higher in 36/36**, and none set a new BCE minimum in that window. This is a window comparison, not a claim of monotonic deterioration.

| Fit group | Fits with higher final-50 mean AUC | Median AUC change, percentage points |
| --- | ---: | ---: |
| Native bodies | 8/12 | +0.18 |
| Shared-BN factorial | 3/12 | −2.15 |
| Private-BN controls | 2/3 | +0.42 |
| Private-BN factorial additions | 3/9 | −0.67 |

For example, shared-BN base GNNM at seed 202 recovered **4.32 AUC points** in the window mean, yet its last-50 best remained **0.26 points below** its selected epoch 101; BCE worsened by 0.58. Native body 1 at seed 101 recovered 2.17 points and tied its earlier best, which correctly did not reset strict-improvement patience. Late recovery should not be mistaken for a new selected result.

## What this says about undertraining

The epoch cap did not truncate these fits. Continuing TRAIN-loss reduction means optimization stationarity is unestablished, but it does not demonstrate that more training would improve the selection metric. The runtime's `convergence_established` flag means only `stop_reason == patience`.

TRAIN losses are noisy, non-graph-weighted minibatch update objectives with training-mode dropout/BN; objectives also differ between methods. No fixed-model full-TRAIN evaluation or gradient-stationarity test was recorded. These measurements support within-fit trends, not cross-method optimization comparisons.

Additional epochs could conceivably produce a later VALID rebound; no continuation was observed, so that possibility cannot be excluded. The current evidence supports retaining the frozen stopping outcomes and describing them as **validation early stopping**, without claiming complete optimization convergence or a demonstrated need for extra epochs.
