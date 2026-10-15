# Completed normalization–geometry–correction study

The nine new fits completed the factorial comparison on MolBACE/GIN. Each method uses its own AUC-selected checkpoint, three paired seeds and the same 151-graph scaffold VALID set (130 positive, 21 negative). Fifteen factorial controls and twelve independent native bodies were reused; M1 is body 0, not another fit. TEST remains closed.

## Results

All AUC values are percentages; means use all three seeds. BCE below uses the common graph reduction; native loss scalars remain separately preserved.

| Method with private BN | Seed 101 | Seed 202 | Seed 303 | Mean AUC ± SD | Mean BCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neither ingredient | 74.84 | 73.00 | 73.00 | 73.61 ± 1.06 | 1.086 |
| Geometry | 75.20 | 71.94 | 73.81 | 73.65 ± 1.64 | 1.122 |
| Input correction | 73.37 | 72.56 | 70.62 | 72.19 ± 1.41 | 0.816 |
| Both | 74.87 | 72.27 | 71.94 | 73.03 ± 1.61 | 1.030 |
| Independent ensemble, I4 | 74.40 | 75.38 | 74.76 | 74.85 ± 0.50 | 0.890 |

**The earlier ingredient gains mostly do not survive the matched normalization control.** Without private BN, geometry improved mean AUC by 3.39 points and correction by 3.98. With private BN, geometry changes it by only +0.04 (two seeds improve, one worsens); correction changes it by −1.43 and worsens every seed. Geometry adds +0.84 when correction is already present, but both together still lose 0.59 points to private BN alone. Geometry remains 1.20 points below I4 on average: its paired differences are +0.81, −3.44 and −0.95.

The geometry–correction interaction changes from −5.87 points with shared running state to +0.81 with private state. Private-state interactions are positive in all seeds (+1.14, +0.77, +0.51). This means combining ingredients is better than their additive contrast predicts; it does **not** mean the combination beats neither ingredient, the native single (73.32%), or I4.

## Competence and probability quality

Private BN raises mean individual-route AUC from 62.01% to 72.63%; its averaging gain is 0.99 points, versus 5.77 for base GNNM. Larger averaging gains therefore do not establish better models. Under private BN, geometry lowers mean member AUC to 72.19% while increasing averaging gain to 1.47; correction gives 71.53%/+0.66 and both 71.88%/+1.14. I4 gives 73.41%/+1.44. Geometry's near-tie with the private-BN baseline comes from greater averaging benefit compensating for weaker members.

Correction reduces mean BCE relative to private BN alone, while hurting AUC; the reduction occurs in two seeds, with worse BCE in seed 101. Geometry slightly worsens mean BCE. Private BN's original +5.84-point AUC improvement also worsened mean BCE (0.719→1.086). Checkpoints were selected by AUC, not BCE; these tradeoffs preclude a general prediction-quality or calibration claim.

## Cost and decision

The **nine new fits** used **1,373 complete epochs, 52,174 optimizer updates and 15,365.76 seconds summed invocation wall** (4.27 worker-hours). They stopped by patience after 149–178 epochs, with no cap stops or failed fits. Fifteen reused controls incur zero new training, not zero historical cost. Nine new route collectors add 180 VALID forwards and 60.89 summed seconds, separately from training. Summed worker wall is not study elapsed time; patience is not proof of scientific convergence.

Retain private BN as a known comparator and preserve original GNNM unchanged. These fixed geometry/correction recipes do not establish an improvement over that comparator or I4. Future extensions need to improve competent members without sacrificing probability quality. Three selected-validation seeds, differing selected epochs and repeated examination of one split establish neither significance, novelty, fixed-weight causality nor transfer.
