# The two ingredients did not add together

For each seed, subtract the separate geometry and private-correction AUC gains from the combined gain. This describes interaction on the AUC scale; it is not a new model score.

| Seed | Geometry gain | Correction gain | Combined gain | Interaction |
| --- | ---: | ---: | ---: | ---: |
| 101 | +8.28 | +4.95 | +5.57 | -7.66 |
| 202 | -0.95 | +2.49 | -1.39 | -2.93 |
| 303 | +2.86 | +4.51 | +0.33 | -7.03 |

The average interaction is -5.87 AUC points (sample SD 2.57). All three observed interactions are negative. The approximate paired t interval is [-12.25, 0.50]; three selected-validation seeds are insufficient to confirm a negative population interaction.

The original motivation to combine improvements was reasonable, but this particular combination does not deliver additive gains. We should retain its complete results and ingredient controls. The result does not justify increasing the loss coefficient or adding more perturbations without a specific explanation.

Possible interference between representation supervision and private corrections remains a hypothesis. This analysis does not measure gradients or establish why the effects combine poorly. It also does not establish that other combinations, backbones or graph tasks will behave the same way.

No predictions were recalculated, no training ran, and no test result was inspected.
