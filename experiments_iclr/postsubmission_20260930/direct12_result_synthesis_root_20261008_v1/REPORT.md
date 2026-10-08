# Complete classifier diagnostic: no useful gain from extra head freedom

8 October 2026. The backbone and its learned node features stayed fixed. We fitted the final prediction layers on all 580 original training nodes and assessed every fit on all 5,274 original development nodes. Three actual seeded model banks and four fixed conditions produced twelve optimization bundles. The unrestricted and factorized BE versions began with the same effective predictions, common bias, feature scaling and training objective. Ordinary ensemble and single-model refits were included as references.

| Prediction rule | Mean development accuracy (%) |
| --- | ---: |
| Factorized BE refit | 80.5714 |
| Unrestricted BE refit | 80.5714 |
| Ordinary ensemble refit | 81.2919 |
| Single-model refit | 80.4766 |
| Untouched unit+contrast | 81.6332 |
| Untouched ordinary ensemble | 82.0440 |
| Untouched single | 81.4752 |

Extra BE head freedom changed the pooled correct count by zero, minus one and plus one node relative to the matched factorized refit. Its mean accuracy gain was zero. Both acquired the same numbers of newly covered native all-wrong nodes (74, 80, 79) and the same common-rival pooled repairs (70, 74, 78). The unrestricted version's small global coverage advantage came from avoiding a few coverage losses. It did not add new correct alternatives on the targeted original errors.

Relative to the untouched native unit+contrast model, unrestricted refits produced 77, 76 and 82 pooled repairs but 131, 168 and 104 harms. All four refit conditions reduced their corresponding native mean accuracy. The fixed refit therefore worsened the practical deficit rather than repairing it.

Six endpoints converged under the declared gradient rule. Six finite endpoints did not meet that rule and remain in the report. The frozen affirmative screen is inconclusive. Regardless of that optimizer limitation, there is no useful measured full-versus-factor predictive signal to justify a private-head extension. No optimizer tightening, penalty grid or broad78 probe is admitted.

This closes this particular regularized frozen-head direction. It does not prove that every head, aggregator or complete network has the same capacity. The features and original checkpoints were selected on this development role, and all evidence comes from one graph/split. No original paper score changed. No end-to-end superiority, independent confirmation, novelty or acceptance follows.

The remaining investigation targets how routes learn and aggregate graph evidence before the final classifier. Existing full context-target, alignment/residual and molecular internal-credit studies continue under their original gates. The private local-attention hypothesis remains conditional and attributed to related work.
