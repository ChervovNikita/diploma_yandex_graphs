# Representative studies: 8 October continuation

The objective is better graph predictions from four learned predictors sharing a backbone. A loss decrease, different embeddings or a successful short execution does not establish this. The final pooled prediction must improve while preserving useful individual predictors.

The completed Wiki24 error profile supplies a concrete target: compared with the ordinary ensemble, the contrastive routes lack correct-member coverage throughout the graph's degree groups. Merely changing how their unchanged predictions are averaged cannot fix nodes on which every member ranks the same wrong class above the truth. New training must acquire useful correct decisions and limit newly introduced errors.

## What the running studies test

| Study | Real task and complete training budget | Question it can answer |
| --- | --- | --- |
| Context9 | Full 11,701-node WikiCS graph; three target assignments at three fixed seeds; 1,100 epochs each | Do route-specific graph-derived same-class targets improve predictions over common and shuffled target controls? |
| Wiki12 | Same full graph; plain, alignment, residual and combined objectives at three fixed seeds; 1,100 epochs each | Which part of the earlier contrastive package changes competence and complementary predictions? |
| SupCon3 | Same full graph; canonical supervised contrastive loss at the same three seeds; 1,100 epochs each | Does a published loss with a different denominator explain the observed behavior? |
| Mol18 | Full official ogbg-molhiv scaffold task; 32,901 TRAIN and 4,113 development graphs; six configurations at three seeds; 100 epochs each | Does the fixed private-factor training rule improve molecular graph prediction over capable single, independent and sharing controls? |

All seeded fits and failures remain in each roster. The selected-development comparisons are exploration, not independent test confirmation. Whole-family closure precedes opening comparative outcomes. A supported method still requires capable ordinary and objective-matched controls, unused confirmation and an honest prior-work comparison. Three seeds on one split do not establish broad graph generalization.

## A scheduling failure to preserve and resolve

At 06:26 UTC, Context9 had seven of nine completed fits. Mol18 had five of eighteen, with its current fit at epoch 33/100. SupCon3 had one completed fit; another was at 998/1,100 and the third was queued.

Wiki12's original controller terminated with ten of twelve completed fits. Its 1,800-second fresh-memory wait expired before starting seed6307 residual-only; neither remaining output directory exists. No scientific child ran for either remaining cell. The old failed family closure and wait cost are preserved. A separate continuation is being prepared to run exactly those two unused releases after SupCon3 terminates, with the same scientific source, graph, optimizer, selector, epochs and original fit resource limits. It must report an explicit union closure rather than rewrite the original failure.

No accuracy improvement is inferred from these progress counts.

## Theory sharpens a hypothesis, but supplies no result

The frozen symmetric alignment loss responds to symmetrized target differences. At identical route paths, the route-specific differences cancel in the shared derivative. Under an explicitly idealized identical, independent path law, route-specific targets add a nonnegative instantaneous shared-gradient covariance term. Neither statement proves an advantage along trained Adam trajectories.

A separate local calculation shows why targets can change the cost of within-class representation variation while having zero first derivative at ideal class prototypes. Different hidden geometry may still leave predictions unchanged. This reinforces the requirement to measure acquired correct predictions, served repairs, introduced errors and member accuracy rather than use embedding separation as the success metric.

A TRAIN-only read of the original target bank confirms that its operative symmetrized targets actually differ. It measures target operators, not learned gradients, noise or predictive quality. The proposed common-shared/route-private training rule remains an inactive prior specification; it was not newly discovered or launched here.

A stronger eight-view single comparator is being assessed separately. It should preserve training views and supervision while removing learned private predictors, so a later comparison can distinguish extra training information from useful persistent routes. Its feasibility assessment does not change the frozen current studies.
