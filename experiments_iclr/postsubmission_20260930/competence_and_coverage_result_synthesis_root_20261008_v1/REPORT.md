# The remaining deficit is coverage of correct decisions

Each node receives ten class scores from each of four routes. We use the saved predictions from the complete Wiki24 study. These are development predictions from the checkpoints selected on that same development role. This analysis changes no model, selection rule or original paper score.

Unit factors plus the combined contrastive objective make individual routes stronger. Their mean individual accuracy is slightly above the ordinary ensemble's mean individual accuracy. However, the ordinary ensemble has substantially more nodes on which at least one member is correct. More of those extra correct decisions are lost when probabilities are averaged, but enough survive to make its final accuracy higher.

For the contrastive model, the numbers of nodes with any correct member are 4,315, 4,325 and 4,305 out of 5,274. The ordinary ensemble's pooled prediction is correct on 4,327 nodes in each of these seed blocks. Even an oracle that knows the true label and picks a correct contrastive route whenever one exists therefore remains below the ordinary pooled prediction in every block. The oracle is an analysis bound and cannot be used during inference.

There is a further property of these saved predictions. Every node where all contrastive routes are wrong also has at least one identical wrong class strictly above the true class in all routes. For each such node, every convex mixture of the existing class probabilities keeps that wrong class above the true class. This bounds any convex reweighting of those probabilities. The saved native predictions contain no common-rival/pool-correct coincidence in these three cells. The mathematical bound does not impose a floating-point parity gate.

This result directs the next learning experiment. The routes must acquire useful correct rankings on different nodes while keeping their individual quality. Increasing individual accuracy through the same correction in every route may improve the model without closing the ensemble gap. Arbitrary embedding separation may increase coverage while making the members worse, as the randomized-factor cells already illustrate.

The active context9 study addresses this particular deficit. Every route still receives every training label. Its additional target relations come from different feature and neighbourhood signatures. The common-target and shuffled-target comparisons test whether those assignments help. We will read all nine complete fits together, report repairs and introduced errors, and distinguish a single route beating all previous common rivals from different routes reversing different rivals. The latter alone does not establish a correct member or pooled decision.

A learned combiner that creates new scores from hidden states is outside the fixed-probability bound. It remains a possible next hypothesis. Its benefit would need comparison with a similarly trained combiner over ordinary ensemble states and with a capable single model. The present counts do not establish that such a combiner will work.

These conclusions concern one previously inspected graph and selected development checkpoints. They establish a concrete error pattern for exploration. They do not establish general superiority or methodological novelty. New families, controls and heldout confirmation remain necessary.
