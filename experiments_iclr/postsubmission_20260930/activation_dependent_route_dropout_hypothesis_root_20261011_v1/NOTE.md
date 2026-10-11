# A falsifiable sampling hypothesis for shared graph routes

Status: root mathematical proposal, unimplemented and untested. The new-method researcher is checking closest adaptive antithetic/dependent-rounding/multi-sample-dropout sources. No new primitive or prediction-quality result is asserted.

## Observation and scope

Complete Physics E members are 5.1747 accuracy points weaker than capable independent members. More union coverage does not survive probability pooling. Receiver context explains almost all the E-versus-BASE gain. A new sampling mechanism should improve acquisition, rather than merely increase representation distance or select a favorable heldout pool.

A uniform balanced mask can reduce the noise of equal route values while increasing it for opposite signed values. Private factors and hidden activations make this distinction relevant. This observation does not diagnose the cause of the Physics deficit.

## Proposed conditional mask law

At an existing dropout site, collect four current route activations for each source-node/channel. Treat their values as fixed when drawing masks. Choose one of the three perfect matchings among the four routes to minimize the total pair covariance contribution to variance of their mean.

For a pair with same-sign activation product, draw a fresh U uniformly on [0,1], retain the first route when U<q and the second when 1−U<q. At q=0.8 the mask covariance is −0.04. For an opposite-sign pair, give both routes the same Bernoulli(q) mask: covariance 0.16. Draw independent U for the other pair and for every distinct source-node/channel/site. Scale retained values by 1/q. Matching and sign decisions are detached sampling choices, with no labels, auxiliary loss, router or sampler-gradient claim.

Every pair covariance contribution a_i a_j Cov(mask_i,mask_j) is nonpositive. Consequently the variance of the mean dropped activation is no greater than with independent Bernoulli masks, conditional on these four fixed activations. This is a elementary linear statistic, not a classifier or gradient guarantee.

For a common fixed linear graph aggregation P, independent source-coordinate masks give conditional variance of a pooled message at node v equal to the sum over u of P_vu squared times the corresponding source pooled-activation variance. The local inequality therefore carries through this common aggregation. Attention operators with different route-dependent coefficients are outside this statement; nonlinear layers, whole training dynamics and accuracy are outside it as well.

For fixed parameters, each route retains the original conditional iid Bernoulli(q) mask tensor law, including at later sites when conditioning on full past states. Joint route dependence changes. Shared optimization trajectories and final trained model distributions are not equal to independent-mask training. The ordinary own-CE stochastic gradient should retain its marginal expectation under the stated law; a complete implementation and full-law check remain needed.

## Costs, harms and decisive experiment

A joint layerwise forward must expose all four route states at a dropout site. Peak activation memory may grow relative to sequential route backwards. No efficiency improvement is assumed. Pair common masks can increase all-route blackout probability even while decreasing the chosen linear variance, and lower linear variance can harm beneficial regularization, private exploration or final accuracy. Preserve those adverse outcomes.

The intended comparison is original native shared GNNM with independent masks, uniform balanced masks and activation-sign paired masks, together with capable single and genuine independent models and same-compute multisample controls. Complete graphs and all paired seeds are required. No heldout result or tiny readiness update chooses a version. The primary question is final/member prediction quality, confidence and cost; a lower local variance alone cannot make a successful method.

This proposal must be checked against earlier saved sampling studies and the strongest primary prior methods before admission. It remains distinct from the current native original-Physics reference acquisition and from the inactive Gaussian fast-factor proposal. No literature reading count is added by this note.
