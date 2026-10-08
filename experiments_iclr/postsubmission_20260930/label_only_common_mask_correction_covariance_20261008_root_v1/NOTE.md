# What the common label mask can and cannot provide

## Scope

This is a source-based calculation for the frozen label-only corrector. It adds no empirical result or novelty claim. The first-moment statement is already documented in the core packet. The covariance calculation is the standard finite-population sampling identity applied to this operator. It is not a new learning guarantee.

## Fixed state and one TRAIN query

Condition on the training history, current parameters, current feature-only native capture, and the event that TRAIN node i belongs to the common query set Q. The fresh mask is independent of this history. There are N = 579 other TRAIN anchors, of which t = 290 are visible. They form a uniform sample without replacement. The query's own label is absent in every route.

For route m, let a_mj be the unscaled residual-logit contribution of anchor j to query i. It is zero when j is not a nonself neighbour. Feature-only attention scores and their all-neighbour denominator are fixed before the mask; the label-to-logit operation is linear and bias free. Thus

    delta_m(Q) = (N/t) * sum_{j visible} a_mj.

Repeated neighbour records, if any, are accumulated under the same anchor label indicator; they are not independent sampling units.

## First moment and cross-route covariance

Let A_m = sum_j a_mj and mean_a_m = A_m/N. Then

    E[delta_m(Q) | i in Q, fixed state] = A_m.

Define the finite-population cross covariance

    S_mn = 1/(N-1) * sum_j (a_mj - mean_a_m)(a_nj - mean_a_n)^T.

The common-mask correction covariance is

    Cov(delta_m, delta_n | i in Q, fixed state)
        = N*(N-t)/t * S_mn.

Each visibility indicator has mean p=t/N and variance p*(1-p). Distinct indicators have covariance -p*(1-p)/(N-1). Expanding the weighted sum and collecting the centred terms gives the expression above. The matrix need not be symmetric for m != n; reversing the order gives its transpose.

For the arithmetic mean of residual logits, the covariance is

    Cov(mean_m delta_m) = N*(N-t)/(t*M*M) * sum_{m,n} S_mn.

The common mask supplies correlated stochastic supervision. Different attention or label-to-logit operators can alter the cross-route terms, but their signs or usefulness are not guaranteed.

## Limits relevant to the experiment

The equality concerns a fixed-state residual logit for a masked TRAIN query. It does not equate the training recipe with unmasked inference, make probabilities or gradients unbiased, or prove generalization. Inference uses unscaled permitted TRAIN labels. Development queries never expose their own truth.

Cross entropy is convex in logits. At a fixed native and corrector state, Jensen's inequality gives expected masked own-route CE at least own-route CE of the expected corrected logits. This is a stochastic training penalty, not a proof that the learned corrector improves accuracy. Learned states depend on earlier masks. The served ensemble averages class probabilities; its variance and accuracy do not follow from the arithmetic-logit covariance formula.

## How this informs the next decision

Finish the frozen three-seed whole-population screen. Its primary comparisons against the joint four-head single and untied correctors stay unchanged. If it fails, embedding distance or mask noise cannot rescue the utility claim.

A later explanatory collector could quantify route cross covariance, class-conflict corrections and same-state native repairs on selected states, with all query targets hidden. It would be an additional diagnostic intervention. It should be authorized only if it distinguishes a concrete next hypothesis.

One conditional design is extra route-specific subsampling of the already permitted common visible anchors, with the corresponding inclusion scale. Every query label must still be excluded from every route, and label exposure, variance and competent controls must be matched. This is established stochastic-context practice rather than an established novel method. No such source, run or coefficient search is admitted here.
