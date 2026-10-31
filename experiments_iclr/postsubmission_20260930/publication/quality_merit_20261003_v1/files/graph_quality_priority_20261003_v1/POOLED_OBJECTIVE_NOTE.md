# Member losses and mean-logit prediction

This is a local mathematical analysis, not an experiment or a novelty claim. It specifies one reason that different member embeddings or logits need not improve the prediction the ensemble serves. The literature followup must compare these observations with prior ensemble loss and splitting methods.

## Exact classification identity

At one labeled node, member `m` produces a class-logit vector `z_m`. The served vector is `z_bar = mean_m z_m`, and its class probabilities are `p_g = softmax(z_bar)`. Let `A(z) = logsumexp(z)` and `L(z,y) = A(z) - z_y`.

Then

    mean_m L(z_m,y) = L(z_bar,y) + D,
    D = mean_m A(z_m) - A(z_bar)
      = mean_m KL(p_g || softmax(z_m)) >= 0.

The identity follows because the target-logit term is linear, and the remaining difference is the Jensen gap of `A`. The KL orientation is from the mean-logit distribution to each member. `p_g` is a normalized geometric pool, not the arithmetic mean of member probabilities.

For binary link logits, replace `A` by `softplus`. The same identity holds with Bernoulli distributions `sigmoid(z_bar)` and `sigmoid(z_m)`. The active BUDDY source averages BCE over the member logits during factorized training and averages raw logits for prediction (`buddy_shared_cache_execution_v5/run.py`, training lines 122–123 and prediction line 56).

Thus the member objective includes a consensus penalty in addition to the loss of the served mean-logit prediction. This explains why adding a hidden-embedding separation loss can fight an existing pressure toward agreement. It does not establish that removing this penalty helps, that the penalty always causes collapse, or that embedding spread corresponds to different decisions. Prior ensemble objectives and SEA must be attributed before proposing any change.

## Small-contrast behavior

Write `z_m = z_bar + c_m`, with `mean_m c_m = 0`. For small contrasts,

    D = 0.5 * mean_m c_m^T H_A(z_bar) c_m + O(max_m ||c_m||^3),
    H_A(z) = diag(p) - p p^T.

This is a local expansion. It describes class-sensitive logit disagreement. Common shifts of all class logits at one node have zero penalty and do not change its prediction. A norm of uncentered embeddings or logits can therefore count differences that have no decision effect.

## Mean trajectories under CE

For a fixed affine logit model `z(phi) = J phi + b`, equal full-batch CE updates of separately parameterized members give

    mean(phi_next) = mean(phi) - eta J_T^T (mean_m p_m - Y),

with the declared TRAIN loss normalization. A single model at the member mean instead uses `softmax(mean_m z_m)` in this gradient. The two differ because `mean_m softmax(z_m)` generally differs from `softmax(mean_m z_m)`. The difference starts at second order for small centered contrasts. This mechanism exists even with an affine predictor, unlike the exact unchanged-mean result for squared loss and a common affine update.

Its sign has no general quality guarantee. For an illustrative binary positive example, two margins `z +/- delta` that both remain positive lie in the concave part of sigmoid. Their mean sigmoid is smaller than sigmoid at `z`, so the mean member update increases the positive margin more than the single-model update. If both margins remain negative, sigmoid is convex and the mean member update corrects the negative margin less. These examples describe a loss mechanism. They are not measurements on a graph, do not satisfy every projected-slice constraint by construction, and do not establish generalization.

Actual GNNM also has shared trainable weights, changing Jacobians and optimizer moments. The affine formula is a diagnostic limit, not its complete dynamics. The relevant empirical question remains whether the graph intervention changes later pooled decisions usefully against common descent, matched perturbations and altered topology.

## Evidence that can support the mechanism

After a complete registered cohort is audited, retained member logits can describe member CE/BCE, pooled CE/BCE, the exact Jensen gap, class-centered contrasts and decision disagreement. These are descriptive analyses of all arms, with no new checkpoint or hyperparameter selection. They must not replace official benchmark scores or become a post hoc quality gate. A separate frozen experiment is required to claim that changing the objective improves quality.
