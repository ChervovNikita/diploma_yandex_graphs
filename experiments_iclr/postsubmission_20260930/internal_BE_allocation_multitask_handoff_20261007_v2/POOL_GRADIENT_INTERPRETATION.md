# What the actual ensemble loss teaches each path

This is a derivative of the stated loss, not a new learning principle, experiment result or recommendation of a coefficient. It explains one limitation of the proposed GNCL control before running it. GNCL supplies the member/pool-risk mixture. The internal-only gradient allocation is a separate attributed adaptation.

## WikiCS: stronger paths receive more of the pool gradient

For one labelled node, let `q_m` be the probability that path m assigns to the correct class. The served predictor averages the four class-probability vectors. Its supervised loss is `-log(mean_m q_m)`.

Differentiating that loss with respect to path m's class logits gives:

`d L_pool / d z_m = [q_m / sum_j q_j] (p_m - onehot(y))`.

Here `p_m` is path m's complete class-probability vector. The expression in parentheses is the usual cross-entropy derivative. Its multiplier is the path's share of the total correct-class probability.

The mean individual loss instead supplies the multiplier `1/4`. For the GNCL mixture `(1-lambda)L_mean + lambda L_pool`, the multiplier becomes:

`(1-lambda)/4 + lambda q_m / sum_j q_j`.

For example, suppose the four correct-class probabilities are 0.8, 0.1, 0.05 and 0.05. Pure pool supervision supplies multipliers 0.8, 0.1, 0.05 and 0.05. Mean individual supervision supplies 0.25 to each. An illustrative lambda of 0.5 supplies 0.525, 0.175, 0.15 and 0.15. This example does not adopt lambda 0.5.

Thus probability-pool supervision can give a weak path less training credit. Retaining the individual component provides a positive multiplier floor for lambda below 1. This is a coefficient on the logit residual, not a bound on the parameter-gradient norm. It does not guarantee that the path improves, because gradients pass through its private factors and interact with shared updates. The proposed adapter also retains the separate alignment auxiliary. The derivative above concerns only its supervised part.

## Collab and MolHIV: the pool residual is the same for every path

These tasks average raw logits. For one finite binary label y, let `bar_z = mean_m z_m`. The pool loss is binary cross entropy applied to `bar_z`.

Its logit derivative is:

`d L_pool / d z_m = [sigmoid(bar_z) - y]/4`.

Each path receives the same pool residual. The mixed supervised derivative is:

`[(1-lambda)(sigmoid(z_m)-y) + lambda(sigmoid(bar_z)-y)]/4`.

Collab additionally averages this expression within each positive and negative group, then sums the two group losses. MolHIV averages across its batch. These reductions are part of the implemented task contract.

The pool term has no preference between different logit allocations with the same mean. Opposing changes to two paths can therefore leave the pool loss unchanged. Different private parameter Jacobians can still give different parameter updates, but the loss does not explicitly require complementary correct decisions.

## Consequence for the experiment

Actual pool supervision avoids rewarding a classifier-invisible hidden code. It can still neglect weak paths or fail to produce useful complementarity. The new control must therefore be judged by member competence, shared mistakes and final served quality against the matched references. Lower pool training loss alone would not establish a better ensemble or a graph-specific contribution.

Both the coefficient and execution remain unadopted. This note changes no running loss, seed, selection rule or comparison roster.
