# Graph support and target orientation in private-correction allocation

Frozen as an internal hypothesis before any held A/VALID/TEST score. The current seven-arm experiment and all original scores remain unchanged. No source, model, new dataset experiment, selection rule or hyperparameter grid is introduced here.

## Observed motivation and unresolved support

The completed training diagnostics show finite response differs from first-order utility and assignments differ from uniform. Aggregate training losses for graph-free, permuted and stop-Q controls are close. Those summaries establish neither pointwise equality nor absence of graph effects. The current protocol uses only direct public edges induced on inner-S nodes of each class pair. Connections through other public nodes are discarded. The actual coverage of these induced graphs must be measured before asserting that the graph prior is weak.

A label-free Dirichlet reduction could retain public paths between supported rows. On components with a support node, minimizing graph energy over unconstrained public-node fields gives the Schur-complement energy on support rows. Components without support provide no responsibility information and require an explicit convention. This is established harmonic graph learning/Kron reduction; the possible contribution is its role in this particular private-correction training map, contingent on comparison with competent controls. No new principle or extra predictor information is claimed.

## A separate target-orientation assumption

For a class pair(a,b), write d_m=z_m(a)-z_m(b), and t_i=+1 for targeta and -1 for targetb. The existing competitor loss is softplus(-t_i*d_m). A positive change of the same canonical contrast d_m decreases the a-target loss and increases the b-target loss by monotonicity. Thus a smooth canonical correction field need not induce a smooth target-specific responsibility field across differently labeled neighbors. This statement alone says nothing about the actual model's graph smoothness or accuracy.

One possible regularizer applies graph energy to h_i=t_i*(Q_i-uniform), while retaining the original Q costs, positivity, row mass, member-column mass, private loss, shared credit and probability-mean serving. With T=diag(t), the quadratic is h^T L h, equivalently(Q-uniform)^T TLT(Q-uniform). T is orthogonal: positive semidefiniteness and spectral norms are preserved. The prior changes because only the energy is transported; the original Q-dependent cost, entropy and feasible constraints are retained. It is not automatically an equivalent variable relabeling.

For a two-row illustration, let the targets bea andb, and assignments beu+v andu-v with sum(v)=0 and both rows positive. Member-column balance holds. Unsigned edge energy is4w||v||², whereas the transported edge energy is0. Opposite target-response costs can prefer this arrangement. This is an exogenous cost/regularizer example, not an executed SGD example, graph experiment or generalization theorem. Other response fields can favor unsigned smoothness and be harmed by the transported prior.

The proposed assumption is that canonical member corrections are smoother on the relevant graph than target-specific usefulness. It must be tested rather than inferred from heterophily, hidden separation or graph degree. The softplus susceptibility, nonlinear member responses, class imbalance, balancing constraints and sparse support can all undermine it. Class-only and graph-permuted controls are necessary to separate label-derived orientation from useful topology.

Endpoint orientation may be applied after a label-free public-path reduction: only the already supervised support labels supply T; unknown public-node labels are unnecessary. Intermediate canonical fields would be real vectors, not deployed probability assignments. Exact positivity/balance, unsupported components, normalization, finite solver steps and complete resource costs need specification before implementation.

## Decision requirements

First measure full ten-pair support and inspect close signed-graph/harmonic/meta-reweighting priors. Then specify one small prospectively fixed comparison if justified. Keep the existing held-score gates and controls. Current supervised labels are permitted; A/R/VALID/TEST labels cannot supply this graph prior. A useful attributed optimization rule need not add a predictor class, but it must improve full served quality beyond competent baselines. Literature search absence and the elementary algebra above do not establish methodological novelty.
