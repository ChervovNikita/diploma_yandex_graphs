# Aligning structural mixture training with served link prediction

## Status

This is a derived, untested hypothesis, saved separately from the frozen Collab and DDI experiments. It is not a new distribution, theorem, novelty clearance, admitted experiment or predictive result. No paper was read for this note. It reuses the recorded shared-latent analysis and its counterexamples. Existing source, serving rules and continuation decisions are unchanged.

## The unresolved transfer

The current auxiliary lets one member explain the neighbour identities at both endpoints. The served ranker averages raw scores uniformly after removing that auxiliary. Better structural likelihood therefore need not improve the queried-edge score; the saved tied-embedding counterexample proves this limit. A possible next question is whether the learned structural evidence can inform which member predicts an edge. That question is separate from whether the current fixed auxiliary transfers through shared parameters.

## A coherent probabilistic alternative

For the probabilistic thought experiment, let x contain query features and specified supports/counts, but exclude the realized neighbour observations S. Let Z be a uniform latent member, Y a binary link label, and S the pair of endpoint neighbour subsets. Suppose each member supplies a normalized structural law q_m(S|x)=q_Lm(S_L|x)q_Rm(S_R|x), and a Bernoulli probability p_m(x) for the link. Conditional independence of Y and S given Z,x is a modelling assumption, not an established property of graphs.

The joint model is P(Y,S|x)=sum_m P_m(Y|x)q_m(S|x)/M. Its conditional link probability is

    g(S,x) = sum_m rho_m(S,x) p_m(x),
    rho_m(S,x) = q_m(S|x) / sum_j q_j(S|x).

This follows directly from Bayes' rule. Likelihood-derived gating, latent mixtures and mixtures of experts are established operations. A graph-specific implementation or empirical benefit would need its own closest-prior comparison. This note does not establish either.

The probability pool is essential: replacing p_m with uncalibrated native ranking scores, or averaging raw logits and then applying sigmoid, does not preserve the joint model above. The native AUC recipe and its sampled negatives do not provide calibrated population link probabilities. They cannot silently inherit this probabilistic interpretation.

## Exact scope of a possible advantage

Under the correctly specified model, for a fixed x the prior mean p0(x)=sum_m p_m(x)/M equals E[Y|x], and g(S,x)=E[Y|S,x]. The expected Brier-risk difference between those two predictors is exactly E[(g(S,x)-p0(x))^2|x]. This is the standard conditional-expectation decomposition. It is positive only when S changes the conditional label probability. Likewise, the ideal expected log-loss difference is I(Y;S|x), when the probabilities and conditional distributions are correct. These generic identities supply a mechanism and assumptions, not a new theoretical contribution or a guarantee for fitted GNNM.

If x includes the full allowed graph and therefore already determines S, then I(Y;S|x)=0. A deployed GNN already sees this graph; the note does not show that routing provides new information beyond its backbone. In that setting, a possible advantage concerns a restricted learned readout or optimisation, and needs an empirical comparison. The information decomposition cannot certify it.

A finite example makes the condition explicit. Two equally likely members have label probabilities .9 and .1. A binary structural observation identifies the member correctly with probability .9. The prior predictor is .5; after observing S, the conditional probabilities are .82 or .18. Expected Brier risk drops from .25 to .1476, a difference of .1024. If both members instead have probability .5, even perfectly identifiable structural members give no predictive gain. If learned likelihoods identify the wrong member, gating can harm prediction.

## Necessary graph and evaluation boundaries

- All structural observations must come from the allowed TRAIN-visible graph. The queried edge and its reverse must be removed symmetrically for TRAIN and validation candidates. A positive TRAIN edge cannot reveal its own label through the conditioning graph. Native batch-wide masks also create an observable-context shift that must be addressed prospectively, rather than overlooked.
- Candidate support and subset cardinality must be specified at serving. A normalized local subset law is not a normalized whole-graph likelihood. Overlapping query contexts require a composite-likelihood description.
- Degree conditioning removes a common logit offset, not candidate popularity or every degree effect. Graph likelihood can favour degree patterns without improving labels.
- Evaluating complete likelihoods for every ranking candidate can be expensive. The baseline must pay the same structural context work, and all serving costs must be reported. Cached endpoint evidence is invalid when a candidate changes the excluded endpoints/support.
- A separately calibrated conditional label model, or an explicitly empirical score gate, is required. Calibration uses TRAIN/VALID only, with the negative-sampling distribution disclosed. No heldout TEST fitting.
- Posterior concentration alone is not useful specialization. Primary evidence must concern served predictive quality, with uniform pooling, a matched discriminatively learned gate and a structural reference as controls.

## Decision

Do not start a new family now. Complete the already fixed representative comparisons and preserve their failures. If the auxiliary fits structural dependence but fails to improve served prediction, this alignment issue is one scientifically distinct successor hypothesis. Freeze any successor and its controls before scoring. If the relevant prior already implements the same graph-structure likelihood gate, record that and abandon a novelty claim. A successful pilot would support an attributable adaptation only until broader independent confirmation.
