# Independent challenge: source overlap in shared graph supervision

**Conclusion: no third intervention is presently defensible for promotion as a distinct quality method.** I assessed graph source overlap weighting, an operation different from the active learned-view and cavity proposals. Its statistical motivation is conditional, it can change the prediction target, and sharing supplies no established advantage over independent4 receiving the identical intervention. This is a scientific recommendation about the current candidate set, not a claim that a shared ensemble cannot outperform independent4.

## Concrete error source and exact candidate

Many labeled receivers reuse the same hubs or intermediate nodes. If an erroneous source representation reaches many receivers, averaging their training signals need not average away its error. Four shared members can additionally inherit the same source error. This is a graph-specific source of correlated evidence; counting four heads as four independent observations would be wrong.

A precise candidate would use a fixed, label-free receiver–source matrix R from the native TRAIN graph and the declared propagation depth, with each row summing to one. For example, R can be the one-hop row-normalized adjacency without self loops. Set K = RRᵀ on labeled receivers, then choose

    w* = argmin_{w ≥ 0, 1ᵀw = 1} wᵀ(K + λI)w,
    L = Σ_v w*_v [¼ Σ_m CE(p_m(v), y_v)].

Use this ordinary loss to train the shared backbone and all four private branches, retain full member trajectories and the existing uniform probability pool. R, λ and w stay fixed; there is no output repulsion, learned router, altered graph view, posterior conditioning or spectral member bank. A single and independent4 can receive exactly the same weights and loss. A split version that weights only the backbone update would be another operation; it should not be silently substituted after observing results.

This is covariance-aware sample weighting, not a new optimization principle. The possible modest distinction is using receiver–source overlap to choose a graph ensemble's supervised risk weights. I found no exact operation match in the preserved scopes I checked, but that does not establish novelty.

## Why the quality argument is currently inadequate

There is a clean *conditional* variance argument. If every receiver estimates the same desired gradient g and has error Σ_u R_vu η_u, with independent, equal-variance source disturbances η_u, then K is its error covariance up to scale. The stated quadratic program reduces the variance of the weighted gradient estimate, with λ modeling independent receiver noise. Sharing could then improve estimation of the common representation while private branches retain different task responses.

Actual classification gradients do not generally estimate one identical g. They depend on labels, features, class frequency and the evolving member state. RRᵀ is overlap, not a measured covariance of label-relevant errors. Hubs can supply valuable repeated evidence. The candidate therefore reweights the empirical prediction task without establishing that the removed repetition is nuisance. If source errors have opposite signs, overlap may even predict the wrong covariance.

An analytic counterexample exposes the risk. Let eight positive TRAIN receivers share one message source and two negative receivers have distinct sources; all visible features are uninformative. With λ approaching zero, the program allocates total weight 1/3 to the eight positives and 2/3 to the two negatives. An intercept predictor then learns positive probability 1/3, whereas uniform node risk has optimum 4/5. It reverses the majority decision. These are constructed quantities, not opened outcome scores. Class normalization or clipping can mitigate this example, but introduces further choices and does not prove that within-class source overlap is nuisance.

Sharing does not add independent labels or graph sources. Independent4 can use the same correction; sufficiently well regularized independent models can also learn the common signal. Any benefit of tying would be a finite-sample or optimization effect that must be demonstrated. General representation-quality benefits of sharing already have CAMERO ancestry. A cheaper model or greater spread would not answer the predictive objective.

## Minimal falsifiable study, if this candidate is reconsidered

Before a fit, require a development-only diagnosis that source overlap predicts *signed, label-relevant residual covariance* after controlling class and degree. Existing source overlap alone is insufficient. Do not use test labels or select an overlap operator from outcome scores.

The smallest complete comparison has six arms: competent native single; ordinary independent4; ordinary shared4; and each of those three with the same fixed overlap weighting. Use the same TRAIN labels, features, topology, native optimizer, stopping/selection allowance and pooling rule. Independent4 plus weighting receives the same R, K, λ and w: it cannot be denied the proposed graph information. Fix one λ and one graph depth before the comparison. One shuffled receiver–source assignment, preserving row profiles and class/degree composition, is a targeted follow-up only if the core result is positive.

The intervention must improve the served pool over both the competent single and ordinary independent4. A sharing-specific claim additionally requires weighted shared4 to beat weighted independent4; improvement over ordinary shared4 alone is inadequate. Report paired pooled accuracy/NLL, own-member competence, and residual covariance within high versus low overlap regions. If gains are reproduced by weighted single or weighted independent4, credit the weighting intervention rather than sharing. If the same effect survives shuffled source identities, abandon the source-error interpretation. If the proposed covariance diagnosis fails, close this candidate before training.

**Recommendation:** do not promote or implement this weighting candidate now. Complete the existing two graph-specific directions and compare them against independent4 with the same information/intervention. The present source memory supports no additional distinct quality mechanism from this assessment.

## Preserved basis and scope

Read literature_memory/index_v56/LITERATURE_INDEX.json and relevant saved conclusions: accuracy_ensemble_quality_prior_gap_20261004_v1/REPORT.md (CAMERO/shared-head quality); graph_specific_error_gap_search_v1/PAPER_CONCLUSIONS.json (AdaGCN, GRAND and stacking); graph_error_specialization_gap_search_v1/PAPER_CONCLUSIONS.json (SEA and persistent shared modulation); graph_factor_distinct_gap_scout_v1/PAPER_CONCLUSIONS.json (GMNN); graph_shared_gradient_routing_20261003_v1 and graph_shared_gradient_gem_closure_20261003_v1 (PCGrad/OGD/GEM ancestry); graph_structural_response_gap_literature_20261003_v1 (structural perturbation and source attribution limits); and graph_conditioned_prediction_disagreement_primary_scout_20261005_v1 (GRE). These are retained scoped conclusions, not new full-paper or author-source audits. No remote work, training, new project outcome read, protocol/audit harness or literature expansion was performed.
