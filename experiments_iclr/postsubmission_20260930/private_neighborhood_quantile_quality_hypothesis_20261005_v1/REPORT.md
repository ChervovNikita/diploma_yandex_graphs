# Literature synthesis into two concrete quality mechanisms

5 October 2026. Source proposal only. No scientific execution, server access, dataset inspection, new outcome access, canonical edit or acceptance claim. Original scores remain unchanged. The retained v63 conclusions were consulted before the two targeted primary scopes below. No full-paper reading or global count update is claimed.

## The main learning-rule hypothesis

The useful synthesis is a **shared encoder trained for the predictions that its private learners will actually serve after learning**. The currently proposed rule draws TRAIN outer link queries and excludes their endpoints from the current private inner-label batches. It makes a virtual private update, differentiates the competence-anchored ensemble loss through that update to change the shared core, then recomputes and commits each private update at the new shared state. Final inference uses those committed weights without adaptation. This is the exact retained proposal; this note admits no implementation or fit.

| Retained work | Established ingredient | What the proposed recipe still has to demonstrate |
|---|---|---|
| BatchEnsemble and TabM | Shared matrices, private factors, deliberate initialization and member supervision. | Sharing supports better corrections than both a competent single and independent members. |
| Learner Collusion; ONE; CAMERO | Collective ensemble objectives, member anchors, shared branches and collaborative/consistency training. | The improvement comes from learning credit through private adaptation, rather than another mixture of established losses. |
| ANIL and BMAML | Head adaptation, differentiated shared features and shared-feature/private-classifier meta-ensembles. | Shared-backbone ensemble meta-learning itself is **direct prior**. The exact persistent update and graph supervision recipe must earn utility. |
| Meta-Graph | Link-prediction adaptation and outer held-edge learning across related graphs. | Endpoint-conditioned episodes within one correlated graph are useful; a multiple-graph justification cannot be borrowed. |
| MLDG and MetaReg | Training-only transfer optimization and deployment without adaptation. | No inference adaptation is **already established**, and is not the new principle. |
| GMNN and the retained graph routing papers | Graph-dependent supervision and context-sensitive prediction are established. | The endpoint exclusion changes useful learning transfer beyond generic random query splitting. |

The potentially different mechanism is narrow: **different persistent private learners provide learning trajectories, and shared features receive credit for how those learners subsequently predict on queries whose endpoint labels were excluded from the current private update**. The live derivative includes the mixed inner-loss term; a detached-update control removes it while retaining the paid schedule. Endpoint exclusion is temporary: past states and graph context can still contain information about the outer endpoints. This is a conditional regularizer, not unbiased cross-fitting, an independence theorem or a new meta-learning operator.

The decisive controls remain the adapted capable single, the same-rule ensemble with four independent encoders, the detached-update shared ensemble, and matched random query separation/support masking. Better inner loss, a useful toy derivative or increased disagreement cannot substitute for replicated served quality. Private recomputation is indispensable: otherwise training scores virtually adapted heads but deploys a different persistent state.

## One genuinely separate graph quality hypothesis

Retain, at most, **private supervised corrections from the distribution of neighbor messages** as a separate utility proposal. It changes the served information interface rather than endpoint multiplication or the optimizer. It does not use an inner/outer update, a diversity penalty, a gate, conditional edge-pattern likelihood, return-message memory or initialization selection.

### Changed prediction operation

Use one source-qualified modern graph encoder and retain its complete native root/neighbor conventions. Let `h_i` be its shared final node state, and `b_m(h_i)` the existing private nonlinear classification route. For each of the four routes, learn a unit direction `u_m` and project the **observed nonself neighbors' shared states** into scalars `u_m^T h_j`. On every complete neighborhood, compute the fixed 0.25, 0.5 and 0.75 interpolated order quantiles, subtract their neighborhood mean, and append those three values to that route's classifier input. Add the same `log(1+degree)` input to every matched arm. The classifier receives `[h_i, centered_quantiles_m, log_degree]`. Its existing nonlinear head is enlarged only at this declared input; all dense layers that were previously shared remain shared, with the same private factors.

Quantile interpolation is fixed: sort `n` scalar messages and linearly interpolate at position `(n-1)q`. For an empty neighborhood set the three centered values to zero and degree to zero. For one neighbor the centered values are all zero. Self evidence remains in the native root representation, not an artificially added quantile observation. Direction normalization needs an explicit nonzero initialization and norm safeguard. Positions and neighbor weights are **not learned**. Exact unit-norm handling, tied-value subgradients and the enlarged-head initializer must be source-qualified before any later numerical test.

Train with the unchanged mean of member task losses and serve the unchanged mean raw logits. Labels therefore supervise corrections to the actual predictions. No reward is assigned for arbitrary embedding separation. An architecture that already averages probabilities needs a separate declared pooling contract; it must not be quietly combined with this raw-logit recipe.

### Why graph dependence matters

A private transform after a neighborhood mean cannot recover distinctions already discarded by that mean. Different projected neighborhood shapes can carry information about mixed local populations. This is a **local interface** motivation; a learned encoder, attention, nonlinear message function or multi-statistic single can already recover such information elsewhere.

The finite-sample quality hypothesis is that one representation supporting four separately supervised distributional correctors can limit encoder-specific overfitting while retaining nonlinear local corrections. Independent encoders can represent the same shared solution and may learn better representations; the capable single can also represent the entire pool. Any advantage must therefore be measured as learning/regularization bias under the declared protocol, not universal capacity or superiority. Member supervision is an attributed competence anchor, not a new objective or a guarantee of complementary mistakes.

For a fixed scalar message interface, the two eight-message neighborhoods

`{-2,-1,-1,0,0,1,1,2}` and `{-2,-sqrt(2),0,0,0,0,sqrt(2),2}`

have the same mean, standard deviation, extrema and size, but different lower/upper quartiles. That algebra identifies information a fixed mean/min/max/std summary loses. It is not an empirical experiment, a whole-network expressivity separation or evidence that real target labels need the distinction. Three quantiles are also noninjective: many distributions remain indistinguishable. Degree-one/constant neighborhoods provide an explicit inactive case. Shared encoder smoothing can erase the necessary variation before this new interface, and all routes can still learn redundant directions.

### Closest prior resolves the operator claim

Two concrete overlaps were checked after consulting v63:

- **PNA**, arXiv:2004.05718v1, Section 2 and the architecture introduction, already combines mean, min, max, standard deviation, higher moments and degree scalers. Neighborhood-distribution information and multi-statistic message aggregation are prior.
- **Fourier Sliced-Wasserstein Embedding**, arXiv:2504.02544v1, Sections 2.2–3 and Appendix A.1 passages, already projects multiset distributions and embeds their quantile functions. It explicitly discusses graph neighborhood aggregation, cardinality and isolates. Projected quantiles, their graph use and Fourier alternatives are prior. The authors also note discontinuities when quantile positions or distribution weights vary; this proposal fixes both and uses a fixed-cardinality order statistic during feature differentiation. It does not inherit FSW injectivity, Wasserstein preservation or arbitrary-measure guarantees.

Learned aggregation functions and earlier sliced-Wasserstein embeddings were discovered at metadata/cited-lead scope only. They remain unresolved nearest-prior leads, not new reading credit. The private factors and member supervision come from BatchEnsemble/TabM. Thus **the proposal does not establish a new aggregation or ensemble principle**. Its remaining empirical question is whether deliberately allocating compact, separately supervised distribution signatures to private routes gives useful corrections under shared features. Composition alone does not establish methodological novelty. A positive result would first establish a useful sharing/learning bias; a paper-level novelty claim would still need a closer complete-recipe assessment.

### Indispensable quality controls and falsification

One later full-task, prospectively frozen comparison needs:

1. A competent source-native modern single and ordinary independent four-model ensemble, trained on all allowed TRAIN labels with complete native schedules.
2. Unmodified shared four-route GNNM, with the same extra degree input, loss scale and selector.
3. The proposed private-signature four-route model, and a four-route model receiving one common learned signature. The latter tests privacy versus a generic richer input.
4. A **capable single receiving all four directions' twelve centered quantiles**, degree and shared node states together, with a competent joint nonlinear readout and the same paid schedule. Restricting it to one direction would be an inadequate control. It can represent the candidate's pooled prediction; there is no whole-model function-class superiority.
5. Four independent encoders/routes with the **identical signature operation**, supervision, initialization opportunities and eligible schedule. This isolates sharing from a generally helpful architecture.
6. A competent source-qualified PNA/multi-statistic single or an exact matched statistic-rich adapter. If that established single matches the quality, the result supports ordinary distribution-aware aggregation rather than a new ensemble contribution.

The falsifiable prediction is a replicated improvement in complete VALID classification quality over both the competent native single and ordinary independent ensemble, plus an improvement over the common-signature, capable all-signature single and same-operation untied controls. Prediction quality, not signature dispersion, is the decision variable. If the untied version matches or exceeds the shared version, retain architecture utility only. If the all-signature single matches it, do not claim ensemble necessity. If common signatures match private ones, discard the privacy explanation. No favorable degree or heterophily subgroup rescues a failed complete comparison.

Use a complete, previously unconsumed representative node task chosen before outcomes, with several paired seeds/splits and native full schedules. Root owns benchmark choice; this packet selects no dataset by test performance and admits no training. The earlier five-dataset scores are not rerun. A small label set or a truncated neighborhood screen cannot establish this quality claim. Confirmation must use a fresh task/split and frozen choices.

Computing four projections costs proportional to `4*N*d`; sorting all four neighbor lists costs proportional to `4*sum_i degree_i*log(degree_i)`, plus sparse gathers, state storage, gradient work and every enlarged head. Sharing does not eliminate this arithmetic. Segmented sorting, ties, complete high-degree neighborhoods, full backward and peak memory require qualification. No runtime saving or GPU-hour measurement is claimed. Cost is a scheduling question, not a theoretical rejection of the hypothesis.

## Disposition

Prioritize qualification of the existing committed private-adaptation rule as the main learning hypothesis. Retain this distributional correction proposal as one separate, explicitly attributed utility avenue; it is not cleared methodological novelty. The endpoint-frame result is not used as a subgroup rescue or a reason to predict this new method succeeds. Source proposals and literature synthesis are work toward the goal, not evidence that the goal has been achieved.
