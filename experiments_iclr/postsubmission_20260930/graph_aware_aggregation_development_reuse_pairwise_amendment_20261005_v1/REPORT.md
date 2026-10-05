# Development label reuse and pairwise aggregation amendment

5 October 2026. Bounded amendment to the saved graph aggregation memo. The original packet remains intact.

## Correct the validation claim

The original memo made labels reserved before backbone fitting and checkpoint selection a necessary condition for an honest test. That condition is too strict for evaluation of a completely frozen pipeline on genuinely untouched TEST data.

A legitimate development pipeline may train experts on TRAIN, select their checkpoints on VALID, and use VALID to fit or select temperatures and a stacker. Reusing those labels makes the stacker's performance on the same VALID population a biased development estimate. It also prevents describing that population as independent calibration data or describing aggregator-only cross-validation as cross-fitting the complete predictor. Once the entire expert, calibration, fusion and selector pipeline is fixed, genuinely untouched TEST can still evaluate that pipeline under its declared protocol. All development reuse, competing configurations and selection opportunities must be disclosed. The final evaluation does not retroactively make the reused VALID estimate independent.

For the current research, TEST access remains governed by the unchanged frozen contracts. This amendment authorizes no scores, changes no gate, and establishes no unseen TEST custody. A new aggregation rule is a separate estimand; it cannot silently replace the fixed pool in an existing comparison. If all available outcomes have already influenced the rule, fresh confirmation is still required. Graph dependence also means connected queries are not independent replicates.

The nearest saved priors support this distinction. Link-MoE explicitly fits its Collab gate on 80% of official validation and selects on 20%. RBS fits temperatures on validation. GATS' saved Appendix A.3 scope fits calibrators on validation and uses a disclosed selection procedure. These are model-construction roles, rather than guarantees that their development estimates are independent. META-DES uses separate base/meta-training and dynamic-selection reference roles, which offers a more separated design but is not the sole valid route to final pipeline evaluation. These conclusions reuse the authenticated scoped records; no primary payload was reopened. [1–4]

## A pairwise definition for Citeseer HeaRT

Citeseer-HeaRT predicts **links**, so the original node operation `P r_m(v)` cannot be transplanted directly. A member's scalar score for a query pair is not a node-class probability vector. Neighbor summaries over an arbitrarily batched set of candidate queries would also make the result depend on that batch or candidate panel.

An explicit link adaptation can reuse a fixed TRAIN support graph, frozen node states `h_v`, and four logits `z_m(q)` for query `q=(u,v)`. Define `mu_v` as the declared normalized neighbor mean of the frozen node states on that support. Use symmetric pair context:

`phi_H(q) = [h_u * h_v, h_u + h_v, abs(h_u-h_v), mu_u + mu_v, abs(mu_u-mu_v), (h_u-mu_u) * (h_v-mu_v)]`.

Here `*` is elementwise multiplication. Add the declared Link-MoE structural-heuristic features `phi_S(q)` and the four own-query logits/residuals `phi_Z(q)=[z_m(q), z_m(q)-mean_j z_j(q)]`. A small gate maps this fixed pair context to softmax weights `alpha_m(q)`, and serves

`z_fused(q) = sum_m alpha_m(q) z_m(q)`.

Sigmoid is monotone in this score; ranking uses the complete unchanged candidate universe. This operation has no propagation between candidate-query scores and no dependence on which queries share an evaluation batch. The support excludes forbidden heldout target facts; a TRAIN out-of-fold construction must mask its target facts and all disallowed label-derived inputs. Original target-edge/support rules remain authoritative.

**Nearest unchanged aggregation comparator:** Link-MoE's structural-heuristic and endpoint-Hadamard MLP gate, followed by the weighted expert-logit sum, on the same frozen expert bank and the same eligible gate labels. The concrete proposed delta is the added frozen neighbor-state contrast and own-query residual context. The bank substitution is disclosed; it is not a reproduction of Link-MoE's original independently trained heterogeneous experts. A baseline with its same gate architecture given the entire expanded feature bank separates additional evidence from a new aggregation operator. If that baseline matches, use the attributed prior implementation.

A joint single stacker receives every feature above and the same softmax/multiplication primitives, so it can implement the candidate exactly. Give a competent independent bank the same fusion supervision and opportunity. Independently learned hidden coordinates stay separate; their equal width does not justify coordinate averaging. The pair adaptation is an explicit task variant of the same context-conditioned pooling hypothesis, with no second hypothesis or node-MoE reproduction claim.

## Cheap evidence and later compute

Start with authenticated existing predictions when authorized. Mean raw logits, mean probabilities, per-member scalar-temperature averaging and parameter-free complete-query rank fusion are necessary cheap baselines. Global weights and ordinary logits-only stacking precede the expanded pair gate. Fusion fitting on already selected VALID is usable for development and must be labeled retrospective; it supplies no independent validation claim. An untouched final population can assess a subsequently frozen pipeline only under a separately satisfied existing or prospective contract.

If hidden states were retained, their neighbor summaries need only a declared cached sparse operation. If states, independent controls or proper confirmation populations are missing, their acquisition is a cost and scheduling requirement. It does not refute a supported accuracy hypothesis. Cheap evidence may justify requesting representative GPU fits or replay work, with all acquisition and inference costs charged. Current resource availability determines what can run now, rather than which scientific explanation is true. No such request, fit or replay is performed by this amendment.

The hypothesis remains unconfirmed. Equal shared errors limit weighted pooling; a graph-conditioned gate can also overfit. A practical improvement must survive the cheap baselines, the unchanged prior with equivalent evidence, the capable joint stacker, and appropriate independent-bank comparisons. A favorable development subgroup or better calibration alone does not establish the requested complete ranking gain. There is no acceptance prescription or novelty clearance.

## Reused sources

1. [Link-MoE](https://arxiv.org/abs/2402.08583v2), index69 record 79: Section 4, Equations 1–3, Algorithm 1 and saved setup/appendix scope.
2. [RBS](https://arxiv.org/abs/2206.01570v1), saved Sections II-C2 and IV-A/B.
3. [GATS](https://arxiv.org/abs/2210.06391v1), saved Section 5 and Appendix A.3.
4. [META-DES](https://arxiv.org/abs/1810.01270v1), saved Section 3.
5. [Original aggregation memo](../graph_aware_saved_prediction_aggregation_prior_scout_20261005_v1/REPORT.md), SCOPED_CONCLUSIONS.json and READ_SCOPES.json. Its node-specific probability context remains a node proposal; this amendment defines the link variant explicitly.

Zero new public retrievals, primary reads, full-paper reads, target payload accesses or scientific executions. Canonical indices, ledgers and frozen families are unchanged.
