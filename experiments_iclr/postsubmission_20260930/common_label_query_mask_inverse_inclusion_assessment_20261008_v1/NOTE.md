# Common query masks and inverse inclusion for label messages

**For root, 8 October 2026.** Pre-implementation algebra and operation choice only. No new paper search, prototype, arm grid, score/model/data access, server action, fit or canonical mutation.

## Exact conditional identity

Let A contain n allowed TRAIN-labeled nodes. Draw Q uniformly from the k-subsets of A, independently of current parameters, feature-forward randomness and attention randomness. Require `1 <= k < n`. All routes hide every label in Q; corrected losses score only targets in Q.

Fix a scored target `i in Q`, current parameters, allowed graph, label-free feature states and realized mask-independent attention scores. Normalization is over **all observed nonself neighbors**, not only visible labeled neighbors. Define the complete permitted label contribution

`M_m(i) = sum_{j in N(i) intersect (A minus {i})} alpha_m(i,j) v_m(y_j)`.

The learned label embedding/value and bias-free output maps are fixed for this expectation. Conditional on `i in Q`, the other `k-1` queries are a uniform subset of `A minus {i}`. Therefore every `j in A minus {i}` has visible inclusion probability

`pi = P(j not in Q | i in Q) = (n-k)/(n-1)`.

Consequently

`E_Q[ sum_j alpha_m(i,j) 1[j not in Q] v_m(y_j) / pi | i in Q ] = M_m(i)`.

This follows from linearity of expectation; independent neighbor inclusion is unnecessary. Duplicate edges use the same node-label indicator and do not change the first-moment identity. Their correlated contributions do affect variance. An additional linear bias-free output preserves the identity, as does adding fixed label-free base logits. Thus the specified one-hop corrector's **message and residual-logit first moments** match the full permitted nonself label context at fixed parameters.

No forward sees any hidden query label. The full-context quantity is a mathematical reference, not permission to insert Q labels into a scored forward. This is standard inverse-inclusion/dropout reasoning, not a new sampling principle or ensemble theorem.

## Strongest limits

1. **Nonlinear predictions and risk.** In general `E softmax(z_Q) != softmax(z_full)`. A nonlinear head after aggregation likewise breaks the logit identity. Mean-probability pooling does not restore it; shared Q also correlates members' context noise. CE is convex in one member's logits, so for this restricted logit identity `E CE(z_Q,y_i) >= CE(z_full,y_i)`. That is a standard Jensen bound, not equality, a descent guarantee or a corresponding bound for the actual pooled NLL. The expected masked-loss gradient is generally not the full-context-loss gradient. Better first moments do not imply better accuracy, calibration or useful alternatives.
2. **Mask-dependent scoring.** The identity fails if attention, features, sampled support or dropout realization depends on the current Q. Label-augmented keys/queries, label-conditioned recurrent states, visible-label-specific normalization and Q-dependent subgraph sampling are outside the stated operation. Conditioning on fixed parameters after previous training is allowed; drawing the current Q using current scores or outcomes is not covered. Use an isolated mask RNG and mask-independent feature/attention forwards.
3. **Visible-only renormalization.** A random denominator changes the operation. For two potential anchors with different full-context attention weights, retaining exactly one makes its visible-only normalized weight one. Inverse scaling then estimates an unweighted total, not the original weighted full-neighborhood message. A global inverse-inclusion factor cannot repair that ratio in general.
4. **Multi-hop self-return.** Nonself edges do not prevent `i -> neighbor -> i` paths. If fixed linear propagation is applied after a single input mask, inverse inclusion can recover only the other-anchor contribution; it cannot recover a hidden target's own returned label. The legitimate TRAIN reference must exclude that target everywhere. Remasking intermediate states introduces products of indicators and different joint-inclusion probabilities. Label-dependent attention/nonlinear multi-hop propagation has further failures. Keep the proposed one-hop, label-free-score operation; do not extend its identity to a general UniMP network.
5. **TRAIN fitting and heldout use.** The result conditions on a TRAIN query and a fixed learned state. It is not cross-fitting, an unbiased generalization estimate, independent-node evidence or protection against overfitting/selection. At a heldout target `i not in A`, a masked A would instead give inclusion `(n-k)/n`; the TRAIN conditional scale would not be correct there. Actual heldout inference uses all permitted A labels with scale one. Do not reuse a masked TRAIN scale for evaluation.
6. **Degeneracy and variance.** Hiding all labels gives zero inclusion and no estimator of nonzero label context. If there is only one TRAIN anchor, a masked TRAIN target has no other anchor contribution. Small inclusion can amplify mask noise and incorrect neighborhood evidence. No clipping, tunable correction strength or variance guarantee is introduced here.

## Contrast with ordinary unscaled masked-label training

For **this restricted fixed-score label-linear corrector**, unscaled masking has expected message `pi M_m(i)` on TRAIN queries, whereas inference supplies the full nonself label message. Inverted scaling removes that first-moment attenuation. It does not remove the different context distribution or the nonlinear-risk effect; training may compensate for attenuation, and either training convention can generalize better.

Saved UniMP v5 §§3.2–3.3 already add label embeddings to features, mask a random portion of labels, predict masked labels and use all allowed labels at inference. Its scores/hidden states can depend on those labels. The inspected paper scope does not certify its exact native sampler or an inverse-inclusion implementation. Therefore the identity is **not** asserted for native UniMP logits, and inverted values would be an attributed modification, not a faithful reproduction claim. The saved conditional-Bernoulli/fixed-cardinality and dropout scopes establish sampling ancestry; the displayed conditional marginal is elementary algebra for the uniform subset assumed here.

## Recommended operation

For the retained one-hop proposal, recommend a single prospectively specified **fixed-size common Q**, mask-independent feature-only keys/queries, normalization over all nonself neighbors, zero unavailable-label values, and linear bias-free label-value/output maps. Multiply visible TRAIN-label values by `(n-1)/(n-k)` only during Q-supervised training. Every route receives the same masked label field. Use all allowed TRAIN labels **unscaled** at inference. No label-derived field may bypass Q exclusion, and unscored TRAIN nodes' scaled correction outputs must not be cached as future context.

This is a clarity choice about the corruption contract, not an extra method lane, tuning grid or implementation admission. If the future operator uses nonlinear post-aggregation heads, label-dependent scores, visible-only normalization or additional propagation, either redesign that operation explicitly or drop the first-moment claim; do not silently retain the guarantee. Preserve native unscaled UniMP and faithful C&S comparison obligations without expanding an experiment protocol here.

The capable same-context single and untied ensemble must receive the same corruption/scaling opportunity. The masking identity neither establishes ensemble benefit nor methodological novelty.
