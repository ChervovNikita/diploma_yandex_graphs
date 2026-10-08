# Serving cost of the new correction architecture

The original GNNM variants can share backbone weights while still running a different hidden-state trajectory for each member. The current label-only screen changes that boundary: one feature-only native forward supplies the same H and base logits to all four correction routes. The complete C4 selected state contains one coherent native model and all four correctors at the same epoch.

This is a concrete opportunity to reduce backbone execution. It is not a measured latency or memory result. The joint four-head single and same-backbone untied controls also reuse one native forward, so that reuse cannot explain a claim of superiority to those controls.

## Declared work to compare

| Served pipeline | Native feature forwards | Label attention heads | Served predictions |
| --- | ---: | ---: | ---: |
| C4 | 1 | 4 | 4, probability average |
| Joint four-head single | 1 | 4 | 1, joint readout |
| Same-backbone untied correctors | 1 | 4 | 4, probability average |
| Genuine independent native4 with one corrector each | 4 | 4 | 4, probability average |

The last row is a reference design, not a completed experiment. Four distinct selected native states cannot generally reuse one hidden-state forward. This work count does not predict wall time: sparse operations, graph preparation, launch overhead and allocator behavior must be measured on identical inputs and hardware.

## Exact value-side folding is possible

The saved operator note derives each residual as delta_m(i) = T_m g_m(i), where T_m is a C-by-C class-to-logit map and g_m is the feature-conditioned weighted histogram of permitted neighbor labels. The current value/output maps are linear and bias free. At a fixed selected state, their composition with the label embedding can therefore be cached as T_m. Aggregation can operate on C label bins rather than the 64-dimensional value embeddings.

Here C is 10. The attention denominator must still include every nonself neighbor, including unlabeled neighbors. Grouping only visible-label edges into that denominator would change the method. Duplicate records share the same label indicator and must retain their weight. Zero label context must still give exactly zero residual.

This is standard linear-operator folding, not a novelty claim. No folded implementation, timing, prediction comparison or quality result is admitted here. Training autograd and float32 reassociation require separate treatment; the benchmark continues to use the unchanged original executor.

## Next decision

First finish the frozen three-seed screen and compare quality with capable controls. If the direction is supported, measure its actual serving cost against those controls and genuinely independent models. Quality and mechanism claims remain primary. Folding is a later engineering option; it cannot rescue an unsuccessful quality comparison.
