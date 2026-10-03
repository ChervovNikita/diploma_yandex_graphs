# CP relation conditioning: predictive-quality gap review

3 October 2026. Reuse of saved scoped literature conclusions and the complete CP/common-gradient/frozen-c proposal. No new primary read, data or fitted-state inspection, numerical experiment, HGEN execution, adoption, or predecessor edit.

## Decision

The strongest defensible opening is **supervised regularization of relation-evidence use**, with a compact committee retaining competent private nonlinear HGT trajectories. It is a conditional predictive-quality hypothesis. The saved literature does not support a new operator, a new label-relevant diversification principle, or a reason this committee should generally beat existing heterogeneous ensembles.

Keep the hypothesis narrow: can learning the relative member positions along one relation/channel modulation axis improve the served mean-logit predictor? The full/common-gradient/frozen-c contrast can address that update-policy question. It cannot establish that the complete member-by-relation interaction is necessary, or that its gain is caused by useful diversity.

## Strongest plausible route to better predictions

For each layer, the residual coefficient is `c_m q_r u_j`. Different c values place members at different positions along the same learned relation/channel axis; signed q values can change which relation messages are amplified or attenuated. Unlike a global output factor, this acts before relation messages are combined. Private attention, nonlinear states and recurrence can turn those changes into different uses of evidence. Ordinary supervised gradients can select q, u and c directions that preserve or improve correct-label predictions.

This could be useful when a small supervised set supports a common typed representation but several relations provide partly redundant, unequally reliable label evidence. Sharing the core and restricting the residual may reduce estimation variance while allowing members to respond differently to that evidence. A compact continuous modification of a complete graph also avoids requiring each member to learn from a separately selected metapath view. These are plausible advantages over some independent-view constructions, not demonstrated advantages over HGEN, LHGEL, unrestricted relation adapters, or global BatchEnsemble.

The important limitation is that the CP residual supplies **one** relation/channel tradeoff per layer. Its scales are static across nodes. It can underfit several unrelated reliability patterns, and cannot directly choose a different tradeoff for each node. Native HGT attention and the base fast factors can compensate; if they do, the apparent benefit may concern ordinary model capacity or optimization rather than the proposed relation mechanism.

## What the saved primary literature already supplies

| Saved scoped prior | Consequence for this proposal |
|---|---|
| GNN-FiLM, arXiv:1906.12192v5 | Featurewise relation-message modulation is established. Static member/relation conditioning with zero extra shift changes the conditioning contract, not the scaling primitive. |
| Conditional CP, arXiv:1611.09345v1 | End-to-end conditioned tensor factors and equivalent diagonal gated paths are established. The member/relation/channel outer product is not a new tensor operation. |
| R-GCN, arXiv:1703.06103v4, and saved BE/diagonal-adapter equivalence | Typed parameter sharing and diagonal fast factors are prior. Their existence does not supply predictive complementarity or collapse private trajectories. |
| HGEN, arXiv:2509.09843v1, and LHGEL, arXiv:2510.03432v1 | Heterogeneous graph ensembles already diversify graph views and train fusion/correlation machinery. CP has a different complete recipe, but heterogeneity and diversity are not an open task-level gap. Saved HGEN source-repair work does not make its numerical quality qualified. |
| FoRDE, arXiv:2306.02775v3, and SEA, arXiv:2508.04948v1 | Explicit true-label explanation/error-directed diversity is prior. The present own-CE recipe does not implement those objectives; their reported benefits or bounds do not transfer. |
| Wood et al., JMLR 24(359), 2023 | Diversity must be assessed jointly with member quality under the actual combiner. Mean logits correspond to normalized geometric probability pooling. |

These scoped conclusions exclude a primitive-level novelty claim. They do not identify an exactly equal complete published HGT training algorithm, nor establish global nearest-prior closure or predictive superiority.

## Main quality and identification gaps

Own-member CE is ordinary supervised learning, with no requirement that members explain different labels or correct different errors. The saved identity is

`mean member CE = pooled mean-logit CE + Jensen ambiguity A`.

A is label-independent at fixed logits. Own CE penalizes its increase when pooled logits are fixed. More spread can therefore accompany weaker members; rescaling deviations around the same mean leaves the served predictor unchanged. Label-relevant diversity is not implied by a nonzero c gradient, changing c direction, or separated embeddings.

All three proposed arms train q, u, base factors and the shared core using labels. Frozen c still permits the relation and channel axes to change. Common-gradient c fixes its centered direction under the declared AdamW recurrence, while learning its common component and decaying its centered magnitude. Thus full versus common tests permission for relative c updates; common versus frozen also changes decay and common learning. Later q/u/core trajectories differ. None is a pure isolation of useful diversity.

A full-CP win can come from improved average member competence, a common confidence shift, initialization-dependent optimization, or a better fit of the relation adapter. These can be valuable predictive improvements. They should not be renamed complementarity without evidence. The c profile diagnostic and interaction norm are construction checks only.

## One next direction

**Sharpen the existing proposal around the relation-evidence regularization hypothesis; add no new arm or operator.** At its already planned selected states, use the existing loss-matched pool/own/A accounting and correct-label member competence evidence to distinguish a quality gain from mere profile motion. Treat lower served NLL with retained member competence as the practical result. A favorable own/pool/A tradeoff is descriptive support, not causal attribution; the three c arms still identify only their stated update policies.

The quality hypothesis fails if profile motion and interaction remain active without served-quality improvement. A useful-diversity explanation remains unestablished when the gain is also consistent with a common adapter/member-quality change. A gain over these c controls alone leaves comparison to competent heterogeneous ensembles and ordinary relation adapters unresolved. The existing two-graph, native-challenger, heldout and cost requirements remain; no new grid is justified by this review. If the required contribution is a distinct label-directed ensemble mechanism, this proposal currently does not supply it.

## Evidence boundary

Source conclusions, scopes and hashes are bound in REUSED_BINDINGS.json. New primary reads: zero. New experiments and executions: zero. No source-paper numerical benefit is imported. This packet proposes claim refinement only and authorizes no execution.
