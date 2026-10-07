# Node and channel factors as a graph ensemble hypothesis

A separable node × channel factor can change a fixed linear graph's node mixing, while channel factors alone leave that node-mixing matrix fixed. This algebra supplies a possible capacity difference under specific conditions. Existing graph modulation, attention and learned masking already cover the operation, and the current nonlinear backbones need not suffer that restriction. Retain a conditional utility question; no new graph primitive or experiment adoption follows.

## What commutes and what changes

Let `X` be node × feature states, `P` a fixed feature-independent propagation matrix, `R_m` a channel diagonal matrix, and `W` a shared dense map. Associativity gives:

```text
P (X R_m) W = (P X) R_m W.
```

The node-mixing matrix remains `P`. The full node × feature linear map still differs through `R_m W`, so feature evidence and class responses can already differ. Factors do not create new graph-hop support in this block; it would overstate the result to say they cannot diversify evidence at all.

For a source-node diagonal `D_m = diag(d_m,j)`:

```text
P D_m X W generally differs from D_m P X W
(P D_m − D_m P)_ij = P_ij (d_m,j − d_m,i).
```

It changes the contribution of source nodes across an existing edge. A separable mask `a_m,j c_m,k` restricts node × channel modulation to rank one at a fixed state; one source-node scalar is reused for every receiver. It is less flexible than a general edge/receiver/channel gate. Constant node factors, factors constant on connected components, or neighbor sets with no differential source weighting can reduce to amplitude changes. Positive masks preserve hop support; zero masks can remove existing paths. No missing graph evidence is created automatically.

Placement matters. A positive scalar multiplying an entire node vector immediately before LayerNorm largely cancels through its input statistics, apart from epsilon. Applied after LayerNorm, it creates node/member-conditioned effective affine coefficients: `alpha_m,j = a_m,j c_m alpha_shared` and `beta_m,j = a_m,j c_m beta_shared`. That is a restricted FiLM/conditional-normalization parameterization. A node-ID lookup also lacks a general inductive or permutation-equivariant construction; such properties require an appropriate feature/structure-based generator.

The fixed-linear assumption must be established for the claimed baseline. In GINE, bond addition and ReLU already produce state-dependent channel admission. With feature-dependent attention recomputed for each member, channel factors can change the propagation matrix itself through different states or queries/keys. Noncommutation with a fixed `P` therefore identifies neither the current model's limiting capacity nor the cause of its common errors.

## Closest graph message ancestry

| Prior and scope | Relevant operation | Boundary for this hypothesis |
| --- | --- | --- |
| [GNN-FiLM](https://arxiv.org/abs/1906.12192v5), reused §2.1 primary method | Receiver/relation-conditioned affine incoming-message modulation; nonlinearity before aggregation. | Direct dynamic graph modulation ancestry. A rank-one node × channel restriction and member conditioning are operational choices within this family. |
| [DIVE](https://arxiv.org/abs/2408.04400v1), reused saved primary scope | Independent mask extractors learn continuing hard edge-mask distributions; private feature predictors use own task loss plus overlap regularization. | Explicitly unshared encoders and validation-selected single-member inference. Its learned graph-view diversity is prior; its output protocol does not establish shared-body uniform-pool utility. |
| [GEENI](https://doi.org/10.1145/3489517.3530416), indexed abstract only | Suppresses outgoing messages from likely error nodes and describes diverse ensemble creation. | Particularly close source-node message-suppression ancestry. Publisher HTML/PDF returned403; exact conditioning, sharing and training rule remain unresolved. |
| [GNN-Ensemble](https://arxiv.org/abs/2303.11376v1), freshly inspected §§III–IV | Trains independent GNNs on sampled graph substructures and feature subsets, then averages probabilities. | Input graph/evidence diversification and ensemble complementarity are established. The inspected method uses sampling rather than this compact member factorization. |
| SuGAr, reused saved mechanism assessment | Label contrast and diverse predicted graph evidence. | Graph-evidence diversity is already an explicit target; the current proposal supplies no new diversity principle. |

Three additional queries targeted graph-conditioned BatchEnsemble, ensemble node masking and separable node/channel modulation. Most returned unrelated metadata. They do not resolve an exact complete predecessor, and they cannot support an absence or novelty claim.

## The meaningful untested prediction

If a competent baseline demonstrably retains insufficiently flexible source weighting, a compact post-normalization member-conditioned node gate might turn relevant differences in neighborhood contributions into **net complementary correct predictions**, with less resource use than competent independent models. This is a hypothesis about constrained sharing and learned evidence utilization. It is stronger than increased mask disagreement, a larger commutator, or increased oracle coverage.

Root's reported common wrong-competitor pattern motivates changing member predictions: if every member puts the same competitor above truth in the declared pooling coordinates, an arithmetic convex pool cannot reverse that ordering. It does not diagnose a fixed-linear propagation failure or predict that node gating will help. A useful result would protect mean and weak-member competence, reduce full-population common-competitor support, and improve the served predictor through net repairs. It would also need neighborhood-margin evidence beyond score amplitude, and attribution beyond ordinary shared/member modulation or stronger feature training.

The immediate disposition is **no adoption**. First establish the remaining capacity restriction and the useful outcome; the operation itself has close prior. This alternative would add architecture and conditioning choices, so it remains separate from the unchanged MolHIV I/message-own question and the active family. No gate recipe, initialization, strength grid, source edit, or launch is supplied.
