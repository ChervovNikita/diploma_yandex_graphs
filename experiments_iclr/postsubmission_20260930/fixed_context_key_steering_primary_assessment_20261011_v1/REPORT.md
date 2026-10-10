# Fixed graph-feature keys for shared SAGE steering

**Disposition:** one coherent conditional construction is retained, with exact first-moment and projection limits. No executable model, source, qualifier, outcome read or fit is created. Three bounded primary method scopes are saved; zero whole papers, proofs, author implementations or reported performance results adopted.

Root supplies the completed affinity39 disposition: private−BASE−.006320pp, private−common−.025281pp, with two negative seeds. It supplies no private weighted-target accuracy gain and does not show that student embeddings became class-coincident. Root's separate positive F-sharing accuracy clue remains separate from this proposed auxiliary. No affinity/F outcome file was opened here.

## Exact gradient distinction

For nonzero student vectors, the derivative of cosine between parallel normalized vectors is zero with respect to either vector. At a same-class coincidence, every weighted-positive peer term therefore has zero tangent derivative, regardless of the ratios in`Q`. Incoming peer-key terms also vanish. Denominator, cross-class and classification gradients can remain. This identifies a local first-order obstruction; it does not diagnose the learned models or imply full stationarity.

Now use a common bias-free auxiliary projection`A`, live query`z_i=Ah_i`, normalized`q_i=z_i/||z_i||`, and fixed keys`k_j`. Let

\[
\mu_{Q_i}=\sum_jQ_{ij}k_j,\qquad
p_{ij}=\operatorname{softmax}_{j\in E_i}(q_i^\top k_j/\tau),\qquad
\mu_{p_i}=\sum_jp_{ij}k_j.
\]

The outside-log loss and its hidden-state gradient are

\[
L_i=\log\sum_{j\in E_i}e^{q_i^\top k_j/\tau}-q_i^\top\mu_{Q_i}/\tau,
\]

\[
\nabla_{h_i}L_i=
\frac{A^\top(I-q_iq_i^\top)(\mu_{p_i}-\mu_{Q_i})}
{\tau\|Ah_i\|}.
\]

For the same query, keys and denominator, changing target rows changes this gradient by

\[
-\frac{A^\top(I-qq^\top)(\mu_{Q_a}-\mu_{Q_b})}
{\tau\|Ah\|}.
\]

Fixed heterogeneous keys can preserve this difference when students coincide. For example, with`q=e₁`, identity projection, keys`e₁/e₂`, and targets`(¾,¼)/(¼,¾)`, the gradient difference is`e₂/(2τ||z||)`. This is a symbolic witness, not a model or toy execution.

The condition is precise: different target ratios need different key means whose difference survives the normalization tangent and `Aᵀ`. Distinct rows can have equal means; parallel differences are removed; the projector and native parameter Jacobians can annihilate or cancel remaining signals. With different context key banks, both softmax and target means change, so complete gradients can cancel. Two identical queries with opposite tangent errors outside the range of`A` have zero hidden gradients and cancel in the shared projector gradient. Fixed keys therefore do not guarantee escape from class coincidence. Zero queries and numerical norm clamps require separate derivatives.

## The target distribution reduces to one feature mean

For fixed linear-dot-product keys, all dependence on`Q_i` is exactly through`μ_Qi`. Rows with the same mean produce identical loss for every query. This loss cannot transfer the entire teacher pair-distribution geometry.

An exact positive-alignment control is

\[
R_i=\sum_jQ_{ij}(1-q_i^\top k_j)/\tau
=(1-q_i^\top\mu_{Q_i})/\tau.
\]

It has the same target-dependent query gradient. The fixed-key loss adds the dictionary partition term:

\[
L_i=R_i+\log\sum_{E_i}e^{q_i^\top k_j/\tau}-1/\tau.
\]

Do not normalize`μ_Qi` in this control: doing so changes its per-row gradient amplitude. This comparison decides whether dictionary repulsion adds useful predictions beyond the same asymmetric feature supervision.

## Closest primary methods

| Primary source and exact scope | Established overlap | Boundary |
| --- | --- | --- |
| Tian, Krishnan & Isola, [Contrastive Representation Distillation](https://arxiv.org/html/1910.10699v1), §§3.1–3.2, blocks15–48 | Teacher/student feature comparison, normalized linear projections, contrastive critic and stored negative features. | Binary NCE critic and learned teacher representations. The proposed keys are raw fixed graph measurements; no independent teacher is fitted. Teacher-side projector freezing is not inferred from this scope. |
| He et al., [Momentum Contrast](https://arxiv.org/html/1911.05722v1), §§3.1–3.2, blocks18–37 | Query/key dictionary InfoNCE, asymmetric backpropagation, queue and momentum encoder. | EMA keys evolve. The proposed finite TRAIN dictionary is fixed throughout fitting and needs neither key encoder nor queue. |
| Hu et al., [Graph-MLP](https://arxiv.org/html/2106.04051v1), §3.2, blocks31–54 | Graph-defined contrastive supervision together with CE and projection/classification layers. | Its keys are live peer embeddings and its positive mass is inside the logarithm. It does not establish this fixed-feature-key construction. |

CRD's printed Eq10 differs from its later correct negative log-likelihood expression; Graph-MLP's standalone piecewise connection rule is reversed relative to its adjacency-power definition. No proof or executable rule is imported from those inconsistencies.

Saved **MLCL** supplies objective/positive-relation allocation to shared private heads. Saved weighted-InfoNCE/y-Aware definitions supply continuous pair weights; the original y-Aware paper was not independently audited here. Saved **NCA §2 Eq6** already supplies log class-mass supervision; this outside-log weighted objective is a different, attributed choice. Asymmetric comparison, normalized projection, graph filtering and supervised neighbor learning are established primitives.

## One conditional operation

Use the same factual native SAGE forward and all580 TRAIN pre-output-head rows. A **single shared bare projection128→300**, outside private factor wrapping, produces normalized live queries. Route`m` uses fixed normalized TRAIN keys from its corresponding`X`, `X−PX`, `PX` or`P²X` signature and the matching classmate-weighted target`Q_m`. All other TRAIN nodes enter the denominator; exact self is excluded by node ID. Zero signatures map to zero keys. Keys and targets are detached buffers prepared once from permitted public graph/features and TRAIN labels.

Retain own CE plus0.05 auxiliary, student/teacher temperatures0.2, native parameters/starts/schedule and probability-mean serving. One separate deterministic semi-orthogonal initial projector can preserve query norms without consuming native/dropout streams; it has not been constructed or qualified. The projector is removed at serving. Native shared/private upstream parameters receive ordinary auxiliary pullback; the native output classifier receives own CE only. No corrupted input, extra GNN pass, teacher ensemble, selective gradient guard or projector grid is introduced.

The proposed construction uses frozen graph-feature means to steer live predictive routes through a common train-only map. It adds no external information to the public graph/features. Its possible useful role is an acquisition bias that preserves a target-dependent tangent unavailable in the stated peer-coincidence configuration. No new asymmetric contrastive or reconstruction principle is claimed.

## Decisive controls

| Control | What it decides |
| --- | --- |
| Same shared projector with **live projected peer keys** and the same`Q/E` | Matches added projection dimensions/initialization and tests the key-source/gradient change. A quality difference would not prove prior collapse. |
| Exact weighted-mean alignment`R_i` | Tests whether the dictionary term earns utility beyond the identical positive steering signal. Matching it supports the simpler attributed alignment ingredient. |
| Common shared mixture of four **complete `(K,Q,E)` losses** | Tests private allocation beyond common task learning. Different key denominators must be averaged as complete objectives;`Q̄` against one or a union dictionary is invalid. |
| Fresh ordinary/factorized M1 and genuine I4, each receiving all four complete objectives | Capable same-information/target references, with native independent own selection and equal paid selection/calibration opportunity. |
| Genuine I4 with the exact assigned key/target pair per body | Tests sharing at the same actual objective. |
| Fixed keys with one class-preserving permutation jointly applied to both axes of every`Q` | Preserves row mass/entropy/class support/zero diagonal and every denominator, while breaking target-to-key incidence. Keeping keys fixed avoids self-key leakage from a key-row shuffle. This tests whole incidence, not a unique topology cause. |
| Uniform classmate targets with the same fixed keys/projector | Tests necessity of nonuniform means. Uniform leave-one-out means can also vary, so it is not a zero-steering control. |

These are controls for one candidate, not separately activated proposals. Root must choose a finite complete roster before source work; none is admitted here.

## Utility prediction and contrary history

The candidate should improve complete served quality and competence/proper risk beyond unchanged shared, projected-peer and common controls, acquire useful rival-margin reversals and retain enough corrections after averaging to repay harms. Nonuniform-incidence and sharing interpretations additionally require their corresponding controls. Nonzero gradients, lower auxiliary loss or greater hidden spread are insufficient.

Failure is plausible: feature means can be redundant or noisy, query/projection/Jacobian geometry can remove their differences, the auxiliary can live in classifier-unused directions, or feature alignment can damage CE and calibrated risk. Correct alternatives can again be lost in serving. A result matched by capable common/M1/I4 references does not establish private allocation or sharing benefit.

The saved complete PubMed **CORE9** masked classification/reconstruction recipe fails its exact utility increment, with CORE−own mean−.0592pp. Nonco-moving raw reconstruction targets already supplied asymmetric gradients there. This is contrary utility evidence, not a rejection of every fixed-target objective. The present operation uses unmasked factual queries, class-conditioned smoothed means and a dictionary term; its exact mean-alignment control makes the reconstruction overlap explicit. A later source-only masked-SSL proposal is not counted as another completed failure.

## Cost boundary

The projector adds38,400 train-only weights. Four fixed key banks occupy2,784,000 float32 bytes; cached target means occupy the same amount. Their one-time `QK` reductions cost403.68million MACs. Per route, TRAIN query projection costs22.272million MACs and one580×580×300 dictionary score matrix costs100.92million MACs. Private4 auxiliary forward arithmetic is492.768million MACs per update; common four-dictionary shared4 pays1.703808billion, reusing each route's query projection.

Native GNN forwards and serving remain unchanged, but backward, optimizer states, graph-signature preparation, normalization, log-sum-exp, buffers/transfers and restoration are paid. Score width300 exceeds the former128, so no lower total step-time claim is supported. Reuse requires exact target/key/order hashes. No runtime budget, source expansion or fit is authorized by this assessment.

Detailed identities, scopes, saved failures, controls and arithmetic are sealed in `EXACT_GRADIENT_AND_REDUCTION.json`, `READ_SCOPES.json`, `ONE_CONDITIONAL_MECHANISM.json` and `COMPUTE_EXPECTATION.json`.
