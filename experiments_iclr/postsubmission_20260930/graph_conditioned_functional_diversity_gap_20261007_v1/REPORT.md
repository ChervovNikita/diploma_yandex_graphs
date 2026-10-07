# Graph-conditioned functional diversity: the remaining gap

7 October 2026. **A defensible empirical gap remains, but no distinct new learning mechanism survives this scoped comparison.** The gap is whether a shared-backbone ensemble can learn complementary, useful dependence on graph evidence at its paid capacity/compute budget. End-to-end response measurements are better evidence for that question than hidden distances; turning those measurements into another repulsion loss is already close to functional/sensitivity ensemble priors.

This is a scientific follow-up, without a manuscript verdict, novelty certificate, or experiment result. The frozen pilot is unchanged.

## 1. Define the functional object before comparing methods

For target node v and competing class c, use the shift-invariant margin `f_m(v,c;G)=z_m,y(v;G)−z_m,c(v;G)`, or true-label log-probability. For fixed edge/neighbor interventions T_k, define

`d_m(v,c,k) = f_m(v,c;T_kG) − f_m(v,c;G)`.

Small differentiable interventions give `d_m≈J_m δG_k`. This measures a predictor's response, including message values, attention, nonlinear gates, skips/global paths and decoder. Compensated hidden-coordinate changes and common logit shifts do not change it. Retain response magnitudes and a numerical floor: normalizing tiny or arbitrarily scaled derivatives can create apparent diversity. Different confidence responses are functional differences, but are not automatically complementary correct evidence.

The actual graph intervention matters. Edge deletion, node-feature zeroing and edge-score perturbation define different dependence questions. A changed graph can remove information or change the correct conditional prediction. True-label scoring alone does not certify semantic invariance.

## 2. Exact overlaps and non-equivalences

| Previously inspected primary method | Optimized object and label use | Relation to graph-conditioned functional diversity |
|---|---|---|
| [SuGAr, §4, AAAI2025](https://arxiv.org/html/2410.22228v2#S4) | Independent invariant GNNs, own risk and label-aware contrast; sum of pairwise dot products of predicted **edge weights**. ENS then aggregates edge weights to a selected subgraph and combines classifier outputs. Saved pinned source uses min-max-normalized edge activations and a conventional negative supervised-contrast loss. | Direct prior for different label-relevant graph evidence. Its edge overlap is not the derivative/change of the final class decision; different masks can select the same top-k subgraph or be compensated by classifier/message values. Shared live W is a parameterization difference, not a new evidence-diversity principle. |
| [HGEN, §§3.1–3.3, IJCAI2025](https://www.ijcai.org/proceedings/2025/0685.pdf) | Meta-path graphs and multiple learners; residual attention fusion, summed decoder predictions and `CE+λ||HHᵀ||₁` on graph-pooled meta-path representations. | Explicit graph-ensemble correlation regularization. Graph pooling and an uncentered Gram measure coordinates/norms, not each node's end-to-end evidence response. Published objective and saved implementation differ; no implementation equivalence is asserted. |
| [CDLG, §III](https://arxiv.org/html/2306.11344v1#S3) | Separate dense channel maps, channel-dependent neighbor routing; same channel/node across views positive, same node/different channels negative. Concatenated channels feed a subsequently trained logistic classifier. | Direct ancestry for cross-view alignment plus same-object channel separation, including different neighborhood routing. Channels are components of one predictor; this is not a bank of independently label-supervised predictors or a margin-response objective. Printed negative-loss/cosine-probability defects remain unresolved. |
| [Multi-Head Attention with Disagreement Regularization, §3](https://arxiv.org/html/1810.10183v1#S3) | Task likelihood plus fixed-weight disagreement; negative cosine of projected values, elementwise overlap of attended positions, or negative cosine of head outputs. | Already distinguishes representation, attention-position and output diversity. A graph port of attention overlap is an attributed adaptation. Head output differences need not survive output projection or later layers into class decisions. |
| [FoRDE, §3.2](https://arxiv.org/html/2306.02775v3#S3.SS2) | Kernel on normalized **true-label log-probability input gradients**, with particle repulsion; each member remains task-trained. | The direct sensitivity-diversity ancestor. Treating graph features/edge coordinates as inputs, or projecting onto a fixed graph-intervention bank, changes the domain/statistic. A finite-difference CE-plus-response penalty is not literally its particle update, but is not a new functional-diversity principle. |
| [Repulsive Deep Ensembles, §2.2](https://arxiv.org/html/2106.11642v1#S2.SS2) | Repulsion in function space, approximated through finite input evaluations and parameter/function derivatives. | Response vectors are differences of function evaluations on `[G,T₁G,…]`. Applying a fixed differencing map to this function vector gives d. Renaming these finite projections as graph-conditioned evidence does not establish new ancestry. |

These are exact comparisons of the inspected optimized objects. They do **not** prove literal equality of every complete architecture, training update or graph port. Nor does the absence of a literal duplicate establish novelty.

A useful algebraic separation is explicit. Let a single attention output be `o=Σ_u a_u t_u`, with `a=softmax(b)` and message values t held fixed for this derivative. Then

`∂o/∂b_e = a_e(t_e−o)`.

If all t_u are equal, any attention distribution gives the same output and every edge-score response is zero. Distinct attention/mask distributions therefore need not imply distinct functional dependence. Conversely, two members with identical a but different task-visible t can have different responses. The final margin additionally multiplies this by downstream derivatives and includes paths in which values/scores depend on inputs. This explains the mismatch; it does not invent a new regularizer or prove causal importance of attention.

## 3. Common errors and capacity are separate constraints

With the current mean-probability pool,

`p̄_y−p̄_c = M⁻¹Σ_m(p_m,y−p_m,c)`.

If every member assigns the **same competing class c** higher probability than y, the pool cannot rescue y. “All members are wrong” alone is insufficient: members can favor different classes and the pool can recover. Exact common wrong-class identities and signed probability margins are more informative than a generic disagreement/correlation score. Functional repulsion can increase response differences while leaving every member's same-c margin negative; it then has not repaired this limitation.

Capacity is different again. If every retained member path only sees an aliased common representation h(G), graph inputs with identical h remain indistinguishable to all downstream heads. A response penalty cannot recover information absent from all paths. Shared W can learn a better representation when the architecture permits it, so this is a conditional limitation, not a universal collapse theorem. Likewise, a single nonzero linear BE map has common entry cross-ratios across members; stronger repulsion does not remove its row/column-separable coupling. Deep nonlinear gates, private attention and root/hop paths invalidate blanket extension of that single-map restriction.

Thus a useful result must distinguish three explanations: hidden-coordinate separation with unchanged functions; functionally different but equally harmful evidence; or actual complementary correct decisions that improve the served pool. None follows from a diversity objective's value.

## 4. Candidate mechanism screen

Three plausible proposals were checked:

1. **Replace hidden residual contrast by margin/graph-response repulsion.** This changes the optimized object and avoids hidden gauges, but is an attributed FoRDE/function-space adaptation. It can steer existing private paths; it does not add missing capacity or guarantee fewer common errors.
2. **Diversify learned attention/edge coefficients.** This acts before aggregation and can change the information path, but attention disagreement, SuGAr and DIVE already supply close graph/operator ancestry. The previously suggested bounded row-stochastic private correction is a constrained operator-capacity control, not an established new graph mechanism.
3. **Increase private affine rank or add a private propagation branch.** This can relax a demonstrated map/information restriction; additive low-rank ensembles, learned graph filters and shared-trunk/private-branch ensembles are established. Adding functional repulsion does not make that capacity extension novel.

**No distinct mechanism is promoted.** A novel claim would need a specific operator/update consequence that survives these equivalences and a representative test showing that it repairs a demonstrated limitation. That evidence is currently absent. This is not a proof that no such mechanism can exist.

## 5. One representative prospective test

Use WikiCS as the already chosen representative, keeping the current24-cell `be_init_contrastive−be_init` comparison and separately frozen TRAIN intervention panel intact. On their fixed probes, relate changes in first/last attention to changes in class/probability margins, exact common wrong classes, pool rescues/harm and member competence. A strong contrast arm with unchanged probe responses would undermine the proposed evidence mechanism; larger responses without pooled utility would undermine useful complementarity. The TRAIN panel is descriptive because all580 targets were fitted.

If a later predictive confirmation is warranted, apply the same prospectively frozen probes to the permitted heldout confirmation targets; do not select nodes/neighbors using their outcomes. Retain initialization-matched BE without contrast, the current contrast package, capable packed untied members and the separately required eight-view single control, with the same label/update/selection budget. A response-loss arm, if explicitly elected, must be labeled **an attributed graph-functional diversity adaptation**, retain the same alignment loss, and receive one predeclared strength/compute allowance rather than a grid. It tests empirical utility, not a new mechanism. Record every member path, intervention forward and any mixed-derivative work.

Failure modes include unequal stochastic-path opportunities, checkpoint-selection differences, response normalization near zero, harmful graph perturbations, a weak single/untied comparator, extra capacity explaining the gain, and attention/decoder compensation. Improvement restricted to fitted TRAIN nodes does not confirm generalization. The current freeze should not be changed to accommodate this follow-up.

Scope: all decisive primary passages and source caveats were already inspected and saved. They were reused for the algebra/equivalence analysis; **zero new source reads/retrievals, full-paper credits, author-code scopes, model/data/checkpoint/score/GPU/77 actions or frozen-source edits**. Exact reused excerpts and hashes are in `evidence/REUSED_DECISIVE_PRIMARY_PASSAGES.json`; read limits are in `READ_SCOPES.json`.
