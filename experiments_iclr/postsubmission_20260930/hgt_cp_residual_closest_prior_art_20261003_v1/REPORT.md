# Closest prior art to the HGT member × relation CP residual

2026-10-03. Static literature assessment only. No model execution, study data/labels/outcomes, author-code execution, remote compute, baseline admission, or protocol changes.

## Assessment

**There is no defensible new-operator claim here.** The implemented operation is a conditional diagonal activation adapter with a CP-factorized coefficient residual, composed with an existing affine HGT message branch and ordinary BatchEnsemble input/output factors. Conditional feature scaling, CP parameter sharing, typed graph transforms, and graph ensembles are established ingredients. The particular constrained HGT committee can remain an empirical configuration worth testing. Its local coefficient-rank witness establishes a distinction from a fixed-site global-BE parameterization; it does not establish whole-HGT novelty or an expressive-power theorem.

No exact identity with one complete published graph system was established in this bounded search. That is a limit of the assessment, not evidence that the combination is new. In particular, replacing GNN-FiLM's receiver-state conditioner with a member/relation lookup produces the same modulation primitive, while changing its complete graph model. HGEN/LHGEL use different views, parameter sharing, fusion, and objectives. Their published benefits cannot be transferred to this candidate or used as native baseline scores.

## Exact implemented object

The implementation README binds, for layer ℓ and canonical raw relation r:

```text
z_mr = reshape(V_source(h_m ⊙ a_m), heads, head_width) @ R_r
message_mr = z_mr ⊙ reshape(Γ_mr, heads, head_width)
Γ_mrj = b_mj + c_m q_r u_j
```

`V_source` includes its shared bias: input factors do not scale that bias, while the output factors scale the relation-transformed bias. This detail belongs in any equivalence check. Γ is constant over nodes for a given member/relation/layer. The additional tensor is exactly `c ⊗ q ⊗ u`. Every member retains its own hidden states, Q/K, attention, message frames, recurrence, and dropout stream; sharing matrices does not remove these trajectories.

An ordinary conditional FiLM/diagonal-adapter path `FiLM(z; condition)=γ(condition)⊙z+β(condition)` implements this exact message scaling by setting `condition=(m,r,ℓ)`, `γ=Γ`, and `β=0`. The coefficient lookup is constrained by CP factors. Expressing the same operation as `z+(Γ−1)⊙z` introduces no new function. A conditional gated network of the tensor-sharing literature can also realize the same diagonal layer using identity input/output projections and the appropriate generated diagonal. These are operator equalities; they do not imply equal objectives, initialization, regularization, optimizers, attention, or complete graph algorithms.

## Retained prior reused first

Index v25 SHA-256: `b002c330d859ab233eed620ada863c84373ba4067226a7d71d20d8431003d9de`. Six selected retained conclusions were reused, with zero retained-primary reread credit:

| Retained work | Established ingredient and limit |
| --- | --- |
| BatchEnsemble, 2002.06715v2 | Shared cores and private rank-one input/output scales; complete per-member work remains. |
| R-GCN, 1703.06103v4 | Relation-specific maps and shared relation bases; typed low-rank sharing is prior. |
| HG-Adapter, 2411.01155v1 | Low-rank typed structural adaptation and label-informed optimization; complete algorithm differs. |
| CHoE, 2605.15888v1 | Frozen meta-path experts, structure-conditioned routing, load balancing and prompt fusion. Native code previously resolves global/meta-path routing rather than per-node weights. Search surfaced v2 metadata; v2 was not method-read here. |
| REEF, 2505.12027v2, updated 2026-09-28 | Relation-conditioned parameter generation and heterogeneous HGB adaptation already coexist; text/pretraining/generator recipe differs. |
| DRSA, 2605.00731v1 | Endpoint-type low-rank bilinear relation alignment; printed same-endpoint-type tying differs from canonical-relation CP conditioning. |

The candidate report/specification and implementation README were read. Existing initialization nulls, affine-site caveats, unrestricted-factor containment, and private-state accounting remain applicable. No original result artifact was reopened.

## Six genuinely new scoped primary method reads

“New” means absent from the retained v25 paper-ID records and method-read in this packet. It does not mean a newly published identity or a full-paper review. Exact URLs, raw hashes, metadata and scopes are saved in the retrieval receipts and `READ_SCOPES.json`.

| Exact primary/version date | Closest method evidence | Relationship to candidate |
| --- | --- | --- |
| [GNN-FiLM, 1906.12192v5](https://arxiv.org/html/1906.12192v5), 2020-06-26 | §2.1 generates relation-specific γ and β from the receiving node's hidden state and applies `γ⊙W_r h_source+β` to incoming messages. Eq.7 places σ after aggregation; Eq.8 places σ per message and adds a post-aggregate layer. | Direct graph-message conditional modulation prior. Candidate uses static member/relation coefficients, zero extra FiLM shift, native HGT attention/normalization and private trajectories. Original GNN-FiLM is not exactly this complete model. |
| [Unifying Multi-Domain Multi-Task Learning: Tensor and Neural Network Perspectives, 1611.09345v1](https://arxiv.org/html/1611.09345v1), 2016-11-28 | §§3.1–3.3 generate weights by contracting a parameter tensor with a descriptor, then derive CP sharing and gated paths. Eq.13 is `U_Cᵀ diag(U_B z) U_D x`; Eq.10–11 use a sum of outer products. Training factors end-to-end is explicit. | The CP residual coefficient tensor and conditional diagonal gating are known parameterizations. Member/relation/channel mode names do not create a new tensor operator. Complete HGT/ensemble equality is not asserted. |
| [ADaMoRE, 2510.21207v1](https://arxiv.org/html/2510.21207v1), 2025-10-24 | §3 builds cohesive/dispersive structural views, sparse foundational experts plus dense residual GNN experts, nodewise fusion, masked reconstruction, load balance and CKA diversity. | Recent graph residual-expert and specialization prior. Heterogeneity here concerns expert architectures/structural regimes; the scoped method does not supply canonical typed-relation CP factors or BE cores. Its reconstruction/diversity claims are unverified. |
| [HGEN, 2509.09843v1](https://arxiv.org/html/2509.09843v1), 2025-09-11; metadata DOI `10.24963/ijcai.2025/685` | §§2–3.3 form homogeneous target-node meta-path views, train multiple allele GNNs using initialization/feature dropout, fuse via nodewise residual attention, sum decoded view predictions, and add pooled meta-path correlation L1 to CE. | Genuine heterogeneous graph ensemble prior. Ensemble diversity, relation/view semantics, and multiple graph predictors are established. Independent view learners and learned fusion/objective differ from same-graph shared HGT with own-member CE and fixed mean logits. |
| [LHGEL, 2510.03432v1](https://arxiv.org/html/2510.03432v1), 2025-10-03 | §§III–IV-D train learners over canonical relation groups and batch-size sampling views, use relational transforms, fuse batch sizes then relation groups with residual attention, and add correlation L1 to classification loss. | Very close heterogeneous relation-ensemble task prior. “Batch View Aggregation” refers to sampled graph views; it is not source evidence of rank-one BatchEnsemble. Native sampling/view construction differs from complete-graph HGT trajectories. |
| [Self-Routed Tensor Adapters, 2608.16384v1](https://arxiv.org/html/2608.16384v1), 2026-08-17 | §3 uses a shared Tucker core, shared low-rank input/output projections and representation-derived softmax routing to blend core slices. A frozen visual backbone is adapted with class CE plus domain-routing supervision. | Recent conditional tensor-adapter prior. It mixes additive matrix updates using data-dependent routing, rather than a static member/relation diagonal coefficient residual on trainable HGT. No superiority or exact whole-model equivalence transferred. |

Nine bounded arXiv metadata searches cover graph BatchEnsemble, recent heterogeneous graph ensembles/adapters/experts, and conditional tensor/hypernetwork adaptation. A narrow graph-title × BatchEnsemble query returned zero entries; broader queries returned off-topic records. These searches are incomplete title/abstract metadata discovery, not a global novelty or absence certificate. No “first graph BatchEnsemble” claim is supported.

## Falsifiable graph-specific hypothesis

The useful remaining hypothesis is: **under the specified complete heterogeneous graph protocol, canonical-relation-dependent member scaling provides a compact regularizer that improves served predictive utility beyond global BE's existing route differences, ordinary dropout, native HGT capacity, and generic conditional-adapter parameterization.** Rank one may be a useful restriction relative to unrestricted relation factors; it may also discard needed freedom. A better coefficient fit, lower cosine similarity, parameter count, or preserved relation names alone does not demonstrate this hypothesis.

The existing global-BE, CP, unrestricted same-site factors, wider BE, native challengers and untied-member controls can test utility of the configured operation. Evidence must include both graphs and paired runs, validation-selected predictive loss/calibration plus competence metrics, and actual private-state/cost accounting. A gain shared by wider BE is consistent with a capacity explanation. A gain shared by unrestricted factors supports relation conditioning but does not identify CP sharing as its cause. Matched initial CP/free-table functions isolate one initial-map difference between those two arms; their optimization coordinates still differ. Global BE starts from a different map, so current CP-versus-BE utility includes initialization as well as learned interaction.

Two invariance cautions matter. Renaming/permuting relation IDs while remapping all rows consistently is a reparameterization, not a meaningful falsification control. Conditioning on arbitrary unique relation IDs still encodes relation identity. Collapsing citations/references by endpoint types changes the graph model and cannot silently replace the canonical-relation experiment. None of these checks proves semantic understanding.

Evidence against the hypothesis includes negligible served gain after calibration, gains explained by capacity or initial perturbation, disappearance across complete graphs/seeds, or reliance on increased training/selection work. Nonzero factor Jacobians and a rank-two local coefficient witness cannot rescue such outcomes. No observed outcome is supplied here.

## At most one future variant

One optional, distinct refinement worth considering **after the current comparison** is a destination-type block CP direction:

```text
Γ_mrj = b_mj + c_m q_r u_{destination_type(r),j}
```

It retains canonical relation coefficients and complete private trajectories but lets the channel direction differ across destination types. The reason is concrete: HGT has typed input/output maps, so a single shared u ties modulation directions across potentially different typed coordinate systems. The hypothesis is that this restriction sacrifices useful type-specific response. For T represented destination types, its additional parameter count is `M+R+T*d` per layer instead of `M+R+d`; that increase and all graph work must be reported.

This is known block tensor/typed conditional-adapter sharing, with no new-operator claim and no guaranteed gain. A representative future comparison would retain the current CP arm and the same-site unrestricted table; initialize that table from the block-generated values before outcomes and disclose budget/optimization differences. HGT's learned typed maps may already absorb the change. This is a literature suggestion only: no new arm, seed, tuning search, implementation, or adoption is created, and the running comparison stays under the root's control.

## Read and qualification limits

Six scoped methods: 162 paragraph blocks, 34 heading blocks, and 42 selected MathML nodes/fragments. Fragments are not 42 separate equations. No complete algorithms, full proofs, full papers, author implementations, result tables or image pixels were inspected. HGEN method block37 incidentally displayed qualitative regularization/result commentary; it is recorded and supplies no benefit claim. Initial metadata output exposed abstracts for the first three papers; empirical claims were ignored. Truncated combined outputs receive no whole-read credit; critical tensor passages and formula fragments were reread in small scopes.

The source/configuration correspondence, native baseline feasibility, study execution, heldout decisions and resources belong to the root. All recommendations are conditional; measured utility and costs are unknown. The packet writes only its own directory and leaves predecessors, literature index/ledgers, implementations and study artifacts untouched.
