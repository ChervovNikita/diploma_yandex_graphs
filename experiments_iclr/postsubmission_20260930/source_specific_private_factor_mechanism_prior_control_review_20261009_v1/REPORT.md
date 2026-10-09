# Source-specific private-factor ensembles: nearest mechanisms and one analysis

## Conclusion

The closest saved published graph collision is **HGEN (IJCAI2025)**: separate meta-path graph learners acquire different neighborhood evidence and feed a classifier through residual attention and summed path logits. **SuGAr (AAAI2025)** learns diverse graph selectors in independent GNNs. **BatchEnsemble** establishes shared slow matrices with private rank-one factors; **GNCL** establishes individual/aggregate-risk training. These ingredients already cover source-conditioned experts, parameter sharing, diversity and ensemble credit. The immutable proposal is an attributed configuration with an unresolved utility question. The inspected scopes establish no new loss, general absence, novelty clearance or benefit.

One useful addition is a **complete member-by-family source-utility analysis in two peer contexts**. It uses the full/ablated logits already required by the proposed heldout assay, adding no fitted arm or derivative. It can show whether source assignments affect correct predictive evidence in the normal full-input committee.

## Precise overlap

| Primary precedent; reused scope | Training and ownership | Collision and operational boundary |
| --- | --- | --- |
| [HGEN](https://www.ijcai.org/proceedings/2025/685), published pp2–4 §§3.1–3.4/Eqs1–9; saved pinned source | Per-path learners have separate feature encoders and GCN parameters. Residual attention fuses learners, path logits are summed, and fused CE is combined with a graph-pooled embedding Gram penalty. | Semantic-neighborhood expert ensembling is a direct collision. HGEN's Gram loss is coordinate/norm dependent and does not measure useful observed-label source supply. It differs from shared native slow objects/private factors, fixed probability averaging and a signed factual/probe private correction. Printed/source penalty and attention discrepancies remain unresolved; no regularized native reproduction is inferred. |
| [SuGAr](https://arxiv.org/html/2410.22228v2#S4), §4/Eqs3–4; publication DOI10.1609/aaai.v39i18.34065 | Independent invariant GNN/selector parameters, own risk and supervised contrast, pairwise learned edge-weight overlap penalty, followed by aggregation/averaging. | Diverse graph evidence is prior. Selector disagreement does not imply correct probability mass. The printed method supplies neither these fixed semantic source families nor this same-state signed committee risk/permission rule. |
| [BatchEnsemble](https://arxiv.org/abs/2002.06715), saved §3.1 | Shared slow W with private outer-product input/output factors; complete member predictions are averaged. | Exact parameterization ancestry. Sharing changes the feasible functions and optimization; it does not establish independent learners, complementary source utility or free graph computation. |
| [GNCL](https://arxiv.org/abs/2011.02952), saved §4.1/Eq5 and pinned fit path | Mixes mean individual risk and aggregate risk. Inspected implementation uses one global scalar backward through all estimator parameters; base-output aggregation follows the supplied task/model. | Direct ensemble-risk ancestry. Match probability versus raw-logit pooling and reductions before claiming literal equality. It has no fixed family removal, negative probe coefficient or private-only guarded correction in the inspected path. |
| [FoRDE](https://arxiv.org/html/2306.02775v3#S3.SS2), ICLR2024 saved method | Particle repulsion compares normalized true-label input-logprob gradients, encouraging different explanatory features. | Classifier-facing evidence diversity itself is prior. Angular/gradient separation does not score the target likelihood gained by supplying a named graph family to a particular committee. |

The adjacent saved DIVE preprint explicitly says its encoders are unshared and penalizes mask overlap (blocks10,37,39). DVERGE's saved NeurIPS2020 scope cautions that output/gradient diversity proxies need not match the intended vulnerability. Neither provides an ordinary node-task utility guarantee. No paper's results or proofs are imported here.

### The proposal's exact training identity

For each query and Bernoulli label, let q_m denote the probability of the **observed label event**: p_m for a positive target and 1-p_m for a negative target. With the other members' same-family-ablated q values frozen, the derivative of J_m=S_m-A_m is

`E[rho_F * grad_eta CE(z_m^F,y) - rho_-a * grad_eta CE(z_m^-a,y)]`,

where each responsibility is the recipient's event probability divided by the corresponding recipient-plus-peer event-probability sum. This is a signed adaptively weighted CE gradient. A detached-weight implementation is identical only when it also retains the scalar J values used by Armijo/guards, both native pullbacks, frozen peers, state/RNG realization, parameter permissions and update contract. It does not justify another duplicate fitted arm.

The full rule is a recomputed Jacobi surrogate with private correction after own Adam, rather than GNCL's one globally differentiated scalar mixture. Its absent-risk anchor prevents obtaining a guarded mean contrast gain solely by worsening the declared absent pool; TRAIN means do not guarantee every query, heldout utility or the all-updated peers' absent pool. Fixed source labels also do not force complementary full-input members.

A truly independent counterpart has every complete learned body/storage disjoint. Coupling its losses through detached committee predictions does not create parameter sharing. A shared-versus-independent claim needs the existing matched independent comparison. The proposed analysis below cannot replace it.

The actual SeHGNN source views remove raw typed support before normalization and rebuild native feature/TRAIN-label channels. Its factors steer the semantic processor over those precomputed channels. This is a source-access intervention and an ensemble-learning placement; it does not establish a new live edge transport or sheaf mechanism.

## One additional mechanistic analysis

Use the restored selected state, every permitted VALID row, all five Bernoulli labels, all four members, all three fixed families and every admitted seed. Use fresh FP32 eval outputs of the **current** members, with the qualified buffer/mode/cache/RNG transaction. Do not reuse TRAIN-mode reference tokens or frozen training-peer predictions.

Let p_m^F be member m's full-input probabilities and p_m^-a its probabilities after removal of family a. Let R(P) be complete-population mean marginal BCE, including positive and negative label events. Form

- `P_-a = mean_k p_k^-a` (all current members lack family a).
- `P_only(m,a) = [p_m^F + sum_(k!=m) p_k^-a]/4` (only m receives a).
- `P_F = mean_k p_k^F` (actual full-input serving).
- `P_remove(m,a) = [p_m^-a + sum_(k!=m) p_k^F]/4` (only m loses a).

Report both complete 4×3 matrices:

`U[m,a] = R(P_-a) - R(P_only(m,a))`,

`D[m,a] = R(P_remove(m,a)) - R(P_F)`.

U measures useful source supply to a source-absent committee. D measures whether that member's source access matters when every peer retains the source, which is closer to actual deployment. All terms use actual normalized probability pools. Embedding distance, attention disagreement and agreeing class labels do not substitute for them.

For the three assigned members and the fixed family bijection, report

`Spec(X) = mean_three assigned diagonal X[m,a_m] - mean_six off-diagonal X[m,a]`.

Recipient-only and family-only additive offsets cancel in this balanced contrast. Keep the complete matrices, including the unassigned member; it need not have zero source dependence. Compare this description across the **already proposed** own-only, pool-credit, view-supervision, COMMON and source-supply branches when admitted. Do not add fits or select an assignment, family, seed, source strength or checkpoint from this analysis.

Also retain per-query/per-label signed BCE changes and counts of label events repaired versus harmed by restoring that source at the native fixed threshold. Report the entire population, including negative contributions. The existing ablation assay requires four full and twelve ablated member forwards per checkpoint. If these logits are already retained, both matrices and the repair/harm summaries need only arithmetic, not extra forwards or VJPs; all existing assay costs still count.

Interpretation:

- Positive U with negligible or negative D indicates source-supply surrogate utility that is redundant or harmful in the normal all-full committee.
- No increase in diagonal specificity over COMMON/view supervision fails to support useful member-specific assignment, even if responses become more different.
- Positive assigned-specific D, favorable repair/harm and better actual served quality are stronger evidence of classifier-visible complementary source use. They do not establish causal source semantics, geometric novelty, generalization or a parameter-sharing advantage.

This is a proposed analysis, not a new selector or acceptance gate. Freeze its reporting before outcomes; it has not been implemented or run here.

## Scope and unresolved recent leads

This note reuses saved conclusions, exact primary excerpts and source-audit findings. It performs **zero new public requests, primary scopes, full-paper reads or author-code reads**. HGEN's published method/source scopes are the concrete collision; an additional download was unnecessary. HeCo's body/method/experiments were left to root's separate lane.

Root's current Crossref hits and failure receipts were inspected as metadata only. ML-HAN (DOI10.1016/j.neucom.2025.131962), Multi-Source Information Graph Embedding with Ensemble Learning for Link Prediction (DOI10.3390/electronics13142762), and the counterfactual diversified-recommendation paper (DOI10.1016/j.is.2023.102322) still have no primary method read after HTTP403. Titles do not establish training equivalence or clear novelty. This bounded comparison is not a generic survey or manuscript verdict.

`READ_SCOPES.json`, `REUSED_EVIDENCE.json` and `INPUT_BINDINGS.json` record exact scope and custody. The proposal, source seals and recipes remain unchanged. No numerical provider/model, scientific dataset/labels/arrays/checkpoint/outcome or research-server execution was used.
