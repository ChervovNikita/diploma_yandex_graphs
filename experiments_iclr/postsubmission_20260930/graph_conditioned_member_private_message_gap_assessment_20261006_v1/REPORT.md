# Graph-conditioned private messages: bounded gap assessment

**Decision: close the generic architecture-novelty path.** Small member-private factors that condition a shared message map on public edge/node context are a useful possible parameter restriction of known conditional filters and attention. The consulted sources do not establish a missing mechanism beyond conditional BatchEnsemble-style factors, GNN-FiLM, edge-conditioned convolution, dynamic multihead attention, graph experts and shared-bottom/private-top banks. No exact published duplicate of every training/serving detail is certified, and no global absence is claimed. The remaining defensible question is a measured quality/storage operating point or a diagnosed task-specific correction, not a new conditional-message or ensemble-capacity principle.

## Concrete proposal and what actually changes

For member m, source state h_m, public context c_vu and one shared dense map W, consider

\[
K_m(c)=D_{s_m(c)}W D_{r_m(c)},\qquad
h'_{m,v}=\sigma\!\left(\sum_{u\in N(v)}a_{m,vu}(c,h_m)K_m(c_{vu})h_{m,u}+b_m\right).
\]

All four complete paths remain live and their class probabilities are averaged. Public context means source-permitted graph/node/edge attributes, with no target labels, prediction-error targets or withheld labels as gate inputs. The shared projection and ordinary supervised acquisition/continuation remain competent references. Ordinary member supervision does not mathematically guarantee preservation of competence; it must be checked by served quality and individual member quality.

Three different sites must not be conflated:

1. **Node-independent factors after a common linear aggregate.** This is BE/channel adaptation. It does not change the common graph operator.
2. **Receiver-conditioned factors after aggregation.** This is conditional channel modulation. If the context is unchanged, it still follows the same aggregate; if its conditioning path sees otherwise discarded input, it has a separate information path.
3. **Edge/source-dependent modulation before aggregation, or a different per-member neighbor score.** This can change the induced operator P_m or nonlinear incoming-message map. It is a real local capacity change relative to a restricted post-aggregate model. It is also the site already supplied by ECC, FiLM and attention.

The current full native member-dependent trajectories cannot be assumed to have a single fixed linear P. Private stems can change hidden states and attention even when dense projection parameters are tied. Residual/root paths can carry unaggregated information. The proposal would add direct private context parameters; it does not establish that the current model erased the information or cannot emulate the operation.

## Closest primary sources

| Source and consulted scope | Established operation; exact boundary |
|---|---|
| BatchEnsemble, arXiv2002.06715v2, saved §3.1–3.2 conclusions | Shared W and private input/output factors equal diagonal feature scaling; all members are evaluated. Replacing static r/s by functions of c is the conditional-BE algebraic family. No independently verified paper with the exact title “Conditional BatchEnsemble” was found in the consulted saved metadata, so that phrase here names a parametrization, not an invented citation. |
| GNN-FiLM, arXiv1906.12192v5, inherited method scope/Eq7–8 | A receiver-hidden-state generator produces affine incoming-message modulation. The saved source explicitly distinguishes nonlinearity placement. A public-context/private-generator restriction is the same modulation primitive; full native method/training equality is not asserted. |
| Simonovsky & Komodakis, **Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs**, CVPR2017, new PDF pp2–3/Eq1 | F_l(L(j,i)) generates an edge-specific matrix before neighbor summation. F_m(c)=D_s(c)WD_r(c) is a restricted differentiable filter generator. This defeats novelty of graph-conditioned message matrices with shared factors. The paper's inspected network discussion restricts its applications to graph classification; no present node-classification recipe or performance transfer is inherited. Its fixed degree normalization is not identical to a neighborhood-softmax gate. |
| Brody, Alon & Yahav, **How Attentive are Graph Attention Networks?**, arXiv2105.14491v3/ICLR2022, new PDF pp2–5/Eqs2–7 | GATv1 ranks keys independently of the query within each head; GATv2 scores a^T LeakyReLU(B[h_v||h_u]). A shared B/private a_m scoring restriction is already multihead dynamic attention. Same-context augmentation can expose the proposed public c. Per-channel filters are not claimed identical to scalar attention; ECC/FiLM cover that broader site. Theorem2's appendix proof and empirical comparisons were not audited or adopted. |
| HGT, arXiv2003.01332v1; R-GCN, arXiv1703.06103v4, saved primary/source conclusions | Endpoint-state attention and relation/type-factorized messages, plus relation bases, already supply graph-conditioned sharing. Separate member states generally induce separate transport. A same common-operator commutation rule cannot be transferred to these paths without freezing their scores/states. |
| HGEN, IJCAI2025 DOI10.24963/ijcai.2025/685; MoSE arXiv2509.09337; node-wise filtering MoE arXiv2406.03464, saved scopes | Learned/selected neighborhoods, structural-context experts and node-wise expert filtering/fusion are established. Untied encoders, top-K routing or embedding-level mixture differ from four shared-projection paths and uniform probability averaging. Those operational differences do not establish a new structural-specialization principle. |
| TreeNets arXiv1511.06314v1 §5; CAMERO arXiv2204.06625v1 §3.1, saved assessments | Sharing early layers and giving branches private upper computation is established. Restricting the private part to a tiny edge/context adapter can save stored parameters relative to a full private block. TreeNets is ownership ancestry, not an asserted exact ECC/GAT implementation. The known private-last-block note is not novelty. |

The nearest source for the most general pair/edge-conditioned message kernel is **ECC**, alongside saved **GNN-FiLM**. The nearest source for a positive normalized scalar gate is **dynamic graph attention**. BatchEnsemble supplies the factor sharing and ensemble bank. Graph ensembles/MoE supply structural specialization ancestry. Their combination may be worthwhile; no additional missing operation has been established here.

## Rigorous equivalences and limits

### Fixed-P nullspace: elementary algebra, not a new theorem

For fixed linear node propagation P and fixed node-independent channel map W_m,

\[
PZ=0\Longrightarrow P(X+Z)W_m=PXW_m.
\]

Changing W_m, including fixed fast diagonal factors, cannot distinguish that perturbation at this restricted interface. Also P^kZ=0 for every k>=1; extra positive-hop powers of the same P alone do not restore it. A zero-hop/root/residual X path, a different operator, nonlinear per-message map or context path can retain it. This does not show an entire nonlinear GNN has lost Z. It does not supply a new graph learning theorem.

For conditional factors, the equality needs **unchanged context**: W_m(c(X+Z)) need not equal W_m(c(X)). Even post-aggregate modulation can encode information in a changed pre-aggregate context path. No automatic nullspace claim is made about arbitrary conditional BE.

### What changes an operator and what is a null modification

Receiver-only linear factors commute through a neighbor sum when they are constant across that receiver's neighbors and context is fixed. Edge-varying factors generally do not. A private P_m can distinguish Z if P_mZ is nonzero. This is precisely the already known pre-aggregation/conditional-filter opportunity.

An additive score b_m(c_v) constant across u cancels in softmax_u(e_vu+b_m). The saved relation-conditioned scout also records cancellation of a member/relation/head bias under separate within-relation softmax. Multiplying already normalized messages, changing score temperature or varying c_vu may be non-null, but these are different operations. A zero-effect bias must not be called private neighborhood learning.

A two-neighbor illustration: equal P=(1/2,1/2) kills Z=(1,-1). Fixed channel scalings still output zero. Public edge contexts (1,-1) and scalar score theta_m c give P_mZ=tanh(theta_m), generally nonzero. One attention head computes the same repair. This demonstrates the restricted capacity difference and simultaneously removes an ensemble-only explanation; it is not a new primary theorem or predictive experiment.

### Same-operation capable single contains the bank

Concatenate the four member states and keep their nonlinear/message/norm operations as four blocks. Tie the dense blocks as in the candidate, retain the private gates and classifiers, and let one predictor return (1/4)sum_m softmax(logits_m). That single structured graph predictor is pointwise identical. With identical loss, parameter ownership, optimizer state and RNG contracts it is a renaming of the same training computation, not a useful duplicate fit.

An unrestricted same-operation multihead/branched model contains the candidate at its tying/block settings. A bare one-head GATv1 or a head-only classifier is not automatically that comparator. Standard GAT's feature-head concatenation/averaging is also not automatically the candidate's probability-mean readout: mean softmax(logits_m) generally differs from softmax(mean logits_m). Precise containment requires the readout, nonlinear paths, normalization and context access just specified; it does not assert equality to every stock GAT implementation.

Four same-operation untied encoders also contain the candidate by setting their projections equal. Sharing can affect estimation/optimization/storage, but grants no larger function class than that reference. Disagreement, private parameter distance and low correlation do not establish competence or pooling value.

## Potential distinction worth testing, without novelty inflation

A narrow useful hypothesis is: **at a fixed shared-projection storage budget and competent ordinary schedule, a small explicit edge-context correction learns a corrective neighborhood dependence that projector-only private ownership does not learn well enough.** This is a practical inductive-bias/optimization hypothesis about the current family. It can succeed despite being assembled from known components.

To become a defensible scientific finding, it needs a diagnosed common error that is correctable by the retained context/operation, and quality beyond competent same-information/same-operation references. Current member correlation or a fixed-P toy does not diagnose that task failure. No such new diagnosis was obtained in this task. A generic context gate or untying a last block supplies no distinct missing mechanism beyond the priors above. **The present novelty path is closed; the conditional utility screen below is not an admitted novel-method experiment.**

## One representative falsifiable screen

See SCREEN.md. It is one prospective Amazon-ratings split0/seed17 developmental block with ordinary supervised learning, one fixed context/gate site and no grid. The source-native graph/depth/residual/global path and competent acquisition/continuation stay intact. Official VALID/TEST and the current A scoring gate stay closed; a separate prospective TRAIN-derived development role needs root/use-history approval before any execution.

The screen distinguishes a quality/storage operating point from an ensemble-capacity story: static shared-factor base, conditional edge gate, same gate with broken public edge-context association, competent containing same-context multihead single, and same-operation untied four. An exactly compiled single-bank duplicate is not fitted twice. All training/acquisition work is charged. A gain in correlation or training loss without served complete-population quality is a failure. A capable single matching closes ensemble necessity; untied four matching removes a demonstrated quality benefit of sharing. Positive one-block evidence still needs a separately frozen confirmation and cannot establish acceptance or general novelty.

## Reading and execution boundary

Index72, saved supplements and decisions were consulted as conclusions/scopes, without reopening their primary passages. Two new public PDFs were retrieved: ECC and GATv2. Only declared method pages and title/abstract headers were semantically read; three equation pages were rendered and inspected. Full-document extraction/page locators are mechanical, not full-paper reads. GATv2 intro/evaluation-opening benchmark claims and attention illustration values, and ECC locator/result-table snippets, were incidentally exposed; none are adopted as comparative evidence. No new author-code audit, proof audit of GATv2 AppendixA, global literature count, index mutation or full-paper certification is made.

No fit, server/77 access, dataset, label, tensor, saved-state, history/score payload, held population, implementation or manuscript edit occurred in this research task. The screen is a design only; no execution or GPU request is issued.
