# Balanced structural fast weights and their limits

The proposed operator bank is a support-preserving latent edge reweighting within known graph-filter and attention families. Its exact mean-operator constraint is useful algebra, but it does not preserve each member's normalization, the mean nonlinear prediction, or task competence. Retain one small quality and storage hypothesis; close the generic architecture-novelty claim. No exact published duplicate of the entire constrained recipe is certified and no global literature absence is claimed.

## Exact operator and information access

For a fixed nonnegative normalized TRAIN operator P, shared node gates a_q, and private centered coefficients c_mq,

\[
P_m=P+\Delta_m,\quad
\Delta_m=\sum_qc_{mq}D(a_q)PD(a_q),\quad
\sum_m c_{mq}=0.
\]

Writing A_vq=a_q(v) gives

\[
P_m=P\odot[\mathbf1\mathbf1^T+A\operatorname{diag}(c_m)A^T].
\]

This is a symmetric latent bilinear modulation with a member-specific diagonal metric when P is symmetric. It changes only existing support, including self-loops unless those are explicitly exempted. The score matrix has rank at most Q; the propagation perturbation generally does not. For Q=1 with all a(v) nonzero, D(a)PD(a) has the same rank as P. P=I_2 and constant a=1 already give a rank-two perturbation from a rank-one score.

The primitive is an edge-dependent scalar incoming-message filter. Feature-generated gates are a restriction of a conditional filter generator, and a member coefficient bank is a parameter-sharing restriction. Learned graph filters, attention, and conditional messages already cover this site. A shared feature-based gate avoids a free node-identity table and target-label lookup; this is an information and parameter restriction, not a new message operation. Ordinary TRAIN supervision may train the generator, while labels remain excluded as gate inputs.

## What centering guarantees

In real arithmetic, M^-1 sum_m P_m=P. Therefore, at one common-input interface,

\[
\frac1M\sum_mP_mHW=PHW
\]

for identical H and W. Once member inputs differ, the residual is M^-1 sum_m Delta_m H_m and need not vanish. Average probabilities after nonlinear member trajectories are a different object.

The exact finite example in evidence/ALGEBRA_CHECK.json uses

\[
P=\begin{bmatrix}1/2&1/2\\1/2&1/2\end{bmatrix},\quad
\Delta_\pm=\pm\begin{bmatrix}1/4&0\\0&0\end{bmatrix}.
\]

Its operators average to P. Separately row-normalizing them instead gives first mean row (7/15,8/15). Separate symmetric degree normalization also changes the mean, with first diagonal 7/15 and off-diagonal approximately 0.512282. With H=(1,-1)^T, baseline ReLU(PH)=0, while mean_m ReLU(P_m H)=(1/8,0)^T. Thus re-normalization and nonlinear averaging invalidate a stronger preservation claim.

If the same perturbed operator is reused in two linear layers,

\[
\frac1M\sum_m(P+\Delta_m)^2
=P^2+\frac1M\sum_m\Delta_m^2.
\]

The extra term survives even without nonlinearities. This formula concerns reuse of P_m, not a first-layer-only perturbation followed by an identical linear map.

## Bounded weights and normalization

The bound |sum_q c_mq a_q(u)a_q(v)|<=rho<1 ensures every supported edge weight remains positive, within factors 1-rho and 1+rho. It neither fixes weighted degrees nor implies row stochasticity or spectral norm at most one. Symmetric GCN normalization itself is not generally row stochastic.

When P is nonnegative and ||P||_2<=1, entrywise |Delta_m|<=rho P implies

\[
\|\Delta_m x\|_2\le\rho\|P|x|\|_2\le\rho\|x\|_2,
\quad\|P_m\|_2\le1+\rho.
\]

Constant a=1 with c=+rho gives P_m=(1+rho)P, attaining the larger norm when ||P||_2=1. This is a local operator bound, not a contraction or classification guarantee. Learned dense norms, residuals, nonlinearities, depth and margins still matter.

A sufficient implementation uses a_q=tanh(g_phi(X)_q), centers raw coefficient columns, then applies one common scale so every coefficient row has L1 norm at most rho. The common scale retains centering. Normalizing each member separately would define a changed method and lose the stated identity. Literal floating arithmetic requires reporting centering residuals and bound violations.

## Cost and initialization

For a common H, cache PH and each a_q odot P(a_q odot H), then mix the Q+1 terms for M members. This requires Q+1 sparse propagations at that interface, plus gate generation, diagonal products, coefficient mixing and output storage. It is not Q+1 free propagations for the whole ensemble.

After generic nonlinear divergence of H_m, each member needs its corresponding terms. Special differences such as a common H followed only by node-independent channel scaling can commute through the operators and retain some reuse; this requires an explicit factorization. A sparse multiplication on a concatenated M-times-wider state can reduce launch count but still pays the wider arithmetic and memory. Shared W saves parameter storage; applying it to distinct member states still pays the dense work. Feature-generated gates also change during fitting and cannot be treated as a permanently cached learned backbone.

Identical members with exactly zero c present a symmetry issue. For an identical differentiable loss, gradients with respect to c are identical across members, so centering projects them to zero; gate derivatives are also zero because their correction is multiplied by c. Distinct existing private factors or dropout can break this symmetry. Nonzero bounded centered coefficients and nonzero gates avoid relying on a nonexistent structural gradient at the perfectly identical zero-c state. No quality follows from that initialization choice.

## Closest literature and exact boundaries

| Family | Established operation and boundary |
|---|---|
| Graph WaveNet, Wu et al., IJCAI 2019, new PDF p3/Eqs5–7 | Learns two node embedding dictionaries and SoftMax(ReLU(E1 E2^T)), then uses powers of the learned transition alongside fixed transitions. Latent product adjacency learning is established. It is a single spatial-temporal predictor, with node lookup embeddings, potential absent-edge connections and separate feature maps. The candidate is symmetric, feature-generated, support preserving and centered across served members. These restrictions do not establish a new latent-edge primitive. ReLU followed by finite softmax does not by itself produce exact zero connection weights. |
| Pro-GNN, Jin et al., KDD 2020, new PDF pp3–4/Eqs2–12 | Jointly learns a bounded symmetric adjacency S and GCN weights, with adjacency fidelity, L1, nuclear-norm and feature-smoothness terms. Graph learning and genuine matrix low-rank regularization are prior. Its all-pairs S, task/robustness objective and recomputed normalization differ from the candidate's low-rank score with potentially full-rank masked operator. No robustness guarantee transfers. Printed Eq12 omits gamma from Eq9; this review does not resolve the full optimization implementation. |
| ECC and GNN-FiLM, reused saved assessments | Conditional filters or incoming-message modulation before aggregation already supply the candidate's effective message site. Scalar latent factors are a restricted generator. |
| GATv2, reused saved assessment | Dynamic endpoint-conditioned scalar attention and multihead sharing are established. Neighborhood softmax has a different normalization; candidate edge weights can also be represented by a conditional scalar filter without softmax. |
| LDS, BGCN and Adaptive Connection Sampling, reused conclusions | Learned graph distributions/masks, shared or graph-associated model training, and averaging predictions over graph samples are established. LDS explicitly distinguishes expected nonlinear prediction from prediction under expected adjacency. Deterministic finite centered banks inherit neither Bayesian semantics nor latent-edge discovery for absent support. |
| BankGCN, Specformer, BernNet and LapLoRA, reused conclusions | Learned filter banks, channel-specific operators and structural/spectral adapters are prior. A latent spatial mask need not commute with P, so it is not generally a polynomial spectral filter in P. No filter-coefficient diversity guarantee becomes ensemble quality. |
| GEENI and graph ensemble discovery | Exact efficient-GNN ensemble identity was reused. Its primary method remains unresolved in the latest saved packet. Metadata does not settle sharing, loss, graph correction or complete-recipe equivalence. |

## One prospective quality hypothesis

With a competent shared learnable fixed-P node-classification backbone, M=4 members and one first common-input gate site, use Q=2 bounded gates from the same public X as all controls and rho=1/4. Retain the native residual/root paths and existing private feature factors; learn all shared dense weights, the shared gate generator and centered private coefficients under ordinary member supervision. Serve all four paths and average class probabilities. This is a representative fixed design, with no strength grid or fit authorized here.

**Hypothesis:** a small bounded centered edge-factor bank improves pooled node-classification accuracy over the ordinary shared-factor bank at useful paid storage and serving cost, while retaining at least its mean member accuracy. Its value would be a measured inductive-bias and optimization result. An additional advantage over the same gate bank with an admitted common coefficient component is needed to credit the balancing restriction itself.

Use the ordinary bank, the same bounded gate bank allowing a learned mean coefficient, and a capable learned-gate single as the mechanism contrasts. Keep competent Polynormer, GCNII and H2GCN controls on identical X and the admitted graph/label roles. A same-operation untied four measures the sharing tradeoff if that claim is pursued. The exact structured single that contains all four trajectories and their probability mean is a renaming; it must not be fitted twice or counted as a new baseline. Charge gate work, dense work, all post-divergence sparse work, fitting, selection and memory.

The hypothesis fails if pooled accuracy does not improve, mean member accuracy falls, a capable same-information single explains the gain, or the paid advantage disappears. Lower NLL or greater disagreement alone cannot pass it. If the unbalanced gate bank wins, credit ordinary graph adaptation and reject a demonstrated balancing benefit. Constant gates can create pure propagation-gain differences, so a structural interpretation also requires showing dependence on learned node association beyond that gain effect. A first small block is developmental evidence, followed by a separately frozen confirmation before general claims. This packet contains no launch instruction.

## Reading and execution scope

Two new deduplicated public identities received bounded primary method reads. Graph WaveNet p1 title/abstract opening and pp2–3 were read; only p3 was visually checked. Pro-GNN p1 title/author header and pp3–4 were read; only p4 was visually checked. Surrounding related-work, motivation and temporal-method text was exposed; no numerical result, full proof, native code, full-paper or runtime certification is adopted. All-page extraction was mechanical. Adaptive Graph Convolutional Recurrent Network was metadata-only and is not counted as a primary read. Existing conclusions were reused without reopening their primary sources. Index72 was left unchanged.

The only executed mathematics was a short stdlib finite-arithmetic check. No model import, scientific fit, dataset, label, tensor, saved-state, prediction payload, server, new agent or held population was accessed. The initializer source review performed during this task is saved separately and does not supply evidence for the structural-bank hypothesis.
