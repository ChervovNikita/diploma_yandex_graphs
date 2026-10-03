# Member-specific graph filters on a shared feature backbone

3 October 2026. Bounded quality-gap review; unadopted research only. Reference memory: **index_v30**, 127 saved conclusion records, 83 normalized paper IDs and two software IDs. These are memory records, not full-paper-read counts.

## Decision

**A precise limitation of final-head-only diversity exists, but no new graph operation survives this review strongly enough to justify a new pilot.** Final heads cannot recover evidence discarded by their common representation. Private graph filters placed before that loss and before nonlinear processing can preserve different neighborhood evidence. Learned graph-filter banks, nonlinear processing of different spectral channels, graph-neighborhood ensembles and graph-filter diversification already have close published precedents.

The useful unresolved question is narrower: **does allocating learned neighborhood filters to predictive members, while tying feature maps, improve pooled quality at the paid training and serving budget over a competent multiscale backbone with equally capable heads, a nonlinear filter-bank predictor, all-layer BatchEnsemble with exact member messages, and an independent ensemble?** The inspected sources do not settle that comparison. No target-quality advantage, exact equivalence to every prior implementation, global absence of prior, or novelty certificate is claimed.

**New pilots proposed: zero.** This decision follows the mathematical reductions and close method overlap below. It does not declare the architecture useless. A later experiment could assess an explicitly attributed filter-bank ensemble, but this packet does not promote that as a newly discovered method or authorize execution.

## 1. What a final head cannot do

Write a head-only ensemble as

\[
H=F_W(P,X),\qquad z_m=h_{\phi_m}(H).
\]

For any two graph/input pairs giving the same complete representation \(H\), all member outputs are identical. Within one graph, nodes that have identical rows of \(H\) are also indistinguishable to nodewise heads without additional node/context inputs. Changing the private classifier, its initialization or its loss cannot invert information already removed by \(F_W\).

An analytic witness is two-node averaging,

\[
P=\tfrac12\begin{pmatrix}1&1\\1&1\end{pmatrix},\quad
X=\begin{pmatrix}1\\-1\end{pmatrix},\quad PX=0.
\]

Heads on \(PX\) cannot distinguish the nodes. An identity or high-pass branch applied to \(X\) before averaging can retain the antisymmetric signal. This is a function-class witness, not a dataset experiment or evidence of generalization.

**Placement matters.** If the shared feature backbone has already erased that signal and private filters receive only the erased \(H\), filtering cannot restore it. A private filter remedy needs a non-erasing shared feature stem or access to the missing feature/graph evidence. Residual feature paths, polynomial token banks, multiple propagation scales and local/global processing can already remove the alleged bottleneck. Heterophily alone does not establish that a competent common representation loses useful information.

## 2. Three exact reductions that bound the claim

### Linear mean-logit filtering is one predictor

With shared features \(H_0\), shared classifier \(B\), and linear private filters,

\[
z_m=G_m(P)H_0B,\qquad
\bar z=\left(\frac1M\sum_mG_m(P)\right)H_0B.
\]

For \(G_m(P)=\sum_{k=0}^K a_{mk}P^k\), the served mean raw logits are exactly one polynomial with the mean coefficients. If classifiers \(B_m\) are private instead,

\[
\bar z=\sum_kP^kH_0C_k,\qquad
C_k=\frac1M\sum_ma_{mk}B_m.
\]

That is an ordinary multiple-basis linear predictor. Member training objectives can induce a different regularization or trajectory, but naming these linear components members does not create another pooled function class. This reduction concerns raw-logit averaging; averaging probabilities is a different predictor.

### A complete shared basis can absorb a one-stage private filter

Let a common representation retain the complete chosen basis,

\[
R=\operatorname{Concat}_{k=0}^K\big[T_k(P)H_0\big],\quad
A_m=\operatorname{Col}_{k=0}^K[a_{mk}I].
\]

Then \(G_m(P)H_0=RA_m\), and any one-stage member
\(z_m=q_{\phi_m}(\sigma(G_m(P)H_0))\) is exactly the head
\(q_{\phi_m}(\sigma(RA_m))\) on the common basis. An appropriately parameterized head reproduces its forward operation. With the same parameterization, objective and optimizer, this is merely relocating the coefficient mixing into the head.

This statement assumes the complete same basis, node-independent scalar coefficients, the same feature stem, and head access to every basis channel. It does not cover arbitrary adaptive neighborhoods or repeated propagation of member-dependent nonlinear hidden states. It shows why a narrow linear-classifier control on a compressed low-pass representation is insufficient. A full basis cache has a real memory/preprocessing cost that must be charged.

### Coefficient separation need not separate useful responses

For \(V_k=T_k(P)H_0\), define \(C_{kl}=\langle V_k,V_l\rangle_F\). The squared feature response difference of coefficient vectors \(a,b\) is

\[
\left\|\sum_k(a_k-b_k)V_k\right\|_F^2
=(a-b)^TC(a-b).
\]

Distinct or orthogonal coefficients can give identical responses when their difference lies in the data-response nullspace. They can also separate nuisance high frequencies while leaving errors strongly correlated. Coefficient diversity, graph-frequency diversity and complementary correct predictions are different quantities. A response penalty alone would also need attribution to the existing diversity literature; this observation does not supply a new regularizer.

## 3. Three newly read primary method scopes

### BankGCN — direct nonlinear learned-filter ancestry

*Message Passing in Graph Convolution Networks via Adaptive Filter Banks*, arXiv:2106.09910v1 (18 June 2021). PDF pages 4–7, complete Section 4 and equations 7–20; page 8 Section 5.1 architecture/objective. Header identification on page 1. Results and proofs were not qualified.

Its subspace operations are

\[
r_p=XW_p+b_p,\quad
h_p=\left(\sum_{k=0}^K\alpha_{pk}T_k(L)\right)r_p+r_p,\quad
X^{\ell+1}=\operatorname{ReLU}(\operatorname{Concat}_p h_p).
\]

Different feature subspaces receive learned graph filters, followed by nonlinear processing; later projections can mix preceding subspaces. Its filter-diversity term penalizes the maximum absolute cosine similarity between coefficient vectors. Graph-level readout and one final CE classifier follow the stacked bank.

**Consequence:** learned graph evidence diversification before nonlinear processing is established. BankGCN is a single graph-classification predictor, rather than a shared-weight node ensemble trained with member losses. Those boundaries leave a sharing/objective/utility comparison; they do not make learned private polynomial filters a new graph primitive. Its coefficient penalty has the response-nullspace limitation above. Published graph-classification utility is not transferred to the target node ensemble.

### Specformer — strong spectral capacity control

*Specformer: Spectral Graph Neural Networks Meet Transformers*, arXiv:2303.01028v1 (2 March 2023), ICLR 2023. Complete main Section 4.1–4.4, PDF pages 3–5, equations 2–6, property statements and complexity/scalability discussion. Proofs, complete evaluation, native recipes and author implementation were not audited.

Specformer encodes the eigenvalue set, applies a spectral transformer, learns multiple spectral bases, reconstructs \(S_m=U\operatorname{diag}(\lambda_m)U^T\), and combines them through learned feature-channel operators. Feature mixing and nonlinearity follow; filter and FFN sharing across layers varies by Small/Medium/Large architecture.

**Consequence:** channel-specific learned operators and nonlinear multiple-filter representations are a close capacity control. It is one predictor, not independently trained predictive members. Full eigendecomposition, dense operators/attention and preprocessing belong in its cost. A truncated eigenspace is a changed control and must be labeled; it cannot silently stand for the full spectral reference. Stated approximation properties are not guarantees of pooled quality or calibration.

### HGEN — different neighborhood evidence and predictive fusion

*HGEN: Heterogeneous Graph Ensemble Networks*, IJCAI 2025, DOI:10.24963/ijcai.2025/685. Official eight-page PDF. Main Sections 2 and 3.1–3.4, PDF pages 2–5, equations 1–9 and theorem/remarks were read; page 1 supplied identification, motivation and a GENN locator. The referenced Appendix Algorithm 1 and Supplement D are absent from these bytes and were not retrieved. The theorem was not independently certified.

HGEN constructs different symmetric meta-path graphs, trains multiple GNNs with feature dropout per path, attention-fuses embeddings within paths, and sums per-path classifier outputs for one pooled CE. Its diversity term uses an uncentered Gram of graph-pooled embeddings. That includes scale/diagonal effects and does not by itself guarantee complementary errors.

The linked author source was statically inspected at commit **3caba805b2c3e16ee2dfd7d56d7e79405f66fd01**, dated 29 August 2026:

- `model.py:188–198,314–345`: each path/member receives a separate feature encoder and private GCN layers. Edge/node dropout calls in this path are commented; feature dropout is active.
- `model.py:201–292`: an initial learner pass produces an unused prediction stack, then learners are called again for the attention-fused outputs; decoded path logits are summed and log-softmaxed. This is a static source observation, not a timing measurement.
- `train.py:193–222`: one optimizer over all parameters, fused NLL plus \(\lambda\|S\|_1^2\), one backward/step. The printed paper uses \(\lambda\|S\|_1\).
- `model.py:154–162`: min-max attention has no visible denominator epsilon; the uniform residual is added to members 1 through \(k-1\), but not member 0, unlike the paper's formula.
- Static `ast.parse` of the saved `model.py` raises **TabError at line 355**. No code was imported or executed. These bytes are not a qualified runnable reproduction.

**Consequence:** ensembles receiving different graph neighborhoods are established. The inspected HGEN path has separate encoders and graph maps, rather than a common feature backbone with only private scalar filters. Source discrepancies limit exact implementation/replication claims; they neither erase published ancestry nor provide novelty evidence.

## 4. Strongest controls and what each would resolve

The controls below are comparison requirements, **not a new pilot protocol or launch recommendation**. Native source/recipe qualification remains necessary where it is not already saved.

| Control | Scientific role |
|---|---|
| Same shared feature stem and same private heads, common learned filter | Tests whether allowing private filter coefficients matters beyond ordinary head diversity; use the same nonlinear depth and objective. |
| Shared complete polynomial/token bank and equally capable private nonlinear heads | Exact one-stage absorption control above. Tests an alleged information advantage rather than comparing to a head denied available scales. Charge the complete bank cache. |
| Capacity-matched nonlinear filter-bank single predictor | BankGCN/Specformer operation family; distinguishes ordinary channel capacity from useful predictive-member allocation. Equal width alone does not match parameters or paid work. |
| **TFE-GNN** (NeurIPS 2024, saved primary scope) | Learns low/high-pass power streams and sums/concatenates them before one MLP CE head. A close homophily/heterophily filter-combination backbone. The word ensemble refers to filter streams, not persistent predictive members. |
| **PolyFormer** (KDD 2024, saved primary and source scope) | Precomputed polynomial tokens, order-specific MLPs, nodewise token attention, residuals and FFNs. Modern multiscale control against the premise that the common encoder must discard high-frequency/hop evidence. |
| **Polynormer** (ICLR 2024, saved primary and source scope) | Competent local/global graph-transformer reference; nonlinear local processing plus linear global attention. Each member's local scores, projections and reductions must remain resolved when used in an ensemble. |
| All-layer BatchEnsemble of the qualified multiscale backbone, **exact member messages** | Strong sharing control. A common hidden-message approximation introduces a separate information restriction and cannot represent the strongest rank-one member ensemble by default. Keep complete private nonlinear trajectories. |
| Independent ensemble of the same backbone; graph-diversified experts | Tests whether tying features creates the claimed useful diversity/cost trade-off. Saved *Training Diverse Graph Experts* already diversifies filters/directions/init/data and later learns heldout fusion. Offer competent packing to independent members and report total member acquisition. |
| HGEN for an actual heterogeneous/meta-path task; GraphMoRE/ADaMoRE as qualified adjacent expert controls | Different structural evidence, graph-conditioned expertise and diversity/fusion already occur in graph methods. Domain, objective and serving pool must be declared rather than treated as interchangeable. |
| Raw-feature MLP and competent residual GNN | Establish task utility and whether graph evidence helps at all under the selected version and label regime. |

Saved BernNet and GPNet conclusions additionally establish learnable graph polynomials, bands, signed filters and multiple-hop channels. Saved MORGAN author-source conclusions establish learned eigenvector band operators and eigenvalue-derived gates, while full primary access remains unresolved. GRAND establishes stochastic propagated views with shared MLP and consistency; its deterministic served predictor differs from a persistent learned member ensemble. None was reread as a new primary scope.

GRAIN and AutoSGNN remain metadata/source-limited leads; no primary mechanism or superior performance is inferred from their names or repository trees. The saved v3 heterophily reassessment requires competent symmetric tuning and exact dataset-version labeling. Neither a dataset's heterophily label nor a selected paper table establishes the best control for this question.

## 5. Plausible quality advantage and failure conditions

The plausible advantage is that different neighborhood operators expose complementary label-relevant signals **before an irreversible common propagation step**, while a shared feature stem amortizes learned feature maps. Nonlinearity after private filtering can preserve effects that average-filter reduction loses. Member supervision might make the exposed signals useful to distinct decisions. These are hypotheses about the training outcome, not consequences of parameter or coefficient diversity.

The advantage can fail when the common modern backbone already retains the required scales; the shared stem erases the features first; filters converge to similar responses; high frequencies contain noise; label support cannot fit multiple filters; private paths make correlated mistakes; fixed pooling dilutes a strong member; or extra nonlinear capacity explains the gain. Tying feature maps can itself prevent member specialization. Multiple propagations, dense projections, state storage and backward paths remain paid work even when most parameters are shared. A quality gain that is matched more cheaply by a competent single or independent baseline does not establish the proposed sharing benefit.

Useful predictive evidence would require pooled NLL/accuracy and member competence/error complementarity under the declared serving pool, alongside response diversity and total measured cost. No theorem here implies improved CE, calibration, uncertainty or robustness. No new thresholds, experiments, data access or GPU requests are introduced.

## 6. Closed evidence remains closed

Only the three explicitly authorized closed summaries were used; their exact hashes are in `CLOSED_DECISION_BINDINGS.json`.

- The 24-fit/replay CPU PPI spectral study failed every primary 10%-label gate against tied/random/raw controls; full-label results do not rescue that failure. Convergence was unestablished, and four full-width paths/12 graph propagations remained. This is a failure of the tested SAGE output-correction recipe, not all learned graph filters.
- The completed 12-fit PolyFormer-Mono comparison triggered its strict arithmetic rule but improved by only tens of micro-nats, with all arms selecting after one/four continuation updates. It establishes no practical superiority or mechanism confirmation.
- The complete35 DBLP relation-CP study failed both global-BE and shared-relation development gates: CP minus global BE mean NLL −0.0003424 (3/5 wins), and CP minus shared relation +0.0002235 (2/5 wins), far below the original 0.005 threshold. It does not prove universal impossibility of relation conditioning.

No raw artifacts, live outcomes, initializer outcomes, mixed40 outcomes, native15 outcomes or BUDDY outcomes were accessed. These closures are adverse bounded evidence, not a reason to manufacture a filter remedy or restart a frozen study.

## 7. Search, read and custody limits

Seven OpenAlex metadata queries returned 78 rows, including duplicates and many irrelevant broad-query hits. These counts do not establish coverage. Three previously unread primary method scopes were read: BankGCN, Specformer and HGEN. **Full-paper certifications: zero.** New author-source audit: one bounded HGEN path; source execution/training: zero. The native PDF scope and incidental exposure are recorded in `READ_SCOPES.json`; saved sources and source coordinates are bound in `EVIDENCE_MAP.json`.

The closest unresolved metadata lead is *Graph ensemble neural network*, Information Fusion 2024, DOI:10.1016/j.inffus.2024.102461. Crossref identifies the paper and authors; its ScienceDirect abstract landing returned 403. HGEN's introduction describes GEN as integrating ensemble operations throughout GNN training, but that is a secondary characterization. This packet does not infer its primary parameter-sharing, graph operators, loss or pooling, and does not use unavailable access as novelty evidence. No fourth method was read.

All retrieval receipts, exact downloaded bytes, extracted page text, the three inspected PDF renders, pinned source, candidate dispositions and reused conclusion bindings are retained. Downloaded arXiv PDF URLs were unversioned; PDF footers and landing histories identify the exact v1 bytes. The HGEN supplement was not retrieved. Index_v30 and prior summaries are hash-bound and unchanged. No index append, canonical ledger/status edit, frozen-study amendment, model fitting or outcome-dependent design took place.

**Disposition:** retain the head-information limitation and the unresolved sharing/utility comparison; attribute the graph operations to the closest priors; propose zero new pilots. Parent review owns any later adoption or explicitly attributed baseline study.
