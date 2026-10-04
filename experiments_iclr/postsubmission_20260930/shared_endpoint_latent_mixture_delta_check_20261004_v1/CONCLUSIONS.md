# Shared endpoint latent member: bounded literature and theoretical check

## Finding

The generic operations are established prior: reconstructing neighbourhood observations, sharing latent-class preference parameters, combining coordinate likelihoods under one latent class, responsibility-weighted learning, and conditioning an attachment/count model on a row total. The remaining proposed composition is **one link-specific latent member coupling both endpoint residual identity subsets, through member-specific graph filters shared with the GNNM that serves the uniform mean predictor after the auxiliary is removed**. Neither of the two newly read scopes establishes an exact duplicate of that complete training/serving composition. This bounded finding is not novelty clearance or a manuscript verdict.

A shared latent variable can explain endpoint dependence. It supplies neither a general specialization guarantee nor a general predictive-gain guarantee. Degree conditioning alone also does not establish degree invariance.

## Prior coverage and limits

| Intended ingredient | Evidence already available | What the inspected evidence does not establish |
|---|---|---|
| Neighbourhood reconstruction | Index v47 records 198–199: Generative GNNs for Link Prediction; GAD-NR | Benefit of this residual-subset auxiliary to the served GNNM |
| Shared mixture component across several graph observations and responsibility gradients | Index record 157: GRAN; Newman–Leicht Eqs. (4), (10)–(11); Amini et al. Eqs. (4)–(5) | Exact two-endpoint query-local subset law tied to the served member filters |
| Member assignment and specialization as a learning idea | Index record 167: sMCL | A specialization or uniform-mean prediction guarantee for this likelihood |
| Exact cardinality-conditioned binary subset law and ESP normalizer | Index record 195: Chen–Liu (1997); record 194: diagonal fixed-size DPP equivalence | A new distribution family; these operations are already attributed prior |
| Conditioning to remove row degree amplitude | Amini et al. §2.2; prior degree-check packet | Removal of candidate-specific popularity, whole-graph degree conditioning, or community recovery in the proposed neural model |
| Auxiliary removal and predictive transfer through shared member filters | The proposed construction | This is the unresolved empirical relation; likelihood fit or responsibility agreement cannot establish it |

### Precise statistical and implementational delta

The intended local observation is a pair of **binary identity subsets** \((S_{u\setminus v},S_{v\setminus u})\) for a queried edge \((u,v)\). One edge-specific latent index selects the member for both sides. Each member uses the already prior fixed-cardinality law

\[
q_m(S\mid C,k)=\frac{\exp(\sum_{w\in S}\eta_{mw})}{\sum_{T\subseteq C,\ |T|=k}\exp(\sum_{w\in T}\eta_{mw})},
\]

with logits tied to that served member's embeddings/private graph filters. The served predictor retains its uniform member mean; the auxiliary is a training score. The fixed-cardinality family and its ESP computation are **not** the proposed contribution: index record 195 explicitly identifies the single-side law with Chen–Liu's conditional Bernoulli family. At \(k=1\) it is categorical (also recorded at index 190); there is no new single-choice probability algebra.

Newman–Leicht instead assigns a class to each **node** and learns global class-to-node preference tables. Its undirected extension uses the two endpoint node classes in each edge factor, not one queried-edge class jointly assigned to two masked residual subsets. Amini et al. assign one latent node class to an aggregated **row block-count vector**, then use a multinomial law conditional on its row total. Those specified models do not tie latent classes to neural ensemble filters that also serve a link ranker, nor do they specify this auxiliary's removal at serving. The binary distinct-subset support is different from the native multinomial count support, although the size-one categorical case overlaps exactly.

Thus the complete composition and placement are not established as duplicated in these scopes. Each statistical ingredient remains attributable prior, and the useful claim to test is predictive transfer through the tied served-member filters. This is an implementation/statistical distinction from these two models, not a certification that no earlier method made the adaptation.

**Newman and Leicht (2007), method only, PDF pages 2–4, Eqs. (1)–(13).** A node has one latent class whose globally shared preference vector explains all its observed outgoing neighbours. With class weights \(\pi_r\), attachment preferences \(\theta_{rj}\), and adjacency \(A\), the written likelihood contains \(\prod_i\pi_{g_i}\prod_j\theta_{g_i,j}^{A_{ij}}\). Its posterior responsibility is proportional to \(\pi_r\prod_j\theta_{rj}^{A_{ij}}\); its preference update is \(\theta_{rj}=\sum_i A_{ij}q_{ir}/\sum_i k_iq_{ir}\). This is attachment-preference likelihood algebra, not the current exact fixed-cardinality binary-subset law or an audited Bernoulli whole-graph density. Source: <https://arxiv.org/pdf/physics/0611158v3>.

A useful derived fixed point is \(\theta_{rj}=d_j^{\rm in}/E\) for every class, \(E=\sum_i k_i>0\), with positive class weights. Then \(q_{ir}=\pi_r\), and the M-step returns the same weights and identical popularity vectors. Zero-popularity coordinates can be omitted or handled by the usual support convention. Thus even this classical neighbourhood model has a collapsed popularity-only fixed point. The paper discusses symmetric initialization and random perturbation; its stability or convergence claims are not transferred to GNNM.

**Amini, Chen, Bickel and Levina (2013), §§2.1–2.2 only, PDF pages 4–7, Eqs. (2)–(5) and EM updates.** Their block sums \(b_{ik}=\sum_j A_{ij}1(e_j=k)\) form an approximate independent-Poisson vector given one latent node class. They ignore dependence between rows to construct a graph pseudolikelihood. Conditioning on the row degree gives a mixture-multinomial pseudolikelihood with \(\theta_{lk}=\lambda_{lk}/\lambda_l\), proportional per row to \(\sum_l\pi_l\prod_k\theta_{lk}^{b_{ik}}\). Shared-class explanation of several count coordinates, conditional-degree normalization and posterior training are prior. Identical parameter rows are an identifiability problem. No consistency theorem, initialization guarantee or empirical result is adopted. Source: <https://arxiv.org/pdf/1207.2340v3>.

## What the joint term rewards

For one fixed query context and normalized component subset laws, put

\[
a_m=q_{Lm}(S_L),\quad b_m=q_{Rm}(S_R),\qquad
q_J=\frac1M\sum_m a_mb_m,\quad
q_F=\left(\frac1M\sum_m a_m\right)\left(\frac1M\sum_m b_m\right).
\]

For positive likelihoods, define \(r_{Lm}=a_m/\sum_j a_j\) and \(r_{Rm}=b_m/\sum_j b_j\). Then

\[
\frac{q_J}{q_F}=M\langle r_L,r_R\rangle.
\]

The joint term rewards endpoint responsibility agreement on the observed subsets. It does not directly reward member diversity. Identical components give uniform responsibilities and ratio one. This same-logit identity does not equate separately trained models. It matches the already fixed endpoint-agreement diagnostic; it adds no new experiment protocol.

## One recommended theoretical proposition

**Proposition: shared latent subset likelihood has no general specialization or served predictive-gain guarantee.**

**Assumptions.** Condition on a query context \(x\), including finite candidate sets and fixed selected counts. Let \(\mathcal S_L(x),\mathcal S_R(x)\) be their finite supports. Each component supplies normalized endpoint laws, and their joint is the uniform mixture \(q_J\) above, with \(M\ge2\). The objective considered is population auxiliary log loss, with no extra member-repulsion term. For the first assertion, the true law is

\[
P(S_L,S_R\mid x)=p_L(S_L\mid x)p_R(S_R\mid x),
\]

where one member can represent both strictly positive factors simultaneously, and copying that member to every slot is allowed. The supports and factors may be informative and nonconstant.

**Assertion and proof.** A collapsed global auxiliary optimum exists. Set every member to \((p_L,p_R)\). Then \(q_J=P\). For any other normalized mixture,

\[
\mathbb E[-\log q_J]=H(P)+D_{\rm KL}(P\|q_J)\ge H(P),
\]

averaged over contexts if necessary. The copied solution attains the bound and has uniform responsibilities. Thus correct subset learning need not yield different members. Separately, an exactly symmetric initialization remains symmetric under a differentiable member-permutation-invariant loss and a deterministic equivariant update with equal optimizer states and matched stochastic decisions: permutation symmetry makes every member gradient identical at the symmetric state, and induction preserves equality. Independent dropout, asymmetric initialization or other asymmetric updates can break that state; no inevitable collapse or stability claim is made.

**Dependent-subset counterexample and predictive limit.** Let \(M=2\); each side selects one of two candidates \(A,B\). The true observed pairs are \((A,A)\) and \((B,B)\), each with probability \(1/2\). Member 1 chooses \(A\) on each side with probability \(1-\varepsilon\); member 2 chooses \(A\) with probability \(\varepsilon\), for \(0<\varepsilon<1/2\). The joint mixture gives each aligned pair probability

\[
q_J(A,A)=q_J(B,B)=\tfrac12[(1-\varepsilon)^2+\varepsilon^2],
\]

while the separately mixed marginals give all pairs probability \(1/4\). The joint expected-NLL advantage is \(\log(2[(1-\varepsilon)^2+\varepsilon^2])>0\), approaching \(\log2\) as \(\varepsilon\downarrow0\). This establishes the dependence mechanism, using ordinary latent-class algebra.

Even **tied embeddings and the same scoring formula** need not transmit that gain to the queried edge. As a concrete witness, use the bilinear score \(s_m(i,j)=h_m(i)^\top h_m(j)\), with enough filter capacity to realize the following vectors. For both members set \(h_m(u)=h_m(v)=e_1\). Put \(t=\tfrac12\log[(1-\varepsilon)/\varepsilon]\), and set

\[
h_1(A)=te_1,\quad h_1(B)=-te_1,\qquad
h_2(A)=-te_1,\quad h_2(B)=te_1.
\]

Using those same bilinear scores as the count-one residual-subset logits gives exactly the two components above on both sides. Yet the served queried-edge logit is \(s_1(u,v)=s_2(u,v)=1\), and its uniform mean remains 1 for every \(t\). The target label can be independent of the auxiliary common-candidate variable, with its conditional probability already represented by that fixed served score. Auxiliary specialization and a strict joint likelihood advantage coexist with exactly unchanged served prediction. This witness requires sufficient filter capacity and this bilinear scorer; it is not a certification of the actual GNNM decoder's nullspace. More generally, a representation/readout nullspace gives the same failure of implication. A transfer theorem for the actual architecture would need assumptions that align the auxiliary relation with the target and exclude directions that alter subset probabilities while leaving served query scores unchanged.

This single proposition is the recommended discriminating result. It directly separates local dependence modelling, member specialization, and served prediction. The previously adopted joint-versus-separate served-ranker comparison remains the empirical requirement; no new heldout access or model run is recommended by this packet.

## Degree bias and probability scope

For the current count-conditioned subset law, adding the same scalar to every candidate logit cancels. Adding candidate-specific \(\log(1+d_w)\) does not: with two candidates and count one, selection odds are \((1+d_1)/(1+d_2)\). This is the prior degree-check conclusion. Correlated degree/popularity patterns can also generate aligned endpoint responsibilities, so concentration or agreement alone is not proof of a substantive graph pattern. Retain the already specified uniform and visible-degree references; do not add an audit ladder or infer degree invariance from the fixed count.

A normalized query-local law over \((S_L,S_R)\) does not make a product over overlapping queries a normalized whole-graph generative density. Without a compatibility/normalization proof for that larger object, describe the training score as local conditional/composite likelihood. Classical pseudolikelihood supplies a relevant scope precedent.

## Provenance and action boundary

The initial inputs were index v47 and the prior degree-check conclusions, scopes and prospective diagnostic plan. Reused index records were consulted through memory, with zero indexed-primary rereads. Exactly two previously unindexed primary papers were read in the bounded method scopes above. Four HTTP requests succeeded (two exact arXiv identity queries and two version-pinned PDFs); there were zero failed requests and no broad absence search. Full PDFs and mechanical extractions are provenance carriers, not full-paper certifications. Exact hashes, semantic boundaries, visual checks and incidental exposure are in `READ_SCOPES.json`.

No heldout reads, model runs, project source edits, scientific-server actions, extra agents, index edits or changes to the sealed independent PENCIL source review were performed. No execution is authorized by this packet.
