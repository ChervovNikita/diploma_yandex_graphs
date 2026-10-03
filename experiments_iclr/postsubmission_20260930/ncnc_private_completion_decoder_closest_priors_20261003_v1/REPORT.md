# NCNC private completion/decoder pairing: bounded current-prior comparison

**3 October 2026. Source reading and symbolic comparison only.** Six public metadata discovery calls were used, followed by five exact selected-source retrieval attempts. Two first scoped method identities were read (IECNC and E-GAE), and PENCIL's retained v4 scope was extended for a changed operation-level question. No full-paper or author-code certification, prototype edit, model/data operation or new experiment is supplied.

## Conclusion

**Learned completion weights coupled to a model's graph features already have direct ancestry.** NCNC is the original reference; the newly read **IECNC (2025)** also estimates residual-neighbor probabilities and combines their transformed weights with context-aware MPNN features. Native independent NCNC members already preserve each member's completion/decoder association. Ordinary combination of complete expert scores preserves their internal operations too.

The proposed delta is narrower: retain that association **under the specified shared encoder and factorized decoder bank**, instead of replacing each branch's post-clamp weights with their mean before its own feature aggregation/decoder. This is distinguishable from that matched pooled-weight control. It is not a new completion primitive, a universal ensemble function-class advantage, or a guarantee that preserving association helps. A capable deterministic grouped model can implement the identical final score.

The current primary comparison adds two useful boundaries. **PENCIL v4 (2026-09-28)** constructs its propagation adjacency from the observed subgraph's token fields; it does not learn an NCNC-style missing-edge probability bank. Its learned query-conditioned transformer still makes it a capable structural competitor. **E-GAE (2025)** explicitly fuses model embeddings before a common decoder, establishing a concrete pre-decoder pooling design whose operation differs from combining complete predictions. Neither source establishes equality to the exact proposed factorized NCNC composition.

The saved paired hypothesis remains scientifically interpretable, with no present advantage evidence. Full published overlap is unresolved; the bounded search and inaccessible 2025 framework source cannot certify novelty or a complete prior frontier.

## Starting map and custody

Read the saved `link_prediction_quality_gap_20261003_v1` report/conclusions/citations and retained index before retrieval. Its source map supplies NCNC's detached depth-1 completion scorer, native clamp and recursive transformations; BUDDY's shared deterministic cache; LPFormer's learned pair context; Link-MoE's complete-expert score combination; and source/protocol differences for collab. Those existing method scopes were reused rather than repeated. Pinned NCNC/prototype files were not reopened or changed in this lane.

The earlier report includes historical outcome-summary sentences; those were incidentally exposed as starting context. No original outcome files, live state, dataset, labels, caches, tensors, checkpoints or server was opened. Public paper empirical text exposed during source extraction is recorded separately and is not adopted as a quality/cost result.

The separately prepared `literature_memory/index_v32` preserves all **130** v31 conclusion records, **165** catalog entries and **87** normalization groups exactly, adding the three sealed conditional-graph-response priors. It contains **133 conclusion records, 88 paper identities and 2 software identities**; these are not full-paper reading totals. This LP packet is **not** appended to that index. Canonical ledger/status and old indices remain unchanged.

## 1. IECNC: a direct current learned completion prior

Zhengyun Zhou, Guojia Wan and Bo Du, **Common Neighbor Completion with Information Entropy for Link Prediction in Social Networks**, *Data Science and Engineering* 10:40–53, [DOI 10.1007/s41019-024-00267-6](https://doi.org/10.1007/s41019-024-00267-6). Published online **2025-01-24**. The exact Springer PDF was retrieved and hashed; it has 14 PDF pages. Selected scope: PDF p4 entropy/link-context definitions and §4.1; p5–p6 §§4.1–4.3/Eqs5–9; p7 method text/Eqs10–11 before §5. Visual checks p5/p6/p7. Global keyword locators and incidental p7 experiment description were exposed; full experiments and proofs were not reviewed.

### Decisive operations

- PDF p5 §4.1/Eq6 represents endpoint features together with union, common-neighbor and unilateral-neighbor logical features, computed by MLPs. Eq7 predicts a connection from those features and context.
- PDF p5 §4.2 assigns \(P_{uij}=1\) to observed common neighbors, learned softmax values to unilateral residual-neighbor sets, and zero otherwise. The text describes predicting a missing adjacency and using it as the probability of becoming a common neighbor.
- PDF p6 Eq9 explicitly forms \(\operatorname{PPNC}(i,j)=\sum_{u\in q}P_{uij}\operatorname{MPNN}(u,M,C_{ij})\).
- PDF p7 Eq11 replaces the weight with \(-P_{uij}\ln P_{uij}\): \(\operatorname{IECNC}(i,j)=-\sum_{u\in q}P_{uij}\ln P_{uij}\operatorname{MPNN}(u,M,C_{ij})\).

Thus learned completion-related weights and learned contextual graph features are already associated **within one predictor before its final score**. This is closer ancestry than general graph attention. It also shows that applying a confidence/uncertainty transform to completion weights is an established direction. The current hypothesis's member axis, factorized sharing, native NCNC clamp and post-clamp pooling twin were not established in this source scope.

### Qualification limits from the printed method

The source calls \(-p\log p\) the uncertainty of one node's predicted common-neighbor event and says it is greatest near 0.5. For a scalar \(p\in(0,1)\), its derivative is \(-\log p-1\), so the printed term peaks at \(p=e^{-1}\). It omits the \(-(1-p)\log(1-p)\) term of Bernoulli entropy. Categorical entropy would require a specified normalized distribution; the softmax axis and combination of observed-common-neighbor ones with unilateral probabilities are not sufficiently specified by these passages to certify that interpretation.

There is also a direct semantic difference: the displayed \(P=1\) for an observed common neighbor yields zero weight in Eq11, whereas native NCNC preserves actual-common-neighbor contributions. The printed \(q\) and feature-output dimensions, softmax versus scalar missing-link semantics, and final loss/decoder interface require source qualification. The acronym MPNN is described as Message Passing Neural Network in one passage and Multi-Perceptron Neural Network in another. No author code was read or executed.

These observations preserve the exact operator ancestry while limiting reproduction and uncertainty claims. They do not establish that the paper's implementation follows every printed ambiguity, that its reported predictions are invalid, or that the candidate performs better. A future IECNC-informed comparator must resolve the essential semantics first; an arbitrary entropy-weight stand-in would not be a competent source control.

## 2. PENCIL: current learned structural capacity, with a precise adjacency distinction

Quang Truong et al., **Plain Transformers are Surprisingly Powerful Link Predictors**, saved [arXiv:2602.01553v4](https://arxiv.org/html/2602.01553v4), dated **2026-09-28**. Exact saved HTML SHA256 and prior scope IDs are bound. No new retrieval was needed. Prior reading selected nine §§3.1–3.2 paragraphs, but not their equations or the relevant Appendix A/B passages. This lane read those missing equations and appendix passages rather than repeating the selected paragraphs.

### What is reconstructed, and what is learned

Main Eqs1–3 give token embedding \(H^{(0)}=\tilde XW_0+G(X)\), a learned transformer \(Z^{(k)}=T_k(H^{(k-1)})\), and the residual \(H^{(k)}=Z^{(k)}+P_k(\tilde A Z^{(k)})\). Appendix A Eqs6–8 reconstruct \(\tilde A\) from **observed adjacency and identifier blocks in the input tokenization**, adding identifier/self links and zero outgoing columns for the two task tokens. The appendix states that the propagation branch uses row normalization.

This adjacency reconstruction is deterministic conditional on the sampled/tokenized subgraph. It must not be described as learning missing-edge probabilities or maintaining a member-conditioned completion bank. The learned attention, feature projection and propagation transform still adapt to the query subgraph. The native feature/config and sampling/target-removal differences remain those already qualified in the saved source report, not a cache-compatible NCNC/BUDDY substitution.

### The comparison theorem does not settle the proposed gap

Appendix B.9 aligns ELPH, Neo-GNN, NCN and SEAL with the cited link-representation framework. Its primary theorem compares **PENCIL with LRP to SEAL under the same sampling constraint**, for suitable parameters. The proof sketch aggregates over endpoint-preserving relabelings; the source expressly says **LRP is not used in its experiments**. Its ordering contextualizes NCN, but this is not a theorem about NCNC's learned completion, the proposed private-versus-pooled twins, fixed finite compute or realized optimization quality. The full attention-enabled expressive power remains open in the source.

The practical advantage gap therefore cannot be reduced to a small parameter count or terminal-head argument. PENCIL is a current single model that can use richer query-conditioned structural context. The paired NCNC comparison must first resolve the specific association under the same architecture, then face a competent current single predictor for any broader LP-quality/cost claim. No published superiority or collab benefit is transferred here.

Incidental newly exposed §3.3 heuristic-regression claims and appendix empirical statements were not benchmark-verified. This is a targeted retained-identity scope extension, not a new paper identity, whole-paper review or author-source revisit.

## 3. E-GAE: a concrete current pre-decoder pooling prior

Chengxin Xie, Jingui Huang, Yongjiang Shi, Hui Pang, Liting Gao and Xiumei Wen, **Ensemble graph auto-encoders for clustering and link prediction**, *PeerJ Computer Science* 11:e2648, **2025-01-22**, [DOI 10.7717/peerj-cs.2648](https://doi.org/10.7717/peerj-cs.2648). The publisher PDF returned 403; the exact PMC full-text article [PMC11784894](https://pmc.ncbi.nlm.nih.gov/articles/PMC11784894/) was successfully retrieved. Selected primary scope: “Proposed model,” nested sections `sec4`–`sec9`, Eqs1–5 and Algorithms1–2. Abstract/metadata and incidental Table1 text were exposed. Full empirical evaluation and author code were not reviewed.

Algorithm2 computes three node-embedding matrices from RWR-GAE, GATE and EGSRWR-GAE, forms \(Z=w_1Z_1+w_2Z_2+w_3Z_3\), then reconstructs adjacency with a decoder. Eq5 repeats the adaptive embedding combination. The EGSRWR-GAE subsection describes GCN/GAT/SuperGAT feature combination before inner-product reconstruction; the loss section separately combines component losses.

This is an explicit **latent pooling before decoding** design, not a demonstration of preserving each complete learner's final score or its private completion/decoder bank. It is not the same operation as the candidate's averaging of post-clamp residual-neighbor probabilities. It establishes that the location of graph-ensemble fusion is already a design choice in primary literature.

In the scalar-weight/inner-product special case, decoding the fused embedding gives \(h_u^\top h_v=\sum_{a,b}w_aw_bz_{a,u}^\top z_{b,v}\), including cross-component terms. Pooling complete component inner-product scores would retain only their own component terms with the specified score weights. These are different functions; fusion before decoding is not uniformly an information loss or uniformly worse. The paper's adaptive-weight dimensions and implementation are not certified here, so this is a symbolic special case, not a reproduction claim. Its cited datasets and published gains do not qualify official collab Hits@50 or full cost.

## 4. What is already preserved or pooled

| Prior/design | Fusion location and retained association | Relation to the exact hypothesis |
|---|---|---|
| Native NCNC; independent complete NCNC members | Each predictor's own missing-link scorer/clamp, weighted features and decoder; final prediction pool can occur afterward | Direct attribution. Independent members already preserve the own-weight/decoder pairing. |
| Link-MoE and retained LP stacking | Complete expert scores combined by a learned gate/combiner | Internal expert operations remain intact. No source equality to the proposed homogeneous factorized NCNC bank was established. Validation supervision/cost must be matched. |
| IECNC 2025 | Learned neighbor probabilities or their entropy transform multiplied by contextual learned features before a score | Direct current completion-feature association prior. Different transform and unresolved source semantics; not a verified shared ensemble implementation. |
| E-GAE 2025 | Adaptive embedding mixture before common adjacency decoder | Explicit pre-decoder pooling precedent; not query residual-completion probability pooling. |
| PENCIL v4 | Learned query-conditioned feature/attention transformations; observed token-derived propagation adjacency | Capable current structural single. No learned missing-edge completion bank in the inspected reconstruction. |
| Proposed matched twins | Same four native scorers and clamps; own weights versus their post-clamp mean passed to each branch's own features/decoder | Specific restricted-sharing comparison. No new completion primitive or universal function-class advantage. |

The saved affine reduction remains useful: with fixed residual features and affine decoder responses, the private-minus-pooled mean score is the covariance-like association \(\operatorname{mean}_m\sum_w(w_m-\bar w)A_m\tilde h_{m,w}\). Equal decoder-feature responses make this difference zero; different responses can make it nonzero. This is a testable operation difference, not a positive ranking sign. Pooling may denoise inaccurate completion. Detached completion weights are trained through their own native scorer roles, not automatically credited by the downstream completion gradient.

## 5. Precise advantage gap and competent resolution

The source-supported gap is **whether preserving the branch-specific completion/feature/decoder association improves served ranking sufficiently to justify the restricted-sharing cost**, relative to a matched post-clamp pooled-weight twin and capable alternatives. It is presently open.

The existing source-map comparison roles remain appropriate:

1. Own weights versus post-clamp mean weights, keeping all four scorers, candidates, private features, decoders, objective, warm/initial state conventions and paid work matched. Verify the equal-branch reduction and exact grouped-network correspondence without fitting a duplicate equivalent model.
2. Native capable single NCNC and a prospectively specified capable single with sufficient structural capacity; independent complete NCNC members with competent packing. Four independent decoders on one encoder do not qualify independent full predictors.
3. BUDDY/same-cache independent BUDDY for the existing shared-cache architecture question, and source-qualified current PENCIL for a broader LP-quality claim. IECNC is a close attribution/completion alternative that needs printed semantics/source qualification before a competent fitted comparator.

Full graph access, target masking/sampling, provided features, label visibility, negative streams, clamp location, detach boundaries, normalization/dropout state, checkpoint budget/ties, raw-logit pool and official Hits@50 contract must be predeclared. Native NCNC, PENCIL and active BUDDY graph/test conventions differ in the saved source report; their scores must not be silently merged into one setting. Learned gate/extra-validation supervision and independent encoder costs remain charged. Current source downloads and static sizes are not timing or predictive qualification.

No comparison, seed, launch, continuation gate or prototype change is authorized or frozen by this packet. No current outcome has been inspected to choose these conclusions. A gain over only a pooled-weight twin would support this attributed architecture effect; broader merit requires capable single/independent/current structural alternatives and measured total budget.

## Discovery/access and reading limits

Six saved OpenAlex metadata searches were used. The first two broad queries were noisy; title/date filters improved relevance. Their exact URLs, times, results and hashes are retained. Bibliography locators in saved PENCIL identified the accepted 2025 framework **Bridging Theory and Practice in Link Representation with Graph Neural Networks**. Its selected OpenReview PDF and exact-record API route returned 403. No blocked route was repeated, and no method scope is claimed for it. PENCIL's secondary description cannot substitute for reading that source or prove full-method absence.

IECNC and E-GAE identifiers/titles were checked against index_v32 before counting them as two first scoped method identities. PENCIL is explicitly retained. The remaining third-new-identity allowance is unused. No other discovered abstract/title is counted as a method read. This search is bounded rather than exhaustive and provides no absence certificate.

`READ_SCOPES.json`, `PRIMARY_EXCERPTS.json`, retrieval receipts and `INPUT_BINDINGS.json` bind exact bytes, coordinates, prior scopes and limits. The verifier checks hashes and scope/custody accounting using local stdlib operations. No numerical/model code is imported or executed. All earlier reports/reviews and the sealed conditional-graph-response packet remain intact.
