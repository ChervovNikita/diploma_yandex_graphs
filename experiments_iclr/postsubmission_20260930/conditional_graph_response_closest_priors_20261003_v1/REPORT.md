# Closest priors for conditional graph-response diversity

**Date:** 2026-10-03. **Scope:** source reading and algebra only. Three new primary method scopes were read, within the maximum of four. The existing candidate and prospective pilot remain unchanged. This packet supplies no novelty, utility, acceptance, numerical qualification or publication verdict.

## Main finding

The proposed squared-cosine penalty is exactly a function-space degree-two kernel energy. A comparator with the same response map, groups, centering, normalization, coefficients, CE terms, allowed private coordinates, optimizer and guards is the candidate itself. It should be checked as an algebraic/source identity, not fitted as another experimental arm.

The meaningful operation to investigate is the **conditional finite graph-removal probability-response construction under frozen common maps and competence/energy constraints**. The cosine penalty, functional diversity, graph edge removal and constrained private adaptation each have established ancestry. The three scoped sources below do not establish published equality of the complete operation. They also do not exclude it; a close 2025 publication remains inaccessible at method level.

## Inputs and read accounting

Read first: `graph_contrastive_private_paths_quality_gap_20261003_v1/REPORT.md`, `CANDIDATE.json`, and `literature_memory/index_v31` metadata/accounting. The retained index has **130 conclusion records, 85 normalized paper identifiers and 2 software identifiers**. These are not whole-paper reading totals. Exact input hashes are in `INPUT_BINDINGS.json`.

The new primary identities are absent from the retained index by arXiv identifier/title checks. This establishes their new status relative to that index, not a comprehensive historical-literature absence claim. No DICE, FoRDE or ADP primary source was reopened. Their saved conclusions and custody records were reused; archived primary quotations inside those saved records were incidentally exposed and are recorded as such.

`READ_SCOPES.json` distinguishes selected method reading, metadata/abstract exposure, keyword-locator exposure, visual equation verification and unread material. All PDF pages were mechanically extracted to preserve page coordinates; extraction does not count as reading. No full-paper or author-code certification is made.

## Exact kernel identity

For fixed fit targets, masks, groups and active set \(S\), let \(f_m\) be member \(m\)'s class-probability function evaluated on the native graph and the two fixed removal graphs. Let \(T_s(f_m)=U_{m,s}\) perform the candidate's concatenated native-minus-probe response, within-group target centering, flattening and normalization. Reference energy bands keep the active denominators nonzero.

Define

\[
k(f_m,f_n)=\frac1{|S|}\sum_{s\in S}\langle T_s(f_m),T_s(f_n)\rangle^2.
\]

This is positive semidefinite: use the feature map
\(\Phi(f)=|S|^{-1/2}\operatorname{Concat}_{s\in S}[T_s(f)\otimes T_s(f)]\), so \(k(f,g)=\langle\Phi(f),\Phi(g)\rangle\). Squaring also means antiparallel and parallel normalized responses receive the same maximum penalty. Orthogonal centered responses receive zero; neither fact establishes independence or complementary useful errors.

The candidate is exactly

\[
D=\binom{M}{2}^{-1}\sum_{m<n}k(f_m,f_n).
\]

For private coordinates \(\theta_m\), its gradient is the chain-rule pullback through the same finite function evaluations and \(T_s\): \(\nabla_{\theta_m}D=J_{f_m,\theta_m}^{\top}\nabla_{f_m}D\). This includes the centering and normalization Jacobians. Freezing other coordinates restricts that pullback; it does not invent a new diversity operator. Fixed active groups are essential to the stated equality. A different epsilon, stopped-gradient normalization, evolving group membership or different feasibility/optimizer rule would define a different implementation.

This identity is **algebraic**, established from the saved candidate. It is distinct from the broader published operator ancestry below. It does not assert that a paper already published the candidate's complete response map and constraints.

## Three new primary scopes

| Source and exact version | Decisive source scope | Operational overlap and limit |
|---|---|---|
| D'Angelo and Fortuin, **Repulsive Deep Ensembles are Bayesian**, [arXiv:2106.11642v3](https://arxiv.org/pdf/2106.11642v3), revised 2023-03-28 | PDF p4 §2.2, Definition 1, Eqs4–5; p7 §§3.3–3.4, Eq20. Visual checks p4/p7. | Kernels on finite function evaluations and Jacobian-transpose pullback already exist. Generic kernel-gradient repulsion is explicitly discussed. The paper's KDE/SGE/SSGE posterior updates include score/prior and estimator requirements; the candidate is not a faithful posterior sampler. No exact class/neighborhood graph-response recipe was established in the scoped reading. |
| Lucic et al., **CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks**, [arXiv:2102.03322v4](https://arxiv.org/pdf/2102.03322v4), revised 2022-02-22; AISTATS 2022 | PDF p3 §§3–4/Eq1; p4 §§5.1–5.4/Eqs2–5; p5 Algorithm 1 and continuation. Visual checks p4/p5. | A fixed predictor is evaluated after edge deletion, with self-loops preserved and degrees recomputed. Its learned local mask seeks a minimal **prediction-changing** explanation. The candidate uses fixed supervised masks and adapts private model paths subject to competence retention. The common edge-removal probability operation is established; the optimization variable and target differ. |
| Suresh et al., **Adversarial Graph Augmentation to Improve Graph Contrastive Learning**, [arXiv:2106.05819v4](https://arxiv.org/pdf/2106.05819v4), revised 2021-11-03 | PDF p3 §3.1 opening, p4–p6 §§3.1–3.2, Eqs6–9; p18 Algorithm 1. Visual checks p6/p18. | AD-GCL learns an edge-dropping augmenter by an adversarial min–max objective, with a drop-ratio regularizer and InfoNCE on projected graph representations. The encoder and projection head update. It does not provide the candidate's inter-member centered probability-response objective, frozen classifier/common maps or per-member CE/energy feasibility checks in the read scope. |

### Repulsive ensembles: equality does not imply Bayesian convergence

The p4 source states that the kernel is evaluated on the canonical projection \(\pi_B(f)=\{f(b)\}_{b\in B}\), and Eq5 pulls the functional force into parameter space with the particle Jacobian transpose. Probability values at a finite list of native/removal graph-target pairs fit that operator pattern after the specified deterministic response transformation.

The p7 source says, “A repulsive effect can always be created in the ensemble by means of the gradient of any kernel function that is measuring the similarity between two members.” It immediately requires a consistent score estimate for asymptotic posterior convergence. Eq20 has a KDE-normalized force \(\sum_j\nabla k/\sum_j k\), together with a functional posterior gradient. The candidate's unnormalized pairwise energy, CE continuation, restricted coordinates and accepted/rejected private steps do not inherit that theorem. The degree-two kernel is positive semidefinite, but is not thereby a suitable probability-density KDE kernel on this space.

Thus a “matched degree-two functional-kernel repulsion” arm would collapse to the candidate. A source-informed RBF/KDE alternative is distinct, provided its response transformation, denominator, bandwidth, likelihood/prior choice and private-path adaptation are specified and competently qualified. It must be labeled as an adaptation; replacing the posterior score with CE and adding backtracking cannot be represented as an exact reproduction or Bayesian guarantee.

A concrete distinct **kernel-energy control family** is \(k_h(f_m,f_n)=\operatorname{mean}_s\exp[-\|U_{m,s}-U_{n,s}\|^2/(2h^2)]\), with \(D_h=\operatorname{mean}_{m<n}k_h(f_m,f_n)\). It retains the candidate's exact response/groups/CE/private permissions/guards and changes only the similarity geometry. It distinguishes opposite response directions, unlike squared cosine. A bandwidth and strength must be fixed or competently qualified under a declared source-label budget before outcomes, with equal qualification effort for the candidate. This is an RBF energy adaptation informed by functional-kernel ancestry, **not** the paper's KDE-normalized posterior update. A KDE-force comparator is a separate family and needs its denominator, prior/score and constrained-update adaptation declared. Neither family is added to the saved pilot here.

### CF-GNNExplainer: removal response is older; its objective is different

Eqs3–4 compare a softmax GNN on \(A_v\) with one on \(P\odot A_v\), holding \(W\) fixed. The source explicitly contrasts learning weights at fixed data with generating data at fixed weights. Eq5 uses a negative original-prediction NLL, gated while the class has not flipped, plus an edge-removal distance in Eq1. This establishes relevant finite graph-removal probability dependence, but is not ensemble response decorrelation or competence-preserving adaptation.

A fitted CF-generator is unnecessary to establish this ancestry. If a future study uses learned counterfactual masks, the generator's search, masks, per-member queries and selection become part of its charged method. The original predicted class and target-flip success must be kept separate from the candidate's fit-label evidence masks and CE-preserving guards. No causal validity follows from either finite edge removal alone.

**Qualification limit:** p4 says loss is minimized, while p5 Algorithm 1 prints \(\hat P\leftarrow\hat P+\alpha\nabla_{\hat P}L\). Thresholding/gradient details are not source-code qualified here. The printed sign ambiguity is preserved; this packet does not silently repair or certify an author reproduction.

### AD-GCL: view generation and contrastive training are meaningful alternatives

Eq6 learns an augmentation family by \(\min_T\max_f I(f(G);f(t(G)))\). Eqs7–8 parameterize edge decisions with a GNN augmenter and add a graphwise dropping regularizer. Eq9 uses cosine similarity in an InfoNCE estimator after a two-layer projection head. Algorithm 1 updates the augmenter in the adversarial direction and the encoder/projection head by the representation loss. Those are different roles from lowering inter-member redundancy of fixed class-probability response vectors.

A private-only, node-level graph adaptation would change the paper's graph-encoder setting, update permissions and supervision; an added projection head/augmenter must be disclosed and charged. The source theorem is for the stated idealized objective/encoder assumptions, not a guarantee for the candidate or this adaptation.

**Qualification limit:** p6 first says an edge is retained when \(p_e=1\) for \(p_e\sim\mathrm{Bernoulli}(\omega_e)\), but later calls \(\omega_e\) the probability of dropping it. The relaxed formula uses \(\omega_e\) additively, and Algorithm 1 regularizes sampled \(p_e\). Algorithm 1 also assigns \(z_{1,n}\) at both view lines while the loss uses \(z_{2,n}\). These are visually confirmed printed inconsistencies. No author implementation was read. A faithful comparator needs a pinned source check to resolve them before execution; this is a qualification limit, not evidence against the general adversarial-view method.

## Retained priors and remaining delta

The earlier saved report already credits:

- **DICE, arXiv:2101.05544v1, §2.1/Eq1:** conditional feature redundancy \(I(Z_1;Z_2\mid Y)\) while retaining label information. The candidate's squared-cosine statistic is not its MI estimator.
- **FoRDE, arXiv:2306.02775v3, §§3.2/3.4/3.5:** normalized true-label input-gradient kernel, particle repulsion and extra differentiation. Finite edge-removal sensitivity does not erase this task-relevant response-diversity ancestry.
- **ADP, arXiv:1901.08846v3, §§3.1–3.3:** class-probability/non-target diversity plus entropy/member CE. Its determinant has a rank limit; its probability pool setting does not automatically transfer to the fixed mean-logit/private-path setting.

Those retained conclusions are unchanged. Combined with the new sources, the remaining substantive delta is a fully specified composition: fixed TRAIN-label-based same/different-class edge-removal masks; probability differences; within-class/neighborhood target centering with declared fallback; fixed active-group normalization; existing intermediate private coordinates with common/stem/head/statistic freezing; and repeated per-member native/probe CE plus reference energy feasibility.

This can be a distinct operation to test without being a new cosine primitive. The guards protect their declared training control sets only, and can overfit them. Energy bounds can reject legitimate invariance. Conditional centering leaves group means unrepelled. The private restriction can make useful changes unreachable. These are material limits of the candidate, not source-established performance outcomes.

## Distinct resolving controls

These are source-review recommendations, not edits to the saved eight-arm pilot or an authorization to fit additional arms.

| Unresolved explanation | Distinct comparison or check | Competence and cost requirement |
|---|---|---|
| The method is simply renamed kernel repulsion | Symbolic equality above, then source-code comparison of the complete response pipeline and permitted update | **No duplicate fit.** Verify centering, normalization derivatives, masks/groups, CE coefficients, active set, optimizer and guards are equal. |
| Finite graph response adds value beyond ordinary probability-function diversity | Native-probability function energy versus graph-removal response energy, using the same kernel and update setting | Match targets/normalization/coefficient qualification. Keep all native/probe CE and existing guard/energy evaluations where isolating the response object. Disclose any changed measurement energy constraints. |
| Neighborhood conditioning adds value | Global response versus class-only versus class/neighborhood response construction | Existing class-only arm addresses one part. Fix masks, fallback/active coverage, normalization and CE/guards. Any global arm needs a prospective amendment. |
| Degree-two geometry explains apparent benefit | Source-informed RBF/KDE functional repulsion using the same finite responses, with honest restricted/private adaptation labeling | Resolve bandwidth, force normalization and prior/likelihood semantics before outcomes. Equal source-label qualification effort and full extra kernel/score/differentiation cost. No Bayesian claim. |
| Auxiliary view supervision or mask structure explains benefit | Existing native/probe CE-only arm and degree/count-matched label-permuted mask arm | Preserve edge-count/degree restrictions and label visibility; keep groups, guards, warm state and budget fixed. Mask construction and three complete graph/token caches are paid. |
| Learned view selection is a stronger explanation | Qualified AD-GCL-inspired node/private-path view-learning adaptation as a separate future comparison | Resolve printed source ambiguities with pinned author code first; disclose added heads/augmenter, supervision and update changes. Charge augmentation search, stochastic passes and rejected/guard evaluations. Do not substitute an unqualified weak stand-in. |

The most economical resolving sequence is the identity check, the already-saved augmentation-only/class-only/permuted-mask comparisons, then a prospectively qualified distinct function/kernel comparison if needed. Fair controls use the same competent warm bank, native inference graph and pooling endpoint; gains must not be credited to extra labels, relaxed guards, extra active parameters or uncharged search. No current utility claim is made.

## Access gap and retrieval limits

**Adversarial Contrastive Graph Augmentation with Counterfactual Regularization**, Tao Long, Lei Zhang, Liang Zhang and Laizhong Cui, AAAI 2025, [DOI 10.1609/aaai.v39i18.34101](https://doi.org/10.1609/aaai.v39i18.34101), pp19086–19094, is a close unresolved lead. Its metadata abstract refers to minimal-sufficient positive views, hard negatives and counterfactual regularization. The official landing/PDF and DOI route disconnected; the alternate AAAI PDF returned 403; the arXiv query returned 429. Crossref/OpenAlex metadata were retained. **No method scope was read**, and exact objective, conditioning, update permission and full overlap remain unresolved.

All retained successful downloads and failed access attempts have URL/time/status receipts in `RETRIEVAL*.json`; local content hashes are bound in `MANIFEST.json`. Some initial broad metadata queries and browser-search attempts were not retained; `PROVENANCE.json` names them and does not present them as auditable search results or primary reading. This is bounded closest-prior work, not an exhaustive literature search or an absence proof.

## Preservation and verification

Only this new packet was written. The 270 previously snapshotted `REPORT.md`/`REVIEW.json` files were rehashed byte-for-byte without inspecting live outcomes. `PRIOR_REPORT_CUSTODY_AFTER.json` records preservation against the before snapshot. `VERIFY_PACKET.py` verifies packet hashes, source receipts, excerpt coordinates, input custody, scope counts and prior-report custody using filesystem/stdlib checks only. No pilot, fit, data access, tensors, server action, canonical ledger/index/manuscript edit or acceptance verdict occurred.

Exact source bytes, passage extraction hashes and reading limits allow reuse without another primary reread. No change to the canonical literature index is implied by saving this packet.
