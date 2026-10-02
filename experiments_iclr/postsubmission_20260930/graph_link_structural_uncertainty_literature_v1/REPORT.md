# GNNM disagreement under weak structural evidence

This report assesses whether recent link prediction and conformal inference papers justify a distinct extension to the compact GNNM predictor. The outcome is **no-go for a new learner or calibration guarantee, with one concrete prospective diagnostic retained**: test whether compact member disagreement preserves the error information of an independent ensemble on future links with no observed common neighbors. This report contains a test specification, not an experimental result or authorization to run it.

The literature memory was consulted before acquisition. Three new primary papers received scoped method reads; none received a full read. Existing NCN/NCNC, BUDDY/ELPH, Link-MoE, negative sampling, adaptive sharing and joint-risk conclusions were reused. No active graph data, checkpoints, Torch, GPU, SSH, training, source modifications, research-ledger edits or memory-index changes occurred.

## Evidence from the three papers

| Primary paper and pinned version | What the inspected text establishes | Boundary for this task |
| --- | --- | --- |
| [LPFormer: An Adaptive Graph Transformer for Link Prediction](https://arxiv.org/html/2310.11009v4), revised June 27, 2024 | GCN node representations feed target-pair attention over context nodes. Learned relative encodings use PPR; distinct encoders cover common neighbors, other one-hop nodes and more distant nodes. Fixed PPR thresholds limit the context. The scalar decoder combines endpoint features, pair encoding and context counts. | This is a direct prior for adapting structural evidence to each pair. Its factor-wise diagnostics already partition links using CN, PPR and feature similarity. Reported performance and epoch timing were not reproduced; author code and graph-policy parity were not inspected. |
| [Conformalized Link Prediction on Graph Neural Networks](https://arxiv.org/html/2406.18763v2), revised July 18, 2024 | CQR fits lower and upper quantile predictors on edge embeddings and calibrates nonconformity scores. A degree-based sampling procedure aims to improve interval efficiency. The paper uses balanced positive/negative subsets, training plus validation positives in the observed graph, and five random calibration/test splits with ten repetitions. | Its listed datasets omit collab. Marginal conformal coverage does not establish conditional coverage for CN strata, validity under chronological shift, or deployment edge probabilities. Extra quantile fitting is real work: Appendix A specifies 1,000 quantile epochs on DDI. |
| [Conformal Network Link Prediction with False Discovery Rate Control under Unstructured Missingness](https://arxiv.org/html/2507.07025v2), revised June 25, 2026 | A weighted graphon model supplies exchangeability of the complete network. Assumption 1.1 requires the missingness matrix to be independent of the realized network. Local calibration/test splits use common fully observed rows to construct exchangeable scores. Local BH decisions become e-values, repeated splits are averaged, and e-BH aggregates decisions. An undirected extension tests the upper triangle. | Unknown or heterogeneous missing rates are permitted within those assumptions. MAR depending on latent node variables and MNAR are explicitly excluded. Chronological future-edge formation is not validated by this result. The real example randomly masks 10% of a static 1989 trade network of 158 countries. |

These are verified descriptions of the inspected primary text. Numerical improvements, coverage and FDR performance remain author-reported. The theoretical proofs were not independently audited in full. The local FDR result states a no-ties condition, with randomized tie-breaking offered for ties. The FDR paper explicitly describes its practical e-value inflation factor as lacking a rigorous theoretical characterization; its basic theorem must not be transferred to that inflated variant without qualification.

LPFormer is a 2024 revised source whose first arXiv posting was in 2023. KDD 2024 and DOI metadata were obtained from OpenAlex, not independently verified against the publisher. The search is bounded and is not an exhaustive inventory of 2024–2026 research.

## Source qualifications for conformal claims

CLP Section 3.1 moves from order-invariant scores and a fixed unordered score multiset to exchangeability. Order invariance alone does not imply a distribution invariant to permutations of calibration and future cases. A simple analytic counterexample keeps the predictor and both label prevalences fixed: let the edge feature be equally often 0 or 1, predict that feature, and let calibration labels equal the feature while future labels equal its complement. Scores are independent of case order. Calibration residuals are all zero, future residuals are all one, and the standard conformal singleton set has zero future coverage. This is an assumption counterexample, not evidence that the paper's random-split experiments failed.

The pinned CLP HTML also prints the interval in Eq. (7) and Theorem 3.2 with upper endpoint `q − upper_quantile`. Its MathML operators, TeX annotation and `alttext` agree. Inverting the nonconformity score printed in Eq. (6) instead gives `upper_quantile + q`. The theorem's quantile argument places the division by K inside the ceiling, whereas its proof uses order index `ceil((K+1)(1−alpha))`. These are verified inconsistencies in this HTML version. The PDF and author implementation were not checked, so an implementation error is not claimed. The diagnostic below requires neither expression nor a conformal guarantee.

The newer missingness paper supplies a more explicit boundary: complete-network graphon exchangeability plus missingness independent of the realized network. Its structured splitting changes the prediction construction to preserve this symmetry. Merely fitting a calibrator to GNNM outputs would not inherit that guarantee.

## The concrete diagnostic gap

**Proposed question:** when a future link has zero common neighbors in the fixed observed training graph, does the compact predictor's low member disagreement still identify unreliable predictions as effectively as four independent predictors receiving the same graph evidence?

The information target is the association between member disagreement and held-out error, conditional on observed structural support and pooled confidence. It is distinct from measuring average member correlation, adding an embedding repulsion penalty, routing experts, or allocating layer ranks. A homogeneous ensemble can agree because the available representation is inadequate; independent parameters alone do not add structural observations. Weight sharing may further remove useful error variation. These are competing hypotheses, not findings.

**Closest diagnostic prior:** LPFormer Sections 4.3 and E.4 already stratify links by local structure, global structure and feature proximity. Its main text gives 90% as an example threshold, while its collab appendix uses 80%, partly to balance group sizes. The proposed test therefore uses the exact predicate CN=0 rather than choosing a factor percentile from held-out outcomes. LPFormer's attention and NCNC's missing-neighbor completion are capable structural controls if a later mechanism claim is sought. BUDDY already supplies pair-specific sketch evidence; this report does not assert that BUDDY has no structural information.

**Closest uncertainty priors:** CLP supplies quantile calibration on GNN edge embeddings; the newer conformal paper supplies structured score construction and FDR aggregation under explicit graph assumptions. Neither paper establishes that four GNNM factor members constitute calibrated uncertainty on chronological collab. Saved uncertainty-guided negative sampling and joint-risk proposals remain prior work for this project and are not reopened here.

## Fixed paired falsification test

The following is a prospective companion analysis. It must be frozen before inspecting its evaluation outcomes. If those outcomes have already been inspected, this specification is exploratory and confirmation requires a separately reserved evaluation cohort. No current checkpoint-selection rule or 15-cell family is changed by this report.

### Inputs and arms

Use the complete fixed collab training graph, native features and identical deterministic BUDDY caches already defined by the sealed protocol. Do not add validation edges, future topology, new negatives, a heuristic-selected negative pool, or outcome-based graph filters. Evaluate the unchanged official positive pairs and global negative pool once after all family checkpoints are locked.

Pair `factorized4` with `independent4` at the existing optimizer seeds 0, 1 and 2. Retain `single256`, `matched_single` and `native1024` as capacity and competence anchors. The independent predictors receive equal cache reuse. Record member scores separately without retraining, checkpoint reselection, refitting a calibrator or altering the published pooled score. Any additional member forward or exact structural computation requires a separately authorized companion run and its own cost receipt.

Define the primary structural strata from the training graph only: CN=0 and CN>0. Compute exact CN rather than using a noisy sketch estimate as truth. A secondary descriptive degree table uses training-node degree quartiles fixed from the full training-node distribution, with an explicit isolated-node bin. Neither strata nor cutoffs may depend on held-out labels or errors.

### Scores and errors

For candidate pair e and member m, record the frozen raw logit `s_m(e)`. Preserve native pooling `s_bar(e)=mean_m s_m(e)`. Let `p(e)=sigmoid(s_bar(e))` and define member disagreement

`u(e)=mean_m (sigmoid(s_m(e)) − mean_j sigmoid(s_j(e)))²`.

These sigmoid values are scores for the sampled binary objective; they are not certified probabilities that a real-world link exists. Define sampled binary error as `L(e)=1[(s_bar(e)>0) != y(e)]`, with the official positive label 1 and sampled negative label 0. Ties at zero predict 0. Some sampled negative pairs may be future or unobserved true links; that label uncertainty remains part of the protocol.

Use pooled confidence `c(e)=abs(p(e)−0.5)` as the necessary score-only control. Within each CN stratum, retain the lowest-u 50% of the complete candidate population and separately the highest-c 50%. Both rules use the same number of pairs and do not use labels. Break all selection ties by the unchanged official pair-row identity. Also report prespecified 75% and 100% coverage descriptively; they cannot rescue a failed primary comparison.

Evaluate each retained set S using weights `w(e)=1/(2*N_y,g)`, where `N_y,g` is the count of the corresponding official label in the complete stratum g. Set

`R_g(S)=sum_{e in S} w(e)*L(e) / sum_{e in S} w(e)`.

This is conditional sampled binary risk with equal starting mass for each label. Report retained positive/negative counts and error rates separately, since uncertainty selection can change the retained label mix. A stratum with either label absent is not evaluable. Report raw probability spread and, as a fixed saturation check, spread of member negative-pool rank percentiles; do not select whichever uncertainty measure looks better.

### Primary comparison and falsifier

For each seed, define `A_arm = R_CN0(confidence retention) − R_CN0(disagreement retention)`. Positive A means disagreement retains fewer errors than pooled confidence at the fixed 50% coverage. The paired compact deficit is `T=A_independent4−A_factorized4`.

The proposed extension claim is operational: compact disagreement preserves the independent ensemble's additional error information within 0.5 percentage points in the CN=0 stratum. **Reject this claim** if the mean paired T is at least 0.005, T is positive in all three seeds, and the independent ensemble's mean A is at least 0.005. These are fixed practical falsification thresholds, not a significance test or a population confidence guarantee. Report all seed values, including failures of the criteria.

For an uncertainty-specific interpretation, additionally require the absolute full-stratum binary-risk difference between the two ensembles to be at most 0.01 in every seed. Otherwise mark the result confounded by pooled competence. The anchors identify whether both small ensembles are simply weak predictors. If independent disagreement itself adds no benefit, there is no evidence supporting a compact uncertainty extension; this does not establish that every conceivable uncertainty score is useless.

### Unanimous retrieval failures

Separately preserve official Hits@50 and its strict threshold tie rule. For each member obtain the 50th-largest score over the unchanged global negative pool. Record positives missed by every member and by the pooled predictor, tabulated by CN stratum. Describe their u and c distributions, including the fixed event `u<=0.001 and c>=0.45`. Never interpret its rate as a probability-coverage guarantee.

Do not compare disagreement against pooled-confidence retention on positive Hits@50 misses: those misses are already a deterministic threshold function of the pooled score. Such a comparison would give the score control an identity-level advantage and would not test added uncertainty information.

If both ensembles exhibit similar confident unanimous errors in CN=0, the result is compatible with a shared evidence limitation, model misspecification, training bias or saturation. Stratification cannot choose among these causes. A later structural-control comparison must use capable NCNC or LPFormer implementations under the same graph and negative policy and account for every extra fit, precomputation and query pass. A matched correction of these failures could motivate a mechanism study; it would still not by itself prove that absence of common neighbors caused them.

The chronological evaluation diagnoses behavior on a future cohort. It does not identify temporal shift as the cause without a separately specified horizon or distribution intervention. Validation outcomes used for checkpoint selection cannot be reused as independent calibration evidence. No calibration is fitted in this diagnostic.

## Why the learner remains no-go

Adding CN bins, PPR attention, completion, a structure gate or a conformal wrapper would reproduce established ingredients without specifying a new learning mechanism. This bounded read does not establish a new efficient way to expose unavailable graph evidence to compact members while preserving cost and a valid temporal inference target. It also supplies no theorem that converts member spread into uncertainty under chronological graph evolution.

The saved result is therefore a diagnostic hypothesis with a falsifier and attributed controls. A successful diagnostic would support an empirical limitation or preservation claim on the fixed sampled protocol. It would not establish state of the art, calibrated deployment probability, a new learner, or novelty through absence of a matching paper in this search.

## Read and preservation accounting

- New scoped primary method reads: 3. New full primary reads: 0. Retained-primary rereads: 0. Author-code reads or executions: 0.
- LPFormer: parsed blocks 18–91, 102–110 and 180–193. CLP: 27–83, 90–118 and 130–135, plus the three explicitly recorded MathML nodes. Missingness/FDR: 5–87, 105–113 and 147–176. Abstracts and heading inventories receive no additional method-read credit.
- Algorithm captions and surrounding descriptions were read. Complete pseudocode listings and figure pixels were not inspected. A complete downloaded HTML file is not a full-paper read.
- Discovery metadata includes 15 arXiv query entries and 5 OpenAlex search entries; selected primary scopes are recorded separately. Cumulative project read totals remain uncertified.
- Input hashes, source quotations, scoped passage identities, discovery dispositions and analytic checks are saved in the accompanying JSON files. The sealed BUDDY execution and staging manifests are preserved. This packet does not alter literature memory or the active study.
