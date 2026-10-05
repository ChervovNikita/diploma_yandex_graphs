# Training hypotheses after the complete Amazon error and fusion screens

5 October 2026. Scoped literature/method note; **two conditional interventions, no admitted fit or novelty claim**. Root owns the completed recurrence diagnostic and its pending independent interpretation/adoption check. This note uses only root-permitted aggregate files, saved literature conclusions and bounded public primary methods. No scientific payload, live result, SSH, source implementation or new compute was accessed.

## What the evidence supports

The adopted complete error diagnosis reports shared versus independent mean member accuracy 52.2402% versus 52.5859%, pooling gains +0.1633 versus +0.5907 percentage points, and mean pair error correlation 0.9292 versus 0.7304. Shared members are slightly weaker and add less ensemble benefit. A useful intervention must address both properties.

The adopted fixed saved-prediction screen is negative. The graph-moment processor was not selected. Complete processed shared/independent/single accuracies are 52.5450%, 53.2528%, 53.1276%; shared loses to both references. The processed single also has better Brier and NLL than shared. Local-full moments do not establish a useful local/off-diagonal ingredient. Preserve that failure; do not reopen its grid or infer that another probability combination repairs the common mistakes.

Root authorized use of the completed recurrence aggregates: all three 6,123-node VALID panels, 53,112 rows, full SUPPORT/STRATA, and all 192 primary rows. In the fixed primary stratum, both pools predict the same wrong competitor and both raw banks are unanimous at it. Same-class VALID neighbors are compared with matched same-class nonneighbors. Equal target weighting gives:

| Neighbor-minus-control excess | Split 0 | Split 1 | Split 2 | Three-split mean |
|---|---:|---:|---:|---:|
| Shared pooled wrong competitor | .200655 | .236147 | .227375 | .221392 |
| Independent pooled wrong competitor | .195300 | .245951 | .252418 | .231223 |
| Single wrong competitor | .190453 | .222720 | .249827 | .221000 |
| Shared wrong-member fraction | .201822 | .241155 | .227483 | .223486 |
| Independent wrong-member fraction | .188122 | .228559 | .246297 | .220993 |
| Shared four-member wrong unanimity | .198959 | .231245 | .211399 | .213868 |
| Independent four-member wrong unanimity | .134189 | .174467 | .190026 | .166227 |
| Primary shared-minus-independent unanimity Delta | .064770 | .056777 | .021373 | .047640 |

The primary Delta is positive in all three descriptive splits. **The excess is primarily coherent four-member agreement, rather than a larger marginal rate of the wrong competitor.** Shared-minus-independent pooled excess averages -.009831; the wrong-member-fraction contrast averages only +.002494. The capable single has comparable pooled recurrence. This matters: a general label smoother or a loss suppressing every graph-local competitor could help all predictors, and would not explain a shared-ensemble benefit.

Primary population sizes are 1,234/1,164/1,204, approximately 19-20% of VALID. Only 393/391/386 targets have matches, with 574/560/514 directed pairs. All primary controls and every fixed support stratum remain bound in the source files; other strata are not substituted to rescue the hypothesis. These conditional descriptions neither account causally for the full accuracy deficit nor establish general effect size. Same graph, overlapping masks, reused selected VALID labels/checkpoints and dependent pairs preclude iid or causal claims. Same-class edges exclude heterophilous neighbors. Matching on true class, degree bin, both banks' correct-member counts, single correctness and confidence limits some marginal differences, but does not identify a harmful message, gradient conflict, information bottleneck or transferable TRAIN regime.

### Training information boundary

Do not import the VALID mistake mask, competitor identity, class/degree outcomes or matched pairs into fitting. Construct an analogous training signal prospectively from permitted FIT/TRAIN labels and public graph/features only. The saved FIT pooled-error upper bounds are 12 shared, 5 independent, 8 single; literal in-sample error mining has very little support. A new hypothesis requiring errors should either use continuous margins on every training target or explicitly acquire predictions from a fixed cross-fit within the training population. In that cross-fit, omit the target fold's labels from the probe's supervised objective, retain only the declared transductive public context, and count all probe fits. This supplies training supervision, not untouched evaluation or an independence theorem. No acquisition is authorized here.

## Intervention 1: allocate additional class-competitor supervision along training regions

**Question:** Can graph-stable responsibility for particular class-versus-competitor margins produce a competent local specialist, while every member retains ordinary all-target supervision? This extends the saved graph-smoothed loss-allocation hypothesis to the observed error object. It is an attributed MCL/boosting/graph-regularization adaptation.

Let training items be `(i,k)` for every permitted training node i and competitor k != y_i. Use a fixed nonnegative difficulty h_ik from a prospectively declared training-only probe, for example its out-of-fold probability of class k; normalize total mass once. Keeping continuous items avoids an arbitrary hard-error cutoff and preserves empty-support disclosures. Set

    b_ikm = softplus(z_mk(i) - z_my_i(i) + mu).

Here mu is a fixed margin. At a declared cadence, evaluate current member margins and detach an assignment Q after solving

    min_Q  sum_(i,k,m) h_ik Q_ikm b_ikm
           + tau sum_(i,k,m) Q_ikm log Q_ikm
           + gamma sum_m Q_m^T L_K Q_m,
    Q_ikm >= 0, sum_m Q_ikm = 1,
    sum_(i,k) h_ik Q_ikm = sum_(i,k) h_ik / M.

K is one fixed nonnegative affinity between training items with the same true class and the same competitor, using direct public adjacency between their target nodes; L_K is its combinatorial Laplacian with a fixed scale. Uniform Q is feasible. Positive tau gives strict convexity on the feasible support. Zero-difficulty items can be kept at uniform assignment. The class/competitor restriction uses only permitted training labels; unlabeled/heldout labels never define K. No graph or class cell is selected from VALID outcomes.

Update shared and private parameters from the **same scalar objective**:

    L = mean_(i,m) CE(z_m(i), y_i)
        + lambda / sum_(i,k) h_ik
          * sum_(i,k,m) h_ik Q_ikm b_ikm.

Every member keeps the unchanged CE on every training node. Extra supervision is allocated, rather than removing nonspecialists' target losses or repelling hidden embeddings. Refreshes use fixed cadence/iterations/tolerances and detach Q; they are not gradients through a learned validation selector. Serve the existing **arithmetic mean of member softmax probabilities**. The saved raw-logit-pool allocation proposal is ancestry, not a substitute serving rule.

This objective can concentrate additional correct-versus-rival learning on different regions, including regions where every member's top-1 answer is currently wrong but their continuous margins differ. It cannot create specialization from exact equality: equal member margins, entropy and balance favor uniform Q. More stable assignments also need not improve the arithmetic pool. Ordinary CE is a competence anchor, not a guarantee that finite shared-parameter updates preserve every member's performance.

**Conditional priority:** retain only if a prospectively acquired training-only margin/winner field has nontrivial, transferable graph structure and sufficient support. The positive VALID unanimity Delta motivates this question; it does not establish that prerequisite. If the training signal is uniform, graph-unstable or supported only by a few mistakes, stop this route rather than select another fold, time or graph operator after observing outcomes.

**Discriminating controls:** ordinary shared own-CE training; the same extra margins with uniform Q (generic hard-margin training); pointwise balanced assignment with gamma=0; one prospectively fixed same-class/competitor permutation of graph alignment; and ordinary training with the same extra probe/evaluation/paid budget. Permutation keeps label/competitor cells but need not preserve every target's degree, so it tests the complete alignment change. The single receives the same difficulty/margin information with sufficient nonlinear capacity and a complete schedule; a capable multi-branch single can receive all regional information. Ordinary independent4 retains separate full encoders and schedules; an additional untied four uses the identical allocation rule to test whether any gain depends on sharing. No weak factor-only single is admissible.

Reject the graph claim if aligned Q fails to outperform pointwise and permuted Q on the prospectively fixed full pooled quality criterion. If uniform extra margins match it, the useful ingredient is generic supervision. If the same operation benefits single/untied models equally, the evidence supports a general training regularizer. Lower assigned loss, lower unanimity, a better oracle member or a recovered selected subgroup cannot replace complete pooled quality and retained member competence.

## Intervention 2: train private graph views while anchoring each deployed full-graph member

**Question:** Does exposing different private routes to different local evidence during training reduce coherent mistakes on the deployed original graph? This directly tests an upstream explanation that the recurrence diagnostic cannot establish. It adopts graph augmentation/denoising, not a new diversity principle.

Keep the original graph and native full-graph serving path. Give each member a private, label-free edge scorer before the local neighborhood aggregate, such as a symmetric sigmoid MLP on the two endpoint raw/shared-stem features. Draw a declared symmetric edge mask V_m with fixed expected retention; protect the same explicit root/self path in every arm. Share appropriate feature transforms while preserving complete member-specific masked propagation, normalization and nonlinear states. A mask applied after a common lossy aggregation does not test the alternative-evidence premise.

One fully specified training family has

    L = mean_m [ CE_F(z_m(G),Y)
                 + lambda E_(V_m) CE_F(z_m(G masked by V_m),Y) ]
        + beta * mean_(m<n) soft_overlap(a_m,a_n),

with fixed per-member retention constraints on edge probabilities a_m. A soft Jaccard overlap of inclusion probabilities is one explicit option; its nonempty denominator and the exact mask gradient/sampling law must be declared, rather than copied from DIVE's ambiguous printed hard sampler. The view CE uses only FIT labels. The overlap term is a known structural-view diversity control; it supplies no theorem about correct competing classes. At inference use each trained member on the **unmasked original graph**, then the unchanged probability pool; no mask average, new gate or validation-selected member is substituted.

Full-graph own CE trains the deployed path, and fixed retention prevents winning the overlap metric merely by making masks sparse. Neither prevents the shared representation from ignoring views, the masks from retaining duplicated nuisance evidence, or the objective from degrading members. A feature/root path can preserve useful information when an edge is removed. Do not assume cross-class edges are noise, or import PTDNet's low-rank/community-homophily penalty into Amazon based on this same-class diagnostic.

**Conditional priority:** a secondary prior-adaptation question only if useful alternate local evidence survives masking and the member-private path lies before the relevant mixing. Its causal premise is unverified. Prefer the ordinary graph augmentation control if it gives the same benefit. If complete global attention already reintroduces the alleged nuisance or the architecture cannot provide a mask-aware local path, the interpretation fails; do not silently mask a final embedding instead.

**Discriminating controls:** identical private scorers/retention/full-graph CE with beta=0; common masks across members; independent per-member random DropEdge views at the same retention/exposure; learned single-model denoising; and a capable single trained on the same view distribution and original-graph anchor. Include ordinary independent4 and untied four with the identical view training. Match labels, graph information, degree/retention descriptions, selection opportunities and complete schedules, while reporting actual differing costs. Lower mask overlap is not success. Reject an ensemble-specific claim if a capable single or untied view-trained ensemble matches/exceeds quality, if only masked serving helps, or if member competence is lost.

## Strong antecedents and scope limits

Saved index72 and supplements V3 were consulted before new requests. The closest saved antecedents are:

| Antecedent | What is already established |
|---|---|
| SMCL, arXiv:1606.07839v1 | Winner/member assignment and oracle-loss specialization are direct prior. Its oracle endpoint differs from a uniform served probability pool. |
| AdaGCN, arXiv:1908.05081v3; B3F-GNN, DOI:10.1007/s10994-026-07041-x; BGNN, arXiv:2210.05920v2 | Error-weighted graph training, sequential specialization/warm transfer and local error-guided modules are established. They do not prove the graph-smoothed parallel probability-pool recipe works. |
| Saved graph allocation note, `graph_distinct_quality_literature_20261003_v1` | Balanced entropy/graph-smoothed member responsibility plus positive background supervision is already proposed locally. Intervention 1 specifies class-competitor add-on margins; it is not a rediscovered unused idea. |
| DIVE, arXiv:2408.04400v1 | Continuing learned private graph masks, own task CE and mask-overlap regularization are direct prior. Its encoders are independent and it serves a validation-selected single for graph-level OOD; the original-graph probability-pool adaptation changes that endpoint. |
| GRAND, arXiv:2005.11079; HGEN, DOI:10.24963/ijcai.2025/685 | Shared-parameter stochastic graph views/consistency and graph-view ensemble representation diversity are prior. |
| GNCL and saved graph-kernel residual analysis | Same-node or fixed-diffusion quadratic residual overlap has an exact own/pool Brier identity. Do not rebrand either as this new graph mechanism. |
| Saved FoRDE/DICE/pulse assessments | Task-gradient repulsion, label-conditioned redundancy and graph-filtered safeguarded tangent interventions have direct ancestry. The old pulse requires a new fixed-time state and native AD qualification; it is not licensed by the current recurrence outcome. |

Two genuinely new **scoped method reads**, no full-paper certifications or author-code audits:

1. **DropEdge**, exact arXiv:1907.10903v2, *DropEdge: Towards the Very Deep Graph Convolutional Networks for Node Classification*. Section4.1 removes a random fixed-size subset of edges and renormalizes; implementation text serves the whole unmasked graph. This directly precedes training-only graph-view perturbation. The introduction also describes independent Bernoulli dropping, so a native adaptation must state its law. The linear random-walk/mixing argument is not a proof about nonlinear Polynormer or complementary wrong classes; no proof or paper score is adopted.
2. **PTDNet**, exact arXiv:2011.07057v1, Luo et al., *Learning to Drop: Robust Graph Neural Network via Topological Denoising*, WSDM2021 metadata DOI:10.1145/3437963.3441734. PDF pp3-5, Sections4.1-4.3, Eqs1-13, Algorithms1-2 and Figure2 were read and rendered. It learns layerwise endpoint-MLP edge distributions with task loss and sparsity, using a binary-concrete/hard-clipping relaxation; inference can also denoise. Low-rank regularization uses SVD/power-iteration machinery and a community-label premise. This is a strong capable-single denoising antecedent. It is not a member-ensemble method or a literal recipe adopted here. Full author source, numerical stability, exact inference rule and native optimization are not qualified. Its printed density/logit conventions should be resolved before faithful reproduction.

The first remembered PTDNet locator mistakenly used **2011.07071**, an unrelated physics paper. Its header exposed the mismatch; its primary body was not read. Two v2 requests returned404, the mistaken v1 acquisition remains explicitly excluded, and one exact-title OpenAlex locator supplied2011.07057. No GENNN retrieval route was retried. Failed access and missing exact duplicates provide no originality evidence. Exact scopes, primary hashes and all requests are retained in READ_SCOPES.json, SOURCES.json and RETRIEVAL.json.

## Advancement boundary

These are **conditional prior-adaptation hypotheses**, with Intervention1 the closer test of the observed unanimity pattern. Neither supplies a currently defensible novel learning principle. The recurrence alone does not justify another cohort, a numerical threshold, a source port or an implementation. Root's independent interpretation check and eventual training-only feasibility decision precede any protocol.

A future protocol must freeze all coefficients/cadences/masking, training probe acquisition, complete member-strength margins, full pooled accuracy/NLL/Brier requirements against both competent references, native selection and paid budget before outcomes. Account for probe fits, assignment solves/refresh forwards, every original/masked member forward/backward, graph renormalization, private scorers, activation memory, selector opportunities and failures. Storage sharing does not remove private propagation work.

If an intervention improves calibration alone, merely lowers conditional unanimity, sacrifices member quality, or gains only over the weaker shared baseline, stop the corresponding promotion. Subsequent untouched confirmation would be required for a predictive claim; the exposed Amazon VALID development cannot become that confirmation. No original manuscript/canonical index/current pilot was modified.
