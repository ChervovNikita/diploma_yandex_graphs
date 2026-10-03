# Graph-conditioned member training: two close primary reads

3 October 2026. Literature and local algebra only. No graph data, labels, current study outcomes, remote execution, manuscript edits, or new training source were used. The saved v27 literature index was consulted first. Exactly two previously unread primary methods were selected; their complete relevant method and evaluation text, including recipes and limitations, was read. This is **not** a claim of two complete every-page/proof reads.

## Decision

Retain one **unadopted ordinary comparison hypothesis**: smooth the allocation of TRAIN-node cross-entropy across members using a graph, while giving every member positive background supervision and serving the unchanged mean raw logits. This changes the optimization when member losses differ. Its graph-free assignment limit is softened multiple-choice learning, and its graph component is a structural regularizer. The inspected literature does not establish that the complete composition has already been published; it also does not establish its novelty.

The assessment supplies no new GNNM method recommendation or implementation. There is a concrete but conditional predictive hypothesis: graph smoothing might remove isolated, unstable winner assignments and improve later mean-logit predictions compared with matched pointwise assignment, with actual costs reported. The scientific weaknesses are also concrete: identical members give uniform assignments; graph-neighbor membership need not predict the same useful expert on heterophilic graphs; and improving an assignment/oracle objective need not improve a uniform logit pool. These are reasons to keep this as an attributed comparison, not to promote it on embedding spread or a literature gap. Resource availability played no role in this disposition.

## 1. What the new primary papers actually do

| Paper | Operation and error signal | Sharing, prediction, and scope |
|---|---|---|
| Guo et al., **Boosting Graph Neural Networks via Adaptive Knowledge Distillation**, AAAI 2023, DOI `10.1609/aaai.v37i6.25944`; inspected arXiv `2210.05920v2`, revised 5 April 2023 | Train a teacher, freeze it, train a newly initialized student with weighted supervised CE plus soft-label KD. Repeat sequentially across different GCN/GAT/GraphSAGE architectures. A teacher-logit/confidence MLP chooses nodewise temperature in [1,4]. A printed SAMME.R-style update raises weights of poorly predicted TRAIN samples. | No shared trainable body between teacher and student. KD sums over all graph nodes; supervised weighting uses TRAIN labels. The final student alone predicts. The comparison ensemble averages raw logits from separately supervised models. This is a close error-guided heterogeneous-backbone training prior, not a shared BE ensemble objective. |
| Duan et al., **Unifying Homophily and Heterophily for Spectral Graph Neural Networks via Triple Filter Ensembles**, NeurIPS 2024, DOI `10.52202/079017-2966`; official proceedings PDF | Learn weighted sums of powers of a low-pass operator and a high-pass operator, then sum or concatenate those two filtered feature streams before an MLP and one CE head. Self loops and generalized normalization are part of the experimental recipe. | The word ensemble denotes combinations of filters. There are no independently supervised predictive members, teacher-error weights, or persistent member-specific errors in the complete inspected method. The sum variant can be rewritten as a polynomial filter. The concatenation variant keeps two streams before the nonlinear predictor. Neither is a BE error-specialization result. |

### BGNN evaluation and recipe boundaries

The complete evaluation scope covers four node datasets (Cora, Citeseer, Pubmed, Amazon Computers) and three graph datasets (COLLAB, IMDB-BINARY, ENZYMES), using two-layer GCN/GAT/GraphSAGE. Citation tasks use standard splits; other datasets are described as random splits, without numeric split ratios in the inspected appendix. Node training is full batch; graph training uses batches of 32 and intermediate batch normalization. Adam weight decay is `5e-4`, and the stated search spans KD weight `[0.1,0.5,1,5,10]` and learning rate `[0.005,0.01,0.05]`. Exact stopping/checkpoint-selection criteria and the temperature MLP architecture are not fully specified in the inspected recipe.

The paper reports classification accuracy, ablates temperature and error weighting, compares six single-teacher pairs and six multi-teacher orders, and includes ordinary KD, LSP, BAN, MulDE, and a logit-average ensemble. The main single-teacher table reports the better of two teacher choices per student; the criterion used to choose that better teacher is not stated there. The main text says mean and standard deviation over ten rounds, while the appendix says graph-classification reporting keeps the top five of ten rounds. This makes the graph-task reporting selection conditional, even though the rule is applied to the baselines as well. No paper score is inherited as evidence for the proposed modification.

Two printed-math issues matter for a port. Equation 2 gives the correct per-class soft-label CE gradient `(p_student-p_teacher)/temperature`. Equation 3 then sums this gradient across classes, which is identically zero, and treats it as a sample gradient; that scalar sum does not justify the subsequent gradient interpretation. Equation 4 takes entropy of `t`, although `t` was called raw logits; entropy needs normalized nonnegative probabilities. Equation 7 uses the paper's one-hot `y` inside a SAMME.R-style expression; a literal one-hot port differs from standard multiclass signed coding. The complete author implementation was not inspected to resolve these notation/implementation questions. The operational ancestry remains clear: sequential KD and label-safe TRAIN error weighting.

The graph affects the teacher/student predictors. The added temperature and weight rules act on their outputs and labels, rather than explicitly transporting errors along edges. Thus this paper cannot establish that an extra topology intervention in the loss is useful. Heterogeneous **architectures** here are not typed heterogeneous graph processing.

### TFE-GNN evaluation and recipe boundaries

The main evaluation uses eleven node graphs, spanning citation, coauthor, Wikipedia and WebKB datasets, with per-class random 60/20/20 splits and ten reported splits. The appendix adds Roman-empire, Amazon-rating, FB100-Penn94 and Genius using splits from the cited benchmark papers, plus six semi-supervised datasets. For the latter, citation tasks use 20 TRAIN nodes per class, 500 validation and 1,000 test nodes; WebKB tasks use per-class 2.5/2.5/95 splits. Main recipes use hidden size 64 or 512, maximum 1,000 epochs and early-stopping patience 200. A stopping metric is not explicitly settled by that phrase alone.

The full recipe has separate optimizer groups/rates/decay for filter coefficients, stream-combination coefficients and MLP; dataset-specific low/high orders, dropouts and generalized-normalization exponent; and sometimes different optimizer families. The appendix states that reproducing effects may require tuning parameters **and seeds**, and that the additional four datasets use different DGL/PyTorch versions. Some comparison results are copied from earlier papers and others are reproduced; the main assertion that all models share splits must be read alongside that provenance qualification. Baselines were not all extensively retuned.

The paper ablates streams and random-walk normalization, plots training/validation losses, explores orders and reports a Cora-only timing study: ten repeats of 100 epochs for several filter orders. A smaller training/validation loss gap is a descriptive observation, not a theorem that greater generalization follows. Its approximation theorems do not prove superior predictive risk. The appendix acknowledges that selecting low/high filter orders using approximate TRAIN homophily or dataset knowledge can be wrong. The complete code and supplemental ZIP were not inspected or executed.

Both low/high stream names and homophily ratios are useful descriptions, not certificates of complementary class decisions. In particular, with matched symmetric affinity `P`, the sum operator is a polynomial in `P`, since the high-pass operator is `I-P`. Experimental generalized normalization can make the chosen high-pass operator differ from `I-P`; it remains a single learned feature operator in the sum variant. No typed HGT/Simple-HGN/SeHGNN or shared-member result is supplied.

## 2. One concrete training modification and its falsifier

This section defines the ordinary comparison precisely. It is not an adopted recipe, new source implementation, or permission to run a cohort.

Let TRAIN contain `n` target nodes, let `M` members have shared weights `W` and private factors `phi_m`, and let their recent deterministic TRAIN CE losses be `a[i,m]`. Refresh the following assignment at a fixed declared cadence using current model outputs, then **detach it completely** for the following optimizer interval. No ground-truth label outside TRAIN is required.

Choose a label-free symmetric nonnegative affinity `K` between TRAIN target nodes. One homogeneous construction is `K = R_T P^2 R_T^T`, where `P` is a symmetric normalized adjacency with self loops and `R_T` selects TRAIN nodes. Intermediate unlabeled nodes supply paths, not labels or output-loss roots. Use the combinatorial Laplacian `L_K = diag(K 1)-K`, scaled by a fixed bound on its maximum degree. For a typed graph, the target-node affinity needs one prospectively declared valid relation/path operator; there is no qualified typed construction in this packet. A sparse operator implementation must avoid materializing dense `K`.

Find the unique assignment `Q` for `tau > 0`:

    minimize_Q  (1/n) sum_i,m Q[i,m] a[i,m]
              + (tau/n) sum_i,m Q[i,m] log Q[i,m]
              + (gamma/n) tr(Q^T L_K Q)
    subject to Q[i,m] >= 0,
               sum_m Q[i,m] = 1 for every TRAIN node,
               sum_i Q[i,m] = n/M for every member.

Define `0 log 0 = 0`. The affinity term is convex for `gamma >= 0`; positive entropy regularization is strictly convex on the feasible interior. Uniform assignment is always feasible. Balance controls supervision mass, rather than forcing any predictive statement.

Then update **all trainable parameters from the same scalar loss**:

    L_update = (1/n) sum_i,m [(1-alpha)/M + alpha Q[i,m]] CE(z_m(i), y_i),
    0 < alpha < 1.

Every member receives every TRAIN node with positive weight. The same detached coefficient multiplies the member contribution to `W` and to `phi_m`; there is no shared-pool/private-own gradient split. Deployment remains `softmax(mean_m z_m)`. Selection must also evaluate this unchanged pool, rather than a winner, an assignment-weighted mixture, or a probability pool.

### What is algebraically changed, and what is known

- At `gamma=0` without the column-balance constraint, `Q[i,m]` is `softmax_m(-a[i,m]/tau)`. It is soft winner assignment. The small-temperature limit concentrates on the lowest-loss members. With balance it is an entropy-regularized allocation of the same losses. Saved MCL/TreeNets conclusions establish oracle/member assignment and shared-prefix training ancestry.
- With an exact assignment solve, this is alternating optimization of one joint objective: positive-background mean CE plus `alpha` times assigned CE, assignment entropy, and graph smoothness. During the model step, the latter two terms are constants because `Q` is detached and the affinity is fixed. This identification applies to the scalar loss, not a finite Adam-step descent guarantee or a deployed-pool-risk theorem. It shows why shared parameters do not require inventing a separate complementary-target gradient rule here.
- For varying `Q`, the update differs from ordinary mean-member CE by `(alpha/n) sum_i,m (Q[i,m]-1/M) grad CE_m(i)`. The graph term therefore can change both shared and private training; it is not algebraically identical to unweighted BE.
- The graph term smooths **member responsibility**, rather than feature embeddings or class labels. It is not BGNN's sequential teacher upweighting, TFE's forward filter combination, a full-node error cotangent lift, a projection of optimizer updates, or a CP relation parameterization. Its graph-regularized assignment is an attributed combination, not an established original learning principle.
- If member losses are identical at each node, the data term is independent of `Q`. Write `Q[:,m]=1/M + delta_m`, with `sum_m delta_m=0`. The graph objective differs from its uniform value by `sum_m delta_m^T L_K delta_m >= 0`, and entropy uniquely favors uniform rows. Thus `Q=1/M` and the update is exactly ordinary mean-member CE. It cannot manufacture specialization from an exactly collapsed predictor. Existing private initialization/dropout needs to be kept identical in the controls.
- Because the allocation favors a smaller member CE rather than the served mean-logit CE, improved assignment risk does not imply improved served risk. For two binary members on a positive label, logits `(2,-2)` give assignment-min loss about `0.127` but pooled loss about `0.693`; `(0.5,0.5)` has assignment-min and pooled loss about `0.474`. The ranking reverses. Positive background supervision may reduce this risk but supplies no guarantee.

### Conditional predictive premise

If the identity of a competent lower-error member varies smoothly along the selected graph operator, the graph term could stabilize noisy pointwise assignments across refreshes. That might produce useful private learning trajectories without losing all-node competence. It does not require labels to be homophilic, but it does require the **winning-member/error field** to have useful graph structure. Heterophily can make this premise false, and a learned modern backbone may already express the same regimes. This is the precise extra premise to test.

### Matched controls and rejection conditions

The topology-free control uses `gamma=0` with the **same** model, ordinary graph message passing, initial states/RNG policy, TRAIN loss probes, refresh cadence, `alpha`, `tau`, column balance, optimizer, fixed assignment-solver iteration budget and served pool. It removes topology only from the added assignment mechanism. Include ordinary unweighted BE as the reference. Solver qualification must verify normalization, balance, convergence tolerance and actual paid costs for every arm; a fixed iteration count alone does not certify the minimizer or equalize wall-clock costs. The topology-free solve can be cheaper, and any quality claim must include that overhead rather than silently treating it as free.

Equal `alpha/tau` and exposure totals do not match all realized weight entropies or functional effects. To distinguish useful alignment from generic smoothing, add `L_perm = Pi L_K Pi^T` using one declared TRAIN-node identity permutation. It preserves the Laplacian spectrum and degree multiset, and uses the same solver operations. It changes degree-to-node alignment as well as edges; it is not a control preserving each node's degree. A further row-permutation of `Q` preserves its exact matrix row multiset and column totals and can diagnose allocation-strength effects, but destroys pointwise error alignment too. None of these controls should be mislabeled an exact isolation of a particular graph frequency or an exact match of functional spread.

Freeze these choices before outcomes and use later mean-logit validation NLL under one common selector as the primary descriptive comparison, with accuracy, all member CE/accuracy, Jensen gap, error overlap, assignment entropy/churn and cost secondary. Subsequent confirmation needs untouched labels and fresh context. No outcome-selected graph operator, assignment hardness, cadence, new pool or error-subgroup selector is allowed. Do not adopt a numerical advancement threshold from this source-only packet.

The claimed graph utility fails in a tested context if the later served predictor has no consistent improvement over pointwise assignment and the permuted-topology control. More stable assignments, more member disagreement, a lower assigned TRAIN loss, or a better best member are insufficient. Uniform assignments, degraded weak members, or a winner field with no transferable graph structure are adverse outcomes to retain. Better quality over unweighted BE alone would support the complete weighted-training operation; it would not identify graph alignment. This is a falsifiable comparison hypothesis, not a new-method claim.

### Cost and TRAIN safety

Each refresh needs deterministic member logits/CE on all TRAIN nodes; if these are not already cached, charge a full member-resolved evaluation and its dropout/RNG handling. Each assignment-solver step needs `O(nM)` elementwise/balancing work and a graph product on `M` scalar responsibility columns. For the two-hop construction, sparse application through the complete allowed graph costs two adjacency products per solver iteration plus degree setup, restriction and balancing. The solve is detached: no graph-product backward pass or additional outside-TRAIN output Jacobian is used. Full graph working columns can cost `O(NM)` memory even though labels and update weights remain TRAIN-only.

These costs are incremental to ordinary BE's complete private message/activation work. Shared parameter storage does not remove that work. No timing, peak memory, solver, native graph representation, checkpoint replay, or predictive performance is qualified by this packet. Code and native execution remain unadopted.

## 3. Reused boundaries and accounting

Saved conclusions were used for PreGS, GraphMoRE, CHoE/Chimaera, BernNet, AdaGCN, C&S, GRAND/GraphMix, B3F-GNN, SEA, TabM, normalization-based implicit ensembles, FoRDE, MCL/TreeNets, and the recent heterogeneous HGEN/LHGEL/residual-expert methods. The consulted generic quality assessment already rules out relabeling conditional stacking, arbitrary repulsion and private-gradient compensation as newly discovered mechanisms. Its full-node amendment remains a separate active question.

The new BGNN read strengthens attribution for error-guided heterogeneous-architecture transfer; the new TFE read separates filter combinations from predictive ensembles across homophily levels. Their evidence does not resolve a typed shared-factor graph training effect. The inaccessible full primaries GEENI, FAGEL, MORGAN and the metadata-only *Graph ensemble neural network* remain access-limited; no failed route was repeatedly retried. Their missing text is not evidence of absence.

- New primary papers selected and method/evaluation read: **2**.
- New every-page/full-proof paper certifications: **0**.
- Previously read primary-method rereads: **0**; saved conclusions/reports only.
- Author-source reads/executions: **0**.
- Scientific/data/remote executions, current outcomes inspected, new source implementations or recommended pilots: **0**.

Exact byte hashes, URLs, read scopes, caveats and reuse pointers are in `PAPER_CONCLUSIONS.json`, `READ_SCOPES.json`, `REUSED_REFERENCES.json`, and `PROVENANCE.json`. The two new PDFs and extracted text are retained once under `primary/`; no existing source tree is copied. Parent review owns any memory-index adoption or further experiment.
