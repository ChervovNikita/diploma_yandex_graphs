# DIVE: retained-primary method scope and GNNM quality boundary

**DIVE already occupies continuing learned graph-view diversity. It does not establish shared-GNN ensemble quality.** Its individual predictors explicitly do not share an encoder, and inference selects one validation-best model rather than pooling their predictions. A shared-factor, fixed-pool adaptation remains a plausible utility question, but neither mask repulsion nor tying weights establishes a new learning principle or a quality gain. Zero pilots are promoted.

## Source custody and read scope

Xin Sun et al., *DIVE: Subgraph Disagreement for Graph Out-of-Distribution Generalization*, [arXiv:2408.04400v1](https://arxiv.org/html/2408.04400v1), 8 August2024; KDD2024 DOI [10.1145/3637528.3671878](https://doi.org/10.1145/3637528.3671878). The printed header lists six author entries, including two entries named Liang Wang with different email addresses; this packet preserves that header without resolving author identity metadata.

The retained HTML SHA256 is `a4bba9cc6916a8b2ae47b431fd9b705821dbcfca576f2cc196a56886c3e67875`. It came from the previous sealed scout and was parsed here into exact indexed blocks. **Zero new HTTP requests or downloads.** The previous scout manifest remains `d824d4b5b112cbe3579f93abefa4adea084b2c769476b39d511d84c931ad17bc`.

New scope: header/abstract; related-work blocks8–16; complete method blocks17–41 (Sections3.1–3.7, Eqs1–13 and Figure1 caption); experimental setup blocks44–53; AppendixA dataset description blocks79–80. MathML and TeX alttext were inspected for Eqs4–6 and12–13. Method claims below use those exact passages. Results tables/analyses, AppendixB/D outcomes, full references, figure pixels, author implementation and later versions were not inspected. The figure caption was read; the figure image was not.

This is **one first primary-method scope from previously retrieved locator-only bytes**, zero full-paper certifications and zero author-code audits. Parsing all HTML blocks is not reading them. v55 and saved prior conclusions were reused; EMR's newly saved root method scope was not reread. No dataset, cohort/TEST artifact, model, server, GPU, selector, gate, paper or index was accessed or changed. An earlier saved filter-review closure summary appeared incidentally in a text query; no raw results were opened, and those closure values are not used here.

## Exact operation

DIVE targets **graph-level distribution shift**, not the single-graph node-classification problem. It assumes both invariant and spurious structural patterns can predict training labels. The goal is to expose patterns that SGD's simplicity bias may otherwise overlook, then use validation quality to choose a predictor.

For each model `m`:

1. A graph encoder `GNN_mask` computes node states `z_u` from the input graph.
2. A sigmoid edge MLP computes `p_m,uv` from concatenated endpoint states; molecule edge attributes are also included.
3. Each training forward samples a hard edge mask with a stated straight-through estimator. The predictor's graph is `A_m=M_m ⊙ A`.
4. A second `GNN_feat` encodes that masked graph; mean readout and an MLP produce the graph prediction.
5. The main loss is CE for classification and squared error for regression.

The collection objective is

\[
L=\frac1M\sum_m L_{\mathrm{main},m}+\lambda L_d,\qquad
L_d=\frac1{M(M-1)}\sum_{m\ne l}\operatorname{Jaccard}(M_m,M_l).
\]

Equation12 prints set intersection/union rather than a complete differentiable implementation; the formula above expresses its stated scalar overlap operation. Equation13 and setup fix `λ=0.5`. No pooled prediction loss, output-error correlation objective or conditional mutual-information estimator is used in the inspected method. Labels influence mask learning through each model's task loss; the mask-overlap auxiliary itself is not label-conditioned.

The learned mask functions persist through training, while realized masks are resampled in each training forward. Thus DIVE is a prior for **member-specific learned distributions of graph views and their structural overlap regularization**, not only fixed randomized views or one-time parameter initialization.

### Sharing and gradients

Section2.1, block10 says: **“our individual predictors do not share the same encoder”**. Section3.4's “identical architectural design” means the same architecture, not tied parameters.

Each member receives its own task gradient through its masked-feature GNN and mask extractor. The cross-member mask penalty couples the extractors. Eq6 explicitly stops the derivative through one copy of `p`; its other copy provides the straight-through derivative. No shared/private block-routing mechanism is described. Actual optimizer update order, scaling and auxiliary gradient implementation remain author-code-unread.

The `ε`-optimal predictor set in Section3.3 is a conceptual target. The actual objective contains mean task loss plus overlap penalty, not a hard per-member `ε`-competence constraint. The prose does not prove that it discovers every predictive subgraph or every invariant one.

## Serving and model selection are decisive

Section2.1, block11 states: **“Ensembling means that the results from the diversity models are aggregated for inference. Rather, we train a collection of models and select on model for inference.”** It says the goal is to discover predictive patterns missed by SGD, rather than combine uncorrelated errors to lower variance.

Section3.7, block41 selects the validation-best member using an **OOD validation set** and deploys that single model. Baselines use that same validation set. The paper says it also reports ID-validation variants, but their result appendix was not read here.

The exact inference-time mask rule (sampled, deterministic threshold, or multiple draws) and checkpoint-selection implementation are not specified in the inspected passages. Section3.7 says highest validation “accuracy”; Section4.1.3 assigns task-specific ACC/ROC-AUC/MAE metrics. No exact cross-task selection implementation is inferred from this wording.

Consequently, DIVE's quality mechanism includes **candidate discovery followed by validation selection**. It supplies no uniform raw-logit-pool result, no proof of complementary member errors, and no shared-body quality result. Replacing selected-single serving with GNNM's uniform pool changes the scientific endpoint.

## Native datasets and authored recipe

| Dataset | Task and stated shift |
|---|---|
| GOODMotif | Synthetic graph classification: motif determines label; base type/size define domains. |
| GOODHIV | Molecular binary classification; scaffold/size domains. |
| GOODZINC | Molecular graph regression, graphs at most38 heavy atoms. |
| GOODSST2 | Sentence-graph sentiment classification; length domains. |
| DrugOOD IC50 | Molecular binary classification; assay domain used. |

The read recipe specifies Adam, learning rate`1e-3`, weight decay0, dropout0.5, three convolutional layers, hidden width300, ReLU, mean global pooling, batch32 and at most300 epochs; it also says train until convergence. The exact convolution type, temperature, initialization, collection-size default, convergence rule and mask inference policy remain unresolved in this scope. The paper describes14 baselines including9 graph-specific OOD methods; this is an authored comparison list, not a runtime qualification of them or a transferred graph-node quality verdict.

Training requires each member's extractor and masked-graph encoder, plus mask comparison; serving uses only the selected member. A pooled adaptation pays for every served member. Sharing feature transforms does not eliminate member-specific graph propagation, normalization, mask construction or backward work. A frozen full-graph spectrum/cache cannot automatically represent arbitrary masks that change throughout training.

## Printed sampling ambiguities: do not silently repair

Equations4–6 and their MathML agree on the printed rule

\[
q=\sigma((\log p+G)/\tau),\quad
G=-\log(-\log U),\quad q'=1[q>1/2],\quad
m=q'+p-\operatorname{stop}(p).
\]

For positive`τ`, its hard draw satisfies

\[
\Pr(q'=1)=\Pr(G>-\log p)=1-\exp(-p),
\]

rather than the claimed `Bern(p)` law. For example, even the limit`p=1` yields inclusion probability`1-e^{-1}`, not1. This is a symbolic consequence of the displayed equations, not a sampled experiment. A usual Bernoulli/logistic-noise relaxation would be a different formula; native code might implement one, but it was not inspected.

Eq6's backward path uses `p`, not the soft sample`q`. With the threshold derivative stopped as stated, `∂m/∂p=1`. Under the printed rule, positive`τ` does not change the hard threshold or this direct backward path. Temperature is unspecified. Additional porting details are also unresolved: treatment of empty-mask union, a differentiable Jaccard surrogate, self-loops, undirected-edge symmetry, normalization of extracted graphs, and the mask matrix's printed`N×N` shape despite`N` earlier denoting number of graph samples.

These ambiguities limit a faithful native recipe. They do **not** erase the established learned-mask/view operation or justify novelty from failed access/qualification.

## Why mask repulsion alone is not useful error diversity

A graph can contain two disjoint isomorphic copies of the same spurious motif with identical features. Two masks can retain different copies, giving zero edge overlap, while permutation-invariant graph classifiers return identical predictions and make identical shifted-domain errors. This directly falsifies the implication “disjoint masks imply complementary prediction errors.”

Independently resampled sparse masks can also have low realized Jaccard overlap even when their learned inclusion distributions are identical. Under an ideal Bernoulli keep rate`r`, the large-edge-count intersection/union ratio tends to`r/(2-r)`, which decreases as masks become sparse. DIVE's printed sampler changes the keep-rate map but retains this issue. Retention-matched stochastic controls are therefore material; low overlap need not demonstrate learned specialization.

Both causal and spurious structures are intentionally admissible during training. Validation selection can choose a better candidate under a representative OOD validation distribution; the structural penalty does not itself identify the invariant structure. Own-task CE can discourage damaged members, but its soft tradeoff is not a quality guarantee.

## Closest saved priors and the GNNM distinction

| Prior | Closest operation and remaining boundary |
|---|---|
| DIVE, this new scope | Learned ongoing edge-mask disagreement, own task supervision, independent encoders, selected-single serving. Closest structural-view ancestor. |
| HGEN; Training Diverse Graph Experts | Different graph neighborhoods/views and independent experts already occupied. HGEN embedding-Gram repulsion is not a complementary-error guarantee. |
| BankGCN/Specformer; BernNet | Learned structural/filter channels before nonlinear prediction already occupied; a competent complete shared bank can absorb some one-stage private filters. |
| GRAND/GraphMix; CAMERO | Stochastic views/shared weights and consistency are prior. DIVE differs by learned edge-mask overlap and independent encoders; a tied-mask version would be an attributed adaptation. |
| GNCL; saved graph-kernel NCL | Predictive-loss/residual correlation trades individual competence against pooled objective. DIVE does not use that predictive objective. Fixed graph-error metrics remain kernel NCL. |
| DICE; FoRDE | Label-conditional redundancy and task-dependent input-gradient diversity are prior. DIVE's mask-overlap term is neither of those operations nor an output-error guarantee. |
| C&S | Graph-error correction/smoothing is a capable control; mask specialization cannot be credited for ordinary correction gains. |
| Shallow ensembles | Shared encoder/private prediction heads are prior; they do not establish the benefit of private learned message paths. |

GNNM private feature/channel factors do not by themselves instantiate node/edge-conditioned masks. A node-independent transform of an already aggregated neighborhood cannot generally replace a member-specific edge gate before aggregation. Conversely, if all useful graph evidence has already been mixed or erased by a common stem, masking a later representation may retain leaked shortcut information or fail to recover alternative evidence. This matches the saved placement/complete-basis cautions.

The plausible shared-factor extension must put private graph pathways **before** the relevant information loss and maintain each member's nonlinear message trajectory. It changes the architecture and paid computation; it is not achieved by changing final-head initialization. DIVE already supplies the graph-view diversity principle. Tying transforms, retaining a fixed pool and demonstrating a quality/cost tradeoff remain empirical adaptation questions.

## One concrete conditional falsifier

**Status: a prior-adaptation utility falsifier, not an admitted pilot or novel-method proposal.** If the existing runner can represent member-specific masked propagation, a representative native node-graph test could reject the putative benefit of learned structural specialization:

- Share a non-erasing feature stem, with per-member mask scorers and complete private masked message trajectories. Preserve the existing fixed raw-logit pool and native task/selection schedule. First pin the sampler/straight-through/Jaccard behavior as an explicitly defined adaptation; do not claim a DIVE reproduction from its ambiguous printed formulas.
- Compare the same masked architecture with and without overlap regularization, plus degree/retention-matched fixed or stochastic graph views. Use the competent independent-member architecture under the same pool as the sharing reference. Existing nonmasked capable GNNM remains the quality reference. The overlap-off arm isolates repulsion; matched views isolate learned specialization from ordinary graph augmentation or sparsity.
- On one prospectively chosen already admitted small graph/seed, require pooled NLL/accuracy improvement under a predeclared criterion while tracking every member's competence, retention and prediction-error overlap. Reject the mechanism if lower mask overlap has no pooled benefit, if retention-matched random views match it, or if the gain depends on switching to a validation-selected single member. A later broad claim would also need the independent-reference and paid-cost gates.

No graph, seed, coefficient, threshold or execution is selected here. Source/runtime qualification and a prospective root protocol precede outcomes. The current factor-only runner was not audited for dynamic masks in this scope; if it cannot expose an exact mask-aware propagation hook, this test is not yet executable. Testing masks only after common propagation does not establish the proposed information-path mechanism.

**Disposition:** attribute DIVE's continuing learned graph-view diversity and selected-single OOD mechanism. Do not promote mask or embedding repulsion as complementary prediction-error training. A shared-factor/fixed-pool adaptation is a falsifiable utility question; no unoccupied learning principle, target predictive gain, native reproduction or manuscript contribution is established by this read.
