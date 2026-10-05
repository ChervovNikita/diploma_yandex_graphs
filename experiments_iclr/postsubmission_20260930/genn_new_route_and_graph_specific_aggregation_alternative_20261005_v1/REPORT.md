# GENNN access follow-up and a close adaptive aggregation alternative

5 October 2026. Public literature and safe saved metadata only. No SSH, training, target inference/payload access, paper edit, index modification, execution admission or cumulative reading-total claim.

## Decision

**Graph ensemble neural network remains an unresolved primary-method prior.** The newly inspected SSRN registration matches its title and four authors, strengthening the manuscript locator but supplying no body. Do not claim that GENNN does or does not implement error-moment transport, shared/private member fusion, adaptive depth, or a particular serving pool.

The requested fallback is **Synergy and Diversity in CLIP: Enhancing Performance Through Adaptive Backbone Ensembling**, arXiv:2405.17139v2. Its Neural Logit Controller (NLC) is a close ordinary feature-conditioned late aggregator: concatenate features from pretrained backbones, predict per-example model coefficients with a small MLP, and combine existing logits under a supervised CE objective. It does not use graph topology, labeled residual cross-moments, or a member×depth feature bank in the bounded primary method. Exact coefficient constraints and the temperature-to-weight map remain unspecified in the inspected prose; no author code was qualified.

## GENNN: reuse and genuinely new routes

Target journal DOI: **10.1016/j.inffus.2024.102461**, *Graph ensemble neural network*, Rui Duan, Chungang Yan, Junli Wang, Changjun Jiang, Information Fusion110, article102461. Prior safe reports/receipts were consulted before requests.

The October3 packet already identified a grouped SSRN locator **10.2139/ssrn.4535927**, but its delivery request was403. The October5 closure additionally retained Elsevier coredata only, FULL401, publisher-abstract403 and unsuccessful title/author searches. Those URLs, the earlier exact-title arXiv/GitHub requests, ORCID lookup, OpenAIRE lookup and scholarly locator requests were not retried here.

| New route | Receipt and scope |
|---|---|
| Crossref record for SSRN DOI10.2139/ssrn.4535927 |200. Exact title and four authors match the journal registration; primary resource is the SSRN abstract4535927. Created2023-08-09; registered published year2023. No abstract, download link or registered relation supplied. Metadata corroborates a candidate manuscript locator; it does not certify version equality. |
| Public SSRN abstract page4535927 |403. Error bytes retained; no primary abstract or method obtained. No challenge workaround attempted. |
| New Google exact-title PDF author-copy query |200, but the returned page contains a JavaScript/redirect interface and no usable external result links in the static response. No manuscript located or method inferred. |

This bounded follow-up ends those routes. Access failures and empty/blocked searches are not negative novelty evidence. A supplied authorized manuscript, an accessible author copy or materially changed access would permit the remaining operator comparison.

## NLC: precise primary scope and operator

Primary source: [arXiv:2405.17139v2](https://arxiv.org/html/2405.17139v2), updated2025-02-16; authors Cristian Rodriguez-Opazo, Ehsan Abbasnejad, Damien Teney, Hamed Damirchi, Edison Marrese-Taylor and Anton van den Hengel. ArXiv metadata comment states ICLR2025; the accepted-paper/version correspondence was not independently audited.

Substantive scope: complete §3 Proposed Ensembling Method, CLIP score notation, AppendixA controller setup, Figure1 caption text, and §4.2 frozen-feature/label-role construction. Broader selected appendix/role prose and captions were visible and are disclosed in `READ_SCOPES.json`; no numerical result or guarantee is adopted. No full-paper reading, proofs, code, figure pixels or native reproduction is claimed.

The method constructs class scores from each pretrained CLIP backbone. Let `h_b(x)` be that backbone's image representation and `z_b(x)` its class logits. Its controller has the general paper-level form

`t(x) = MLP([h_1(x),…,h_B(x)])`,

`z_NLC(x) = sum_b a_b(x) z_b(x)`,

where `a_b` denotes the effective scalar coefficient induced by the predicted temperature. The **second expression is a faithful schematic of the stated weighted logit sum, not a published explicit equation**. The prose explains ordinary temperature division `z/t`, then says the predicted temperatures weigh logits. It does not provide the final conversion, a positive-output function, normalization across backbones, or a served implementation. The predicted vector is described as `t in R^B`; therefore neither simplex weights nor positivity should be assumed from the background `t>0` discussion.

Features condition the coefficients; they are not directly concatenated into a new feature classifier in this native method. Scalar temperatures are shared across classes for each member. Even positive scalar rescaling that preserves each member's own argmax can change the *ensemble* argmax when relative model weights change. No ensemble accuracy-preservation claim follows.

AppendixA states a small MLP hidden width128, Adam learning rate2e-4 and weight decay0.01. The phrase “one-layer MLP” and the hidden-layer description do not resolve the exact affine/activation count. The controller is supervised with CE on a held-out part of each target dataset's training set. In the separately described linear-probe construction, classifiers use frozen visual features and90% of target TRAIN; the other10% fits NLC. These are extra aggregator labels, not a label-free postprocessor.

The dense NLC gate consumes features produced by the backbone bank, so every backbone must execute in that dense recipe. The paper discusses combining NLC with a separate confidence cascade. Its conditional execution, thresholds and accounting are different from a gate after already computed states; no cascade cost or quality result is transferred.

## Relation to the two current aggregation hypotheses

| Question | NLC finding and remaining delta |
|---|---|
| Ordinary learned aggregation of fixed member evidence | Direct close prior. A feature-conditioned per-example model weighting mechanism is established. Concatenation does not require coordinatewise alignment of private hidden channels. |
| Graph-local member-error moments | NLC fits a CE-trained feature-to-coefficient map. It does not explicitly form or diffuse `G_u[m,n]=(p_m(u)-y_u)·(p_n(u)-y_u)`, estimate a graph-local uncentered error moment or solve a PSD quadratic probability-pooling objective in §3. That operator distinction does not prove a useful target gap or exclude unknown GENNN. |
| Probability versus logit pooling | The proposed simplex probability pool has conditional squared-error objective `w^T Sigma(v) w`; NLC is a conditional scaled-logit sum. Softmax nonlinearity prevents silently transferring that Brier identity to NLC. Compare complete predictive objects, including the actual native mean-probability anchor. |
| Frozen private member×depth fusion | NLC consumes final features to choose output weights. A member×depth bank and a residual feature classifier offer more input evidence and a different output family. JK/DAGNN/GAMLP and FFL/PCL/VFusion already cover their principal depth/fusion ingredients; retain their saved scopes. |
| What remains worth distinguishing | Label-relevant evidence in earlier/private states; useful generalization of the readout; and whether actual neighbor information contributes beyond an ordinary feature-conditioned readout. Hidden disagreement, attention weights and latent decorrelation do not establish these facts. |

The saved moment identity still applies to probability errors: `G_mn=(G_mm+G_nn-||p_m-p_n||^2)/2`. Off-diagonal moments combine labeled diagonal competence with labels-free disagreement; they are not an independent additional label statistic. A supervised feature gate may learn competence indirectly. Neither operator has an accuracy guarantee from this relationship.

## One graph-specific comparison

For the proposed **labels-free neighbor-context fusion head**, keep the member×depth bank, attention/readout choice, native pool anchor, normalization, parameter count, labels and selection opportunity fixed. Compare only:

- `g_A(v) = [T_A p_bar(v), degree(v), isolate(v)]`, using the predeclared normalized graph-neighbor operator;
- `g_self(v) = [p_bar(v), degree(v), isolate(v)]`, replacing the neighbor-probability input by the node's own pooled probabilities.

The dimensions and trainable head are identical. The own probability vector is already present in member-score evidence, so the second input adds no neighboring prediction information. This small comparison isolates the contribution of the **neighbor-probability field**, while retaining degree/isolate context in both arms. It neither tests all graph structure nor certifies graph-local error covariance. If a moment field or C&S is also used, it must be common to both arms; otherwise the difference confounds multiple graph operators.

A gain supports the usefulness of neighbor information for this fixed head/bank and protocol. A tie gives no support for that graph-context ingredient; keep the simpler context. This is separate from the prior memo's same-bank attention-versus-concatenation question and allocates no new grid or execution. It also does not compare true graph neighbors against every capable feature-neighbor kernel, prove a novel graph operator, or establish independent confirmation.

## Roles, costs and custody

The source-bound study currently assigns no honest fusion-fit role. TRAIN-control is its frozen evaluation endpoint; VAL already selected base checkpoints; TEST scoring is unauthorized. Retrospective aggregation-only reused-VAL splits do not become independent confirmation. Held-out evaluation labels must be excluded from every supervised moment/diffusion field. NLC's reported supervised holdout recipe does not supply missing labels or authorization in this study.

Actual logits retention is source-supported; private penultimate/depth states are not cached. Exact intermediate interfaces still require source review and a separately admitted complete selected-checkpoint export. Charge all backbone/member work, extraction, storage/I/O, graph context and head fitting/selection/serving. A late coefficient predictor saves no work already executed.

`SOURCE_BINDINGS.json` binds safe priors; `RETRIEVAL_LEDGER.json` binds new request receipts and failure bodies; `READ_SCOPES.json` distinguishes NLC method text, role construction, navigation and incidental result prose/captions. GENNN has no new primary-method scope. The manifest/seal attest saved bytes and unchanged inputs only.
