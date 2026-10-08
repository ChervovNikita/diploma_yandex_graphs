# Common shared targets and private route targets: two new primary checks

**Recommendation: abandon a new-gradient-principle or new-complementarity-principle claim. Retain the exact common-shared/route-private field only as an inactive, attributed policy ablation. These two reads do not establish a complete prior match, but they also supply no ready replacement that is justified for frozen WikiCS.** For the accuracy goal, complete the representative scalar comparison and competent references already specified. If later evidence identifies insufficient private capacity, the previously recorded private-last-propagation-block control is a more direct, materially different hypothesis than adding more representation repulsion. No implementation, arm or fit is adopted here.

The whole proposed field is already recorded in [the prior disposition](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/context_target_recipient_whole_rule_prior_disposition_20261008_v1.md>): shared parameters receive ordinary mean own CE plus common-target alignment; designated private internal BE rows receive the same own CE plus factual route-target alignment; all blocks are collected at one old state before one native Adam transition. The classifier receives own CE because the captured representation precedes it. This differs from the frozen scalar COMMON and ROUTE arms.

## Saved memory checked before discovery

The current supplement pointer and its prior chain, base index v72, the whole-rule disposition, and targeted saved conclusions were inspected first. The saved record already establishes:

- Song collaborative learning, ONE and PCL: shared/private branch training and explicit differences in loss exposure or shared backward scaling.
- AMCL: one common backbone with multiple contrastive projection heads; different heads alone are not a new ensemble principle.
- PCGrad and ParetoGNN: reconciliation of auxiliary/task gradients on shared parameters; neither supplies this target-to-recipient field.
- GraphMix, MA-GCL, HLCL and the context-target attribution record: learned common graph representations, structural/filter views and graph contrasts are existing ingredients.
- TreeNets/CAMERO and the saved architectural closure: a shared bottom with private upper computation is a known accuracy-oriented capacity control.

These are **reused scoped conclusions**, not new primary reads. SupCon, BE, TabM, FairACE and GrowNet bodies were not reopened. The saved target-linearity and prototype-curvature notes explain why different same-class target graphs can change local geometry without guaranteeing competent complementary predictions. Combining these established ingredients is not novelty evidence.

## 1. GCMAE: shared auxiliary learning, not four predictor roles

[Generative and Contrastive Paradigms Are Complementary for Graph Self-Supervised Learning, arXiv:2310.15523v1](https://arxiv.org/html/2310.15523v1), Sections 3–4 and complete Algorithm 1, provides a close **shared-encoder, complementary-objective** rule.

One trainable GNN encoder receives a feature-masked graph and a second graph made by random node dropping. The feature-masked embedding enters a GNN decoder. Both encoder outputs enter separate two-layer projectors for symmetric same-node InfoNCE. The printed joint objective is feature scaled-cosine reconstruction plus weighted contrastive loss, whole-adjacency reconstruction, and a representation-variance term. Adjacency reconstruction combines squared error, binary CE and a distance-ratio term. Algorithm 1 makes both views, computes all branches, and updates the common encoder using the joint objective; its returned artifact is the trained encoder.

**Gradient rule:** the scalar objective sends the auxiliary branches' gradients to the live common encoder through their ordinary paths. Decoder and projector loss exposure is imposed by the forward graph. There is no common-target/private-route derivative assignment, four BE factor rows, class-compatible positive-neighbor bank, own-label loss per served member, or four-predictor probability pool in the inspected complete method.

**Setting:** GraphMAE-style GNN encoder/decoder pretraining; the inspected setup does not identify a concrete native GAT/GCN encoder recipe. Node tasks use Cora, Citeseer, PubMed and Reddit. Graph classification uses IMDB-B, IMDB-M, COLLAB, MUTAG, REDDIT-B and NCI1. The downstream classification procedure trains LIBSVM classifiers on representations with five-fold cross-validation; link prediction fine-tunes a final layer and evaluates AUC/AP, and clustering uses K-means with NMI/ARI. Five random-seed runs are stated. This is representation pretraining and downstream evaluation, not the fixed full-label WikiCS joint classifier protocol or a test of retained useful predictor diversity. No published score is adopted.

**Operational limits:** the printed positive variance term does not express variance-increasing minimization without an additional sign/hinge convention; the distance-ratio term also needs its distance/sign interpretation resolved. Raw inner products are described as edge probabilities without an explicit bounding operation in the displayed reconstruction. Algorithm 1 drops Eq. 8's weights and explicitly names only the encoder update. These are body-to-operation ambiguities, not resolved by an author-code audit here. Do not instantiate the printed rule blindly as a cure for weak predictors.

**Representative control required:** if a reconstruction/global-structure mechanism were separately elected, compare the same learned native backbone with and without that scalar auxiliary package, then test the same package with a competent single and an untied ensemble under the same TRAIN budget and selector. A direct native adaptation must disclose its changed pretraining/evaluation recipe and charge both views, decoder and adjacency work. Reconstruction loss or embedding spread cannot replace served accuracy, pooled NLL and paired repairs/introduced errors. GCMAE is not presently recommended as a new arm.

## 2. DGE: complementarity from distinct observed graph evidence

[Deep Ensembles for Graphs with Higher-order Dependencies, arXiv:2205.13988v3](https://arxiv.org/html/2205.13988v3), Sections 2–3, supplies the more direct **graph evidence-subspace** alternative.

Start with observed sequences of entities. Construct a first-order network and a higher-order network (HON), in which conditional nodes retain preceding-entity context. For each original labelled node and each member's fixed bootstrap, select exactly one conditional relative, sampled in proportion to its weighted out-degree within that node's higher-order family (Eq. 3). Edge tasks sample a pair of relatives according to normalized conditional-edge weight (Eq. 4). Thus members see different neighborhood subspaces derived from observed sequential dependence.

The complete serving/training alternatives in Eq. 5 are:

| Variant | Learning and serving rule |
|---|---|
| DGE-concat | Independent GNN modules produce hidden vectors; concatenate them, apply one learned classifier, and train the combined graph end to end. |
| DGE-pool | Average independent modules' hidden vectors before the learned classifier; train the combined graph end to end. |
| DGE-bag | Train separate class-probability predictors independently on their bootstraps; serve their arithmetic mean probabilities. |
| Shared starred controls | Tie every GNN parameter. For concat/pool, accumulate each parameter's loss contributions before one update; bag trains the same parameterized learner across bootstrap stages. The paper treats these as single models with repeated conditional-input evaluation. |
| DGE-batch* | One shared model resamples relatives per training batch and samples multiple relatives for mean-probability inference. |

The primary method declares supervised loss against base-node labels but does not specify its exact classification-loss formula in the inspected body. No unstated CE reduction or optimizer equivalence is inferred.

**Setting:** six sequential-data networks—Air passenger itineraries, T2D diagnosis trajectories, Wikispeedia clickstreams, magazine readership Mag/Mag+, and shipping paths Ship. The default is 16 mean-aggregating GraphSAGE predictors on an order-two HON. Identity features are shared between each base node and its relatives. Node classification uses stratified five-fold cross-validation and mean micro-F1; link prediction uses held-out positive/negative node pairs, hides their higher-order-family edges, and reports mean AUPRC across five repetitions. Appendix C states manual tuning and selection of the best configuration averaged across testing folds, which cannot be imported as the frozen prospective selector. The qualitative accuracy/diversity and sharing discussion in Section 4.3 was read; numerical result tables, plots, and claimed effect sizes were not audited or adopted.

**Whole-rule relationship:** DGE changes the predictor's neighborhood evidence and usually its full private parameter capacity. The current Qm intervention changes a supervision relation while preserving the native predictor graph. DGE has no same-class contrastive Qm/Qbar, no BE rows and no allocation of common auxiliary credit to shared weights. Its fully tied controls also lack private BE factors. Therefore its complete rule does not remove the exact recipient-policy distinction.

**Material difference and boundary:** DGE offers a principled mechanism for useful diversity—distinct conditional neighborhoods rather than a larger hidden disagreement penalty. But its evidence comes from observed path histories. WikiCS supplies a preconstructed static graph in the frozen protocol; manufacturing random walks from it is not equivalent to recovering observed higher-order dependencies. The paper itself states this limitation. A DGE remedy is appropriate only on a separately specified sequential graph task with those observations; it is not a source-grounded replacement for current WikiCS targets.

**Representative control required:** on such a task, hold the observed paths, HON construction, bootstrap bank and serving pool fixed while comparing a capable learned shared model, a shared model with the elected private capacity, and separate predictors. Add the matching first-order input reference to distinguish better graph evidence from parameter untieing. Keep label budgets, parameter/cost accounting and prospective selection comparable. A gain from HON inputs cannot establish the benefit of a common/private gradient split.

## Practical disposition for a learned shared backbone and four useful predictors

No inspected complete rule is an exact match. This bounded non-match is neither global absence nor novelty clearance. The already saved partial-gradient and graph-SSL priors remove the broad novelty claims immediately; abandon those claims rather than rename their combination.

Do not elect an additional auxiliary loss on the premise that complementary embeddings imply complementary correct predictions. Complete the frozen scalar COMMON/ROUTE/permuted comparison and its competent objective-matched single/untied and ordinary references. Evaluate served quality and member mean/worst competence together with paired repairs, introduced errors, any-correct coverage and common wrong-rival support on the whole selected population.

If that evidence later shows that useful correct alternatives are available under matched untied capacity but weak under factor-only private paths, the prior attributed **private-last-propagation-block** control targets the missing freedom directly while retaining a learned shared bottom. It is a new capacity hypothesis relative to this target rule, not a new shared/private architecture principle. If a competent simpler shared-head control matches it, abandon the claim that full private propagation is needed. If competent untied predictors also fail to supply useful alternatives, the premise for either recipient allocation or additional private capacity is unsupported.

The recipient ablation, if independently justified later, still needs scalar COMMON, scalar ROUTE and common-shared/route-private at the same full horizon and all paired seeds. A mirror recipient rule is required for a stronger role-allocation claim; outcome-independent target assignment shuffling tests persistence. These requirements were already recorded in the prior disposition and are not new admitted arms. A direct future complete-rule collision should end the novelty search and shift the work to attributed utility comparison; no such collision was established by these two scopes.

## Scope and custody

Exactly **two genuinely new primary method scopes** were read: GCMAE v1 and DGE v3. Both were checked against the existing index/supplement titles and identifiers before method reading. Four arXiv discovery queries and one five-identifier metadata/abstract request are discovery only; the other returned papers add no method-read credit. Neither full paper, proof, author implementation nor benchmark was audited. READ_SCOPES.json binds exact HTML section/block ranges, Algorithm 1 and selected setup text to direct source hashes.

Only public primary literature and saved scoped conclusions were used. No active scientific scores, arrays, datasets, checkpoints, servers or experiments were accessed; no active source, frozen protocol, canonical literature pointer or study was changed.
