# Accuracy of graph ensembles with shared weights: two close prior checks

The user's clarification makes **served predictive quality** the main target: GNNM was motivated by cleaner ensemble decisions, not parameter reduction. This packet adds two scoped primary reads after checking literature memory v52. It does not establish a new learning principle or a positive GNNM result. The existing persistent graph-view comparison remains the right bounded test of whether shared updates help quality.

## What the new sources establish

**CAMERO: Consistency Regularized Ensemble of Perturbed Language Models with Weight Sharing** (ACL 2022, DOI 10.18653/v1/2022.acl-long.495; requested arXiv version 2204.06625v1) is a direct quality-oriented sharing prior. In §3.1, members share their bottom-layer weights and retain private top-layer weights. Each member receives independently perturbed input/hidden representations and its own supervised loss. In §3.2, a consistency loss pulls each perturbed prediction toward the committee prediction. The paper motivates this regularizer by the difficulty of accommodating overly diverse member representations in the common weights. It is not an orthogonality penalty or a learned inference router.

For the BERT experiment, §4.1 states that private classifiers are differently initialized, the encoder is shared, and **inference uses one encoder pass followed by the average logits of the private classifiers**. Thus training perturbation diversity is not the same thing as retaining member-specific graph propagation at serving. Its §5.1 representation diagnostic freezes the trained encoder and fits a new randomly initialized classifier, comparing encoders from CAMERO and ONE. This supports the authors' representation-quality interpretation in their setting, but is not a tied-versus-untied intervention with all other ingredients matched. Its perturbation and consistency analyses also do not prove that greater diversity is always better. The qualitative conclusion is a trade-off; no CAMERO numerical performance is transferred to GNNM.

**Uncertainty Quantification in Graph Neural Networks with Shallow Ensembles** (arXiv 2504.12627v1, 2025; Tirtha Vinchurkar, Kareem Abdelmaqsoud and John R. Kitchin) is direct graph prior for a shared encoder with many private output heads. It adapts the cited DPOSE approach to SchNet by replacing its final layer with 64 energy-output heads. The heads' mean and variance enter a Gaussian NLL. Its OC20 fine-tuning example freezes all weights except the final layer. The studied endpoint is uncertainty under molecular/material shifts, not a clean classification-accuracy benefit of shared updates against matched independent graph ensembles. The authors' conclusion explicitly reports difficulty distinguishing some structurally similar out-of-domain gold systems. More head spread or a changed input graph is therefore not, on its own, evidence of reliable uncertainty or improved predictive quality.

The underlying DPOSE paper is a **cited, uninspected lead** here; no method or theorem is adopted from it. This packet inspects the SchNet adaptation itself. Neither newly read method supplies a theorem that arbitrary graph-view sharing preserves competent complementary errors.

## Exact overlap with the current candidates

| Candidate ingredient or proposed claim | Prior overlap | Remaining bounded question |
|---|---|---|
| Sharing weights can improve learned representations and committee quality | CAMERO states and studies this quality objective | Does tying improve the native pool of otherwise equivalent persistent graph-view members? General sharing quality is not itself new. |
| Different training perturbations while common weights remain trainable | CAMERO; retained AM-GCN/GRAND/GraphMix and graph ensemble conclusions | Are persistent equal-class/different-class TRAIN edge views useful beyond shuffled exposure and degree/count-matched random deletion? |
| Shared graph encoder with private output heads | SchNet shallow ensembles, plus retained adjacent shared-base ensemble methods | Do complete member-specific graph trajectories add useful neighbourhood reliance beyond a head-only shared-encoder ensemble? |
| Mean-preserving private-head covariance chosen by one own-CE trial | The new sources do not specify this operation | Retained MAML/SAM/splitting/GNCL ancestry remains decisive; random/permuted/fixed graph allocations and competent quality references remain necessary. |

The new sources do not establish an exact published equivalent of the complete graph-view or covariance protocol. **This bounded failure to establish equality is not novelty clearance.** The claimed contribution cannot simply be “shared graph ensembles improve accuracy” or “perturb members to improve diversity.”

## What changes in the actual comparison

**No queued family, source contract, outcome gate or four-bank graph-view comparison is changed by this packet.** The current tied-persistent versus exactly matched untied-persistent comparison remains necessary: CAMERO's strong quality claim would not answer that causal contrast. Keep persistent-versus-shuffled and semantic-versus-random comparisons, the competent all-TRAIN augmented single, the native single and conventional independent-four references. Current v6 references with a different TRAIN-label budget cannot provide decisive all-TRAIN controls.

A **positive** initial graph-view result would justify one additional prospectively specified control before a broader claim about complete private trajectories: a common modern graph encoder with four private heads, trained with the same persistent graph views and ordinary per-head supervision. Attribute this architecture to the close shared-head prior. Charge its separate training-view passes; at native serving it may share the encoder pass. It must use the same complete TRAIN labels and served pooling rule as the candidate. This is an architecture control, not a faithful CAMERO reproduction: a faithful CAMERO comparator would additionally require its consistency term, native selection budget and mean-logit pooling to be reproduced or the adaptation disclosed. Do not tune a new consistency term retrospectively to explain a failed current comparison.

The all-TRAIN **ordinary native GNNM** reference, if absent, also remains necessary for a claim that graph views improve GNNM rather than merely establishing a tying effect within a new augmentation recipe. It cannot be supplied by the label-budget-incompatible running v6 fits. This requirement follows from the causal claim, not from the new sources, and this packet authorizes no extra fit.

Two different claims should remain separate:

1. Tied beats untied under the same persistent views: evidence for beneficial coupling in that development setting.
2. Complete private trajectories beat a competent shared-encoder/head ensemble: evidence that retaining private neighbourhood computation matters.

A gain in the first does not prove the second. A shared-head control matching the candidate would favour a simpler model. If random views or shuffled roles match, the corresponding graph semantics or persistence interpretation fails. Negative development outcomes close this fixed hypothesis rather than justify a new post-outcome mask, regularizer or target.

## What to measure and how to interpret it

Report native pooled accuracy/NLL together with individual competence and paired error overlap. Useful complementarity means that otherwise competent members supply correct evidence that improves the served pool. Probability spread, response decorrelation or a large Jensen gap is insufficient, as the retained DICE/FoRDE/GNCL/JMLR conclusions already explain. This packet adds no replacement diversity objective.

The fixed-source shared gradient geometry is a diagnostic, not an invariant predictor property. The retained prior already warns that negative pairwise shared-gradient dots do not determine the actual summed update and that coordinate changes can alter them. Native coupled Adam requires its actual state and preconditioner; an SGD inner-product slogan is not a theorem for that optimizer or a generalization guarantee. Therefore this read does not introduce a gradient-surgery arm or elevate the selected covariance trial into a global competence safeguard.

CAMERO's frozen-encoder/new-head probe suggests a possible later representation-transfer diagnostic, but it costs a separately selected fit and would not establish the native committee's accuracy by itself. The current quality comparison should be completed first.

## Reading, discovery and scope

`READ_SCOPES.json` lists exact paragraph blocks and displayed-equation nodes. `PRIMARY_PASSAGES.json` binds selected verbatim passages to saved source hashes. Primary HTML was successfully obtained for both requested v1 URLs; abstract pages provide title/author/date metadata. Three OpenAlex queries are discovery metadata only. The SchNet paper's selected method/setup/conclusion text and CAMERO's background/method/setup and selected §5 analysis were read; neither complete paper, implementation, benchmark or proof was audited. Published result numbers were incidentally visible in extracted CAMERO figures/tables and setup/result prose, and in a keyword locator view of the SchNet source; none are adopted as verified GNNM performance or comparative evidence.

The task opened no server, dataset, fit, checkpoint, logits or scientific outcome. No reviewer verdict is requested or issued. No model was implemented or trained. Deliberate files and parser scripts are confined to this packet in the project repository; no temporary tree outside the project was used. Literature memory and canonical research state are left for root adoption.
