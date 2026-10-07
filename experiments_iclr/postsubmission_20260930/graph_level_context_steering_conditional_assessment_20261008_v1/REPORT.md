# Conditional molecular graph-context steering

2026-10-08. Contingency planning while context9/Mol18 run; their outcomes were not opened. Current literature-memory pointer/base/prior chain and retained molecular/diversity scopes were consulted first. One narrowly relevant fresh primary method scope was needed: MoCL2106.04509v1. No second fresh scope, fit, model import, GPU/data/heldout access, or running-study/source change.

## Assessment

**Selective label-compatible chemical positives can define different private route objectives without requiring every inactive molecule to share one embedding. They do not guarantee prevention of collapse, chemical semantics, stronger members or scaffold generalization.** Molecular similarity positives are already explicit in MoCL; activity-aware graph contrast and prediction ensembles are also retained priors. The open question is utility of persistent context assignment through the existing shared/factor model, not a new molecular-contrastive or ensemble primitive.

The binary inactive label is especially coarse: molecules can be inactive for unrelated chemical/assay reasons. Treating all label0 graphs as mutual positives makes their class identity an insufficient semantic target. Bounded structure-selected same-label relations retain distinctions among inactive chemotypes, but their connected relations can still homogenize representations, while denominator repulsion can split chemically related molecules excessively. Same-label compatibility is not equivalence of pharmacology or a label-preserving biological intervention.

One full ogbg-molhiv experiment is specified below, conditional on complete context9/Mol18 interpretation retaining a reason to test this extension and on source/resource qualification. It is not a launch, staged grid, second node study, or superiority recommendation.

## 1. Closest molecular and ensemble priors

| Source and read status | Exact relevant conclusion | Limit |
|---|---|---|
| **[MoCL,2106.04509v1](https://arxiv.org/html/2106.04509v1)**, NEW scoped primary §§3.1–3.4 | Learns molecular graphs without property labels. Local bioisostere/substructure substitutions provide views. ECFP/Tanimoto similarities supply global neighbors selected by threshold or neighborhood size; global objectives are cosine-to-similarity least squares or positive-neighbor versus background contrast. Uses encoder plus projection head and combines local/global terms. | Direct chemical-similarity-positive prior. Its printed local ratio excludes the paired positive from its denominator; its global ratio sums neighbor exponentials against nonneighbors. Neither is silently equated with the proposed row-normalized cross-entropy. No code/proof/results reproduction or accepted-version equivalence audit. |
| **[ACANET/ACA](https://doi.org/10.1038/s41467-026-75713-2)**, retained molecular assessment, NO new primary reread | Regression plus label-conditioned online triplets; positive/negative activity-difference thresholds and label-dependent margins; graph backbones and averaged submodel predictions. | Direct property-aware molecular contrast plus ensemble ancestry. Continuous activity differences carry more information than one HIV binary label. Its PNA/MAE recipe is not this GINE/BCE experiment. |
| **[MolCLR,2102.10056v1](https://arxiv.org/html/2102.10056v1)**, retained method conclusions, NO new primary reread | Same-molecule augmented-view alignment with other molecules as distractors; molecular graph pretraining and downstream scaffold-split classification. | Generic molecular views are prior. Removing atoms/bonds/subgraphs can remove property-relevant structure; no pretraining cost/performance is inherited. |
| **[Chemprop package paper](https://doi.org/10.1021/acs.jcim.3c01250)**, retained method conclusions, NO new primary reread | Bond-directed molecular encoder, complete independent model/cross-validation ensembles and uncertainty evaluation. | Strong molecular ensemble ancestry; current Chemprop v2 source/recipe was not audited here. |
| **[DeepDelta](https://doi.org/10.1186/s13321-023-00769-x)** and corrected MoleculeACE, retained conclusions | Signed property-difference prediction and chemically similar activity-cliff evaluation are established. Molecules must be split before pairs; cliff-preserving splits differ from unseen-scaffold transfer. | Binary HIV does not supply quantitative activity-cliff margins. Neither pair multiplication nor corrected potency tasks certify this scaffold-split extension. |
| **BE/TabM, GNCL, DICE/CDLG/DNCC**, saved diversity/sharing scopes | Shared/private ensembles, collective versus member risk, task-information preservation, channel pairing and shared-backbone diversity all have prior. | A molecular application, fixed fingerprints or one auxiliary per route do not establish novelty. Binary four-member ADP's original non-target determinant is rank-deficient; no silently regularized replacement is proposed. |

MoCL's local property-similarity assumptions are not asserted for HIV efficacy. The proposed views below retain the same molecule and change only native stochastic dropout. No bioisostere substitution, bond deletion, teacher, external pretrained model or drug-target network is added.

Read accounting: one NEW primary method document in this assessment, zero full-paper/proof/code audits. Existing molecular and diversity conclusions are REUSE, not fresh primary reads. No global novelty exclusion is inferred from the saved-memory search or metadata ranking.

## 2. Mechanism, imbalance and scaffold limitations

For graph representation h_m(G), use fixed positive-target rows Qm in cross-view contrastive CE. Different Qm can yield different private-factor cotangents while the common target Qbar=mean_m Qm matches aggregate mass. At identical deterministic routes, target linearity gives equal mean score cotangents and, with equal shared Jacobians, equal shared gradients. Dropout/Adam and changing representations prevent a trajectory identity.

Chemical context may let private nonlinear atom–bond/virtual-node paths emphasize different class-compatible motifs. Yet contrast acts on h, not on an active–inactive ordering. Head-unused feature differences can satisfy it without changing scores. With perfectly parallel same-label vectors across views, selected-positive cosine derivatives vanish, so different masks can also provide no differentiated target gradient. These retained node-branch limitations remain true for graph embeddings; they are not evidence of actual molecular collapse.

**Inactive-class heterogeneity:** choosing only a close donor prevents an explicit all-inactive positive clique. It does not prove the retained modes are meaningful. Fingerprints can favor molecule size, common rings, stereochemistry encoding conventions or scaffold families rather than HIV-related features. A same-label donor can reinforce a nuisance direction that all members already use incorrectly. Conversely, a structurally similar opposite-label molecule remains a denominator distractor, which can supply useful discrimination, but binary/noisy assay labels do not certify mechanistic antagonism.

**Scaffold shift:** four structural fingerprints strongly encode scaffold identity and overlap. An apparent TRAIN benefit could memorize scaffold families. Requiring off-object donors from a different TRAIN scaffold reduces exact-core attraction; it does not erase shared fragments, series ancestry or chemical-space shift. Cross-scaffold positives may also be chemically remote and harmful. No fingerprints, target thresholds or auxiliary weights may be selected from official validation/test similarity or outcomes.

**Imbalance:** ordinary minibatches can contain few/no active molecules. All-same-label positives then disproportionately steer label0, or active targets reduce to self positives. A graph-level contrast extension cannot assume the node study's almost-full TRAIN panel. Restricting global kNN targets to an ordinary128-graph batch may discard most off-object donors. The explicit panel below supplies donors without replacing the full supervised training set. Its auxiliary anchor classes are balanced, while ordinary BCE retains the original class distribution. This deliberate weighting changes optimization and can magnify noise in the rare class; all context-matched controls share it.

Distinct masks and useful rankings must be measured separately, by label and scaffold. Higher inactive spread, lower hidden loss, better TRAIN BCE or a larger covariance is not success.

## 3. One complete representative experiment

### Scope and fixed learner

Use complete **ogbg-molhiv official scaffold TRAIN**, the already declared complete official VALID development role for selection/readout, and closed TEST. This plan accesses none of them now. It is a fresh successor with the complete five-layer, width256 bond-aware residual GINE/virtual-node learner, categorical atom/bond encoders, shared stateless LayerNorm, original private internal/boundary factor sites and **mean raw-logit serving**. It is not a reproduction of an OGB leaderboard recipe or a claim of established competence.

Retain the full original100 epochs, batch128, Adam learning rate0.001, two ordinary own-BCE stochastic views of every TRAIN graph and strict-first full-development ROC-AUC selection; no early stopping or shortened cap-completion. Keep the source-native dropout0.5 and existing initialization/reset order. Shared route arms use all-unit factors. Independent models use genuinely distinct constructor/dropout streams, not identical copied predictors.

The existing full same-operation single and independent4 are architecturally capable references. Complete Mol18 evidence must establish that their acquisition/recipe is a meaningful quality reference before admitting this successor. If a baseline is materially unqualified/underfit, repair its recipe in a separate frozen specification first; do not reinterpret a context win over a weak reference. Retained PNA/ACA and Chemprop are external quality context, not additional unqualified arms in this experiment.

All contrastive conditions use **one scalar objective**: ordinary own BCE plus0.05 times the context loss, temperature0.2, all ordinary live parameters as recipients. Shared banks use the mean of member losses; each genuinely untied model optimizes its own corresponding objective. No residual-member repulsion, inherited phi-only alignment, O/I/P/G credit rule, class-balanced BCE, new graph branch or projection head is silently retained. The current studies remain unchanged.

### Four fixed chemical signatures and donor rule

Before fitting, pin one RDKit version and the original input-molecule parsing/canonicalization. Do not change salt/tautomer/protonation/stereochemical handling to improve outcomes. Use four binary2048-bit signatures:

1. Morgan radius2 fingerprint with chirality included: local atom environments.
2. RDKit pattern fingerprint: fixed substructure-pattern inventory.
3. Topological-torsion fingerprint: four-atom path environments.
4. Atom-pair fingerprint: atom environments and graph distances.

They are standard overlapping structure descriptions, not four independent biological evidence sources. Compute fingerprints/scaffold keys only for admitted TRAIN graphs. For each TRAIN anchor and route, select **one OTHER same-label TRAIN donor with a different Bemis–Murcko scaffold key**, maximizing signature Tanimoto with strictly positive similarity; deterministic TRAIN-row ties. Use the provider/toolkit's empty scaffold key consistently for acyclic molecules; do not invent an outcome-dependent special class. If the signature is empty or no eligible donor has positive similarity, use self only and mark the row inactive for off-object steering.

A row with a donor gives equal mass to its self cross-view positive and that donor. Every other graph in the auxiliary panel, including other same-label graphs, stays in the denominator. No threshold/k/fingerprint grid is included. This does not assert that same-label cross-scaffold donors are semantically equivalent.

### TRAIN-only auxiliary panels

Every eighth supervised update, choose32 TRAIN auxiliary anchors:16 from each binary class, using fixed cyclic shuffled class lists and an RNG separate from model/dropout. All TRAIN graphs still receive their original ordinary supervised exposure each epoch; these panels are extra regularization minibatches, not a reduced dataset.

Add the union of the anchors' factual and shuffled route donors (defined below). With one donor per route/control, the panel has at most **32+32×4×2=288 distinct graphs**. Encode the same original graphs in two native stochastic dropout views; no chemical edits. Score32 anchor rows against all opposite-view panel embeddings, with equal total anchor weight per class and symmetric view exchange. Keep donor tensors live; there is no stale memory bank or detached teacher.

Every context arm, including the single/untied matched controls, uses identical anchor/donor panels and denominator populations. Qbar is averaged **after** panel normalization. A single Qbar receives all four context relations. Auxiliary RNG is isolated/restored so the added panel forwards do not silently consume the ordinary own-view dropout stream. Ordinary BCE-only references retain their complete native recipe; their absence of auxiliary gradients/work is disclosed rather than padded with meaningless computations.

This panel is a proposed source requirement, not implemented or measured. Charge fingerprint/scaffold preparation, donor search, both auxiliary views, all member trajectories and selected-state replay. Original Mol18 caps/time measurements cannot be imported for it. Qualify the full graph-size distribution and inclusive budget before any admission; stop rather than shrink the dataset, lower epochs or silently approximate targets.

### Factual, common and shuffled targets

The common arm uses Qbar for every route. It matches aggregate context mass, labels, panels and denominator samples, while generally increasing per-route target support/entropy. Report that distinction.

For the shuffled arm, start from each route's fixed off-object donor map. Apply one prescribed seeded sequence of donor endpoint swaps between anchors of the same label, accepting only swaps that retain nonself/different-scaffold eligibility. Self-only rows stay self-only. A fixed ten sweeps of shuffled anchor pairings is proposed; no outcome-based retry. This preserves every anchor's positive count, label compatibility and the donor-count assignment to individual graphs, while breaking factual fingerprint-neighbor incidence. It is a relation control, not a scaffold-causal intervention or a distribution over all random target graphs.

Freeze target/panel/scaffold/toolkit hashes before fitting. Require route-to-common and factual-to-shuffled target total variation of at least0.05 **separately for both labels**, along with nonempty eligible active-class donor support. Inspect empty signatures, same-donor overlaps, donor concentration and scaffold overlap without opening fitted outcomes. If these source-only opportunity checks fail, do not run an undifferentiated family or change signatures/seeds to force the desired result.

### Fixed comparison set, not a parameter grid

One candidate, eight fixed conditions, three fresh paired optimizer blocks proposed as9101/9203/9307:

| Condition | Role |
|---|---|
| Full single, ordinary two-view BCE | Capable ordinary single quality reference. |
| Ordinary independent4, each full own-selected BCE model | Conventional independently acquired ensemble reference. |
| Full single plus Qbar | Objective/context/panel-matched single explanation. |
| Untied4 plus respective Qm | Objective/context-matched independent member explanation. |
| Unit-factor shared BE, ordinary two-view BCE | Same-architecture no-context reference. |
| Unit shared BE plus common Qbar | Aggregate-target control. |
| Unit shared BE plus factual route Qm | Sole candidate. |
| Unit shared BE plus shuffled route targets | Chemical relation-incidence control. |

This is **24 method-seed endpoints**, not24 single-model acquisitions: each independent bank charges four complete fits, and shared banks still evaluate four trajectories. Freeze the complete set/recipe before opening outcomes; no coefficient, signature, sampling, width, loss-routing or selector sweep. Source compatibility could allow an existing ordinary endpoint to be reused only if all contracts match; no such reuse is assumed now.

Single and independent members retain strict-first own development selection; shared banks retain joint mean-logit development selection. Those are complete practical methods with disclosed selection differences. Comparisons do not isolate parameter sharing from selection/optimizer geometry. No secondary shared-bank selector or another fitting arm is added.

## 4. Readout, falsifiers and contingency boundary

Primary contrast: factual routes versus common targets. Relation interpretation additionally requires factual routes outperform shuffled targets. Report every complete/failed endpoint and seed; served ROC AUC, mean/worst member AUC, pooled/member BCE/Brier, selected epochs, and inclusive work. Do not promote a direction that improves only hidden loss or a selected chemotype subgroup.

For active molecule a and inactive b, define d_m=z_m(a)−z_m(b). If every d_m<0, the raw-mean-logit pool reverses that pair too. Freeze common-arm pair-error cohorts before candidate scores are opened; count repaired and newly introduced orderings over the **full** development active–inactive pair population, with ties worth1/2. At least one member must acquire a nonnegative/positive relevant margin for a pooled reversal; one changed member alone need not offset the others. Scaffolds/molecules and derived pairs are dependent units, not millions of independent observations.

Global offsets change BCE without changing AUC. Per-member positive rescalings preserve member rankings but can reweight the mean-logit pool. Distinguish acquired rankings from such score effects. Report both labels' donor activity, within-label representation geometry, common pair reversals and full-population net ranking contribution. Scaffold/chemotype diagnostics are prospective descriptive strata, not endpoints chosen for favorable outcomes.

Failure cases include: active targets collapse to self/common rows; factual and shuffled donors work equally; context preserves mainly scaffold/size cues; inactive spread rises without ranking benefit; rare-active overfitting damages mean/worst competence; or common/single/untied context models explain the gain. Pair-cohort repairs canceled by new reversals are not useful complementarity. A favorable common-target gain supports a general supervised molecular regularizer; a favorable assignment contrast supports only this fixed route-target intervention.

Contingency only: first finish the unchanged context9/Mol18 families and their complete scientific readout. Proceed only if they leave a material reason for chemical target assignment and competent references. This assessment does not monitor jobs, reserve GPUs, admit a fit, open VALID/TEST, change a running source, or infer a positive outcome. Any later generalization claim needs separately specified unused scaffold/task confirmation with capable controls; none is accessed, reserved or launched here.

`CONCLUSIONS.json`, `READ_SCOPES.json`, retrieval/passages and `INPUT_BINDINGS.json` preserve NEW versus REUSE scopes. `CONDITIONAL_DESIGN_DISABLED.json` records this single unadopted comparison set. The manifest covers only this new folder.
