# Attribute feedback between meanSAGE ensemble members

Freeze one inactive architecture study on the competent source-derived meanSAGE backbone. Each member sends its first-layer hidden states through the raw node-feature incidence graph. A shared nonlinear map processes each member's attribute context separately. Each member then receives the other three contexts before its final graph layer. The operation changes the paths by which distant nodes and live member states influence a prediction. It is substantially different from the frozen sign-coupled dropout study, which remains unchanged and has execution priority.

The hypothesis is that peer attribute contexts can help weak shared routes acquire useful class evidence and retain more correct alternatives during native probability averaging. Generic attribute diffusion and continuing-stream communication are established. The proposed interlayer operation is untested; no new primitive, accuracy theorem or broad novelty clearance is claimed.

## The observed failure and the competent reference

The closed Physics E study lost 1.462764 accuracy points to its capable equipped I4, and its mean members were 5.174743 points weaker. It supplied 4,379 additional covered alternatives but lost 9,369 more during pooling. Its endpoint increment over N was only 0.028158 points. Preserve these failures. They motivate separate measurements of member competence, correct-member supply and actual pooled repair; they do not establish that missing attribute paths caused the errors.

The source-derived published meanSAGE single reached 93.058986% in the complete ten-block summary. It uses width 32, different normalization, regularization and selection from the weaker original native recipe. The new study adopts that complete backbone recipe and includes a fresh exact source single. This is a source-derived Torch port, not a bitwise TensorFlow or published-split reproduction.

## Closest prior methods

| Method | Established operation | Consequence for this proposal |
| --- | --- | --- |
| [CONN](https://arxiv.org/html/2307.11981v1), §§III–IV | Augments the graph with attribute-category nodes using the feature matrix; node-to-attribute-to-node paths enrich receptive fields. Its chosen model uses trainable initial embeddings, linear propagation and a reconstruction objective. | Directly occupies the generic incidence graph and two-way attribute diffusion. Applying it to meanSAGE under CE is an attributed adaptation. |
| [TextGCN](https://arxiv.org/html/1809.05679v3), Method | Document-word edges, word co-occurrence edges and a two-layer GCN permit document-to-document exchange through words; supervised document CE. | Supervised learning through feature-like intermediate nodes predates the proposal. The feature/path idea is not new. |
| AM-GCN, retained primary conclusions | Propagates topology and feature-kNN graphs; common transforms are shared, specific transforms remain separate, embeddings are attention fused. | Shared processing of graph and feature evidence is established. Its fused single is a relevant generic explanation. |
| Cross-stitch and Set Transformer, retained primary conclusions | Continuing streams exchange activations; set attention and nonlinear set aggregation are established. | Excluding a recipient and averaging nonlinear donor maps introduces no new communication or set primitive. |
| Saved private feature-kernel SAGE proposal | Adds sparse feature-neighbor support and private learned metrics through own CE. | Already retains a generic absent-edge acquisition question. The present study changes to attribute hubs and live peer-state feedback rather than renaming that proposal. |

Two fresh primary method scopes were read: CONN §§III–IV and TextGCN Method. Exact blocks, public versions, raw hashes and quoted passages are retained in `READ_SCOPES.json` and `PRIMARY_PASSAGES.json`. Eight OpenAlex queries are metadata discovery, including five inherited saved queries. Unselected titles and unresolved prior access supply no absence evidence. No full-paper, proof or author-code reproduction claim follows from these reads.

## One fixed operation

Let `B_ij=1[X_ij>0]` for the full permitted binary feature matrix. Normalize `B` by node row counts to obtain node readback `D`; normalize `B^T` by attribute counts to obtain attribute aggregation `C`. Zero-count rows or columns produce zero outputs. Both operators are sparse and fixed. They use no labels, prediction errors, learned graph or kNN construction.

For four first-layer source hidden states `H_m` of width 32, compute:

```text
U_m = C H_m
phi(U) = clip_row(ReLU(U Bmap) Vmap)
T_m = (1/3) sum_{n != m} phi(U_n)
Hprime_m = clip_row(H_m + .25 D T_m)
```

Here `Bmap` is 32×16 and `Vmap` is 16×32; both are shared over routes, with no biases, added dropout or affine normalization. `clip_row(z)=z/max(||z||_2,1)`. Initialize `Bmap` by Glorot uniform and `Vmap=0`. The initial complete function therefore equals the corresponding backbone: native first-layer hidden rows already have norm at most 1. Donor-path and `Bmap` gradients initially vanish until `Vmap` changes. This is initial function preservation, not protection of the final learner.

Insert only after the first source normalized ReLU and before the second source dropout. Keep source dropout .4, aggregate-plus-skip equations, clipped final logits, LR .001 TF1 Adam port, first-layer regularization and latest-tied-minimum SELECT total-loss criterion. Stop after 50 strictly worse SELECT totals or 100000 updates. The raw binary features are not row-normalized before the source first layer.

For the shared bank, each source aggregate/skip matrix has private learned input/output factors; layer biases are private. Regularize the squared **effective** first-layer matrices, averaged over members, at coefficient .05. Penalizing only shared slow weights would change the comparison through the factor gauge. Factor M1 is a separate control because factor optimization is not the original source learner even when its initial function distribution matches.

Every member's own CE sends gradients into its recipient first layer, the other donors' first layers and the shared feedback maps. There are no detached donors, teachers, reconstruction losses, diversity rewards or pooled training CE. The bank trains by the mean of four own CEs plus the mean effective regularizer. Serving uses one complete dropout-off coupled forward and the arithmetic mean of own class probabilities. No final router or graph smoother is selected.

This deterministic feature graph adds computation paths, not newly observed information. Distant nodes sharing attributes can influence a recipient before its next graph aggregation; the native two-layer social/coauthor support alone may not carry those paths. Their value is a hypothesis: frequent nuisance attributes, common wrong peer contexts, over-smoothing and shared-gradient interference can instead make members weaker or align their mistakes.

## Restricted mathematical explanation

Averaging `phi(U_n)` generally differs from applying `phi` to the peer mean. For example, opposite scalar donor values under ReLU can yield a positive mean of activations while their mean has zero activation. This ordinary nonlinear noncommutation explains what the donor-before-mean control changes. It does not show that useful class evidence survives or that accuracy improves.

Because `C` and `D` are nonnegative row-substochastic and `phi` clips its rows, the added feedback row has norm at most .25. This local bound supplies no gradient, competence or confidence guarantee.

The entire bank is equivalent to one structured width 128 network with grouped private maps and a fixed off-diagonal relation between route states at attribute hubs. Four own losses constrain that model; they do not make its members statistically independent. A single with greater feedback capacity could reproduce the computation. The panel tests a capable source single with the same incidence feedback, while leaving a universal comparison against every wider single outside its scope. A positive result supports this particular architecture and learning/storage point, not an ensemble expressivity separation.

## The decisive panel

Use ten banks per block and three fresh paired mask/optimizer protocols: 10103, 10427, 10949. The complete panel has **30 banks,48 optimizer trajectories,57 physical backbone instances and30 CAL endpoints**. A coupled unshared bank has four physical bodies but one joint optimizer trajectory and selector; it is not genuine I4.

| Arm | Feedback | Purpose |
| --- | --- | --- |
| P1 | None; exact source meanSAGE single | Published-recipe competence |
| F1 | None; factorized meanSAGE single | Factorization competence |
| I4 | None; four genuinely independent factor M1 fits | Capable ordinary ensemble |
| S4 | None; shared factor bank | Common shared baseline |
| PA1 | Self attribute feedback; ordinary source single | A more capable source single with the same extra paths |
| IA4 | Self attribute feedback; four genuine independent factor fits | Capable same-information feedback ensemble |
| SA4 | Self attribute feedback; shared bank | Same maps, site and incidence work; peer-state access removed |
| SM4 | Peer mean formed before `phi`; shared bank | Same parameters and four map calls; nonlinear placement changed |
| SX4 | Mean of nonlinear peer donor contexts; shared bank | Candidate |
| UX4 | Candidate operation, four disjoint bodies and feedback maps | Same live peer opportunity without parameter tying |

All arms use the same full X, graph, masks, native normalization/dropout, optimizer equation and declared source/effective regularizer. Equipped arms have the same rank 16/scale .25/site/clipping and zero-output start. Shared banks select the mean own total SELECT objective; genuine I4 models select their own full objectives independently. UX4 is jointly trained and selected through live cross-body gradients. It tests complete tying, including feedback-map tying. These selection and parameter-count differences are explicit; the arms are not described as equal capacity or equal compute.

Each fresh block uses 20 TRAIN, 30 SELECT and 30 CAL nodes per class; the remaining 34,093 are DEVELOPMENT_EVAL. All selected states and 30 bank exports close before CAL; every fixed CAL temperature closes before one complete DEVELOPMENT_EVAL release. This is development on an already encountered graph. Old TEST truth, prediction arrays and selected states are not reused. Root owns source/provider/resource qualification and later execution admission.

## Predictions and falsifiers

Before a positive attribution, factor F1 and I4 mean-member accuracy must remain within .25 points of source P1 on average, with no block harm over .5 points. A factorization competence collapse defeats the transferred shared-backbone construction; weaker controls cannot rescue it.

SX4 must improve raw pooled accuracy by at least .25 points on average over S4, P1, PA1, I4 and IA4, with positive paired gain in all three blocks. Its mean accuracy must at least match UX4, with no block harm over .25 points. Mean own-member accuracy must improve at least .25 points over S4 in all-positive block differences, and remain within .25 points of IA4 on average and .5 in each block. Calibrated NLL mean harm may not exceed .02, nor any block .05, against the capable references. Raw NLL protection and all raw/calibrated Brier outcomes remain declared in the structured proposal.

For peer-state attribution, SX4 must beat SA4 by .10 points on average with no negative block difference. The same margin over SM4 is required to retain the donor-before-mean explanation. If PA1 or IA4 matches, the known generic feedback single/ordinary ensemble suffices. If SA4 matches, access to peers is unnecessary. If SM4 matches, the nonlinear ordering adds no demonstrated value. UX4 improvement defeats shared accuracy superiority; an honest storage tradeoff remains a separate result.

Count covered predictions, pooling losses and aggregation-only correct predictions exactly. Their identity is `served_correct=coverage-pooling_losses+aggregation_only`. SX4 pooling losses must not exceed S4 or IA4 on average to claim repair of the recorded pooling failure. Report gross repairs/harms, mean and worst members, every class and every block. On the fresh S4 common wrong-rival set, distinguish newly acquired correct members from actually served repairs, including cases outside fresh I4/IA4 coverage. These regions never define training weights or the incidence graph.

Three blocks are an exploratory screen, not confirmatory evidence. Complete vectors, descriptive uncertainty and eight sign-flip assignments cannot establish general superiority across graph domains. A complete useful result takes unused whole-pipeline confirmation priority. Failure closes this exact rank, scale, site and operation without a rescue grid.

## Paid work and disposition

The symbolic candidate count is 607,924 trainable parameters versus 2,227,636 for four equipped factor M1 bodies. The ordinary source single has 538,917, and its feedback version 539,941. These are parameter-storage counts, not peak memory or capacity equivalence.

Let `S=nnz(B)`, which this scout did not inspect. Each candidate forward adds `256 S` sparse scalar MACs and 34,467,840 dense MACs, plus clipping, donor mixing and normalization. Three hub tensors each hold 1,077,120 FP32 scalars; the exchanged node tensor holds 4,415,104. Native tapes, sparse storage, optimizer state, backward, every selector forward, factor regularization, preprocessing, exports and CAL all count. No dense node-pair feature graph is built. The full ceiling is 4.8 million optimizer updates and 9.3 million TRAIN route forward-backwards, with 9.3 million SELECT route forwards; actual early stopping should be measured, not assumed.

The coarse planning envelope is 6–24 GPU hours, with unknown incidence sparsity and optimizer horizons. It is not a runtime bound or ETA. Root must qualify one actual full-input update and finite resources before execution. The old overlapping 90-fit cohort cannot supply an isolated speed comparison.

Retain this single inactive next mechanism. Current sign-dropout closure has priority. No scientific run, model/array import, score recalculation, source admission, reviewer or manuscript edit occurred. Core implementation is not included; the complete control and gradient contract is ready for root to assign a distinct source owner if this study is next.
