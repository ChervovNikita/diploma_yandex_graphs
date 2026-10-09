# A masked-context reconstruction hypothesis for private BE members

10 October 2026. Source/literature proposal only. One hypothesis is retained; no fit, model, dataset, checkpoint, server operation, manuscript change or new scientific owner is created. A15 diagnostics and Q/K36 comparative outcomes remain outside this assessment.

## Purpose and recommendation

Test whether **different persistent missing-feature contexts can teach competent private BE members to use different useful graph evidence**. Every member still learns all factual TRAIN labels. A second, masked-input forward both classifies the same TRAIN nodes and reconstructs the missing raw features through a common decoder. The shared backbone remains trainable. Serving uses the unchanged factual graph and mean member probabilities; the decoder is discarded.

This is an attributed composition worth a small, complete, controlled screen if the root history check finds no identical recipe. It changes the information available during the auxiliary forward and asks the actual classifier to work with that information. It does not repel members, invent different class labels, route private gradients manually, use a teacher, or simply move the old same-class contrast inside the network.

The measured deficit supplied by root is redundant decisions with no stable gain from basic contrast or relation credit. It does **not** establish latent collapse or missing graph information. This proposal could be useful if some correct alternatives require neighborhood context that factual self-features let the current routes ignore. It fails if restoration is task-irrelevant, is solved only in the decoder/nullspace, weakens members, or supplies no net pooled repairs.

## Saved history comes first

The canonical pointer names supplement v38 and the saved base index v72. Their predecessor metadata and existing scope receipts were used for discovery deduplication, with zero new credit for those reads. The saved competence/private-propagation, persistent-positive, MA-GCL recovery, decision-visible-response, GCMAE/DGE and recent contrastive conclusions were consulted before selecting a primary body.

Those records already contain shared/private capacity, graph/filter/model views, persistent same-class positive assignment, output-visible finite responses, reconstruction auxiliaries, bootstrap neighborhoods and function-prior hypotheses. None is reissued as new. In particular, the old persistent-positive recipe changes which same-class observations are aligned while leaving every predictor input intact. This proposal removes a target node's complete raw feature vector from an auxiliary predictor input and requires its own classifier to learn on that input. That is a materially different complete training operation, although its ingredients have direct ancestry.

## One newly read primary scope

**CORE: Contrastive Masked Feature Reconstruction on Graphs**, Jianyuan Bo and Yuan Fang, [arXiv 2512.13235v1](https://arxiv.org/html/2512.13235v1), is the sole new method identity read. The inspected operation is §3 plus §4.2, Eqs. 1–3 and 12–13. §4.1 was also read for its stated rationale, and adjacent §5 setup text was exposed; numerical paper results are not adopted. Figure pixels, full benchmark tables, author code and accepted-version equivalence were not audited. The v1 carries unresolved conference/DOI template placeholders; no publication venue is claimed.

CORE masks node features, applies an encoder/decoder, contrasts each masked node's reconstructed vector `z_i` with its own raw vector `x_i`, and draws negative **reconstructed** vectors `z_k` from other masked nodes. Positives and negatives are restricted to masked nodes. It retains the chosen masked-autoencoder masking/decoder techniques, rather than specifying a unique native supervised backbone. The inspected method is a self-supervised encoder, not a simultaneously trained four-member BE classifier with factual and masked own CE and persistent role assignment.

The proposed adaptation uses this finite-temperature masked-node objective, not a faithful reproduction of all GraphMAE/CORE techniques. Its positive is `(z_i,x_i)` and its negatives are `(z_i,z_k)`; replacing them by raw `x_k` would change the operation.

Do not import the paper's low-temperature objective-equivalence claim. With positive cosine `a=0.5` and all negative cosines zero, InfoNCE tends to zero as temperature tends to zero, while cosine reconstruction error stays `1-a=0.5`, even with perfect unmasked reconstruction. If a negative has higher cosine than the positive, its loss instead diverges. The printed passage's step from a constant zero limiting objective to maximizing positive cosine does not establish equality of the objectives. This algebra limitation does not invalidate the finite-temperature Eq. 12 used as an attributed auxiliary here.

## The complete proposed rule

Use four existing source-qualified internal-BE routes in one capable native backbone, without changing their factor sites or parameter ownership. Start from fresh unit-factor constructors; all shared weights, factors, classifier parameters and one common auxiliary linear decoder learn through one scalar objective and ordinary native optimizer steps.

Before training, partition all nonzero-feature nodes into four balanced sets using a fixed label-free hash ordering of public node IDs. Each member always masks its assigned set to zero in its auxiliary input. Edges, full node coverage and native self-loop/duplicate policies remain unchanged. Zero-feature nodes remain in all classification populations but are not reconstruction anchors. No node-ID embedding, per-node parameter, factual hidden cache, raw-target skip or unmasked target input may enter the auxiliary predictor/decoder.

For every update and every member, compute factual logits on `(X,A)` and masked logits/hidden states on `(X_masked_m,A)`. Use **every** official TRAIN class target in both own CE terms. Decode only masked anchors with the same common linear hidden-to-raw-feature map. At fixed temperature `0.2`, draw 32 other masked reconstruction negatives per anchor, without replacement; all draws have separate logged streams. If a graph cannot provide that many distinct eligible anchors, the family is inadmissible rather than silently changing the loss. Raw features and reconstructed vectors use a declared `1e-8` norm floor.

The scalar objective is

```
mean_m CE(factual logits_m, TRAIN labels)
+ 0.5 mean_m CE(masked logits_m, TRAIN labels)
+ 0.1 mean_m CORE(masked reconstruction_m, raw targets).
```

Neither coefficient, temperature, mask fraction, factor location nor reconstruction dimension is searched. This is a fixed proposal before scientific outcomes. Validation/test class values are never reconstruction inputs or optimization targets. Label-free raw features on the complete graph are used as transductive targets and must be disclosed; this is additional training supervision from already allowed features, not new observed class information.

A common decoder restricts the escape route of giving each member an unrelated reconstruction head. It does not eliminate classifier-nullspace satisfaction. The masked own CE connects the auxiliary forward to the actual classifier, but it still cannot guarantee factual competence, complementary correct decisions or generalization.

## Closest complete-rule collisions

| Prior or saved proposal | Shared operation | Material difference to test |
|---|---|---|
| CORE / saved GCMAE | Masked feature reconstruction, graph encoders and contrastive auxiliary training | Factual and masked supervised own CE; four persistent BE predictors with one live shared backbone; fixed member-owned missing-feature contexts; factual probability pooling |
| Saved GRAND / CAMERO | Shared weights, perturbed inputs and supervised/consistency learning | No cross-member agreement term; masked raw-context restoration and persistent factor-owned predictors remain served |
| Saved MA-GCL / AMCL / CGCL | Model views, common encoders or multiple contrastive heads | Input evidence is removed before the whole auxiliary predictor; the target is raw missing context, rather than old same-class pairs or altered propagation placement |
| Saved DGE / DIVE / SuGAr | Different neighborhood/input evidence and supervised diversity | Static original graph is retained; no invented observed path history, learned subgraph mask, fully untied encoder, mask repulsion or selected single-member deployment |
| Existing persistent-positive and functional-response plans | Stable member roles, graph supervision and useful-decision tests | The complete masked-input, raw-target and masked-classification rule is different; its contribution must be established in deployed factual decisions |

These scoped differences permit a controlled attributed composition. They do not certify originality or a quality advantage. A complete prior that already specifies this rule would shift the work to reproduction/extension; known pieces alone do not disqualify it. The containing untied family can implement the same computation with more freedom, so any sharing benefit is empirical.

## One complete representative development experiment

Nominate **the complete PubMed Planetoid graph and official split**, not an induced graph or favorable cohort. It has informative raw feature vectors, graph context and sparse supervised labels, which makes the information-restoration mechanism plausible. The graph's original paper/current scores are not recomputed. Root must check whether this task/configuration is already exposed; if so label the study developmental, without replacing it using outcome evidence. Use three prospectively fixed optimizer seeds `9101,9203,9307` and all official TRAIN/VALID/TEST nodes in their declared roles.

Use the already source-qualified capable native PubMed configuration, with its full horizon, optimizer, early/global transition and validation selector frozen before this family. If no competent native configuration is qualified, do that source/reference qualification before admission; do not substitute an arbitrary weak GCN or a short run. The exact source/config is a required binding, not guessed in this literature packet.

| Condition | Purpose |
|---|---|
| Own-only shared BE4 | Existing factor sharing without the new auxiliary |
| Shared BE4 + factual/masked CE only | Whether masked supervised augmentation explains the gain |
| **Proposed persistent masked-context reconstruction BE4** | One candidate |
| Candidate with updatewise permutation of mask ownership | Exact same four masked views and anchor/negative mass each update, but no persistent route-context association |
| Candidate with a label-free degree-preserving rewiring only in auxiliary forwards | Whether original neighbor alignment matters beyond generic masked regularization; factual graph stays original |
| Ordinary capable native single | Strong single reference |
| Ordinary genuine independent4 | Strong ordinary ensemble reference |
| Capable native single with all four matched masked views and the same auxiliary | Whether the same graph learning signal benefits a simpler single predictor |
| Genuine independent4 with the same persistent-mask auxiliary recipe | Whether sharing itself contributes beyond objective-matched independent members |

This is 27 condition/seed records, with full independently trained bodies charged where required. It is one fixed experiment, not a coefficient/data/backbone grid. All routes/bodies train from scratch, with no donor or teacher. The updatewise permutation preserves input/view and target mass at each update; a merely common quarter-mask control would confound distinct-target coverage and is not substituted. A fixed whole-fit mask permutation is only a route-renaming equivalence check, not another fit.

The auxiliary rewiring preserves degree, nodes, feature vectors, existing self edges and edge multiplicity policy using a prospectively frozen double-edge-swap recipe. Its swap count, rejection cap and fingerprint must be qualified before fits; failure leaves that cell incomplete. With native global attention, even a masked node can access non-neighbor features. Therefore this is a conditional local-neighbor-alignment control, not proof that the complete predictor uses only local graph information.

Require whole-population improvement in factual served accuracy and NLL against own-only, masked-CE-only, the capable single and ordinary independent4; then compare with objective-matched single/independent controls. Protect mean and worst own-member accuracy/NLL with margins chosen by root before execution. Record every seed, member, introduced error and repaired error, any-correct coverage, strict common wrong rivals, pooled harm, and mask/degree subgroups as diagnostics only. A larger coverage or smaller reconstruction loss cannot rescue failed whole-population quality. Three optimizer seeds are exploratory on one fixed graph; node counts are not independent graph evidence.

Interpretation is deliberately discriminating. If ownership permutation matches, persistence adds no supported value. If masked CE matches, the contrastive restoration ingredient adds no supported value. If the same-aux single matches, the ensemble explanation is unnecessary. If same-aux independent4 matches or wins, shared-bank quality superiority is unestablished. If rewiring matches, the local graph-evidence account is unsupported. If members weaken or repairs are canceled by harms, reject the hypothesis. Passing this complete development screen would justify unused-task/split confirmation, not an acceptance claim.

Charge factual and masked forwards, the decoder and 32 negatives, all view/mask generation, extra single-model auxiliary views, genuine independent training, per-epoch full selections, all attempts, checkpoints, collection/auditing, and unchanged factual deployment. No speed, memory saving or additional-GPU need is established. Root must forecast the entire family before compute; a cap failure leaves this proposal inactive, with no shorter/subgraph fallback.

## Evidence accounting

`READ_SCOPES.json` binds the sole new primary method identity and the actual bounded scopes. `REUSED_MEMORY_BINDINGS.json` records saved memory; `DISCOVERY.json` records three bounded metadata queries and source/abstract exposure. Identifier dedup was only a first pass: ASPECT was rediscovered and treated as saved ancestry even though the automatic identifier check did not recognize it. No false new-ASPECT reading credit is given. Other hits remain metadata/abstract-only.

An early broad pathname-only literature locator used the word “CORE”, which also matched ordinary “core/score” prose. Its result showed filenames rather than comparative content; it was not used for outcome selection and was replaced by precise, protected-path-excluding searches. No Q/K36 or A15 comparison values, raw outcomes or model tensors were opened. The paper's stated benchmark numbers were incidentally displayed by a broad primary-text locator; no performance claim or dataset choice is based on them.

New primary method identities/scopes: one. Full papers, author-code audits, experiments, source changes to active studies and novelty/superiority claims: zero. Canonical-memory adoption and experiment admission remain root-owned.
