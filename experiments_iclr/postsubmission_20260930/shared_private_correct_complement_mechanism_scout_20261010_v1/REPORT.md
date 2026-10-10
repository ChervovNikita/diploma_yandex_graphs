# Private hop-occlusion training: one prospective competence/acquisition mechanism

**Disposition:** one concrete, inactive training hypothesis. No implementation, source edit, solver, fit, grid or launch. Hop/view diversification and selective gradient routing have prior art; methodological novelty is unresolved. This is neither a rescue of the failed CORE screen nor an ensemble-specific interpretation of the ordinary PubMed gain.

## Evidence changes the question

Root's now-closed M1 comparison reports native89.937426%, factorized M1-native90.816844%, factorized M1-fourdrop90.783020%, shared M4 90.901404% and own-selected I4 90.216472% mean VALID accuracy. Shared minus M1-native is only+.084559pp with +/+/− seed signs, and M1-fourdrop has slightly lower mean NLL. Factorized learning reproduces most apparent accuracy improvement. These supplied observations support an optimization/reparameterization explanation; they do not prove its cause or certify a fully parameter-matched M4 comparison.

The closed complete12/error synthesis nevertheless identifies a different deficit. Shared members are strong and almost every remaining bank error has no correct member:343/356/363 common-wrong nodes versus only4/8/2 lost alternatives. Own-selected I4 supplies broader oracle coverage, but weaker members and many more lost alternatives. Unchanged convex pooling cannot solve the shared common-false-rival witnesses. A candidate must acquire new correct decisions without creating larger competence losses. More spread or better masked-view accuracy is insufficient.

## The mechanism

Use the existing native K=2 polynomial token bank `Z=[X,PX,P²X]`. All inference and shared-body training use this full bank. Give four private routes these fixed auxiliary banks:

| Route | Auxiliary bank |
|---|---|
|0|`[X,PX,P²X]`|
|1|`[X,0,P²X]`|
|2|`[X,PX,0]`|
|3|`[X,0,0]`|

Zero a complete propagated token block at the native token interface; keep its slot and native token/position semantics. Preserve X, the graph, normalization, residual/root paths, architecture, all native factors and output boundary. This is an explicitly incomplete-evidence input, not a claim of equivalent propagation on a modified graph. No features are reconstructed and no label/pseudo-label becomes an input. A source audit must verify the exact interface before admission.

The intended intervention is **private learning of alternative evidence reliance**. The shared core continues to learn complete-context classification. Private factors receive additional correct-label pressure under different missing-hop conditions, with complete factual supervision retained. At inference every route again sees the full context. The hypothesis is that this discourages identical shortcuts and yields some correct alternatives on factual common mistakes while retaining the strong factual member decisions. It introduces no new raw information, so this explanation is unproved.

## Exact update, before one native optimizer transition

Let theta include every shared native parameter; phi_m include every existing private parameter of route m. Let L_m be ordinary TRAIN CE from its factual full-bank forward, and A_m the same all-TRAIN CE from its assigned auxiliary bank. Use the previously declared masked-CE coefficient lambda=0.5 as a **fixed prospective value**, with no strength search. Its provenance is the earlier CORE protocol; CORE's observed outcome does not select this value.

At one unchanged parameter state compute

`g_theta = partial_theta [(1/4) sum_m L_m]`

`g_phi_m = (1/4) [partial_phi_m L_m + 0.5 partial_phi_m A_m] / 1.5`.

Each partial treats the other parameter blocks as independent constants. Auxiliary cotangents reach phi_m only. Shared weights/biases/norms receive no direct auxiliary gradient. The division keeps the total private loss coefficient equal to the native own coefficient; all factual TRAIN examples still contribute with weight2/3. There is no claim that private gradients or member competence are unchanged.

For route0, A_0 is exactly L_0 from the same factual stochastic forward, so reuse that scalar/derivative and obtain its native own gradient exactly. For routes1–3, use separately prescribed auxiliary dropout streams paired across controls. Gather all gradients before advancing any optimizer. Advance the existing native optimizer groups/moments once, using g_theta/g_phi, the unchanged native schedule/decay/clipping and selector. Do not advance factual RNG or mutable factual buffers through auxiliary calls; the native buffer inventory and any required preservation must be qualified. Auxiliary forwards do not need or acquire TEST labels/IDs.

This is implementable with ordinary first-order differentiation: accumulate the four factual gradients, keep shared gradients unchanged, scale private factual gradients for routes1–3 by2/3, and add their assigned auxiliary private gradients with coefficient1/(12). Route0's native private derivative is retained. No higher-order graph, teacher, external head, decoder, reconstruction target, contrastive negatives or parameter repulsion is used.

The block update is generally **not** the gradient of one scalar loss. Future shared gradients can change because factual predictions depend on subsequently changed private factors; keeping its immediate objective native does not preserve its trajectory or guarantee competence. The result must be tested in the factual task space.

## The decisive controls

1. **Competent factorized M1 with the same view bank.** One factorized native model receives factual shared credit and private credit from L plus lambda times the uniform mean of the four A views, with the same1.5 normalization. Reuse its full auxiliary view from the factual forward. This is four full forward paths per update, no weak ordinary single substitute. If it matches the candidate, the effect is multiview factorized learning rather than useful route plurality.
2. **Shared M4 with full-input auxiliary passes.** Same update and factual/auxiliary stochastic streams; replace all three occlusions by full inputs. Route0 still reuses its factual pass. This pays seven paths and controls extra dropout samples, gradient averaging and loss-coefficient semantics.
3. **Shared M4 with a common auxiliary view.** At each update every route receives the same one of the four banks, cycling a fixed balanced four-step schedule. Each route sees every bank over a complete cycle; view frequencies, full supervision and immediate shared objective match. Reuse factual derivatives in the full-view step. This isolates persistent different evidence tasks from generic multiview robustness. Its mean path cost is seven per update over a full cycle; endpoint differences are charged.
4. **Same hop banks with all-block auxiliary credit**, if the private-only claim is pursued: theta also receives the normalized factual-plus-auxiliary scalar gradient. This compares selective body credit with ordinary augmentation. It is an inactive necessary attribution control, not permission to expand a current study.

The already closed ordinary shared/factorized-single/own-selected-I4 observations are references and remain unchanged; do not rerun them to alter current scores. Any claim about sharing or superiority over independent factorized ensembles additionally needs a properly individually selected same-information factorized I4. That reference question is not a novel method and is not admitted here. Do not fit a renamed multibranch computational graph as a spurious independent baseline.

Root must freeze native source/interface, streams, complete budgets, selectors, paired seeds/splits, numerical competence and gain gates before any experiment. Use a full representative task. This PubMed hypothesis is development on an encountered split; positive evidence needs unused confirmation. An unused homophilic/heterophilic task cannot inherit the same mask interpretation or performance guarantee without its own protocol.

## What would falsify it

The primary output is factual probability-pooled VALID accuracy against the factorized all-view single and matched M4 controls, supported by NLL/macro and individual-member/class readouts. Also retain factual common-wrong repairs, introduced common errors, acquired-correct-member events, lost alternatives and their served consequences. All contrasts and failures remain visible.

Predicted principal failure: the propagated tokens are genuinely important, so missing-hop supervision weakens factual members—especially the root-only route—and the shared core cannot supply separate features through small private factors. A weaker route may add noisy votes while failing to repair high-margin common false rivals. Another plausible null is that the factorized all-view single learns the same robustness, eliminating any ensemble-specific advantage. Full-input auxiliary passes may explain any gain through additional dropout-gradient averaging. Changed confidence without new correct alternatives is also insufficient.

A gain only on owned/masked views, a better oracle without served improvement, a useful class traded against larger other-class harm, or improvement explained by common-view/all-block learning does not pass the claimed mechanism. Do not chase failed masks with a schedule or coefficient grid.

## Cost that can actually be defended

The token bank is already available; no new sparse propagations or dataset are needed. The candidate uses four factual paths and three additional occluded paths per update: **7 versus4 native member-forward paths**. It needs corresponding first-order backward work, with auxiliary differentiation only to private blocks. Conservatively budget up to roughly twice the current shared fit's forward/backward workload; this is an operation count, not a measured wall-clock ratio. Same full-task optimizer horizon and validation work remain required.

At2000 updates this means14,000 route-forward paths versus8,000, plus the unchanged full-bank validation/checkpoint work. Full-input M4 has the same count; common-view M4 averages the same count over four-step cycles. Factorized M1 all-view has8,000 paths at2000 updates, versus2,000 for its ordinary one-view fit. First-order gradients/buffers can be accumulated sequentially; retaining every factual and auxiliary graph simultaneously is unnecessary. Peak memory and throughput still require a real source-qualified measurement.

Native parameters do not increase. Inference remains four factual paths with the existing arithmetic probability mean; no conditional-compute saving is claimed. For PubMed, one dense19717×500 FP32 block is39,434,000bytes, and the three-block bank is118,302,000bytes. Masks may reuse token views and one zero block, or at worst one bank-sized auxiliary scratch if the native interface requires materialization. Do not retain four full masked banks unnecessarily. A five-condition/three-seed complete screen would be substantial; GPU hours and stopping horizons are unmeasured, and this report requests no allocation or launch.

## Collision and exact difference from the failed CORE screen

**CORE source** zeros the raw features of four persistent node-ID quarters and recomputes every polynomial token from each masked X. It adds all-TRAIN masked CE and instance-contrastive feature reconstruction through a common decoder. The scalar loss `.factual + .5 maskedCE + .1 CORE` reaches shared body, private factors and decoder. The complete screen failed its unchanged continuation gates and remains closed. Its unexecuted masked-CE-only/shuffle/four-view-single conditions provide no positive evidence.

This candidate zeros whole propagated token blocks while retaining every raw X, uses no reconstruction or contrastive target/decoder, and sends auxiliary CE only to existing private parameters. Its native shared factual credit and normalized private coefficients are explicit. Those differences make it a separately falsifiable learning operation; they do **not** establish novelty or prove that private-only masking repairs CORE's failure. A successful result would need the all-block and same-view controls to locate its cause.

**Saved GRAND** already combines feature/node masks, fixed mixed-hop propagation and supervised view losses in one shared predictor. **Saved FAGEL/AdaGCN** already use diversified hop/context evidence and specialist/sequential training. **MIMO2010.06610v1**, newly read §§2 and3.5, trains independent input/target tuples through one live network and repeats one input at serving; input repetition explicitly trades independence against finite-capacity competence. Its operator differs from this fixed graph-view/private-cotangent recipe, but the independence/competence tradeoff and shared training are prior. None of these papers supplies a guarantee that our missing-hop views create complementary correct graph decisions.

A second retrieved primary,2007.07296v1, is titled “Privacy Preserving Text Recognition with Gradient-Boosting for Federated Learning.” Its inspected method mixes client models using local/cross-client validation losses. The2023 journal metadata uses a different title (“FedBoosting…”); body equivalence is unverified. It supplies no qualified gradient-protection mechanism for this proposal and is excluded. DIBS was a metadata lead; its PDF retrieval failed and no method was read. No further search or source acquisition is needed to state this bounded question.

**Scientific contribution remains contingent:** if the fixed intervention beats the competent same-view factorized single, acquires served correct alternatives, retains competence and survives unused confirmation, it establishes a focused shared/private evidence-learning result. It still cannot be presented as newly discovering graph views, masks, ensembles or selective gradient routing.
