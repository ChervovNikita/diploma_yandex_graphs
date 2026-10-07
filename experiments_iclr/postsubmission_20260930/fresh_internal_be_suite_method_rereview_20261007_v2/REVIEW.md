# Technical successor method rereview — v2

Reviewed 7 October 2026. Exact v2 MANIFEST SHA256: `fab86d30e67d8773031adc8dcfc636a516eba656a60f9794d033029812cd7af4`. Predecessor MANIFEST: `3c0c88031dfc30178f0942794e64f4ae3677d3a2def1f2035adaa9e745474211`.

This is a **technical successor rereview using prior source context**, not a fresh academic-paper review, independent literature search, novelty clearance or manuscript verdict. The v1 review and both source packets remain unchanged. The earlier detailed method reading and claim limits remain applicable where source bytes are identical.

**Disposition:** the executable method changes repair M1–M3. No additional method implementation defect was identified in the inspected successor changes. Do not treat this as scientific execution approval for exact v2 while its two incompatible closure/score-opening instructions remain unresolved. Make one prospectively adopted closure policy explicit in a separately bound adoption addendum or sealed successor, and obtain the separate role/runtime/export/resource approvals before release. Real learning, official data export and representative resource use remain unverified.

## Evidence and boundaries

Independently verified all 27 MANIFEST file hashes and sizes, verified the exact requested manifest digest, and parsed all 12 sealed Python files with the standard-library AST parser without importing them. Compared v2 against v1, including the complete run/objective/factor/config/CPU-test changes and new selection helper. Read the updated protocol, repair scope, static/CPU receipts and relevant data/runtime changes. `models.py`, native dependency bindings, ATTRIBUTION and RUNTIME_PIN are byte-identical to v1. Factor arithmetic/installation is unchanged; its parameter-accounting field name changes. All three configs retain identical model, loss, optimizer, initialization, horizon, seed, arm and pool settings; their additions are tranche order and `automatic_next_tranche:false`.

The native dependency source hashes are unchanged and match their named retained files. This rereview reuses the prior inspectable native architecture reading. Named installed remote PyG/OGB providers remain unavailable locally; no remote verification was attempted. No model/framework imports, CPU model execution, dataset/checkpoint/logit reads, GPU work, remote access, exporter execution or source edits occurred. The author's 27-file source closure and 30-row CPU receipt are evidence about source preparation, not proof of competent real learning, actual role custody or production feasibility.

## M1: joint untied transition — repaired

The runner now computes `ordinary_independent = job['arm']=='independent4'` (run.py:48) and passes that exact flag to `local_transition` (run.py:113–114). The helper's own-checkpoint branch validates the ordinary four-member arm and restores each body's own local model/Adam state (selection.py:11–17). Its other branch loads one `selected_local.pt`, checks the optimizer bank length, restores the complete ensemble state and every corresponding Adam state, then enables global mode (selection.py:18–25).

Consequently, `independent4_contrastive` restores the whole pooled joint trajectory; it no longer assembles coupled models from independently selected epochs. The pooled checkpoint contains the complete model and optimizer list (run.py:103–107) and is selected with the pooled metric during the local phase (run.py:142–143). The helper does not restore saved stream/outer RNG state or draw new randomness, preserving the prescribed live end-local forward streams and RNG instead of rewinding them. Single arms now use the same pooled branch; with one member that is the intended one-body selector.

Ordinary independent4 still restores own local candidates and serves its own-selected overall bank with recorded per-body local/global modes (run.py:158–168). The runner continues to write own checkpoints for all `model.independent` arms, including the packed untied control, but those files do not choose or replace the packed control's transition or final pooled bank. This extra bookkeeping does not recreate the v1 defect.

The added CPU fixture calls the same helper for ordinary and packed untied controls, records which files it loads, compares restored full model/optimizer state for the packed case and checks live CPU RNG preservation (check_cpu.py:129–165). Its source supports the engineering receipt. It remains a tiny CPU fixture; the main transition conclusion comes from actual runner/helper bindings, not from assuming that a “passed” field proves the scientific trajectory on CUDA.

## M2: repeated/reversed edge contrastive targets — repaired

The contrastive runner takes `batch['query'][index]` using the exact index already applied to representations and labels (run.py:123–130). Collab `alignment_loss` now requires those TRAIN target identities, sorts each pair's endpoints, and makes all rows with the same canonical unordered pair positive across views (objectives.py:38–44). Distinct targets remain negatives. It also rejects the same canonical pair carrying contradictory positive/negative labels.

This directly repairs the false-negative mechanism identified in v1: repeated or reversed records of the same target are no longer another instance's negatives. Canonicalization concerns loss pairing only; data.py:195–207 still retains every TRAIN positive record, complete tails and matching negative rows for own supervised loss, and support removal still excludes every duplicate/reversed positive target pair (data.py:157–168). There is no TEST-based future-positive filter or target identity input to any loss from VALID.

The public objective helper also accepts and forwards identities (objectives.py:84–87); the CPU collab fixture now contains a reversed target pair and supplies query identities. Its added small test checks finite canonical-pair alignment and contradictory-label rejection (check_cpu.py:172–181). That test does not numerically compare the whole positive mask against a reference objective, but the explicit pair-equality source is sufficient to identify the fix; another tiny numerical parity gate is unnecessary for this review.

Repeated records continue to contribute repeated supervision, alignment anchors and class-center weight. That is the declared complete record-population recipe, not an equal-unique-edge weighting scheme. No real projection was opened to measure duplicate frequency, and no actual performance claim follows from repairing the pair semantics.

## M3: nonlinear GCN description — repaired

The task table now accurately describes the one-layer `GCNConv -> LayerNorm -> dropout(.1) -> ReLU` encoder and retains the correct exclusion of learned edge attention (PROTOCOL.md:12). The native architecture and model settings did not change. Its fixed degree-normalized aggregation may respond to learned channel factors and stochastic encoder-edge dropout, but is not member-specific learned graph attention or learned topology.

Active xdropout `.25`, adjacency dropout `.25`, post-convolution dropout `.1` and the absence of a residual addition for 128-to-64 dimensions remain evident in the unchanged model/native source. The `.05` taildropout argument remains unused in this chosen non-pure native constructor path; it is not claimed as an active mechanism in the corrected table. Parameter reports now name non-factor parameters `non_factor_parameters` instead of misleadingly calling the four untied bodies “shared” (factors.py:80–86).

## Other inspected method changes

The runner now checks both stochastic TRAIN member logits/pools and representations before loss, every post-step model parameter/Adam tensor after optimization, and VALID member logits/pools before metrics (run.py:77–79,119–121,133–137; selection.py:28–41). This closes the possibility that a finite argmax/rank metric silently selects a nonfinite predictor. It does not change the finite model's gradients or any declared selector comparison.

Startup preparation is inside the new failure-capture structure, and runtime/export/resource supervision changes are substantial additional source changes. The separate reviewer owns their provenance and admission verdict; this method review does not certify those subsystems. No new private projection head, architecture, teacher bank, normalization state, supervised-target mask, pooling fit, contrastive coefficient, temperature, seed, horizon or parameter-capacity intervention was introduced.

## M4: incompatible closure instructions — resolve before execution adoption

PROTOCOL.md:41 still requires **all eight arms and all three tasks** to complete or preserve failure status before opening predictive comparisons and says “No score-driven stopping.” PROTOCOL.md:75 instead requires root to review complete first-family WikiCS paired effects before adopting later task families, and permits stopping later-family execution based on null/failing/weak first-family evidence. Both instructions cannot govern the same score-opening decision as written.

The appended staged intent is clear and can be a legitimate prospective exploratory deployment plan. It preserves the complete 24-cell matrix for each released family, original full public/supervised populations and full horizons. The configs add the fixed `wikics -> collab -> molhiv` order and disable automatic next-tranche release. No within-family score-driven arm/seed pruning, dataset shrinkage or recipe replacement is introduced. The code remains a separately authorized one-cell runner, not an automatic score-driven task-selection engine.

Before execution adoption, explicitly state which closure rule governs. For the stated staged plan, a consistent rule would close every released task family's complete eight-arm/three-seed matrix before opening that family's comparisons; later families receive a separate adoption decision; skipped cells and reasons are recorded; no within-released-family score-driven shortening/replacement is permitted. Global claims must refer to the families actually completed under their prospective definitions. A separately declared later question after a null first family is exploratory and cannot retrospectively rescue the original fixed recipe or convert chosen task families into an unconditional three-task confirmation.

This is a protocol/adoption conflict, not a remaining BatchEnsemble/contrastive arithmetic defect. Its resolution requires no model or outcome-driven tuning. The exact conflicting source cannot be silently treated as both an unconditional 72-cell closed suite and a success-contingent family expansion. A hash-bound adoption addendum explicitly superseding the stale sentence would make the interpretation reviewable without modifying v2; a sealed successor may instead reconcile the prose.

## Causal, comparative and novelty limits that remain

The primary `be_init_contrastive - be_init` contrast remains matched on capacity, initialization, member count, own supervision, dropout streams and pooled selection. It measures the combined alignment/residual recipe. The secondary initialization and unit-start auxiliary contrasts remain as originally defined. Single/untied contrastive controls can show effects available to ordinary bodies, and ordinary independent4 remains an independently learned bank under a shared target order.

The eight arms still do not isolate alignment from residual repulsion, individual factor placements, the molecular pre-message vector, or a unique advantage from sharing. Single/independent/BE controls change capacity, gradient sharing and total stochastic computation; identical per-member supervision is not equal total compute. No capable single control matched for both capacity and compute has been added. Previously derived static capacities remain unchanged: Polynormer base 7,537,172 + 122,112 BE factors; GCN+NCN base 33,922 + 4,100 BE factors; molecular BE factors 63,492, with common categorical embedding capacity not locally independently certified here.

All genuine architectural properties supported in v1 remain: shared trainable W/b/attention/normalization/embeddings/virtual-node parameters; private learned internal rank-one dense factors; complete per-member nonlinear paths; molecular categorical encoders before modulation; every member's full own supervised loss; and independent persistent member stochastic views. The GINE + virtual-node LayerNorm adaptation remains a baseline adaptation, not reproduced author benchmark scores.

Class centering does not guarantee preserved predictive label information after network parameter updates. Same-label chemistry remains a coarse grouping, and representation repulsion may exploit irrelevant/gauge directions without useful evidence or prediction diversity. The residual proxy remains neither DICE conditional mutual information nor a verified author CDLG loss/independence theorem. BatchEnsemble/TabM and the closest prior are explicitly attributed; v2 invents no new novelty conclusion.

Three paired seeds per family are weak development evidence. A first-family gain does not establish broad graph validity, cost efficiency or unused-population confirmation. A fixed task-safe evidence-response panel, member competence/common-error analysis, matched capable controls, adequate cost measurements and separately reviewed one-shot unused TEST confirmation remain required for the corresponding stronger claims. Source or synthetic success cannot replace any of those results.

## Execution-source recommendation

M1–M3 are closed by actual code/protocol changes. The method implementation is suitable to proceed to the separate real-data/resource qualification stage once M4's exact prospective closure policy is explicitly adopted and source/role reviews permit that stage. **This receipt has `approved:false` for automatic scientific fit admission of unqualified exact v2**, while recording `method_implementation_repairs_verified:true`; it does not authorize exporter execution, training, predictive score opening, later tranches or TEST access.

After the closure policy is made consistent and independent role/export/runtime/resource gates pass for the exact source/arm/config/data/device identity, no additional method implementation blocker identified by this rereview needs to prevent the fixed full first-family pilot. Retain all failures, selected horizons and skipped families, and limit interpretation to the completed prospective questions. Neither the full 27-file closure nor the author's 30 CPU records substitutes for that release evidence.

Bindings, predecessor-to-successor changes, dispositions and this scope are recorded in `SOURCE_BINDINGS.json` and `REVIEW.json` beside this report.
