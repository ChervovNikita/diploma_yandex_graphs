# Heterogeneous private modulation: source qualification

## Decision

**The fixed one-rank CP relation-output operation is technically implementable at exact HGB HGT message sites.** It is a source-qualified prospect, not an execution-ready recipe, adopted driver or demonstrated quality improvement.

The concrete route is `NC/benchmark/methods/HGT/train_hgt.py` → `model.py` at HGB commit `ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe`. A complete HGB-DBLP comparison is the smallest representative first stage: retain the existing nine proposed arms and five proposed paired seeds, yielding 45 configurations. This stages the dataset scope; it does not change the parent's arm/seed/threshold freeze. The saved two-graph gate still requires ACM.

Literal ACM HGT is unsuitable for the full relation-preservation requirement: its graph dictionary names relations only by endpoint types, so citation and reference relations with the same paper→paper type pair overwrite one another. A declared repair must retain raw relation IDs in unique canonical relation names, identically in every HGT arm. This source bug does not reject the heterogeneous-quality hypothesis.

No model was imported or run. No dataset, split, checkpoint, logits, native results or logs were opened. No GPU, SSH or remote compute was used; no training driver was written. Only this packet was changed.

## Exact insertion and retained native semantics

At `model.py:69`, multiply each member's private source hidden channels by its layer input vector `a_m` **before** the biased source-type `v_linear`. At `model.py:50`, apply `b_m + c_m q_r u` **after** the per-head `relation_msg[r]` transform and before attention weighting. Reshape the output factor to `(1, H, d/H)` in the native flattened head-major channel order. Shared `a_m,b_m` cover all type/relation cores within a layer; `c_m` is member-specific, `q_r` relation-specific and `u` shared output-channel direction. Every layer has its own factors.

The source-bound affine operation is

`[ (h_m ⊙ a_m) V_type + native_bias_type ] blockdiag(R_relation) ⊙ (b_m + c_m q_relation u)`.

Thus input scaling leaves the native V bias unchanged, while output scaling acts on its transformed contribution too. This preserves a shared native bias and binds the affine extension of the saved linear formula. Moving the output diagonal before the relation transform would generally change the operation.

Query/key projections at lines68/70 remain unmodulated shared native maps evaluated on each member's private hidden state. Attention can differ indirectly through recurrence. Native HGB HGT performs neighbor softmax within each relation (`57`), weighted value sums (`58`), cross-relation mean (`73–74`), biased typed output, sigmoid skip mixing, optional LayerNorm and then dropout (`78–83`). Input adapters use tanh; the classifier is biased (`98–110`). Preserve every path and four complete recurrent trajectories, with isolated member scratch frames and dropout RNG. The topology/features may be immutable shared inputs; hidden states, attention and autograd work may not be collapsed or detached.

Same-type citations/references are eligible canonical relations. “Non-self” excludes synthetic self-loop relation types, not all same-type relations. Native HGT adds no synthetic self relation. Released edge-level diagonals and exact raw IDs remain archive-verification items.

Two tempting paths are incorrect: `HGT/run_hgt.py` builds `myGAT`, not HGT; original MIT pyHGT uses global target attention softmax, GELU and a different dropout/norm order. Replacing HGB HGT by pyHGT is a port with different semantics, not the matched native implementation.

The exact map and unchanged source excerpts are in [SOURCE_SITE_MAP.json](SOURCE_SITE_MAP.json), [HGT message sites](excerpts/HGT_message_sites.txt) and [graph/recipe hazards](excerpts/HGT_graph_recipe_hazards.txt).

## Released native recipes

| Baseline | Bound HGB branch | DBLP / ACM feature mode | Source-specific recipe |
|---|---|---|---|
| HGT | `HGT/train_hgt.py` + `model.py` | DBLP target attributes + other types identity; ACM all given features, identity where absent | total hidden64 /8heads; DBLP3 layers + LayerNorm; ACM2 layers, literal shell leaves LayerNorm **off**; actual layer dropout0.2; AdamW/OneCycle300, validation CE checkpoint |
| GAT | `GNN/run.py --model-type gat` + `GNN.py:GAT` | DBLP all identity; ACM target attributes + other identity | hidden64 **per head**, heads[8,8,1], two hidden stages + separately appended class stage =3 convolutions; biased typed input projections, native DGL message projection biasFalse; ELU/slope0.05/dropout0.5; residualFalse; Adam lr5e-4/wd1e-4 |
| Simple-HGN | `baseline/run_new.py` + `GNN.py:myGAT`, `conv.py` | DBLP all identity; ACM target attributes + other identity | same hidden/head/stage convention as GAT; edge64; residual attention alpha0.05 with detached preceding attention; residual feature maps; native final class-logit L2 normalization; Adam lr5e-4/wd1e-4 |
| SeHGNN | `hgb/main.py`, `utils.py`, `model.py` at `e92bd37d0b803457339555684f139b4c8f3e160d` | all provided attributes, identity where absent; native feature and TRAIN-label propagation | hidden/embed512; feature projection2; DBLP feature hops2/task depth3/residual; ACM feature hops4/task depth1; label hops4;200 epochs; Adam lr1e-3/wd0; batch10000/patience50; native AMP |

HGT's CLI `--lr` and `--dropout` are not used as their names suggest: the optimizer leaves LR at its constructor default, OneCycle max LR is1e-3, and the HGTLayer constructor retains dropout0.2. The paper's saved ACM normalization prescription and literal shell differ. The parent must bind an explicitly labeled choice before outcomes, across every matched HGT arm. No numerical author-result reproduction is claimed.

GAT/Simple-HGN `num_layers=2` means two hidden stages plus one output stage, so the paper's three-convolution depth is consistent. Their concatenated hidden width is512, unlike HGT's total64. Native baselines are competence comparisons with disclosed feature/architecture/budget differences; only the HGT arms isolate the factors.

GAT uses an undirected union plus self loops and erases relation labels. Simple-HGN retains ordered-edge relation IDs plus self/reverse categories, but its `(u,v)` dictionary can collapse multiple labels on the same ordered edge; actual release multiplicity must be audited later. Its CUDA-specific epsilon and GAT's missing prediction-file name argument are portability/output-plumbing issues, not qualifications already tested.

SeHGNN's DBLP/ACM feature propagation is computed on CPU before training; dense propagated features still encounter learned per-metapath embeddings/maps in forward. Its label input is seeded only from TRAIN, with the diagonal of each complete metapath product removed before propagation. ACM keeps both PP and PP_r information under a merged/coalesced P-P relation with added diagonal; it omits field/K nodes by author prescription because raw attributes encode field distribution. This is an explicit native preprocessing exception to common graph encoding, not an analyst-selected candidate subgraph. Preserve and disclose it. The final class BatchNorm uses evaluation-batch statistics, so freeze inference batch composition before outcomes. No OGB ComplEx recipe is transferred.

Details, native commands and precise limits: [BASELINE_RECIPES.json](BASELINE_RECIPES.json).

## Release access, split and heldout protocol

The current HGB root README announces a public release including test labels on2023-03-02. The linked Google Drive NC folder returned HTTP200 HTML and lists `DBLP.zip` and `ACM.zip`. **The archive bytes, exact schema, node/edge counts, split hashes, full-test labels and dataset license were not verified.** Old Tsinghua automatic-download URLs returned HTML under HEAD, which does not establish archive access. Later README text still describes half test labels and `xxx_full` renaming; do not blindly apply that stale instruction to an uninspected current archive.

The inspected loaders require `node.dat`, `link.dat`, `label.dat`, `label.dat.test`. Node types have contiguous ID ranges. Raw relation IDs and weights appear in `link.dat`; target TRAIN+VAL membership comes from `label.dat`, with20% of that pool shuffled into validation. Test membership is the released test mask. The saved paper percentages24/6/70 are conditional on the original30/70 labeled pool and integer rounding, not newly observed counts. Bind one sorted TRAIN/VAL split per master seed with a private NumPy RNG and inject identical masks in every arm; native HGB scripts leave the shuffle seed unbound.

Literal HGT copies test labels and uses `labels.max()+1` for class sizing; use TRAIN class schema instead. HGT and SeHGNN compute per-epoch test metrics, and SeHGNN preprocessing `check_acc` computes test metrics even if only its display is disabled. A future wrapper must separate test labels from membership and remove all such accesses, retaining native unlabeled transductive features, TRAIN-only propagated label caches and fixed SeHGNN inference batches. Test labels open once after source/recipe/split/use-history/selection freeze.

Evidence and pending release checks: [DATASET_RELEASE_QUALIFICATION.json](DATASET_RELEASE_QUALIFICATION.json), [DATA_ACCESS_PROBE.json](DATA_ACCESS_PROBE.json).

## Smallest representative first-stage proposal

Retain all nine prior proposed arms on complete HGB-DBLP: native HGT, proper native GAT, Simple-HGN, full native SeHGNN, global message BE, the fixed CP operation, unrestricted canonical-relation output factors, four untied HGT members, and wider global message BE. Retain proposed seeds131/137/139/149/151, pending parent use-history qualification. This is45 configurations,60 downstream pipeline optimizer jobs and120 complete encoder-trajectory equivalents. These are bookkeeping counts, not equal-cost fits or a compute forecast. ACM doubles those counts and remains necessary for the saved both-graph practical gate.

Global BE is the direct mechanism control. The unrestricted output table starts from CP-generated values to match its initial function. Untied members test full parameter independence. Wider BE checks added parameter capacity; choose the smallest width divisible by8 with complete parameter count≥CP before outcomes. Under the bound DBLP source schema,72 is the next width and analytically sufficient, with a substantial budget overshoot to report. Do not call it exact parameter equality.

Keep ordinary uniform mean own-member TRAIN CE and fixed mean-logit serving. Use the saved validation-only calibration and NLL/F1 criteria only if admitted by the parent; five seeds are not a significance/power claim. A DBLP first stage cannot pass the saved both-graph gate. Native author inputs, label preprocessing, output normalization and preparation costs remain intact. No new arm/seed/threshold freeze or tuning decision is made here.

[MINIMUM_COMPLETE_RELEASE_COMPARISON.json](MINIMUM_COMPLETE_RELEASE_COMPARISON.json) binds the proposal and its limits. [RESOURCE_ACCOUNTING.json](RESOURCE_ACCOUNTING.json) derives source parameter formulas without model execution. For source-schema DBLP (T4,R6,H8,L3,d64), CP adds222 parameters to global BE across the three layers, yet retains all four private graph paths; no speed claim follows.

## Initialization, attribution and license limits

Preserve the saved nonzero generic initialization: `c=0.01*(-3,-1,1,3)/sqrt(5)`, balanced relation signs and output-channel signs. It perturbs the initial function. Zero-factor symmetry/dead-gradient cases remain; nonzero product Jacobians do not guarantee useful loss gradients, and antithetic contributions may cancel in a linear common-response pool. CP/unrestricted initialization matches initial function but not optimizer coordinates or decay. Base BE/dropout diversity is separate.

The same-site diagonal activation-adapter construction is exactly the same affine operator. The rank≤2 versus≤1 witness concerns fixed effective coefficient interactions across member×relation, not complete HGT expressive power; native attention/nonlinearities/learned relation maps can already induce interactions. Conditional tensor adaptation, diagonal adapters, BE and typed sharing remain attributed prior. This packet adds source feasibility, not a new novelty theorem or expected predictive gain.

No root or applicable NC HGT/baseline/GNN license was found in HGB's complete pinned tree; unrelated subtree licenses cannot be transferred. SeHGNN has no located license/copying file or README grant. Original pyHGT is MIT; native DGL0.4.3 is Apache2.0. Those grants do not automatically license HGB additions, SeHGNN or datasets. HGT-DGL origin license metadata returned404. Future code reuse/distribution must bind an authorized route; technical qualification itself is complete without assuming such permission. See [LICENSE_SCOPE.json](LICENSE_SCOPE.json).

## Audit boundary

40 small pinned source/recipe/license/dependency units were retrieved and hash-bound;38 were semantically inspected in recorded scopes, including25 author program-source units and one native DGL source unit. Two retained bodies were retrieved only, without semantic use. Four previously retained code/README scopes were deliberately reread under this authorized source task. There were zero new or retained primary-paper reads in this packet, zero published numerical-result reads and zero model/data execution or adoption.

[READ_SCOPES.json](READ_SCOPES.json) distinguishes retrieval from semantic read scopes; [SOURCE_RETRIEVAL.json](SOURCE_RETRIEVAL.json) contains raw URLs, commits, Git blob SHA1 and SHA256. The stdlib read-only verifier checks payload/input integrity, read accounting, pinned blob bindings, source anchors and analytic count arithmetic. PASS certifies those checks, not model/runtime equivalence, scientific utility, author-number reproduction, licenses or complete release bytes.
