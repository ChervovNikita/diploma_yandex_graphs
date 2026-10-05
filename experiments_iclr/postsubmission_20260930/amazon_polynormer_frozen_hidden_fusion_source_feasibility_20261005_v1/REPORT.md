# Amazon/Polynormer frozen hidden-fusion source feasibility

5 October 2026. Source/release metadata only. No remote contact, allocation access, outcome-root traversal/stat, checkpoint/logit/label/data payload read, inspected-source import, model forward, numerical replay, metrics or training occurred. Source-only notes are saved in this folder; original source, releases, registry, selectors, scientific arms and label roles are unchanged. The parent’s 14/15 running report is context only, not independently checked or a closure claim.

## Decision

**Source-feasible, but hidden export is required.** The implemented final fit path retains every raw member logit. It does not retain the private penultimate hidden states. Those states can be captured during one complete selected-checkpoint eval replay per unique physical fit; no new base fit is required. Across the complete three-block paired family, that is 15 model calls comprising 24 complete member trajectories. Wall time and resource admission for a new export were not measured or granted here.

Both GNNM4 and the eligible independent4 bank have exact 512-dimensional classifier-input interfaces. Their states are not an already-saved cheap cache. Full FP32 hidden export over all nodes and both banks would have 1,203,830,784 raw tensor bytes (about 1.20 GB decimal) plus metadata/serialization overhead. All original M4 member work remains charged.

## Actual source and cohort binding

The inspected source is `amazon_polynormer_paired_family_source_preparation_20261003_v6`, manifest SHA256 `d4160d8aeec2b770274875f9a1fb137facc9f93efd4eca6ad91e58802521ff44`, seal `b70beb8868d09c7f066c82cd9c0d1c582229cdfe0124485b32e12d7a7069f373`. Fourteen inspected source/contract files match their frozen manifest rows. All 15 explicit registry fit-release metadata files bind that V6 source, the same consumer release and exact original row/output references, with TEST authorization false.

V6 inherits the registered V5 registry; the registry’s V5 source field is therefore expected. `COHORT_SOURCE_BINDING.json` explicitly preserves its 15 output/release/claim paths and forbids replacement registration. `study.py:11–18,33–45` binds that original cohort; the current fit releases identify V6 execution. No fit-output directory or body was opened, statted or enumerated. Source-promised retention is distinguished from verified physical availability.

## Saved logits and accessible hidden states

| Interface | Exact source locator and meaning |
|---|---|
| GNNM logits | `native_training.py:151–159` enforces FP32 [4,24492,5]. `snapshot:307–310` stores complete `selected_raw_logits` in a selected full-state image. `study.py:171–179,185–194` restores final selected state and writes `SELECTED_LOGITS.pt` with raw logits, selected checkpoint and bindings. |
| Independent logits | Each native call is wrapped as FP32 [1,24492,5] by `native_training.py:153–157`. `evaluate.py:129–154` restores each own checkpoint and concatenates by registry `physical_fit_references`. |
| GNNM local penultimate | `reused_models/backbone_boundary_adapter.py:87–104`: per-member ten-layer local trajectory accumulates `x_local`; active `local_head(x_local,member)` receives[N,512]. |
| GNNM global penultimate | Adapter line 103: active `global_head(core.global_attn(core.ln(x_local)),member)` receives[N,512]. This is after the selected global attention body. |
| Independent local/global penultimate | `reused_models/native_polynormer.py:148–177`, especially172–175: `pred_local(x_local)` or `pred_global(x_global)` receives[N,512]. |
| Member order | Adapter 106–109 stacks complete calls for member 0,1,2,3. Independent order is the exact registry reference list, checked/assembled in `evaluate.py:152–154`. `worker.py:27–36` defines single_author as native member 0 alias. |
| Portable restore | `native_training.py:57–77,217–258` constructs exact source and restores strict model/Adam/modes/RNG/permissions/gradients plus explicit stored `_global`. Selected checkpoint may be local; do not force global. |

The minimal export defines **raw penultimate h as the actual active head argument before boundary R**. BoundaryProjector then computes `linear(h*R_m,W)*S_m+B_m` (adapter 15–37). Thus h and the effective affine-weight input h*R_m are distinct named interfaces. Exporting raw h matches the usual preclassifier feature-fusion interface; this report does not add experiments over both variants. Native classifier input is h directly.

No hidden-return flag exists in these forward functions. Their temporary states are local variables and the model returns logits. The scientific snapshot path saves model/buffer/gradient/Adam/RNG/mode/stage state and raw logits, with no penultimate activation key (`native_training.py:217–224,307–310`). Checkpoint parameters/gradients do not substitute for per-node hidden states. Under bounded retention, final and selected-local images are protected (`retention.py:7–23,57–65`; `RETENTION_POLICY.json`); superseded epoch images may be retired and are not candidate feature sources.

GNNM has one selected stage/checkpoint for all four members in a block. The independent bank uses four independently selected checkpoints, and their local/global stage flags can differ. Both head interfaces have width 512, but source does not establish shared feature semantics across independently learned coordinates. Concatenation is supported; elementwise hidden means/contrasts would need an additional alignment assumption.

## Node population, order and graph context

This is **node classification**, without link-query negatives or a separate candidate-list generator. Full output rows preserve native feature-row/node indices 0…24491; the source never relabels nodes. Class axis is integer 0…4. `common.py:371–389` reads raw FP32 [24492,300] features and native masks, obtains role IDs from mask.nonzero, and checks fixed cardinalities. Class-stratified FIT/control assignment is hash-ranked within each class, while emitted role order retains the original official TRAIN order (`common.py:318–336`). Compact labels are required to match those IDs exactly (`common.py:416–424`).

The consumer binds preprocessing to raw unchanged features and `to_undirected → remove_self_loops → add_self_loops(num_nodes=24492)` (`common.py:404–412`), with edge shape [2,210592] and logical hash `229a8a787ef9120a4d7a1dcdd6481b973619a1c7f44a4abe9ed69d05c256e550`. No dataset/graph payload was read here. Labels-free graph summaries can later be derived from this same public topology and retained probabilities under a separately declared operation; no new context operator was constructed here.

A selected-global forward includes full-node `k*v` and `k_sum` reductions (`native_polynormer.py:54–83`). A selected-local forward still executes all 10 complete local GAT layers. Replaying only control nodes, masking out TEST-feature rows or running an induced subgraph would change the checkpoint’s function. Capture all-node states during the complete forward, then select rows by the fixed role IDs.

## Frozen selectors and label roles

Source blocks are(split 0,seed 17),(split 1,seed 29),(split 2,seed 43). Independent member seeds are respectively17/1026/2035/3044,29/1038/2047/3056,43/1052/2061/3070. The single_author view aliases exact independent member 0, including features, checkpoint and costs.

- **FIT:** per-class floor(4*n/5) of official TRAIN; counts 9795/9796/9795. Only FIT labels enter loss (`native_training.py:162–168`).
- **TRAIN-control:** remaining official TRAIN; counts 2451/2450/2451. Its labels are decoded for prefit class-stratified role derivation, then excluded from model fitting and VAL selection. Predictive control scoring needs separate post-closure authority (`evaluate.py:115–121`; `common.py:400–403`). It is the frozen evaluation endpoint, not a free fusion-training set.
- **VAL:**6123 nodes per block. Every actual update has a strict accuracy correct-count selector. Earliest ties remain; one persistent best spans200 local and2500 global updates. The local-to-global transition restores selected-local model/Adam; final restore may select local or global (`study.py:111–179`; selector arithmetic `native_training.py:196–202`). VAL is already a base-selection sample, not an untouched meta holdout.
- **TEST:** public membership/features participate in transductive inference. No compact TEST-label artifact or reader exists in the current consumer, and authorization is false (`common.py:340–376`; all 15release metadata). Historical producer raw all-node label decoding is disclosed in DESIGN; this report does not deny that history. No TEST scoring is authorized.

The GNNM selector uses argmax of mean member probabilities; each independent fit uses its own native predictor. Their checkpoints are not interchangeable with an ensemble-selected independent epoch. The entire family must close before evaluation (`evaluate.py:54–80`, worker152–154 and supervisor108–110). No fusion/readout training role or new arm is allocated in this frozen study; this report allocates none.

**Correction to the generic literature shortlist:** this actual family serves arithmetic mean softmax probabilities, with stable FP64 log-mixture NLL and FP32 probability-pool selection/accuracy (`evaluate.py:83–100`). A zero-residual head anchored at mean raw logits would change the native predictor. The feasibility report preserves the actual pooling and does not amend the earlier candidate or current study.

## One minimal replay/export design

After existing complete closure and custody gates, a separately reviewed **labels-free, inference-only sidecar** can restore fresh exact copies of the 15 final selected checkpoints. Load only the bound public feature/edge projection and reproduce its preprocessing. The current `common.load_data` is not a labels-free reader because it decodes TRAIN and VAL packs.

Attach one passive forward pre-hook to the selected active head. For GNNM it receives (h,member); for native it receives (h,). Return None and capture raw [N,512] FP32 head arguments in fixed member order. Run one complete `eval/no_grad` forward per physical fit; stream detached CPU copies outside all sealed source/scientific outcome roots. Require raw output parity against both selected-image and final saved logits at the existing fixed rtol=1e-5/atol=1e-6. Record source/runtime/checkpoint/stage/preprocessing hashes, node/member/class axes, exact raw-before-R interface and export hash/size. A mismatch stops export; no reselection or tolerance change.

Do not call `train_step`, `evaluate.run` or `portable_replay` as the exporter. The latter performs real scratch FIT updates and the evaluator additionally opens label packs/computes metrics; the export needs neither. Existing qualified portable replay remains an upstream custody condition. Assemble each independent bank by registry references and let single_author alias member 0.

Source-dimensional raw H cost is 50,159,616 bytes per member and 200,638,464 bytes per four-member bank. Three paired banks total 1,203,830,784 bytes; all unique raw logits occupy 11,756,160 tensor bytes and can be referenced rather than duplicated. Add model/Adam restore memory, temporary full-graph activations, CPU copies, I/O, headers and any future graph/head work. A passive export during an already mandatory authorized replay could amortize the neural forward only through a separately reviewed addition; no such source change was made or assumed here.

## Scope and integrity

`SOURCE_BINDINGS.json`pins the 14 inspected frozen source/contract files and explicit release metadata. `READ_SCOPES.json`records exact passages and prohibited surfaces left untouched; `RELEASE_METADATA_COVERAGE.json`checks all 15 source/registry/consumer references without following their output paths. `FINDINGS.json`and `MINIMAL_EXPORT_PLAN.json`hold the decision/interfaces and prose-only plan. No numerical implementation, new arm, training permission, selector change, closure certification, novelty or gain claim is produced. Manifest/seal certify these report bytes only.
