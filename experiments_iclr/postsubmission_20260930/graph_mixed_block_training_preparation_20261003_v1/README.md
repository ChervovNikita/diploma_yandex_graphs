# Full-training mixed shared/private HGT objectives — source preparation

Status: prepared source only; scientific design adopted separately by root. Training and qualification releases remain false. No native imports, dataset/label/fitted-state access, training, qualification execution, or canonical-record edits were performed in this preparation. Static compilation and the one-call source projection are the only executed checks.

## Bound experiment

- Complete released HGB DBLP and ACM; all five existing paired seed/split blocks: 131, 137, 139, 149, 151.
- Existing immutable native-compatible global_BE HGT, four members. Core matrices, biases, norms, skip and classifier are shared. Every BE layer's member `a` and `b` rows are private. No CP model, router, repulsion, coefficient, or inference change.
- Four full-training assignments, written shared/private: own/own, pool/pool, pool/own, own/pool. Forty ordered terminals are required before comparison.
- Original uniform mean-member TRAIN CE retains its `1/M` normalization. Served inference and checkpoint selection use raw FP32 mean logits.
- Default `fresh_all40` trains all forty models. `reuse_all10` trains thirty only after all ten own/own controls pass a root-observed, outcome-independent exact compatibility audit following complete original-family closure. Any absent/ineligible control requires a new `fresh_all40` freeze/release. There is no partial, favorable, or successful-subset reuse.
- New own/own controls are independent of the older CP continuation decision. A never-launched ACM CP-gated study does not block this independently adopted study.

The exact split previously appeared as a 32-step comparator. Full training tests its utility and duration; it establishes no new gradient principle or verified quality gain. The field is generally nonconservative, with no global descent guarantee. The bounded closest-prior literature disposition is retained in `graph_mixed_block_objectives_scout_20261003_v1`.

Root's bound scientific authority is `graph_mixed_block_objectives_execution_root_v1/FROZEN_SCIENTIFIC_DESIGN.json`, SHA256 `739160acd4ffb0380348f79496e88624db6163f4a3a93902b043d58eaacc18c5`. It specifies mean NLL improvement of 0.005 nat, four of five strict wins, and macro-F1 nondecline per graph against own/own and pool/pool. Reverse own/pool is required for interpreting the proposed role assignment. Source emits complete paired descriptives; root applies the prospective criterion. No additional scientific gate is introduced here.

## Minimal implementation

`block_policy.partition` establishes parameter ownership by exact immutable module classes and object identity. It requires a disjoint, exhaustive, alias-free union of `model.core.parameters()` and every be-mode factor's `a,b` tables. Names and shapes are recorded only after semantic ownership is established.

Own/own calls the imported immutable `train_dblp.fit` directly. For the other assignments, one AST expression is substituted: `train_loss.backward()` becomes `_mixed_backward(model, logits, ids, labels, train_loss)`. `FIT_BACKWARD_SUBSTITUTION.diff` is independently inspectable. Reverse projection must reproduce the entire original function AST. `FIT_BODY_CUSTODY.json` binds original source and AST hashes and the derived AST. The original files are never edited.

Pool/pool uses one pooled backward. Each mixed assignment collects the shared and private gradients from the same pre-update forward graph, then returns to the original one AdamW step and OneCycle step. No second forward, sequential block update, extra buffer update, or member/global RNG transition is introduced. `allow_unused=True` returns actual `None` for unused leaves; these remain `None`, preserving native AdamW skip/decay/moment behavior.

The original own CE is still computed and retained in the existing `train_mean_member_CE` trace field for every policy. That field is a descriptive diagnostic for pooled or mixed policies. Saved bindings identify the actual gradient assignment. The original native recipe, OneCycle state, latest-tie early stopper, selection, full checkpoint/RNG, and selected-state replay functions are reused. The selected checkpoint retains the original structural schema tag; outer dataset and binding records identify the graph and policy.

Selected served/member NLL, micro-F1, fixed-class macro-F1 and own-minus-pool D are computed from the immutable saved `selected_member_logits.pt`. They do not select another epoch, member checkpoint, calibration, or validation winner. Final heldout labels remain closed.

The served own-minus-pool gap uses the original FP32 member-CE and FP32 mean-logit arithmetic. It is left unclamped and carries no exact nonnegativity claim: tiny negative values can arise from rounding. A separate tagged FP64 algebraic Jensen reference converts each saved member logit before averaging. That reference is diagnostic and changes neither served inference nor selection.

The fixed final endpoint comes from the last byte-bound `TRACE.jsonl` row, with no new prediction or selection. Selected optimizer cost counts the logical bytes of unique tensors in the immutable selected checkpoint's optimizer field, excludes scalar/Python metadata, and is reported separately from serialized file bytes. It is not a physical-storage or peak-memory claim. Complete selected and fixed-final paired NLL/macro-F1 vectors against all three controls receive the existing native driver's descriptive SD, SE and illustrative t95 interval, plus all five leave-one-block-out means. These add no scientific acceptance threshold.

## One focused qualification packet — prepared, unexecuted

`qualify_policy.py` contains one focused qualification procedure:

1. A tiny actual-HGT synthetic same-state witness for all four assignments. An independent oracle uses full own/pool `backward` gradients and then chooses the semantic block slices. It compares the complete gradient field and one native AdamW/OneCycle step, buffers, independent member streams and global RNG. It exercises genuine unused shared leaves and counts exactly one production forward. Own/own and RNG/nonfloating state are bitwise checked; mixed floating values use Torch's standard `assert_close` tolerances, not a scientific acceptance threshold.
2. Complete real DBLP and ACM, prospectively fixed seed 131, all four assignments: six consecutive TRAIN updates with observed `/proc` memory before/after TRAIN and eval/checkpoint, followed by full model/optimizer/portable-OneCycle/member-RNG/global-RNG restoration and exact next-step replay. Eval only allocates logits; it does not score validation/test, early stop, or choose a model. Existing readers and family constructors are reused. Resource inability is a technical deferral, not scientific rejection.

The resource loop uses separate `logits` and `own_loss` assignments at the original fit's points, rather than retaining both old values through an entire next step in a tuple. Thus the original forward/loss variable lifetimes are represented before any resource decision. No probe has been executed or repeated in this source amendment.

The single-thread Linux CPU environment is exact baseline Torch `2.1.2+cu118`, Python `3.11.14`, FP32 default, CUDA hidden, and one Torch/inter-op/BLAS thread. Root installs the resource envelope before import and observes the qualification. No full old math suite is rerun. Native/resource qualification is **unverified** until this prepared procedure has an actual successful source-bound receipt.

## Root integration

`STUDY_FREEZE_TEMPLATE.json` is an executable-freeze template around the already adopted science and the two immutable graph authorities. The graph authorities retain their original recipes, archive/development descriptors, directed-relation order and all split descriptors. Their old training admission/CP gate is not imported.

`QUALIFICATION_RELEASE_TEMPLATE.json` and `TRAINING_RELEASE_TEMPLATE.json` are unadmitted templates. Root fills the exact executable-freeze and prepared-manifest hashes, run name and observed qualification/artifact records and makes the corresponding explicit release. The training driver verifies a complete source-bound eight-case qualification receipt before any new fit. Source integrity is rechecked after execution; comparisons require all forty selected, replayed, binding-verified terminals and preserved inputs.

If all-ten reuse is requested, the root audit schema is `HGB_mixed_all10_baseline_reuse_audit_v1`:

- `root_observed=true`, `successful_subset_reuse=false`, `favorable_outcome_filtering=false`.
- `families`, in DBLP/ACM order: dataset, original source-freeze hash, byte-bound `complete_family_receipt` with all35 ordered terminals, and `closed_family_audit` with root-observed all35 closure/audit/preservation and the same source hash. Complete closure may include failures; every selected global_BE baseline must independently qualify.
- `baselines`, in graph/seed order: dataset, seed, initial model fingerprint, original checkpoint bindings and records for selection receipt, checkpoint, saved logits, trace and runtime receipt. The baseline selection must match the original complete-family terminal. Runtime receipt identifies exact Torch/Python and CPU/inter-op threads. The `compatibility` object contains exactly `source`, `architecture`, `initial_state`, `graph_bytes`, `split_bytes`, `runtime`, `recipe`, `member_streams`, `loss_reduction`, `selector`, `checkpoint_replay`, all true after independent root audit.

The source also checks the new paired initial model fingerprint and the old checkpoint's parameter order, source graph/split/implementation bindings and selection. It never resumes a baseline as a challenger or chooses controls based on favorable outcomes. If audit prerequisites fail, no fits start under that release; root must admit a fresh-all40 freeze.

Prepared commands, after the appropriate source-bound root release and native environment exist:

```text
python -B verify_source.py
python -B qualify_policy.py --freeze <mixed executable freeze> --admission <qualification release> --run-name <fresh name>
python -B train_policies.py --freeze <mixed executable freeze> --admission <training release> --run-name <fresh name>
```

The latter two commands have not been run. No transport, scheduler, automated restart, deployment or study launch is created here; root owns execution and custody. Existing canonical status, ledger, design and source trees remain untouched.

## Pre-execution metadata correction

Root corrected only the scientific design's ACM class schema from four classes to the source-authoritative three. `REJECTED_PREEXECUTION_METADATA.json` preserves the prior packet metadata exactly and distinguishes the reported earlier seal discrepancy from the observed live seal. `PREEXECUTION_REBIND.json` binds the root correction proof and confirms unchanged Python source. The executable manifest was regenerated after every payload write; its seal binds the exact final manifest bytes. No model, loss, policy, seed, threshold, fixture, or numerical execution changed.
