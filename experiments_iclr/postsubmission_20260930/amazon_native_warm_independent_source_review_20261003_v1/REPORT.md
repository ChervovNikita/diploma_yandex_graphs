# Independent source review: bounded Amazon native warm study

Target seal: `53e6c4ba3de1df7ea44233068359a8a60ba9b6d0028361fa3a729a312201c434`.
Qualified v4 dependency seal: `cdece1f2a5e317de54d509d6b961ca41235cab880c3becb959b792aa75e03d72`.

One qualification claim mismatch and one narrower state-scope overclaim were found. No source defect demonstrating wrong fits, accidental TEST scoring, wrong fixed mask columns, recipe mismatch, or erroneous selection was identified. Actual numerical replay and full-data/GPU passage were not performed, so this is a source review and cannot certify those results or authorize launch.

## Findings

### AW-01 — Numerical equality is reported as bitwise qualification (P2)

`native_training.py:75–79`, `:211–219`, `:240` and `:247–251` use `torch.equal` for native token, full-logit and next-step state comparisons while emitting bitwise flags. `evaluate_study.py:106–113` does the same for saved VAL logits. `torch.equal` is exact numerical equality: positive and negative floating zero compare equal despite different bytes. The `exact` helper checks dtype and shape, but direct token/logit checks do not explicitly check dtype. Python scalar equality likewise does not distinguish signed floating zero.

This does not demonstrate a real disagreement between any live and restored trajectory. It does show that the required literal bitwise resource/replay guarantee is not enforced by the comparator. A root admission requiring that guarantee needs a correction or a reviewed narrowing of the claim before calling the passage bitwise. The sealed v4 `core/transaction.py:66–79` already shows the suitable tensor pattern: require tensor metadata, flatten contiguous values, view as `torch.uint8`, and compare bytes. Update the source/design/README/receipts together in an unsealed successor, or explicitly choose the weaker exact numerical-equality guarantee.

### AW-02 — Full Adam Python state wording exceeds the checkpoint projection (P3)

`native_training.py:172–177` saves `optimizer.state_dict()` and `:194–208` loads it. This captures Adam numerical state and parameter groups, not the optimizer's complete Python `__dict__`, defaults, hooks and internal flags. The portable/live comparison projects both optimizers through that same `state_dict`. The README's line27 “full Adam Python/tensor state” therefore exceeds the implemented serialization scope.

The driver constructs source Adam with default options, adds no hooks or scheduler, and does not mutate defaults. Consequently this is a nonblocking scope clarification for the current driver; no wrong next update was demonstrated. Describe the actual admitted default-Adam closure precisely, or inventory/save further Python fields if a reusable full-Python-state guarantee is intended. The joint live deepcopy and subsequent reconstructed next-step replay still provide strong numerical-state coverage within the stated default-Adam scope, subject to AW-01.

## Verified source and data behavior

All 107 expected descriptor checks matched, including the requested study and v4 seals, all16 study payloads, all37 v4 payloads and every `SOURCE_BINDINGS.json` descriptor. Both payload inventories are exact. Eight study Python files parse and compile under independent stdlib checks. All14 recipe fields match independently extracted author defaults and the unique Roman-empire mono shell command. Native `PolyFormer`, `PolyAttn`, `FFNNetwork`, `FFN` and `PolyFormerBlock` class ASTs match the pinned author files.

The author CLI omits Amazon, while its loader and `RunExp` explicitly support it. The packet correctly calls these prospective recipe transfers and discloses the TRAIN80/20 adaptation and Torch/PyG environment difference. `native_training.build` preserves the complete M1 native trajectory, freezes every adapter R/S at one, skips those factors in Adam, and preserves the author attention-versus-other rates and coupled L2 rule. Attention LayerNorm/bias-scale and FFN parameters remain trainable. Native initialization and factor allocation are explicit adaptations; no native hardware efficiency claim follows from their parity.

`prepare_data.py` directly invokes the pinned PyG process body with `pre_transform=None`: raw Amazon features, native undirected processing, and masks transposed to `[nodes,10]`. Its physical all-node label decode is disclosed. It emits exactly a label-free public graph and compact TRAIN/VAL channels for splits0/1/2. The readers reject label-bearing public arrays, call the exact author `heter_fixed_splits` AST, and require compact IDs in the same order as the selected official columns. The class-stratified role hash operates only on current official TRAIN labels, with floor(4*n_class/5) fit and the remainder control. Control is supervised TRAIN information. No forced minimum, rebalance or seed retry is present.

The trainer loads each official block's TRAIN/VAL channels, but fitting indexes only that case's fit IDs. Full graph logits include TEST nodes because native propagation and prediction are complete graph operations; TEST labels are not passed to loss, grouping, selector or reductions. TEST masks are public role structure. A node can be TEST in one block and TRAIN/VAL in another, as disclosed. No TEST scoring or accidental all-label tensor reader was found in the training/evaluation path. Resource preparation loads compact TRAIN labels but does not decode VAL channels; preservation hashes physically open the compact files, as disclosed.

## Six-fit selection, replay and evaluation

The six slots are fixed in block-major order, with both recipes and seeds17/29/43. Each epoch performs one fit update and eval-mode scoring. Best accuracy starts at zero, a strict VAL accuracy improvement creates an immutable checkpoint, exact ties retain the earliest, and250 consecutive non-improvements or2000 epochs stop. NLL and fixed-five-class macro-F1 are recomputed at the same selected checkpoint. An all-zero trace creates no artificial selected state, is retained, and makes complete comparison ineligible. Failed and blocked cases remain in `STUDY.json`; no favorable subset is recommended.

The portable checkpoint records source/data/recipe/roles/runtime, optimizer group names, native model and buffers, Adam numerical/group state, gradients, flags, modes, primitive configuration, CPU and selected-device CUDA RNG, Python RNG, and primitive-encoded NumPy RNG. The model classes remain local; only primitive/tensor state is pickled, avoiding unsupported full nested-class model pickling. A joint deepcopy of the selected live model/optimizer preserves parameter binding; a fresh source reconstruction is restored for full-logit and subsequent fit-step comparison. Two extra train steps and two eval forwards are recorded and whole-case elapsed time includes their work. These checks are implemented, subject to the qualification wording in AW-01/AW-02.

Resource preparation uses complete uncached CPU graph tokens for both K values, compares them to the saved native mono body, performs six true fit updates per recipe, allocates full prediction, and exercises checkpoint/inference/next-step reconstruction. It records process RSS/peak RSS, CUDA allocation/reservation/peaks and elapsed time. Native8 CPU evidence does not imply this actual-data or CUDA passage; the packet correctly requires a fresh representative run and declines per-epoch/block, OS isolation, author PyG2.3 or complete linked-dependency qualification.

Evaluation checks exact six-terminal closure, all selected/replayed flags, source/input/checkpoint bindings, the entire selector/bad-counter trace and successful root-observed training process exit before importing Torch or opening projected data. It reconstructs selected native inference, checks saved compact VAL logits, recomputes saved metrics, retains all six rows, and applies the predeclared three-block mean rule with defaults winning an exact tie. A numerical mismatch aborts the comparison.

## Statistical and admission limits

The recommendation maximizes validation accuracy selected repeatedly on development labels. Three overlapping splits on one graph do not provide three independent graph replications, and this comparison is not a confirmatory generalization estimate. The packet discloses these limits, emits no significance test or TEST utility claim, and keeps competence and M4/intervention authorization false. No independently comparable published Amazon native competence reference has been established; that is a later scientific gate, not an invented threshold for this descriptive six-fit study.

Required external evidence remains: root disposition of this source review and AW-01 guarantee; actual commit-resolved official NPZ bytes and CPU projection; fresh selected-device metadata and complete-data both-recipe resource/replay passage; separate six-fit training release and normal job receipt; then full selected/replayed closure and successful process-exit evidence before evaluation. Nothing in this review supplies that evidence or authorizes its execution.

## Review artifacts and scope

`HASH_CHECK.json` records all expected hashes and inventory checks. `SOURCE_SEMANTICS.json` records independent AST/value checks. `INSPECTED_HASHES.json` distinguishes full content review, partial metadata review and custody hashing. `REVIEW.json` preserves both findings and every admission limitation. No sealed packet was changed. Only project source/metadata reads and stdlib hash/AST/compile work were performed; no packet code was imported or run, and no numerical library, dataset, label, checkpoint, predictive outcome, remote code, installation or subagent was used.
