# Fixed Amazon Polynormer-r paired successor

**Status: source requirements prepared; no dataset, scientific module, numerical/GPU execution, SSH, install, commit or subagent was used.** The new training/evaluation orchestration and independent qualification runner are specified, not implemented or admitted. Exact reused neural bodies and the pinned author reference are included. This is one fixed sharing comparison, with no tuning, regularizer or novelty claim.

## Source reuse and exact differences

The latest `modern_backbone_teacher_amendment_v3` supplies byte-identical copies of `native_polynormer.py` (`9b4e533f…`) and `backbone_boundary_adapter.py` (`699b606e…`). `GlobalAttn` and `Polynormer` match the pinned author model classes as ASTs. `BoundaryProjector` and `PolynormerBoundaryFamily` match Section 4's saved layout as ASTs. The original Photo single/GNNM4/independent4 receipts report successful full-graph local/global parity, four probe updates, model/Adam restoration and selected-logit replay under Python 3.11.14/Torch 2.1.2+cu118/PyG 2.7.0. Their exact receipts, physical terminals and source identities are bound in `SOURCE_BINDINGS.json`; they do not qualify Amazon.

The surrounding modern trainer cannot be reused unchanged: it normalizes Photo features, uses a different recipe and seed spacing, a four-config grid, raw-mean-logit NLL selection with a fresh global selector, and one joint independent4 selector/loss. The new contracts replace those interfaces while preserving the neural bodies. Author `main.py`/`eval.py` cannot be executed directly because they evaluate TEST. `REUSE_AND_DEVIATIONS.json` records each difference before execution.

## Frozen comparison

| Item | Fixed requirement |
|---|---|
| Complete blocks | Existing splits/seeds `(0,17)`, `(1,29)`, `(2,43)`; all three retained |
| Families | Native single, boundary GNNM4, independent native4: nine family records |
| Recipe | Author commit `fc8c276c9c5dfbd616d83f65338a3392188a5e08`; 256 per head ×2; 10 local/1 global; 200+2,500 updates; Adam lr0.001, wd0; input dropout0.2, local/global0.3; learned sigmoid gates, shared Q/K, ReLU, no pre-LN |
| Data | Existing label-free raw Amazon projection; native symmetrize/remove/add-loop path; same class-stratified hash80% official TRAIN FIT/20% TRAIN-control; official VAL selector; TEST labels locked |
| Initialization | Common-body/member0 seed is block seed; native member `m` seed is `block_seed+1009*m`; baseline stem Rademacher R, other R=1, S=1, copied native B; archive factor/dropout RNG; no post-wrap reset |
| Training | GNNM mean of four FIT cross-entropies through four complete trajectories; each independent native has its own unscaled FIT loss, Adam, selector and transition |
| Selection | Strict VAL accuracy, earliest tie, first post-update candidate; one selector across both stages; GNNM uses arithmetic-mean probabilities; each native selects itself |
| Transition/final | Restore local-best model and Adam after 200 local updates, preserve current end-of-local RNG, explicitly enable `_global`; complete 2,500 global updates; final selected state may be local or global |
| Pool/report | Mean all four probability vectors for both arms. Primary: GNNM4 minus independent4 TRAIN-control NLL per block and mean; retain control accuracy/Brier, every member/native-single result, selected stages and costs |

Native single is the exact independent member0 fit, referenced by an immutable family record. This preserves the control without a duplicate replicate or copied checkpoint. Complete spec/source/runtime/role/seed/RNG/optimizer/selector identity is required for that alias. There are **15 distinct fits**, 40,500 Adam updates and 64,800 complete member training-trajectory updates. Study totals charge physical work once; arm costs retain their actual member0 contribution. No matched-compute claim follows.

The existing data manifest was read as metadata only: neither its NPZ arrays nor label/checkpoint payloads were opened or hashed. A new exact consumer release must bind its producer chain and roles. The prior raw projection's all-node label decode remains disclosed; this successor has no TEST-label reader. Official TRAIN labels support fixed class-stratified role derivation, including control membership, but control predictive outcomes cannot select a checkpoint or recipe.

## Qualification needed before the full study

Use independently reviewed instrumentation on the complete Amazon graph and exact recipe, first on split0/seed17 with all four prescribed native seeds and GNNM4. Freeze the same existing logit `rtol1e-5/atol1e-6` and gradient `rtol1e-4/atol1e-6` gates before observing discrepancies.

1. Compare native/identity-boundary and actual Rademacher member functions in both stages, shared/private gradients, copied biases, dormant-owner removal and exact Adam membership. Verify each member's complete GAT/global activations and node-axis reductions. Eval-mode gradient parity and real train-mode updates serve distinct checks.
2. Run real full-graph local FIT updates, retain the actual VAL-selected local image, verify a nontrivial scratch rollback of model/Adam tensors and step counters, preserve the current RNG at transition, then perform a real global FIT update. Check all active gradients/member connectivity; dormant stage heads legitimately have no gradient. Short probes prove mechanics, not the full schedule.
3. In both stages, save/reopen safe full state, replay full logits, then compare the same next real FIT update to a preserved live clone. Loss, model/Adam, gradients, permissions, module modes, explicit `_global` flags and all RNG state must match bitwise logical bytes. No unsafe checkpoint loading or tolerance rescue.
4. Measure local/global train+VAL, all-four inference, checkpoint/snapshot/replay memory and whole-process costs. Approve the full 200+2,500 schedule forecast prospectively. Bind exact source/role coverage for blocks1/2 separately from the numerical block0 claim; every actual fit retains validity gates.

`QUALIFICATION_PLAN.json` distinguishes the original qualified Photo runtime from the current Amazon Python3.12.11/Torch2.7.1/PyG2.7 runtime. New Amazon qualification is required in the intended study runtime; the original qualified runtime also needs a bounded Amazon scope to establish its new shape/recipe parity. Neither is the author's stated Torch2.0/PyG2.3 runtime. An exact author-runtime reproduction claim remains unsupported without separately pinned available builds and qualification. No environment change is authorized here.

`CHECKPOINT_CONTRACT.json` makes the missing Python `_global` state explicit. Saving RNG for restart does **not** mean rewinding it at the native stage transition: the author's `logger.load_model` restores only model/optimizer. Earliest ties and an overall local winner must survive final restoration. Safe primitive metadata prevents the former `TorchVersion` loading failure.

## Preservation and release limits

`PRESERVED_FAILURES.json` binds the original failed identity/parity, safe-loading, deterministic APS and launch receipts, attempt registries and prior Amazon source-transfer failure. All originals remain unchanged. New failures must remain distinct and block closure; no favorable subset, hidden retry or changed recipe is allowed.

The three driver/checkpoint contracts are the concrete implementation requirements. Before a run, implement the thin orchestration and independent qualification runner, review their exact source, obtain separately bound qualification/resource/consumer/fit authority, and require all 15 physical successes plus all nine family records before control comparison. No historical PolyFormer/Photo trained artifact is substituted. Passage would support this bounded sharing diagnostic, not published-test equivalence, heldout quality or superiority.
