# Native single-forward capture and label-only training integration

8 October 2026. **Disabled source preparation only.** This new folder supplies a callable full-input WikiCS pilot driver and read-only native capture. It preserves the sealed corrector core and public/native sources. No server, dataset, checkpoint, current run or partial outcome was accessed; no numerical import, training or runtime qualification occurred. No canonical record or publication was modified.

## Concrete operation

`capture.py` attaches additional read-only pre/post hooks to native `pred_local` and `pred_global`. It verifies the active head and captures full H `[11701,512]` and logits `[11701,10]` before the existing `WikiBackbone.forward` indexes batch IDs. Hooks return `None` and detach only their recorded references. They do not replace native tensors, add a backbone call, or draw randomness. The source requires exactly two head captures during an original native TRAIN update and exactly one during a full VALID evaluation.

`train.py` calls the **unmodified** public native `Session.train_step`: two complete own-CE views of all580 TRAIN labels, one native backward/Adam update, the original optimizer and persistent dropout stream. A common half-TRAIN query mask is drawn beforehand using the corrector's private generator. The unchanged corrector trains once on **view B**, the second existing forward's detached full H/logits. This fixed capture choice introduces no view search or extra native forward. The native Adam runs first; the corrector update uses the old view-B activations and detached logits, so its gradients cannot enter that native update.

The sealed core continues to enforce common query-label exclusion, label-only values, all-observed-nonself-neighbor normalization, conditional TRAIN scaling and unscaled legal inference. Corrector construction, mask draw, update and serving have native/ambient RNG-state checks. These are source guards awaiting actual runtime qualification, not measured trajectory or bitwise parity evidence. Sparsity dilution, low native TRAIN CE feedback and representation drift remain unresolved utility risks.

## Native local restoration and final selection

Each complete VALID event calls the original public native `evaluate` once. Its native own accuracy selects `NATIVE_OWN_LOCAL_COHERENT.pt` by strict-first maximum during epochs1–100. The capture from that same native evaluation supplies full context for corrected serving; no second backbone evaluation is performed.

At epoch101, the original native local-transition helper restores the native model/Adam from that **native own-selected** epoch and switches global mode. The integration restores every learned corrector parameter and its optimizer state from the same coherent epoch. Fixed graph/TRAIN buffers are checked rather than overwritten. Native private/ambient streams and corrector mask RNG remain at their **live end-local** states, and actual-work counters remain cumulative. Snapshot mask states are recorded for custody, not rewound. The old-stage restored parameter epoch and logical executed-update count can differ.

Final `selected.pt` is the strict-first ensemble development-accuracy maximum of coherent native/corrector epochs. `NATIVE_OWN_BEST_COHERENT.pt` separately retains the native own final maximum. These final selected epochs may differ. The method therefore preserves the specified native update and own-local policy, while reporting a different final ensemble selector; it does not declare final native-single checkpoint equivalence or baseline competence.

`reconstruct_selected` checks exact source and role identities, rebuilds the selected native stage and learned corrector coordinates, and exposes a serving-only wrapper using one native forward. Exact training resume is unsupported. Full selected-state reconstruction remains unqualified at runtime.

## Callable scope and controls

The numerical entry defaults disabled:

```python
run_complete(
    train=bound_train_npz, valid=bound_valid_npz, output=fresh_cell_output,
    polynormer=bound_native_source, seed=6101,
    condition="label_only4_detached", device="cpu",
    later_execution_authorized=False,
)
```

The source uses the existing full WikiCS split0 recipe: 11701 nodes,300 inputs,10 classes,580 TRAIN labels,5274 merged development nodes, width512,7 local/2 global layers,1100 epochs including100 local, and original native Adam. The public input validator checks complete shape/domain/role contracts; official acquisition and historical exposure custody still need separate approval. There is no TEST reader, shortened horizon, automatic campaign, retry, outcome opening or owner/queue.

`native_single` delegates to the unchanged public full driver with no corrector construction. Existing closed Wiki24 native references remain historical context; this packet does not repeat them or relaunch them. A future matched reference release is a separate decision.

The finite first mechanistic panel distinguishes the retained shared-factor candidate, `multihead_single4`, and `shared_backbone_untied_correctors4`, all around the same native trajectory and common mask. The exact newly sealed control source is bound in `CONTROL_CORE_BINDING.json`; its common protocol supplies actual graph/TRAIN/selection identities, explicit capture provenance, complete learned-parameter/optimizer-bank snapshots and live mask/counters. Native parameter epoch and logical executed-update count are recorded separately, including the local rewind. The candidate's sealed core is not edited.

| Condition | Predictions served | Attention heads | Corrector Adam calls/update | Final corrector-state authority |
| --- | ---: | ---: | ---: | --- |
| `label_only4_detached` | 4 | 4 | 1 | One coherent ensemble-selected epoch |
| `multihead_single4` | 1 | 4 | 1 | Complete four-head single with joint linear256→C readout, one coherent selected epoch |
| `shared_backbone_untied_correctors4` | 4 | 4 | 4 | Entire private bank at one coherent family-selected epoch |
| `one_path` | 1 | 1 | 1 | Diagnostic single, not the required stronger single control |

The shared-backbone untied bank owns four independent correction functions on **one** native backbone. It is not `ordinary_independent4`, which needs four independent native trajectories and their own coherent selections, and is rejected by this driver/factory. Separate head/member route snapshot APIs are not substituted for the four-head single's complete snapshot or the shared-bank family epoch. Selected serving authority comes from this integration's exact coherent checkpoint; it does not claim the ordinary four-GNN per-member selected-state API.

The first mechanistic panel has three conditions × three predetermined seeds = nine full configurations, with the plain native reference as a separate three-cell baseline option. It is source-bound and **not launch-admitted or runtime-qualified**. Four heads alone do not certify single baseline competence. Source-faithful label-aware/C&S, live-gradient, label-free capacity and strong ordinary GNN controls remain later comparison obligations, not implemented cells in this folder. A feature-only baseline gap alone cannot establish ensemble usefulness.

## Remaining qualification

Static AST/JSON/hash inspection can verify the source structure, but no actual full-input update, masking behavior, RNG endpoint, gradient boundary, local restore, coherent reconstruction, memory/runtime, baseline competence or scientific quality has been verified. Full native/core/control source review and real complete-path qualification must precede any separately reviewed cell release. Current research directions, gate rules and jobs remain unchanged.
