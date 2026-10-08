# Serving API map for the post-family collector

This is a read-only implementation map, not executed code or a source amendment. Authority: `label_only_private_corrector_four_bank_first_screen_source_20261008_v1/screen.py` and its sealed integration/control dependencies. Final scientific source identities must match each trusted selected state before use.

## Coherent state reconstruction

- `screen.reconstruct_for_serving(state, train_data=..., polynormer=..., device=..., later_execution_authorized=True)` — lines330–350. Use each arm's own `kind="arm_final_selected"` state only after root's whole-family opening. It checks original source/context/recipe, restores native model/mode/Adam and that arm's learned corrector/Adam, and creates a serving-only session. No training resume, query-mask generation or VALID evaluation is permitted.
- `session.native`, `session.banks`, `session.train_data`, `session.capture`, `session.metadata()` — session fields/methods around lines127–140. A selected-arm reconstruction contains only that arm. Different arm files may select different epochs; never mix files.
- `session.close()` — releases read-only native hooks. Close every reconstructed session; retain no mutable reference to the training owner.

## One forward for both native and corrected predictions

`session.serve_ids(ids)` (lines195–201) returns corrected results but consumes and discards the full native capture. Calling it and then `capture.finish()` would be invalid. For the same-state native comparison, reuse its explicit public session operations:

1. Set the restored native model to eval and enter `native.torch.no_grad()`.
2. Build the same full x/edge batch from `session.train_data`, with the full development ID list, exactly as `serve_ids` does.
3. Call `session.capture.begin("SERVE",1)`, then **one** `session.native.forward(batch)`.
4. `H,base = session.capture.finish()` returns full active-head states/logits; select `base[development_ids]` for feature-only native probabilities.
5. Call `session.serve_banks(H,base,development_ids,native_capture=session.metadata())` — lines187–193. Reuse this identical capture for corrected predictions. No extra native forward, training Q or heldout truth enters serving.

Separate metric code may read the authorized development truths after prediction. Do **not** call `session.evaluate(valid)` in a selected-serving session; it explicitly refuses outside the scientific training/selection purpose. This collector does not select another epoch.

## Available outputs and structural coverage

For each returned arm, use `member_logits[M,5274,10]` and `served_probabilities[5274,10]`. C4/U4 have M4; both singles have M1. `screen._readout` (lines64–78) documents stable selected-state metrics, but it is not a member-head expansion or a new selector.

Coverage is computed from the permitted TRAIN IDs and incoming nonself edges already in the sealed context. Candidate buffers `edge_source`, `edge_target` and `train_ids` express the exact relation; use fixed source buffers or the saved graph, without reading heldout labels or changing edges. Do not interpret a control's constructor seed override as an independent native-backbone acquisition.

## Operator explanation beyond public outputs

The public serving return does not expose attention histograms. C4's compatibility matrix uses its shared `label_embedding`, `value.weight/r/s`, `output.weight/r/s` as derived in the saved operator note. For an untied ordinary correction head, the effective map is `W_O W_V Eᵀ`. For S_joint4head, slice the joint output into four 64-column blocks: its single correction is the sum of four head-specific compatibility-map/histogram products, still **one prediction**.

A later read-only analyzer can derive feature-only attention/histograms from captured H and saved parameters using the exact sealed formulas. It must preserve all-neighbor normalization, multiplicity, scale one and original ID order, and reconcile its reconstructed correction with `member_logits−base`. No such helper is written or admitted here. Charge any extra map/edge aggregation and retain limitations; do not alter the core to export new fields or turn a small floating-point difference into a scientific finding.

Existing source hash/custody checks apply. This map adds no numerical qualification, outcome access, theorem, training operation or revised first-screen gate.
