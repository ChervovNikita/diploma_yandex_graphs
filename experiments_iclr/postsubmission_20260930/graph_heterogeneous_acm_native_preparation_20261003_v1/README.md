# Minimal ACM native15 preparation

**Source prepared; execution unverified.** No archive, development-label payload, heldout label, DBLP outcome, Torch runtime, training, or remote job was opened or executed for this preparation. The root-adopted ACM scientific design is SHA256 `96062ebf89e676d000c62f6a0ec577b8ec59a845196b98e08328bf6c3dcc1c24`. The template binds its measured complete graph, canonical 907-paper development descriptor, and all five frozen 726 TRAIN / 181 validation splits.

## Files and immutable reuse

- `train_native.py`: native GAT / Simple-HGN / SeHGNN, seeds 131/137/139/149/151, all 15 terminals retained. Imports the accepted `NativeStopper`, seed, and global-RNG functions. Its fit/replay body preserves optimizer, AMP, stopping, selected-state persistence, parameter ordering, replay tolerance, and paid-cost behavior; only the dataset tag and classifier-report count change. Class metrics now use the fixed three-class schema. Every GAT/Simple-HGN TRAIN, validation, and replay call passes the ACM feature list.
- `native_inputs.py`: ACM preprocessing only. Reuses the sealed HGT agent's `acm_inputs.stream_schema`, whose shared development reader/split verifier are unchanged. Reuses the accepted native graph materializer and whole-target evaluation-ID function. The accepted `native_models.py` is imported unchanged; no model-math copy or rewrite is included.
- `FREEZE_TEMPLATE.json`: exact data/split/scientific bindings and native recipes. Root still fills adoption, paired executable HGT freeze, one-GPU identity, and admission. It is not a runnable release.
- `PROVENANCE.json` and `MANIFEST.json`: concise reused-source and current-payload bindings, with qualification limits. They do not claim a new numerical PASS.

The sealed DBLP and HGT ACM packets are unchanged. Existing native qualification is reused; this packet does not repeat their numerical suites.

## ACM changes

**GAT / Simple-HGN:** native feature type 2 supplies the 3,025×1,902 paper attributes and sparse identities for 5,959 authors / 56 conferences / 1,902 fields. Input widths are [1902,5959,56,1902]. All raw relation support becomes the native undirected union; graph self loops are removed and one is added per node. Simple-HGN assigns raw labels in verified file encounter order 0–7, overwriting the previous label on the same ordered pair. Raw self labels survive graph self-loop replacement. Synthetic self type is 8; missing reverse types are raw+9; the unchanged model receives 17 types. Full overlap, overwrite, edge-type, self-label and support fingerprints are recorded. GAT ignores relation labels.

**SeHGNN:** source raw row is destination, column is source. Verify the author P/PK, A/AP·PK, C/CP·PK and transpose-support assertions. Merge raw PP and PP_r as binary support, coalesce and set the diagonal, then row-normalize in FP32. Omit K by native `ACM_keep_F=False`; retain P/A/C attributes and operators in order PP/PA/AP/PC/CP. The native lexical DGL feature traversal produces 41 target-paper feature channels through four hops. The existing left-extension/removal logic produces 20 paper-to-paper label products through four hops. Seed only TRAIN one-hot labels, remove the diagonal from each complete product, and do not renormalize afterward. Rebuild label inputs per split and record channel/cache hashes without label values.

The unchanged SeHGNN class receives three classes, one task layer and residual false; its target metadata is P. Preserve accepted port initialization and sorted stacking policies. Its final class BatchNorm uses batch statistics at evaluation: one 3,025-paper batch, sorted TRAIN then validation then topology-derived remaining papers; extract validation metrics only. This is the existing label-safe whole-target policy; literal author's labeled-batch membership equivalence remains unverified until a separate membership receipt exists.

Recipes remain native: GAT/Simple-HGN [8,8,1] heads, 64 hidden channels per head, dropout .5, Adam 5e-4 / decay 1e-4, 300 epochs / patience 30 / latest tied validation minimum. Simple-HGN retains edge64, detached residual attention .05, feature residuals and final L2 logits. SeHGNN hidden/embed512, projection2, task1, no residual, dropout/input-drop .5, Adam 1e-3 / decay0, TRAIN batch10,000, 200 epochs / strict improvement / literal 51 nonimprovements. CUDA TRAIN AMP is retained; CPU and inference autocast are disabled.

## One necessary ACM-specific execution check, proposed only

Before eventual DBLP-gate-conditional fits, root should perform **one bounded full-ACM check at seed 131**, with TEST closed and the exact frozen sources/split:

1. Stream the actual node/link payload once through the reused reader; verify the bound shape/support and raw encounter order. Materialize the native graphs/features, execute the new attribute/PP/overlap assertions, and record the 41/20 ordered channels and TRAIN-only cache/batch fingerprints.
2. Construct each of the three accepted models with its ACM arguments. Run one TRAIN update, one fixed-batch validation pass, then save/load the complete state and replay the validation logits using the same source functions. This specifically exercises explicit GAT feature passing, class-3 metrics, ACM label products and the 61-channel SeHGNN forward. It is not a new baseline score or an old model-equivalence suite.
3. Record actual preprocessing plus per-arm peak RSS/device memory and update/evaluation time; establish the resource bound for the full native15 schedule. Include failed work. Do not infer actual resource feasibility or complete 200/300-epoch checkpoint selection from a syntax check or this one-update result.

No such check has been run here. `ACM_full_preprocessing_verified`, `ACM_actual_resource_verified`, and `ACM_checkpoint_replay_verified` remain false. The driver requires a root admission binding the exact source manifest, freeze, paired HGT freeze, scientific design, DBLP continuation pass, and ACM-specific check before fitting. A later failure retains all 15 statuses and prevents scoring a successful subset. Full selection/replay validity is recorded by the eventual fit itself.

No calibration or final scoring is implemented here. The root's adopted rule supersedes the earlier unadopted SciPy proposal: use the existing DBLP **80-iteration inverse-temperature golden-section** protocol when the separate final evaluator is released. Heldout opening remains separate.
