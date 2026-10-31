# Portable spectral checkpoint replay

This packet prepares external verification of the new spectral PPI study. A reader can replay a selected checkpoint and verify its selection without accessing author calibration or gradient records. **Checkpoint replay is not independent fresh training reproduction.** No performance claim follows from this packet.

The exporter admits only a complete CPU-v3 fit whose source, runtime protocol, preflight, complete artifact inventory and numerical author replay match. It copies the unchanged model and adapter sources, clean selected state tensors, the two official validation graph tensors/targets, selected logits, the selection trace and all 101 saved validation-logit epochs. Frozen bases and member contrasts are reconstructed directly from the state_dict. It copies no optimizer/RNG state, processed dataset pickle, training/test graph archive, credentials, literature files, calibration records or author paths.

Validation reconstruction reads only four frozen `valid_*` raw files: three arrays loaded with `allow_pickle=False` and one graph JSON. Every reconstructed tensor must match the original validation fingerprint exactly. A mismatch refuses export. This reconstruction and its fingerprint checks are pending runtime verification.

The portable harness sets bundle-contained temporary/cache paths before Torch/PyG imports, reads every tensor with `weights_only=True`, verifies all immutable hashes, recomputes six saved metrics across all 101 epochs and independently checks the selected winner. It then reconstructs the complete adapter state and produces fresh selected predictions. It records float32 tolerance, threshold changes and fresh metrics. Outputs use new paths under `bundle/results/`. The executable and metadata use no hardcoded author root.

Minimal runtime versions and the SAGEConv source hash are recorded. Torch core, PyG and NumPy versions must match; the Torch CPU/CUDA build suffix may differ because execution is CPU only. Python 3.11 is the recorded environment. Numeric replay does not certify arbitrary dependency or hardware changes.

**Current status:** source preparation and static checks only. Torch/PyG are unavailable locally, so no synthetic or production export has run. No predictive result/checkpoint was read during preparation. The production engineering fixture is prospectively fixed to `seed0_label10_selected_run01`, selected by its seed and known nonzero rank, before inspecting predictive quality.

`COMMANDS_CPU.sh` supplies the supervised order. First, run the synthetic relocation gate for tied/rank-zero and nonzero bases, including a re-sealed metric-tamper rejection. Second, export the predeclared complete/replayed fit. The exporter runs a fresh standalone bundle replay and creates `EXPORT_ADMISSION_PASS.json` only on success. Preserve failed attempts and use new output names after failures.

Only `exports/<attempt>/bundle/` is the public artifact. Author admission records and supervisor logs remain outside that bundle. Complete101-epoch logits require roughly 1.3 GB per official PPI fit; this is deliberate evidence retention. Exporting one production fixture is an engineering gate, not evidence that the whole 24-fit matrix or every public export is complete.
