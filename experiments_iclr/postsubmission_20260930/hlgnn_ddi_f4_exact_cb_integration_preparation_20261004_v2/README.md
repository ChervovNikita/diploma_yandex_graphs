# DDI exact-CB auxiliary on the saved F4 private-hop HL-GNN

This v2 packet supplies a corrected, disabled scientific bridge to the sealed F4 target-only source. It reuses that model, learned node-ID embedding, private hop coefficients, shared affine map and four native Hadamard pair heads. It copies the verified exact conditional-Bernoulli arithmetic without edits. There are **zero additional trainable parameters** and no serving override.

The scientific priority is **joint minus separate**, with identical parameters, masked contexts, query positions, input/scorer dropout, loss scale and native budget. Target-only is a secondary transfer comparison and has different augmentation and compute. All three arms serve by the inherited uniform mean of four raw pair scores on complete TRAIN; counts, bits and responsibilities never enter served scores.

## Read first

- `PROTOCOL.md`: supports, mask, losses, paired family, interpretation and budget decisions.
- `f4_cb_model.py`: actual training bridge, including native AUC and checkpointed auxiliary scores.
- `ddi_pattern_support.py`: complete TRAIN teacher, predefined stratified query selection and complete residual supports.
- `mechanism_diagnostics.py`: passive same-state uniform/visible-degree references and responsibilities; not invoked during training or checkpoint selection.
- `train_f4_cb_ddi.py`: disabled scientific entry point reusing the native TRAIN/VALID loader, 500-epoch driver and selector; explicit external project output and complete-family comparison.
- `runtime_train_epoch.py`: explicit real-TRAIN one-epoch runtime entry point for a fixed arm; fresh seed0, all17 batches, per-update time/memory/support receipts, no VALID scoring/checkpoints.
- `test_receipt_regression.py`: focused stdlib schema/path tests; no model/numerical imports or performance measurement.
- `INPUT_BINDINGS.json`: exact dependency/evidence hashes. No F4 owner files were edited.

## Evidence entering the proposal

The qualified DDI artifact contains 1,067,911 TRAIN positive records, no TRAIN/graph weights, and the original fixed VALID pools (133,489 positives; 101,882 shared negatives). The native dispatch is therefore AUC, despite the author's nominal `WeightedHingeAUC` setting. The artifact metadata was read; no tensor payload or DDI score was read here. Runtime and full-budget qualification remain false.

The saved F4 encoder evidence establishes synthetic float64 tied-context output/input/parameter-gradient equivalence (109 assertions), including the propagated bias channel. It does not qualify this trainer, float32 DDI execution, speed or memory. The saved exact-CB comparison passed its original loss/gradient rule on one Collab native batch. It does not establish dense-DDI backward feasibility. The DDI Boolean census reports many nonconstant patterns, but uses the NCNC 24,576-record mask rather than this HL-GNN 65,536-record auxiliary mask. Its fractions and timings are not DDI exact-loss throughput estimates.

The degree-prior packet contains conclusions and a prospective diagnostic plan, **no measured degree-only reconstruction result**. Its key conclusion is retained: local count conditioning cancels a common side-logit offset, while candidate-specific degree preferences remain. Uniform and visible-degree references cannot replace the primary served-prediction contrast.

## Status

V2 source/JSON/AST/hash work and focused stdlib receipt/path regressions were performed. Only the stdlib receipt/path modules were imported for checks; no numerical/model imports, real training/resource/GPU/server actions, installs or heldout score reads occurred. Scientific config release remains disabled. This packet neither changes the frozen Collab family nor fulfills the older full-stream NCNC diagnostic protocol. Root must review the explicit DDI adaptation and its runtime budget before releasing anything.


## V2 practical runtime command

After root stages the sealed v2 and its unchanged pinned sibling F4 sources, one real combined TRAIN runtime invocation is:

```sh
python -B /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/hlgnn_ddi_f4_exact_cb_integration_preparation_20261004_v2/runtime_train_epoch.py \
  --arm joint --device cuda:1 --cuda-memory-fraction 0.30 \
  --output-dir /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/ddi_f4_exact_cb_runtime_execution_root_20261004_v2/joint_seed0_run01
```

Root must place this inside its existing supervisor with the declared host/wall/capacity limits. With `CUDA_VISIBLE_DEVICES=1`, pass `--device cuda:0` instead. For the other fixed arms, change `--arm` and give each invocation its own new absolute output directory. There is no automatic arm loop or retry. A completed target-only invocation alone leaves combined exact-CB cost unresolved.

The command uses the frozen qualified `HLGNN_DDI_TRAIN_VALID.pt` contract already in config. It verifies the dataset and report hashes, constructs fresh state, performs real TRAIN forward/backward/Adam updates and writes `runtime_summary.json`, `runtime_updates.jsonl` and complete `paired_stream.jsonl` outside source. The artifact loader reads the fixed VALID tensors to validate the existing contract but the runtime entry point never invokes VALID evaluation or selects a checkpoint. Runtime receipts cannot satisfy scientific-family completeness.

`V1_TO_V2.diff`, `CHANGE_REPORT.json`, `STATIC_CHECKS.json`, `MANIFEST.json` and `SEAL.json` describe the repair and frozen source digests. V1 remains preserved. No speed, memory peak, predictive gain, novelty or acceptance claim is supported before actual evidence.
