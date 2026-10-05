# Operator 9 numerical witness extractor

Source only; not executed. Reconstructs exactly split0/shared/fold0 from authenticated source V2, complete15 closure, canonical public graph, retained logits and official VALID anchors. No features/TRAIN/control/TEST labels, checkpoints, fits, head/calibrator construction, metrics or scored outcomes are accessed. It is a narrow numerical diagnostic, not an automatic study retry.

The unchanged source QP is called in its native 2048-row order. On the first exception, a traceback hook identifies the first actual `chosen` NaN row and captures its exact P[4,5], posterior, raw/scaled A,b and scale. A read-only recomputation exports all15 old normal-equation face KKT matrices, singular values and feasibility residuals for that witness. No sealed source is changed; bytecode writes are disabled. Outputs stay inside the phase/repository, with exact input hashes and CPU costs.

Root command, after separate diagnostic authorization:

```sh
python extract_projection_witness.py \
  --source /absolute/phase/amazon_polynormer_logits_graph_moment_source_preparation_20261005_v2 \
  --manifest-sha256 af15ac11349ac7d9dbc8608b4362d461195409ca6348fdfc67e7ac7b4a273b21 \
  --seal-sha256 2ed8fef0d08e420e140ddcddfa2929e00849797715d16e1e0be626e9cb2ef900 \
  --phase /absolute/phase \
  --closure-freeze EXISTING_COMPLETE_CLOSURE_FREEZE_PATH \
  --closure-sha256 ROOT_BOUND_CLOSURE_SHA256 \
  --output /absolute/phase/NEW_NUMERICAL_WITNESS_OUTPUT
```

Protocol defaults to exact V2 SHA5d667f15640102995fa998ca6932c53c5b71554273c84d4938a331c0a4eae7ba; its preserved reference correction is authenticated by the source custody routine. Complete original descriptors are checked before any array decoding. Only one seed context is built; no inner feature/head work is performed.

Success writes `WITNESS.json`, `BOUND_CONTEXT.json`, and `DIAGNOSTIC_RESULT.json` with `WITNESS_EXTRACTED_NO_FITS_OR_SCORES`. Failure/nonreproduction preserves diagnostic and exits nonzero. Root must review the actual witness before sealing any probability-space solver repair or admitting another study run. This preparation received only stdlib AST/integrity checks.
