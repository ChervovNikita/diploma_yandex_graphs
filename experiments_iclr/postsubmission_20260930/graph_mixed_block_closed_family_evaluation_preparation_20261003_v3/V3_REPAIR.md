# Current V3 source repair

Source only; pending independent review and a separate root release.

The failed V2 evaluation logged `ValueError: Reconstructed full graph schema differs` at line 341. Training writes an integer-key `raw_relation_to_row` map to JSON, which stores the keys as strings. V2 compared that saved JSON directly to the live schema.

V3 changes exactly one evaluator line to compare the saved JSON with `json.loads(json.dumps(schema))`. The live schema remains unchanged for graph construction and checkpoint bindings. All 40 primary cases, 15 native controls, metrics, calibration, contrasts, replay checks and scientific gates remain unchanged. Freeze/release templates, provenance and all other Python sources remain byte-identical to V2.

Both failed launch transports and both subsequent read-only diagnosis transports are pinned in `V3_CORRECTION.json`. Their sources, outputs and costs remain preserved. No scoring, numerical graph reconstruction, archive/array/checkpoint/label read, or remote write occurred during V3 preparation. The source repair is engineering evidence, not predictive progress or acceptance.

`README.md` and `V2_CORRECTION.json` preserve predecessor documentation. `V3_CORRECTION.json`, this note and `V2_TO_V3.diff` describe current scope. Root must bind the exact reviewed V3 manifest and new external output/release before any evaluation launch.
