# Upload package and omitted public data

This archive is the anonymous code supplement intended for OpenReview. It is
kept below 100,000,000 bytes, a conservative interpretation of the submission
form's 100 MB supplementary-file limit. The manuscript's appendix is part of
the paper source and follows its references.

All experimental runs, frozen training source, validation traces, and original
result records remain in this upload. No outcome is removed based on its sign.
For all nine Roman bridge rows, the upload retains exact float32 pooled logits,
every member's hard class, and the original validation/test node IDs and labels.
It omits their individual member float logits. This still permits recalculating
pooled accuracy and cross-entropy, member accuracy, and pooling gain. Recreating
the average from individual member logits requires the original arrays retained
in the full author bundle. `roman_bridge/POOLED_DERIVATION.json` links every
replacement to the unchanged original result and array hashes. Run
`python roman_bridge/verify_compact_pooled.py` for the new representation.
`verify_full_original.py` is preserved as source, but requires omitted arrays.
Large training checkpoints are not part of either compact bundle.

For all 18 earlier external-SAGE rows, the upload retains exact float32
pooled logits, every member class, original IDs and labels, and all source,
trace and result records. Individual member float logits remain in the full
author bundle. `external_sage/PROJECTION_MANIFEST.json` and its independent
derivation audit link each replacement to its original file. Use
`python external_sage/verify_pooled_classes.py` for this representation.
The original `verify_compact.py` needs omitted member-logit arrays.

All 72 primary validation-selected cells additionally retain their original
float32 test member logits in `experiments_iclr/hpo_selected_test_logits/`.
Their mean reproduces each original pooled float32 array bitwise. That
companion's verifier checks every selected cell, official test reference,
score, hard decision and source-array hash. Original pooled validation logits for these same 72 cells are also retained
in the temperature companion. Individual validation member logits and
training checkpoints remain outside this compact upload.

Two additional public raw graphs are omitted relative to the full compact bundle:

- File: `data/roman_empire.npz`; bytes: 20401489; SHA-256: `a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428`; pinned public URL: https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/a431395582e929d88271309716bea4fe24ce6318/data/roman_empire.npz
- File: `data/tolokers.npz`; bytes: 1329769; SHA-256: `dacf3ac94cec53d03cd2adb5255c08b33dee1656c33ca8164a464bd9450a1667`; pinned public URL: https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/a431395582e929d88271309716bea4fe24ce6318/data/tolokers.npz


The existing `experiments_iclr/data_manifest.json` records this exact source.
Run `python experiments_iclr/fetch_datasets.py` before new training to download
and verify it, together with the other public raw graphs already omitted from
the full compact bundle. That script refuses mismatched existing bytes.
The advertised compact score-verification commands use supplied selected
records and arrays and do not need these raw public graphs. Full checkpoint
replay has additional requirements described in each study README.

Descriptive README paragraphs and the Roman bridge compact-verifier entry are
adapted to state these omissions. Frozen training sources and original result
manifests are unchanged. The full author bundle preserves the original arrays.
`BUNDLE_MANIFEST.json` hashes every supplied file except itself.

## Earlier Roman control prediction projection

This upload projects all 14 earlier supplied Roman control prediction files
uniformly. It retains the exact float32 mean logits and all other original
arrays, including member classes, pooled probabilities, labels and node IDs.
Individual member float logits remain in the full author bundle. The original
`prediction_manifest.json` identifies those original arrays and their hashes.
`CONTROL_POOLED_MANIFEST.json` links every derived file to that manifest and
its unchanged selected result row. Run
`python experiments_iclr/verify_control_pooled.py` to recalculate the 14
pooled scores, cross-entropies and member accuracies. The remaining original
control rows still have their summary CSV records and original omission scope.
The original full-array diagnostic scripts require the full author artifacts.

The two 216-cell studies pack every original result and validation trace in exact CELL_RECORDS.tar.xz archives. Their verify_trace_archive.py wrappers verify all original hashes and run the unchanged original verifier after temporary extraction. All cells and trace rows remain.

The primary 432-cell study packs its original 864 result JSON and validation-trace CSV files in RESULTS_RECORDS.tar.xz, with every byte bound to its unchanged COMPACT_BUNDLE_MANIFEST.json. Run experiments_iclr/validation_tuning/verify_trace_archive.py to verify the archive and execute the unchanged primary selection verifier after temporary extraction. The original records and all outcomes remain.

The Roman additional-mask study similarly packs all 210 original results/ files, including complete validation traces, selected hard decisions, source manifests and audit records, in a lossless RESULTS_RECORDS.tar.xz. Its verify_packed_results.py wrapper checks every original SHA-256 and runs the unchanged 48-cell verifier after temporary extraction.
