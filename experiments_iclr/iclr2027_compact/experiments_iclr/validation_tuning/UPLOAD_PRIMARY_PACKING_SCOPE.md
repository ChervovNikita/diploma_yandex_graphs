# Upload transport for the primary 432-cell study

The original COMPACT_SCOPE.md and COMPACT_BUNDLE_MANIFEST.json are unchanged
and describe the full author stage with ordinary results/ files. This upload
stores every one of the same 864 result JSON and validation-trace CSV files
inside RESULTS_RECORDS.tar.xz without changing their bytes. The separate
PACKED_RESULTS_MANIFEST.json lists each original SHA-256. Run
`python verify_trace_archive.py` here. It verifies the archive and all 864
original file hashes, temporarily reconstructs results/, and invokes the
unchanged `verify_compact_tuning.py` selection/decision audit. Omitted model
checkpoints and float logits remain outside the compact upload.
