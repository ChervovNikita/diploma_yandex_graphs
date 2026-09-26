# Upload transport for Roman additional masks

The original result, trace, decision, source-manifest, and audit records remain
byte-identical to the full author bundle. All 210 files formerly under
results/ are in RESULTS_RECORDS.tar.xz. PACKED_RESULTS_MANIFEST.json binds
their original SHA-256 hashes and the archive digest. Run
`python verify_packed_results.py` here. It validates every archive member,
temporarily reconstructs results/, then invokes the unchanged
`verify_roman_multimask_compact.py` audit. Its optional --public-npz argument
still anchors labels and masks to the pinned public graph bytes.
