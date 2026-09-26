# First graph update: compact evidence

This folder records all 12 fixed first-step comparisons of TIED and SYNC
graph updates on Cora, WikiCS, Actor, and official filtered Chameleon, with
seeds 0, 1, and 2. The measurements use the frozen width-128, two-block SAGE
recipe, training indices and labels, and matched initial weights and random
draws. They do not measure validation or test accuracy.

The experiment compares two ways to use four member gradients. TIED averages
the gradients before one AdamW graph-parameter step. SYNC gives each member's
graph copy a separate AdamW state, averages the resulting graph weights, and
keeps the copies equal before the next forward pass. The private graph
gradients receive the factor-of-four correction stated in the protocol.
This diagnostic examines their **first** update only. It does not establish
why either trained model performs better.

## Files and evidence chain

- `INITIAL_UPDATE_DIAGNOSTIC_PROTOCOL.md` fixes the design and formulas.
- `INITIAL_UPDATE_DIAGNOSTIC_FREEZE.json` binds that protocol, source, base
  recipe, and mechanism source. Its SHA-256 is
  `52f825d983c9792ee0d82a68d0a27c781b639458724a09dba6bc6cc38363df75`.
- `INITIAL_UPDATE_DIAGNOSTIC_FREEZE_ABORTED_PREEXEC.json` records a discarded
  source freeze that was replaced **before** any diagnostic execution.
- `initial_update_diagnostic.py` runs the fixed CPU matrix. Its complete
  output is `INITIAL_UPDATE_DIAGNOSTIC_RESULTS.json` (SHA-256
  `911afaf61b8b55c56611bb7c9059e73fcc834312399e2c64b7d7438a5cb42539`),
  with a matching CSV and execution log.
- `export_initial_update_gradients.py` was used after the fixed result to
  export each member gradient and the two actual parameter-update vectors.
  `initial_update_raw_manifest.json` contains the 12 raw array names,
  dimensions, hashes, and replay checks. Its SHA-256 is
  `4deae261a1713347ca501307212bf41d584c5cc3dbcceafbd64aa38f01156c73`.
- `audit_initial_update_arrays.py` independently recomputed the results from
  those arrays. `INITIAL_UPDATE_ARRAY_AUDIT.json` records all 12 passing
  checks and each array's hash. It also records a **post hoc** sensitivity
  check for sign counts. Those sign thresholds were not fixed endpoints.
- `plot_initial_update_diagnostic.py` generated the two-panel figure and
  complete 12-row table from that audit. Its provenance JSON binds the
  plot source, audit JSON, and PNG. The figure shows only update cosine and
  the SYNC/TIED update-norm ratio.
- `COMPACT_STAGE_MANIFEST.json` and `verify_compact.py` check the hashes and
  internal links of this compact transfer.

Across the 12 fixed rows, the first-update vector cosine is 0.641–0.857,
and the SYNC/TIED graph-update norm ratio is 0.517–0.784. The largest
analytic-to-actual coordinate error is below `9.32e-8`. These are
descriptions of the initial training update. The four graph settings are
not a sample of all graph tasks, and each graph uses one published split.

## Verification and recreation

Run `python3 verify_compact.py` in this folder to check the compact evidence
without installing PyTorch. It checks file bytes, the fixed 12-case matrix,
the diagnostic result and audit links, and figure provenance. It cannot
recalculate member gradients because their arrays are omitted here.

To recreate the arrays and their full audit, place the supplied source files
in the complete `optimizer_aggregation_prepared` study folder from the code
supplement, with the frozen `tuning.py`, `mechanism.py`, `models.py`, graph
data, and their two source/data freeze files. Use the documented Python
environment, then run the commands below in a fresh copy **without** the
existing result and raw-output files. The original runner refuses to
overwrite recorded evidence.

```sh
python initial_update_diagnostic.py check-freeze
python initial_update_diagnostic.py run
python export_initial_update_gradients.py
test -f initial_update_raw_gradients/manifest.json || cp initial_update_raw_gradients/MANIFEST.json initial_update_raw_gradients/manifest.json
python audit_initial_update_arrays.py --study . --output INITIAL_UPDATE_ARRAY_AUDIT.json
```

Compare generated hashes and all 12 rows with this folder. The raw array
files occupy about 33 MB and are retained in the author evidence. They are
omitted from this compact public stage to fit the supplement size limit.
Consequently, the supplied audit **records** an independent raw-array
recalculation, while this compact stage alone supports integrity checks and
recreation from the provided source and public graphs. It does not allow a
reader to rerun the array-level audit on the original arrays without
recreating or obtaining those arrays.

The conditional copy handles a filename case mismatch in the independent auditor:
the exporter writes `MANIFEST.json`, while that auditor reads `manifest.json`.
Their contents and SHA-256 are identical. On a case-insensitive filesystem
the copy is skipped.

The graph loader reads the complete published label vector to verify its
frozen checksum. The returned training bundle excludes test indices and
labels, and only training labels enter these gradient calculations.
