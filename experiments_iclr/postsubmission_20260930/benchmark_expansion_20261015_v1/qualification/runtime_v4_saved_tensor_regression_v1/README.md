# Runtime v4 saved-tensor regression

Status: **passed source and saved-tensor contracts only**. Native v4 lifecycle
qualification remains pending.

## Inputs and execution boundary

The fixture bundle contains genuine saved VALID tensors from existing native
molecular, C6 node/molecular and interaction I3 node/molecular engineering
outputs. `ORIGINAL_INPUT_RECEIPTS.json` preserves original byte receipts;
`ACQUISITION.json` pins the 298,269-byte bundle. Its SHA256 is
`49b4f6f90d553c8f21cc1cadbc34f0fcd4954eca52c80ffe41dc917a2542384c`.
Acquisition used the guarded authorized allocation route and read saved
artifacts only. No new model construction, forward or update was performed.

`check_saved_tensors.py` invokes the actual runtime v4 Node/OGB normalization
and mean-export functions through explicitly simulated Python session/provider
views. Each callback returns saved tensors. Views contain only saved VALID
labels, with placeholders at other IDs. Remote source paths are projected to
matching local authenticated files; the original fixture metadata is retained.
Native return dictionaries are reconstructed from recorded same-call fields.

The selected epoch counter, dependency context and checkpoint byte receipt
are unit context. The fixture-file receipt is a byte-authority proxy. No native
source constructor, model inference, optimizer, selector, checkpoint load,
stop/resume or native selected export is executed. The simulated TRAIN entry
raises before native work and tests cache invalidation only.

## Passing checks

All five fixtures pass with exactly unchanged actual outputs, IDs, labels,
loss and metrics. The original ungated molecular path still has no mean.
C6 and interaction means retain the exact saved logits and native metrics
from one recorded callback. Original C6 callbacks may omit optional raw labels;
provider-normalized saved labels remain the authority.

The four mean cases refuse 76 mutations in total: wrong job/epoch/checkpoint,
runtime or extension bindings; altered actual IDs/labels/logits or mean logits/
metrics; older validation transactions; caches invalidated by load, TRAIN
entry or failed VALID; omitted fields; declared providers omitting all mean
fields; and changed provider methods. Missing molecular IDs use the existing
validator's `KeyError` refusal; other tested refusals use `ValueError`.
The passing attempt invokes 41 recorded tensor callbacks, with zero native
construction/forward/optimizer/TEST operations.

## Preserved attempts and cost

1. `attempt_01_fixture_integer_nan_failure`: the molecular test view tried to
   fill integer labels with NaN before any recorded callback. The original
   harness and the prior tool's failure/wall receipt are preserved. Corrected
   by using zero placeholders for integer labels and NaN only for floating
   labels, then restoring saved VALID values at their exact IDs.
2. `attempt_02_integer_fixture_corrected`: the harness expected `ValueError`
   for missing molecular IDs; the unchanged adapter normalizer raised
   `KeyError`. Source, stdout, stderr, cost and explanatory note are preserved.
3. `attempt_03_refusal_expectation_corrected`: passed after allowing that
   existing exception only for missing molecular IDs. No runtime producer
   repair was needed. The actual refusal exception types are recorded.

`COST.json` retains all known attempt costs. Recorded outer wall totals
2.119432833 seconds. Attempt 1 CPU/RSS were not captured and are reported as
unknown. Attempts 2 and 3 cost 0.906258 user and 0.164935 system CPU seconds;
maximum recorded child peak RSS was 201,506,816 bytes on macOS. The passing
attempt's internal check elapsed time was 0.028840333 seconds, excluding import
and process startup; its outer wall was 0.611466208 seconds. No unreported
failed-attempt tensor callback count is relabeled as zero.

`SOURCE_CHECKS.json` records the separate no-model AST/source review.
`READINESS.json` states the exact boundary. The source and evidence manifests
freeze this result without qualifying native v4 full-epoch selection,
interrupted resume, stopped finalization, export persistence or reporting fault
behavior. Earlier runtime checkpoints and lifecycle evidence are not migrated.
