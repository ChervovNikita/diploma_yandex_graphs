# Composed scorer: inherited extra-state repair

This is a separate, source-only successor to
`composable_normalization_runtime_v1/final_scoring`. The original runtime and
scorer remain frozen. Root owns staging, admission and execution.

## Concrete finding

The predecessor's `backend.model_fingerprint(adapter)` delegates composed
producers to `adapter.state_signature()`. Its inherited branch instead reads
`.shape` and `.dtype` from every `state_dict` value. The retained node
`interaction_readout` and NCNC `_interaction_readout` modules register an
aggregator whose `get_extra_state` returns a configuration dictionary. That
dictionary fails the inherited tensor-only branch before prediction.

The composed branch already has a recursive owner signature. The genuine
`ProducerAdapter.state_signature` uses the owner export guard and fingerprints
both model state and private BN banks through the original
`NormalizedAdapter._state_signature`. It supports the actual gate config's
string keys, finite scalar values and feature lists. Its representation and
ownership protocol are unchanged; this audit does not claim it has the same
broader typed-state semantics as the V5 helper.

## Exact repair

The inherited branch uses the exact `_state_fingerprint` implementation from
`final_selected_scoring_v5/backend.py`: recursive tensors, None, bool, int,
float, str, bytes, dict, OrderedDict, list and tuple; unsupported types reject.
The tensor shape/dtype/hash representation remains identical. V5's retained
checks exercise typed distinctions, signed zero/nonfinite bits, unsupported
objects, and gate-config mutations; those sources are pinned in `REPAIR.json`.

The composed branch still delegates directly to the same owner. All three
`score_fit` fingerprint calls and every other serving function are retained.
The contracts, role serving, native link helper, seal, package initializer and
dependency bindings are byte-identical. The source schema changes to V2 and
the expansion-root expression accounts for this sibling package; every
runtime and serving target still resolves to its original path.

## Focused verification

Run `PYTHONDONTWRITEBYTECODE=1 python3 -B check_source.py` in this directory.
The check verifies source receipts and unchanged functions, then uses one
scalar-only fixture containing the retained node and NCNC gate configs. It
reproduces the original dict/shape failure, verifies stable typed fingerprints
and mutation detection, and confirms the original composed owner accepts the
same configs. No model, tensor, array, dataset or checkpoint executes or loads.

This package does not rerun native qualification, rewrite a producer or
checkpoint, change a roster, grant campaign GO, or grant TEST access. Any
future use must explicitly bind this successor's `MANIFEST.json` under root's
existing authority. Existing native evidence remains attached to the source
that actually executed; this package claims source verification only.
