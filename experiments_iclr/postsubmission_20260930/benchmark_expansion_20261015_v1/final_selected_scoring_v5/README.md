# Recursive selected model-state fingerprint

Scorer v5 is a narrow successor to frozen scorer v4. It repairs the core
`backend.model_fingerprint` used by actual `score_fit` before serving and after
VALID and TEST inference. The same function is used by the selected VALID replay.
No qualification-only replacement, metadata omission, checkpoint conversion or
producer change is introduced.

## Concrete failure and repair

`NODE_SELECTED_VALID_REPLAY_26.json` records the failed v2 node replay: one native
construction and one selected-state restore, followed by zero forwards, zero VALID
calls and zero optimizer updates. The failure was `dict has no attribute shape` in
v4's tensor-only fingerprint comprehension. The attempt used 7.310 worker seconds
and 7.976 supervisor seconds. Its evidence and scorer/harness sources remain
unchanged; the work must remain in the cumulative cost record.

The retained conditional aggregator implements `get_extra_state()` by returning
its configuration dictionary. Its strings, numbers, booleans and feature list are
legitimate selected model state and must participate in the immutability check.

V5 recursively fingerprints every `state_dict()` entry. Tensor leaves retain
v4's exact shape/dtype/byte-SHA256 representation. Other supported values carry
explicit type tags: `None`, Python bool/int/float/str/bytes, dict/OrderedDict, and
list/tuple. Dictionary keys and values, container kinds, and sequence/mapping
iteration order are retained. Float bits preserve signed zero and nonfinite
payloads; ordinary equality and JSON NaN handling cannot erase differences.
Unsupported custom objects are rejected with their type name, never dropped or
converted to a string representation. Fingerprints contain detached value
descriptions and do not retain mutable container references.

The original tensor hashing implementation is unchanged. This patch adds no
new tensor layout/dtype support beyond that implementation. Actual tensor/model
execution remains for root's bounded fresh qualification.

## Scope and source identity

`contracts.py`, `role_serving.py`, `native_link.py`, `seal.py`, `__init__.py` and
the complete external `SOURCE_BINDINGS.json` are byte-identical to v4. The
portable runtime, producer registry, training jobs, selected checkpoints,
selection criteria, role data and numerical tolerances remain unchanged.
The actual `score_fit` path keeps all three state checks and now protects extra
state as well as tensor entries. Inference error messages reflect that scope.

V5 retains v4's explicit BUDDY inference ownership and original Collab conversion;
the frozen v4 README documents those unchanged behaviors. `DELTA.diff.txt` shows
the code changes relative to v4. `REPAIR.json` pins the original source and failed
attempt. `REQUIRED_REBIND.json` records the prospective source rebind; no campaign
GO, roster, seal, registry or existing integration artifact was changed.

The source authority is `final-selected-scoring-source-v5`. Existing campaign
authority/result schemas remain unchanged. A future production use must bind
this exact reviewed manifest through its existing source/GO admission workflow.
`score_fit` still proceeds from VALID to TEST under campaign authority; the
bounded replay calls its exact components and stops after VALID.

## Checks and root handoff

Focused stdlib regression uses the actual retained gate configuration expression
for node and link settings. It verifies all config fields and a nested feature
mutation are detected, key/scalar/container types are distinguished, fingerprints
are stable independent snapshots, unsupported objects fail, and signed-zero/NaN
bits are preserved. AST checks establish unchanged tensor leaf hashing and all
three calls in the actual serving path. No fake torch module, mock tensor, model,
dataset or checkpoint is executed. Unchanged v4 checks are retained as provenance.

Root independently reviews the scorer and new
`qualification/final_selected_node_NCNC_VALID_v3` harness, publishes their exact
source closure, and admits one fresh corrective attempt with the unchanged
physical reservations and fixed tolerance. Old scorer v4, harness v1/v2 and the
failed attempt remain history. This source preparation launches nothing and
provides no numerical qualification result.
