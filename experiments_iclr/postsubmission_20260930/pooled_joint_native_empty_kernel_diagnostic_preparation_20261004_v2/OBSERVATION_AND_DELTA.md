# Observed failure and minimal v2 delta

The preserved v1 exact run has `ERROR_OBSERVED_NO_QUALIFICATION`, `ValueError`, and message `Out of range float values are not JSON compliant: inf`. Its traceback enters native encoder/GCN normalization and reaches `deg_inv_sqrt == float('inf')`, then `metadata(args)` and `json.dumps(..., allow_nan=False)` in the observer. The comparison's Python infinity sentinel is a legitimate original argument; v1 failed to encode it. The traceback does not identify the original native CUDA error.

The first-error event is the enclosing `unchanged_direct_native` call propagating that observer exception. It must not be relabeled as a kernel error. Last16events preserve the last synchronized power operation and the failed-row sequence gap. V1 source, receipts and exception bytes are pinned unchanged in SOURCE_BINDINGS.json.

## Literal runtime code change

`operator_trace.py` adds a stdlib `math` import and this branch before its existing primitive return:

```python
if isinstance(value, float) and not math.isfinite(value):
    return {"type": "python_float", "value": str(value)}
```

The comment plus these additions total four source lines. The allowed tag values are the strings `inf`, `-inf`, and `nan`; finite Python floats, including signed zero, use their original return path. Nested metadata lists/tuples/dicts recurse as before. These are descriptive metadata copies. The original `function(*args, **kwargs)` and same-overload dispatch remain literally unchanged. No exception suppression, kernel fallback, numerical replacement, clipping, precision conversion, empty-input skip, profile edit, or fixture/seed change is added.

`OPERATOR_TRACE.diff` is the exact unified diff. `CODE_DELTA.json` binds both observer sources and records that `diagnose.py` and LIMITS.json are byte-identical. The static checker verifies the complete literal replacement, so all remaining source bytes and original exception paths must match v1. It separately extracts only the scalar metadata function AST for a stdlib serialization check using a type-classifier stub; no Torch/tensor/sparse/native/kernel behavior is tested or qualified.

## Fresh attempts and interpretation

The main disabled release lists only `exact` in execution_root_v2. Four later disabled controls remain distinct fresh-process invocations. V2 authenticates the relevant old v1 source bytes, manifest and observer-failure bytes as extra custody, while retaining the same original native qualifier failure and runtime authority as v1. Static checks verify the entire old seal. Root stages those metadata artifacts at their bound relative paths and decides on any execution separately. Protocol schema names remain v1 because the driver bytes are unchanged; preparation/output roots and manifest identify v2.

No future result is supplied by this correction. An actual scientific/kernel error still propagates through the unchanged first-error and worker logic. Diagnostic completion/error never grants native qualification PASS, data/fit permission, donor status, or scientific superiority. No server dispatch occurred during preparation.
