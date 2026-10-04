# V2 device handling correction

Sealed v1 is preserved. Its `str(device).startswith('cuda:')` dispatch omitted the implicit forms `"cuda"` and `torch.device("cuda")` from explicit CUDA seed/RNG handling. V2 normalizes with `torch.device(device)`, detects `.type == "cuda"`, and resolves an implicit CUDA index using the current selected device. Scientific entry resolves once before allocating graph/features/model tensors.

The existing seed sequence remains Python, NumPy, Torch and CUDA `manual_seed_all` when CUDA is selected. Construction, RNG images and driver bindings now record type/index/canonical device. RNG restoration rejects missing CUDA state, unexpected CUDA state on non-CUDA, or mismatched selected-device provenance. State schema v2 records this changed image contract; no old image migration is provided.

`SOURCE_DIFF.patch` and `DIFF.json` record the source changes and predecessor bindings. The four-bank protocol, view construction, 10% sampling rule, schedule, loss coefficients/accumulation, pooling/selection, native200+2500 recipe and immutable neural bytes are unchanged. No fifth bank or reference implementation was added. `REFERENCE_IMPLEMENTATION_PLAN.md` is a separate future plan.

V2 stdlib tests use an explicit stub backend to check CUDA strings/device objects, seed dispatch, RNG capture/restore target and provenance mismatch. They exercise no real numerical package or CUDA device. The inherited numerical CPU suite must run again for this exact v2 source after root admission; real CUDA seed/state/provenance behavior also requires admitted hardware verification. Root's reported five passing v1 CPU tests qualify that exact historical synthetic source only, and are not cited as v2 qualification or predictive evidence.

No numerical/server/data/current-score access, scientific training or release occurred in this successor preparation. Actual fixed-mask/null coverage and the same-full-TRAIN references remain separate outstanding work.
