# Native empty-query kernel diagnostic source

## V2 observer serialization successor

V1's actual `exact` process stopped with observer `ValueError: Out of range float values are not JSON compliant: inf` while serializing a Python `inf` comparison argument in GCN normalization. It did not locate the native CUDA failure. Its receipt, supervisor receipt, first error, traceback, last16events, and sealed source remain preserved and pinned. See `OBSERVATION_AND_DELTA.md`; no kernel attribution is made from that observer run.

V2 changes four source lines: import stdlib `math`, then tag a nonfinite Python metadata float as `{"type":"python_float","value":"inf"}`, `-inf`, or `nan`. Finite metadata values, original overload arguments, exception propagation, synchronization, model/fixture/profile/seeds and limits are unchanged. `diagnose.py` and `LIMITS.json` are byte-identical to v1. Strict JSON remains `allow_nan=False`. The driver/receipt protocol schema remains v1; the new preparation manifest identifies this v2 source.

This is a separately released diagnostic candidate. Preparation and verification use the standard library only, including the extracted scalar-metadata serialization check. No native/GPU numerical work, dataset array, fit, VALID, TEST, or old checkpoint was executed or read while preparing it. Previous sealed sources and failed attempts are preserved.

`diagnose.py --help` is safe without Torch. The worker imports Torch only after source, failed-attempt, ordinary interpreter, device, and explicit root release checks. Execution is disabled in `ROOT_RELEASE.example.json`.

## Future root-owned execution

Stage this complete sealed packet at the admitted research root, alongside its existing authenticated source closure, sealed v1 diagnostic packet, original qualifier failure artifacts, and the five observer-failure artifacts listed in `SOURCE_BINDINGS.json`. Root creates a separate enabled release with the actual manifest SHA, an authorization reference, unchanged limits, and the exact case/output invocation. Each case receives a fresh child process and fresh output directory under `pooled_joint_native_empty_kernel_diagnostic_execution_root_20261004_v2`.

Use the pinned ordinary interpreter; set `PYTHONDONTWRITEBYTECODE=1` and `CUDA_VISIBLE_DEVICES=GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced`. `CUDA_LAUNCH_BLOCKING` must be absent. The existing admitted runtime makes its original deterministic transition after ordinary authentication; there is no kernel, server, interpreter, library, model-source, or profile change.

```text
/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12 -B <research_root>/pooled_joint_native_empty_kernel_diagnostic_preparation_20261004_v2/diagnose.py --case exact --release <separate_enabled_root_release.json> --output <fresh_released_output>
```

`ROOT_RELEASE.example.json` is disabled and lists only `exact`. Inspect its future first error and stage markers, then decide separately whether to release a control case. The four distinct controls remain disabled in `ROOT_RELEASE_controls.example.json`; none is automatically authorized by the exact-first template. The driver never launches a second case, retries, or continues numerical work after failure. A diagnostic receipt has `native_qualification_pass=false`, zero fits/evaluations, and `state_donor=false`. Completion means only that the selected diagnostic process completed.

## Interpretation and instrumentation

Read [DIAGNOSIS.md](DIAGNOSIS.md) for the observed failure, uncertainty, cases, and limits. `OPERATORS.jsonl` is flushed and fsynced at each event; `FIRST_ERROR.json` records the first observer error. `NATIVE_CALL_STAGE_SYNCHRONIZED` identifies each original native call, its parent, train/eval state, effective depth, and query-column count after a successful fence and immediately before that native call.

The observer temporarily wraps the native forward, `adjoverlap`, and `spmm_add`, restoring them in `finally`, and uses TorchDispatchMode to call each unchanged original overload. It observes shapes/dtypes/devices/sparse sizes/nnz only. The explicit synchronization and log I/O change scheduling and wall time and may change whether a timing-sensitive failure reproduces; they make no numerical substitution. Python source identities of the actual sparse operations and TorchDispatchMode are recorded by the future worker.

Static checks are not runtime qualification. A trace cap, process cap, source admission error, or observer error is not a kernel diagnosis. No analytical/portable empty oracle is implemented in this packet. Any native empty-input limitation must be stated separately from full-batch architecture feasibility and from portable empty-input correctness. Callbacks and scientific execution remain deferred.
