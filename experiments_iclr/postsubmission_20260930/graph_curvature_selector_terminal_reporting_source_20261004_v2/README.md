# Terminal reporting for the frozen covariance-initialization continuations

**Source preparation only.** No numerical package or prepared source module was imported or executed; no array, checkpoint, result or server was opened. This packet supplies an observation hook. It supplies no predictive result or launch authorization.

## What it records

The native continuation already evaluates the model after every update. Its last evaluated member logits are still available immediately before it restores the validation-NLL selected model. The separate source copy inserts one optional observer at that exact point. It adds no model forward.

The observer receives cloned CPU member logits and the raw member-logit mean computed on the native device. It saves `terminal_logits.pt` with those tensors and the already authorized VALIDATION node/label pack, plus `terminal_report.json` with served accuracy, NLL and Brier; each member's accuracy, NLL and Brier; actual updates; stop reason; source bindings; tensor hashes; and reporting cost. The artifact records the last completed update. It marks whether the native cap was reached. A Squirrel patience stop is not described as a full-cap endpoint.

The original native-device NLL scalar is retained separately. Additional metric calculations use CPU float64 from the captured FP32 logits, so their NLL can differ slightly from native FP32 reduction. These additional metrics do not replace the original checkpoint metric or the frozen mechanism screen. Accuracy uses the captured native-device pooled raw logits. Brier sums squared error over classes and averages over nodes.

## Source custody and unchanged decisions

The canonical `graph_init_training_adapter.py`, selector, driver, constants and outline remain unchanged. `SOURCE_BINDINGS.json` binds their exact bytes and the separate `graph_init_training_adapter_terminal.py`. `OBSERVATION_ONLY.patch` shows the only native-copy change: optional keyword-only observer/context arguments and one block before selected model restoration. `V1_TO_V2.patch` records this packet's failure-reporting correction against preserved v1.

`SOURCE_EQUIVALENCE.json` records a stdlib AST check. Removing those arguments and that block reproduces the complete original module AST exactly. Existing warm acquisition, training, evaluation, midpoint saving, strict checkpoint comparison, epoch-zero eligibility, stopping, selected replay and return values are unchanged. This is source reasoning, not a numerical replay test. Independent review and exact-runtime qualification remain required before scientific use.

The hook retains the same native Python, NumPy, CPU Torch and CUDA RNG snapshot/restore calls. After a complete snapshot it attempts restoration in `finally`, including observer failure. A snapshot or restoration failure is explicitly retained; it is not evidence that RNG restoration succeeded. A secondary restoration failure does not replace an already pending capture/observer exception. No model, optimizer or graph object is passed to the observer. An observer receives detached copies and scalar metadata; its return is ignored. The provided observer draws no random numbers and writes only inside the supplied run directory within the project research phase.

## Exact caller integration

Keep fresh acquisition and `driver.prepare_from_native_warm` unchanged. Keep every frozen arm and block, including null arms and failures. Use the root's existing verified-source loader to compile the exact manifest-bound `terminal_reporting.py` bytes; do not import an unverified cached module. Replace only the caller's `driver.continue_one_arm(...)` invocation with:

```python
best, metadata = terminal_reporting.continue_one_arm_with_terminal_reporting(
    initialized, graph, train, validation, trace, save_logits,
    report_root=run_root,
    arm_relative_directory=f"{graph_name}/seed{seed}/{arm}",
    identity={
        "graph": graph_name,
        "seed": seed,
        "source_split_index": (17, 29, 43).index(seed),
        "configuration": 0,
        "arm": arm,
        "validation_provenance": verified_validation_descriptors,
    },
)
```

`run_root` must resolve inside the same project research phase as this packet. `verified_validation_descriptors` must identify the caller's already verified role freeze, VALIDATION nodes and compact labels. The observer copies those descriptors and computes tensor digests; it does not establish acquisition provenance or open another label role. Selected and midpoint `save_logits` callbacks retain their original names and arguments.

`load_reporting_adapter()` verifies the bound source bytes and compiles the separate source under a distinct module name, preserving the original integration module. It restores the supplied original continuation RNG before entering the copied continuation. The wrapper returns the original `(best, metadata)` result. Reports are additional artifacts, not additional selection rules.

## Failures and costs

This preserved v2 successor addresses F1–F3 in `graph_curvature_selector_terminal_reporting_fresh_review_20261004_v1/REVIEW.md`. Training, strict checkpoint comparison, stopping, metrics, forward count and frozen sources remain unchanged.

Before constructor or bound-source loading, the wrapper exclusively creates `terminal_reporting_attempt_<sha256-of-arm-relative-path>.json` directly under the caller's run root. A caught failure after ownership is acquired attempts the matching `_FAILED.json` sidecar there. That record covers constructor/loading/initial RNG restore, native continuation, terminal capture/reporting, RNG restoration and failures after the observer returns. The marker is single-use; a collision is propagated without writing into the previous attempt. No retry or alternative endpoint is authorized.

Each observer is single-use. Directory creation is now guarded, and existing terminal evidence is never overwritten. A collision is retained by the run-level sidecar without writing into that existing terminal directory. For a directory created by this attempt, reporting errors preserve partial files and attempt `REPORT_FAILED.json` there. Both failure receipts retain safely available terminal metadata, capture costs, source bindings and availability/verification status. Endpoint fields not yet reached or unavailable are explicitly identified. Inventory failure records an unavailable inventory instead of replacing the original error. The entire best-effort receipt construction and write is protected, and the original exception propagates.

Invalid output paths, inability to create the run root or exclusive marker, and storage failure during receipt writing can prevent a durable receipt. No unsafe fallback location or overwrite is attempted. Those cases require the caller's existing run-level accounting; this source packet cannot guarantee storage availability. Native-runtime exception tests remain unperformed.

Capture wall/process CPU time includes endpoint metadata, RNG snapshot, CPU copies and the native-device mean, including a partial capture that fails. The successful report retains its observer-body timing boundary before final report JSON writing and RNG restoration. Failure sidecars additionally record observer costs, wrapper elapsed wall/process CPU before receipt construction, and available restoration costs. These intervals overlap and must not be summed as native-training cost. Failure-receipt construction/writing and successful final JSON/RNG restoration are not silently presented as fully measured end-to-end cost. No memory or full-schedule feasibility claim is supplied.
