# Exact native qualification runner v2: rejection diagnostics

**Source only; unexecuted.** This is a separately bound diagnostic revision of runner v1. It changes no reconstruction, method, tolerance, native source, input provenance, schedule or trial budget. The original Squirrel failure remains unresolved and preserved.

`DIAGNOSIS.md` explains what the saved report establishes and what it cannot distinguish. `FAILURE_EVIDENCE_SUMMARY.json` binds the existing observation and its exact embedded Squirrel engineering report without duplicating the large report. `DIFF_V1_V2.patch` shows the source change.

The same explicit callable remains `runner.run_qualification(graph_name, output, device)`. It still admits only Squirrel17/Photo17, the authorized one-GPU route, fresh prescribed native warm states and disposable engineering checks. No predictive continuation or CLI launch is added. Input bindings/provenance are byte-identical to v1.

V2 retains strict equality. A rejection now names its recursive state path and supplies compact mismatch details in `exact_state_difference`, without saving new full tensor dumps. Diagnostic maximum differences never become an acceptance tolerance.

This packet contains no retry permission. Any later execution must use a new owned output directory and exact v2 source binding, preserve all v1 costs/failures, and run the complete existing checks. Passing a source AST check does not establish native-runtime correctness or repair the observed failure.
