# Disabled paired bridge qualification V2 preparation

This is a disabled source successor, not a qualification result or execution release. `SOURCE_RELEASED=False` remains unchanged. V1 is preserved. No prepared source/helper was imported or executed; no numerical library, dataset, raw tensor, model, server, staging or launcher was used. Root and joint must review this exact delta before any separately authorized invocation.

The delta addresses two source-review obligations. Every query-logit cotangent now uses the existing `compare(..., gradient=True)` path and its unchanged `GRAD_ATOL=1e-8`, `GRAD_RTOL=1e-6`; its maximum discrepancy joins the derivative ledger. Value checks retain their original tolerances.

Terminal accounting now retains elapsed time, process peak RSS and cumulative CUDA peaks after cleanup, collects every fixed cap breach, and runs again in the terminal receipt path even when earlier setup, body or restoration fails. Independent restoration and resource-sampling failures are recorded. An active body failure is rethrown with its original traceback; cleanup failures and resource breaches are additional receipt fields. A resource-sampling failure cannot establish PASS. A hard termination still requires root's external watchdog and terminal closure.

`PLAN.json` and `SOURCE_BINDINGS.json` are byte-identical copies of V1. Mathematics, all tolerance values and resource caps, inputs/source pins, device, graph recipe, callback schedules and qualification oracles are unchanged. `CHANGES.diff` is the exact source delta. `STATIC_CHECKS.json` contains standard-library AST/diff checks only; it establishes neither exception-path runtime behavior nor numerical parity, memory use, resource sufficiency, fitting or predictive usefulness.

The fixed limits remain 1,800 seconds, RSS 8,589,934,592 bytes, CUDA allocated 79,456,894,976 bytes and CUDA reserved 83,751,862,272 bytes, with the separate 1,850-second root watchdog. The 284 native callback plan, original FP32/device, eight assignment steps, both live controls and target orientation off remain fixed. No retry, cap expansion or gate change follows from this preparation.

`DELTA_BINDINGS.json` binds the sealed V1 reviewer report and the preserved V1 source/manifest/seal. Root and joint review of the V2 source remains separate from that V1 evidence.
