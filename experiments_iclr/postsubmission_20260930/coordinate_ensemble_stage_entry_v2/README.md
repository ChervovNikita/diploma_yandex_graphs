# Stage bridge v2 — source only

Minimal derivative of immutable stage bridge v1. Existing path/hash, route/GPU,
source/protocol, environment/cache, argv, timeout/process-group and complete-update
qualification guards remain unchanged.

Changes:

- Request schema is `coordinate-ensemble-stage-entry-request-v2`.
- Both acquire and qualify require `PHASE/coordinate_ensemble_runner_v3/runner.py`.
- Dataset keys are exactly `AmazonPhoto` and `CoauthorCS`.
- Modes remain only `acquire` and `qualify`; fit/test/preflight are rejected.

The v1 bridge README specifies the unchanged request fields and supervised
execution chain. Root supplies new independently frozen request/protocol/source
and protected-file hashes. `argv` must use the v3 runner path; qualification still
requires an explicit dataset/split/seed/arm/recipe and acquired data-manifest hash,
100 complete updates and eval_every10, with training-only evaluation.

This new source does not retarget any current Photo qualification or change any
v1/v2 source. No import, compile, test, script, download, SSH, or execution was
performed. Root owns runtime admission and measurements.
