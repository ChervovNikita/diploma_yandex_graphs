# Disabled V2 CPU CMCL fixture executor

V2 is a separate source successor to the once-only V1 engineering attempt. V1 remains FAIL: after four pre-Torch disabled-entry guards, Torch 2.1.2+cu118 metadata capture raised `AttributeError` because `torch.backends.mkldnn.deterministic` is absent. No analytic objective, serving, gradient, native callback or model fit ran. The child exited 1 and was reaped; the outer wrapper was FAIL. Original failure and publication receipts remain unchanged and are pinned in `SOURCE_BINDINGS.json`.

## Minimal compatibility delta

Only `cpu_backend()` changes. It records `mkldnn_deterministic_supported` with `hasattr` and records the observed attribute via `getattr(..., None)`. If unsupported, the setting appears explicitly as supported=false and value=null. That discloses unavailable coverage; it does not infer a deterministic value of false, claim determinism or set the backend. If supported, its existing observed value is recorded.

The full before/after `cpu_backend()==previous_backend` comparison is unchanged, so both API coverage and every available observed setting must stay unchanged. Byte and AST comparisons establish that every part of the executor outside this function is identical to V1. There is no new guard framework or supervisor.

## Unchanged scientific and execution scope

The six analytic functions, rational probability constants, closed loss and gradient expectations, exact owner masks, all invalid-input cases, fixed check order and original float64 ATOL/RTOL 1e-12 are unchanged. The 19 checks remain four pre-Torch guards, six analytic cases and nine invalids. The successful call bill remains six positive role-objective calls, one serving call, six ordinary gradient APIs and 13 rejections: 20 direct helper attempts.

`EXECUTION_PLAN.json` is copied byte for byte from V1. Helper SHA `4f04eb87381512c9f7506b6458917b83fd12228f3c7651c6291e902c99fe6ac9`, original fixture plan and helper math review remain pinned. The disabled native H16 callable is unchanged. No checkpoint, labels, predictions or native payload was opened.

V1's metadata failure is not a CMCL hypothesis rejection or numerical/mathematical result. V2 is unexecuted, disabled, and requires a new exact source review plus a distinct prospective root engineering invocation with immutable scope/output/caps and owned child exit/tail/resource closure. Its default root scope is unauthorized; prior V1 review/authorization cannot authorize V2. Utility H16 and native reference jobs retain priority.

## Static preparation

Source AST, JSON, pinned metadata and source hashes are the only checks performed. No prepared source/helper import, numerical fixture execution, staging, SSH, launch, fit or scoring occurred. V1 and other sealed predecessors, actual failures, the original helper and native callable remain untouched. An incidental read of a nonexistent predecessor COUNT_PLAN.json was harmless; the actual EXECUTION_PLAN.json was used instead, and the read error is recorded in COMPATIBILITY_DELTA.json.

Executor V2: 26,488 bytes, SHA `c0553fef2bed858e9e6672642c2216cc0e026cda82d852300caac88a34b69edc`.
