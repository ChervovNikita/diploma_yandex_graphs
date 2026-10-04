# Independent bounded v3 technical source review

**PASS.** No concrete technical source blocker was found for the disclosed pin-memory deviation in this one-rank resource qualifier. This review authorizes no execution and establishes no CUDA root cause, performance equivalence, resource success, predictive result or manuscript acceptance.

| Exact subject | SHA256 |
|---|---|
| v3 source manifest | `211c140f7aa95e9af6395d841206160bb50aa8b20980b4b00d75a5a79a8291e6` |
| v3 source seal | `29789295827a485a0702854e2040a4abf112e0cc8846ff1adeeeac6b49b9e55e` |
| v3 plan | `6c423bc7a28f7acdab622c75472413716230db0a081bdc84e0a5929f0e229aa7` |
| exact V2_TO_V3.diff | `9257cc7100a3f6f2fbdd5d5fb7d31f16bf8e10953316fb16d7b6a39e45e4cafd` |

The reviewer is independent of the v3 author. Agent context is reused from earlier work in this project: `fresh_zero_context_review:false`. The existing v2 source assessment was authenticated and read as inherited evidence; this is a bounded v3 delta assessment, not a fresh-context academic paper review. No favorable verdict was requested and no additional agent was created.

## Concrete failure and repair

The six exact `failure_snapshot_v2/` files match the preserved local v2 execution records byte for byte. Progress records 7 TRAIN batches, 7,168 queries, zero optimizer updates and zero VALID batches/queries. The traceback reports `Caught RuntimeError in pin memory thread for device 0`, `data.pin_memory(device)` and `CUDA error: invalid argument`. It also explicitly warns that CUDA errors may be reported asynchronously. Thus the implicated API is recorded, while the underlying CUDA root cause remains undetermined.

The physical terminal reports worker exit 1, a closed owned session, direct child reaped, no cleanup errors or unresolved cleanup, and no stop or termination actions. Terminal/custody remain `FAILED_NO_RESOURCE_ADOPTION`; their physical/terminal hash links match. These records preserve the failed attempt and its cost. They do not establish a completed update, epoch or VALID traversal.

At `worker.py:159`, the byte-identical native `build_loaders` still constructs the original TRAIN/VALID loaders. Immediately afterward, lines 166–171 assert both native `pin_memory=True` values, set only the two `pin_memory` attributes to false and assert the effective false policy. The override precedes both native loops. The inspected native builder constructs loaders and returns them without creating a loader iterator. No alternate loader, collation function, sampler, generator, worker seed or dataset is introduced.

`worker.py:256` records the exact four-flag `PLAN.loader_pin_memory_policy` in the resource receipt. This makes the effective transfer policy visible in the scalar/metadata result. `V3_REPAIR.md`, README and the added protocol difference explicitly disclose the change in host-to-device transfer behavior and timing. Native device-transfer expressions remain unchanged. Disabling pinning bypasses the API implicated by the traceback; it is a concrete source repair to test under the existing bounded resource workload. It does not demonstrate that the asynchronous error originated there or that a later attempt will succeed.

## Exact verification and retained protocol

`SOURCE_VERIFICATION.json` records 232 independent passing checks. `SOURCE_CHECKS.py` uses only standard-library text, hashes, JSON and AST operations. All 73 v3 payloads (797,588 bytes), 22 declared external text/metadata inputs, the exact manifest/seal/plan and immutable 0444 file/0555 directory modes authenticate. All 26 candidate Python files parse as ASTs; no target code was imported, compiled or executed.

The v2 manifest/seal and all v2 payloads authenticate. The preserved v2 independent PASS, its manifest and three payloads authenticate. Its REVIEW.json SHA256 is `9ab8e021a1979b5795109914ac0e4195b86daa1dbf0e4364a10b42f7b1294fc4`. This review reuses that technical assessment for unchanged controls and scientific paths; it does not rerun prior QA or treat the prior PASS as a resource result.

The exact declared changed/added and unchanged inventories match the original bytes. The whole `V2_TO_V3.diff` was independently regenerated with unified diffs and matches byte for byte. Removing only the four new pin-policy statements and the resource policy keyword makes the entire v3 worker AST identical to v2. Thus native construction, training/evaluation call expressions, data draws/order, observation hooks, precision, optimizer/update logic and cleanup code receive no further runtime source changes. `static_check.py` has new source assertions supporting the repair; it was parsed as source and was not run as a target.

All 20 native Python copies match v2, their saved origins, SHA256 and Git blob identities at author commit `2d32e29dbed533288d9d758138e07547a0a7d8a9`. `supervise.py`, `common.py`, selective data adapter, official YAML, non-weight config, data/runtime authorities, dependency installation/admission templates, installed dependency binding and requirement/constraint closures are byte-identical to v2. The source plan matches v2 after removing only the loader policy, reverting its execution-directory identity and removing the single appended protocol deviation.

Consequently batch 1,024, accumulation 8, 12 workers, scratch BERT architecture, AdamW recipe, bf16/TF32, complete epoch0 and full 160,084-query/157-batch VALID resource traversal remain fixed. The same 7,200s/64GiB RSS/70GiB CUDA allocated/75GiB reserved/64MiB output caps, direct held numerical worker, descendant ownership/cleanup, persistent spent-attempt lock and fail-closed resource collection remain bound. No retry path, smaller workload, new precision, environment expansion or extra engineering ladder is added. The fresh v3 identity preserves the v2 failure as a separate record rather than making it disappear.

The existing external root gate still requires this exact source PASS, a concrete source/plan-bound root release and the already installed overlay admission. Both provided templates remain disabled. The unchanged selective data and resource-only controls exclude TEST, predictive metrics/selection, retained score arrays and checkpoints. State is discarded and is not a donor. The resource attempt does not complete the official 20-epoch scientific recipe.

## Limits and later controls

This is local source inspection with explicit context reuse. No network/server connection, package operation, staging/launch, numerical import, target execution, dataset/checkpoint read or source change occurred. The v2 runtime failure and physical closure were inspected through authenticated local metadata; server runtime/data/install binaries and current remote custody were not inspected. The installed overlay remains existing evidence, not a fresh compatibility qualification.

V3 is unexecuted. The review cannot establish model/operator/update/full-epoch/full-VALID compatibility, realized sampling equality, performance or numerical equivalence to pinned transfers, GPU root cause or resource sufficiency. Pin-memory disabling can change transfer latency and host behavior, so later measurements must be labeled with this disclosed policy. Inherited sampled-RSS, CUDA allocator and finite publication/exit-tail limitations remain. A physical exit alone is insufficient for resource adoption; the unchanged complete-work, cap, custody and collection checks still apply.

Before any future execution, root must separately bind this exact review, source and plan plus the existing dependency admission and authorized release. This technical PASS neither supplies that authorization nor fulfills a fresh-context manuscript acceptance review.
