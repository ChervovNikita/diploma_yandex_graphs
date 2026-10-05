# Independent static review of the disabled sequential scientific worker V1

6 October 2026. **Verdict: BLOCKED_PRE_W_IDENTITY_ORDER. One unresolved blocking source defect.**

## Exact scope

Reviewed `amazon_learnability_responsibility_sequential_train_only_execution_preparation_20261006_v1/six_arm_worker.py`, SHA256 `440c06005a85f116cd1dd7ece2e96daeaec17852b7509d994e499faab0000ee5`, under packet manifest SHA256 `8309188146b1702c4a646364ce44c90c8a93582e304dfb99d3780b16b4e9b701`. This is a source review of the fixed native caller and prerequisite interface. It does not verify any actual synthetic or full-FP32 result, authorize fitting, establish memory feasibility or release the held-A evaluator.

The worker source was read statically. Standard-library AST/JSON/byte checks confirmed the specified pins, relevant new packet files, anticipated flag-only successor hashes, V3 copied-file identities and unchanged named construction/commit/permutation helpers. Historical dependencies were not re-reviewed wholesale. No prepared source was imported or executed; no data, scientific-result, logits or checkpoint payload was opened; no SSH, fitting, persistent model update or A/VALID/TEST access occurred. Author sources remain unchanged.

## Blocking finding

**B1 — Certificate graph/role/native-edge identity is checked after W labels are opened.**

`run_six` calls `accessor.load_public_w` at worker line544. That reader decodes `W_LABELS.npz` at accessor line332 before returning public provenance. The certificate comparison occurs at worker lines548–551. `_prerequisites` (lines279–370) validates an actual all-six certificate and recipe/source/device claims, but receives no `public_b_dir` and performs no comparison against the actual scientific projection or processed edge identity. The event at line531 therefore establishes certificate admission before W access, but does not establish matching scientific graph/role identity before W access.

A certificate with otherwise admitted PASS fields and differing graph, roles or processed edge hash is rejected only after W-label decoding. No acquisition update would precede that rejection; nevertheless it violates the explicitly fixed pre-W identity requirement in this review assignment and `RELEASE_PLAN.json` (`all_six_full_FP32_before_any_W_access_or_fit=true`).

**Required repair:** prepare a distinct disabled successor that verifies the actual public projection roles/source and processed native-edge identity without reading W/S/R/A labels, compares all three identities with the certificate, and only then calls `load_public_w`. Preserve the existing W-only scientific recipe. Re-review the repaired source and rebind its flag-only successor pins before any release. This V1 review cannot supply a zero-defect scientific-source admission row.

## Source checks that passed within this scope

- Fixed G0 values, seed17/split0 native constructor/device/reset/wrap order, unchanged W-only acquisition math and Adam defaults, 200 local then200 global updates, six serial H16 arms in fixed order, fresh reload of one common400 state, and the frozen within-class hash permutation are preserved. Compared named V3 helpers are AST-identical; `_warm_step` adds only attempted backward/optimizer accounting.
- Original warm diagnostic, accessor and held-A evaluator copies are byte-identical to V3. The diagnostic remains16 complete member callbacks,8 private-gradient bodies and10 pair primal maps, with no committed update. S/R labels are opened after immutable common400 storage.
- Continuation calls the reviewed public `sequential_native_episode` signature. The original operator/port retain exact hashes and false source/process guards; only their fixed pure callback/operator/pair helpers are used. Every arm reloads the common state, checks complete finite commits and dormant heads, then writes an immutable endpoint.
- Future worker/accessor/warm/sequential enabled hashes are exactly the in-memory `SOURCE_RELEASED = False` to `True` byte delta. No enabled file was created. The preparation and admission placeholder are disabled. The worker neither imports nor calls the held-A evaluator, whose guard remains false and which is absent from the proposed release list.
- The actual-certificate interface rejects a oneLIVE receipt: it demands all six controls, full24492x300/five-class/four-member FP32 context, exact source/device claims, original-phi commit qualification, finite/feasible states and diagnostics, restoration, zero fitting/updates/scoring, measured per-arm counts, immutable raw evidence and positive actual whole-process resources. Independent candidate and scientific-source evidence plus fixed scientific limits and an external watchdog are required. Actual evidence remains a root-owned prerequisite and was not inspected here.
- A transparent native callback wrapper independently counts physical calls, restores the original attribute in `finally`, and retains elapsed callback time even on failure. Original warm private/map bodies are counted by restored passthrough wrappers. The derivative/map attempt ledger uses the reviewed sequential `Counters.add` path; it is separate from the independent physical-forward counter, rather than a second independent autograd observer.
- Queue arithmetic gives continuation4096 forwards/private2112/native-VJP832/Q-primal2560/Q-VJP640/small-query-VJP96. Acquisition1600 plus warm16 plus continuation4096 gives5712 fitting-phase forwards; seven served states reserve28, yielding the prospective5740 total. Serving is reserved and is not executed here. Ordinary acquisition backward/Adam attempts and successful updates/episodes are recorded separately.
- Attempt events are appended before physical callbacks and staged operations. Python `BaseException` failure retains phase/arm/episode, attempted counters, resources and completed immutable states; no replacement or shortened run follows. Hard termination leaves the closed-per-event JSONL stream and no COMPLETE record, with the required external watchdog responsible for a termination receipt. Worker time/RSS/CUDA checks occur before warm updates/episodes and completion; source review supplies no runtime resource pass.

No other concrete blocker was found in the reviewed source scope. Repair B1 before admitting this scientific worker. Preserve this sealed V1 finding when reviewing the successor.
