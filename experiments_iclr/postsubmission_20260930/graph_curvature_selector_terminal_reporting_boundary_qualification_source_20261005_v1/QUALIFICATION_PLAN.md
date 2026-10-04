# Bounded actual-boundary qualification for terminal reporter v2

Status: **source preparation only; unexecuted and runtime unqualified**. The source was parsed/compiled with Python standard-library tools only. No numerical package, prepared module, server, array, checkpoint or predictive result was imported, executed or opened during preparation. This packet changes no canonical source, scientific schedule, selection rule, frozen arm or endpoint and authorizes no execution.

This harness is suitable for the **first separately authorized continuation of a root-custodied actual initialization**, after the pending native warm/state/transport/selector/trial/runtime prerequisites are satisfied. It runs that full unchanged continuation once and retains its original `(best, metadata)`. Qualification instrumentation and disposable probes are charged. A probe failure is not permission to repeat the already completed primary trajectory.

## Exact input custody and entry

The caller supplies `initialized`, graph, TRAIN and VALIDATION objects; the canonical adapter returned by the frozen v3 verified loader; ordinary existing `trace` and `save_logits` callbacks; the normal reporting identity; and root-owned custody receipt descriptors. No dataset loader or preprocessing function is provided.

**Use the exact captured native preprocessing tied to the supplied actual warm/initialization state. Do not regenerate preprocessing separately.** The reported Squirrel preprocessing repeatability issue is handled by input custody, not by claiming bitwise reproducibility or changing deterministic policy. The harness fingerprints the supplied tensors at entry/hook/return and retains native object/version/device/storage metadata. It never reconstructs native features, teacher tokens or teacher edges. Caller receipt descriptors must bind captured preprocessing, actual warm state, graph/roles/compact labels and runtime/device to this very initialized state; declarations alone do not establish that custody.

Caller prerequisite fields are all required:

- `fresh_actual_native_warm_verified`, `actual_initialization_state_verified`, `continuation_RNG_verified`, `native_runtime_and_device_verified`, `authorized_VAL_provenance_verified`, `exact_captured_preprocessing_bound_to_warm_verified`, and `native_trace_and_save_callbacks_observational_verified`: true.
- `source_manifest_sha256`: `faace8c4ba59e3233a92809c3052c3dc02b37ddf77159c43e3573360818fbdf1`; `old_qualification_inherited`: false.
- `verification_receipts`: nonempty dictionary of exact root-owned verification descriptors, including source/runtime/device bindings and the actual captured preprocessing/warm/initialization pairing.
- `naturally_eligible_noncommon_arm`: report whether the supplied arm is naturally eligible and non-common, or state that coverage is absent; no manufactured eligibility is allowed.

Bind the real interpreter/package versions and authorized visible device inventory before calling. The canonical default RNG helper covers legacy NumPy global RNG, Python, CPU Torch and every available CUDA default generator; custom generators/unrelated backends remain outside scope. The harness synchronizes/draws only in that visible admitted inventory. An existing Python tracer needs separately reviewed integration; this source refuses to overwrite one.

## One primary continuation

Runtime entry (shown for review; it was not executed during preparation):

```python
best, metadata = boundary_qualification.run_once(
    initialized, graph, train, validation, trace, save_logits,
    canonical_adapter=verified_sources["integration"],
    report_root=run_root,
    arm_relative_directory=f"{graph_name}/seed{seed}/{arm}",
    identity=normal_v2_reporting_identity,
    root_custody=actual_state_and_runtime_custody,
    retain_native_result=retain_original_result_before_probes,
    record_attempt_accounting=charge_complete_attempt,
)
```

Load this harness from its manifest-verified source bytes using the root's ordinary verified-source loader. `retain_original_result_before_probes(best, metadata)` must retain the original successful primary result independently, tagged qualification pending. It is called before the disposable probes. `charge_complete_attempt(accounting)` must record all attempt costs, including failed probes and attempted receipts. The caller must additionally bracket the whole harness invocation in its own attempt ledger so pre-path/storage/bookkeeping failures and the root bookkeeping callback's own overhead remain charged. Neither callback may redraw, select another arm, or change native training state/RNG; preserve all null/failed/abstaining arms.

The main path composes the exact supplied `TerminalReporter` with read-only state witnesses and calls the exact bound `adapter.continuation` directly once. It keeps the entire native loop, cap, patience, strict selection and epoch-zero eligibility. It forwards every existing native trace and midpoint/selected save callback with the same arguments. Its exclusive `terminal_reporting_boundary_qualification/ATTEMPT.json` and best-effort `QUALIFICATION_FAILED.json` provide this composition's outer accounting. The ordinary v2 wrapper's attempt-marker and outer failure semantics are tested separately using an isolated copy of its exact AST; the main path does not claim to have executed that ordinary outer wrapper unchanged.

A Python trace restricted to the bound continuation records its actual pre-hook locals after stopping/selection. A read-only raw-model forward hook counts invocations. Both are removed after the primary call. Instrumentation adds copying, hashing, synchronization and tracing overhead; its total is not a bare-training benchmark.

Measurements and checks are:

1. Before constructor/loading and at continuation entry: complete CPU copies of live model/buffers and named Adam state, parameter/buffer/gradient identities/versions/storage metadata, gradient contents/presence, modes, aliases/groups/options/steps/links, specification/stage/member count, input digests and exact default RNG. Entry numerical/state values must match across setup, and RNG must equal the supplied continuation RNG.
2. At the actual terminal hook: the same state, existing native member logits and native raw mean, existing source NLL, completed/cap/patience/stop fields, selected `best` state and Adam snapshot/IDs/NLL, and midpoint status. No new model forward witnesses this endpoint. Native logits remain in memory for after-capture equality/version checks.
3. Around the actual supplied reporter: exact live state/Adam/gradients/modes/inputs/default RNG equality; member/served copy equality against the native witness; CPU payload storage isolation, including the native CPU logits when applicable. The actual native logits' bytes/versions remain unchanged.
4. After the actual hook and selected tail: successful default RNG restoration, selected live model and eval modes, **terminal live Adam/gradients**, selected `best['optimizer']`, exact returned best/metadata, unchanged input fingerprints, callback names/order, and measured primary counts: U updates, U+2 evaluations, 2U+2 raw-model forwards. Hook forward delta is zero.
5. Actual terminal artifact: load the newly created authorized `terminal_logits.pt` only during this later authorized runtime; compare saved member/served/VAL tensors, receipt tensor-file hash/size/identity and actual endpoint to the witness. Check served/member accuracy, NLL and class-summed Brier independently on the authorized VAL pack using float64 logsumexp and direct class-error arithmetic. Accuracy/support must match exactly; float64 NLL/Brier comparison uses the declared absolute 1e-12 engineering tolerance. These metrics do not select anything. The original reporter retains its mandated metrics; qualification summaries hash score scalars instead of presenting new predictive scores.

## Isolated exact-boundary tests

The source creates new test function wrappers around **unmodified AST statements** extracted from the manifest-bound hook and selected tail. There are no training-update statements in these extracted blocks. Each test namespace is a separate dictionary; neither the native module's globals nor source files are patched. Narrow dependency substitutions and probe-only filesystem path objects are explicitly labelled fault injection. The ordinary wrapper AST is likewise copied without editing its statements for failure-ledger tests. These tests are engineering fixtures, not alternative continuations or predictive endpoints.

Disposable models come from the actual post-primary architecture plus the witnessed terminal model/Adam/gradient/mode state, with the same supplied graph input. In-memory selected `best` is copied separately. Each probe starts from the appropriate exact terminal RNG, and the ambient primary post-return RNG is restored after tests. No old fitted state/certificate is loaded.

The paired tail test runs an observer-disabled selected tail and an observer-enabled hook plus selected tail from that identical actual boundary. The deliberate test observer mutates only payload copies and draws from all admitted default RNG inventories. The exact hook must preserve model/Adam/gradients/modes/inputs, native logits and all default RNG, and add zero raw-model calls. Each selected tail adds exactly one native evaluation. State, modes, gradients, Adam, RNG, `best` and returned metadata match exactly. Fresh selected replay logits use the already bound native identity tolerances (`atol=1e-6`, `rtol=1e-5`); the unchanged selected replay NLL guard `<=1e-6` also runs. This is a measured local replay check, not a claim of bitwise native forward reproducibility or full-trajectory numerical equivalence. A divergence is retained and resolved locally before considering complete training repeats.

Bounded failure coverage comprises fifteen cases:

| Case | Actual path/expected check |
|---|---|
| Constructor | Exact wrapper and real constructor reject invalid frozen identity; outer receipt, unknown endpoint. |
| Bound loading | Exact wrapper with explicit loader fault; original error and outer stage. |
| Initial RNG restore | Exact wrapper with explicit initial-restore fault; original error and outer stage. |
| Native before hook | Exact unchanged continuation on a disposable actual terminal copy; epoch-zero trace raises after one initial evaluation and before any update. Unknown terminal endpoint and outer receipt. This is an exception fixture, not shortened scientific training. |
| RNG snapshot | Exact hook with injected snapshot error; available terminal context/capture cost; no claimed restoration. |
| Capture | Exact hook with injected CPU-copy failure; original error and restoration attempt. |
| Observer plus restore | Exact hook with deliberate default-RNG draws, primary observer error and secondary restore error; primary exception object is preserved and FAILED restoration recorded; external cleanup is charged. |
| Restore after report | Supplied reporter succeeds, injected hook restoration fails; completed reporting is not mistaken for successful return. |
| Selected tail | Actual selected replay runs on a disposable boundary and selected callback raises; post-hook outer receipt. |
| Terminal tensor IO | Supplied reporter with probe-only path open fault; inner/outer receipts and original error. |
| Final report JSON | Supplied reporter with probe-only final-JSON fault; partial tensor and inner/outer receipts. |
| Inventory | Tensor IO failure plus probe-only unavailable directory listing; entire receipt fallback protects the original error. |
| Terminal directory collision | Existing test evidence is unchanged; only an outer receipt is created. |
| Exclusive marker collision | Exact ordinary wrapper path/open rejects existing attempt; previous marker/sidecar bytes are unchanged. |
| Complete receipt and receipt-write collision | Actual receipt helper handles unserializable identity/unavailable inventory; an exclusive write collision preserves the pending original error and existing bytes. |

Every probe artifact carries a test-only identity tag or resides exclusively under the probe path. Source-binding verification flags in injected wrapper fixtures describe that simulated branch and must not be promoted to scientific custody. The only numerical native probe evaluations beyond the primary are two paired selected replay evaluations, one pre-hook exception evaluation and one failing selected-tail replay: **four additional native evaluations, zero additional training updates**. Other probes use existing captured logits, state copies, reporting arithmetic and IO. All are charged. The first successful primary remains one of the authorized development continuations rather than an uncharged extra run.

## Costs, outcomes and limits

The saved success receipt reports entry/terminal state and logit digests, equality checks, actual stop coverage, source/review bindings and component wall/process CPU intervals. CUDA synchronization is included in measured component intervals. Host RSS and CUDA allocated/reserved peaks are read as **process-lifetime peaks**, without silently resetting caller counters; they include instrumentation and probes. They are not bare-training peaks or independent component maxima. Cold preprocessing, warm acquisition, selector/trial costs and prior caller peaks must remain in the caller's larger budget ledger.

Intervals nest and overlap; do not sum them. The saved `QUALIFICATION.json` timing boundary excludes its own final completed write. The required root accounting callback receives the elapsed wall/process CPU through that final write and its write component. Failure accounting includes attempted receipts up to the root callback. The caller's enclosing ledger covers callback bookkeeping time and unavailable durable storage. Failed cost/memory components are retained where available; source receipt helpers remain best effort and do not replace the primary exception.

Coverage belongs to the actual graph/seed/arm/stop witnessed. Run the harness on the first separately authorized representative continuation for each prescribed backbone; use distinct arm attempt directories and retain each primary result. A naturally absent Squirrel cap/patience path or non-common arm stays missing coverage; never force it, change patience/cap, invent eligibility or add a shortened predictive pilot. The first witnesses can establish hook/tail engineering for the exercised native runtime, not all thirty-arm feasibility or full-trajectory numerical identity. Original pending fresh native warm/transport/selector/gradient/trial/state checks remain prerequisites. Full training repeats are warranted only for a separately intended empirical full-trajectory claim, unresolved training-affecting entry/state/RNG divergence, or missing cumulative full-schedule feasibility/stop coverage—not for every reporting failure fixture.

This prepared harness itself still needs independent source review and authorized exact-runtime execution. Passing conditions above are a concrete qualification plan, not completed evidence.
