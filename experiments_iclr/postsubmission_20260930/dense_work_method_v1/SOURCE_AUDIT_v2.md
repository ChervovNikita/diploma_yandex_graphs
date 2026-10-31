# Independent shared-message source audit v2

## Decision

**No confirmed static blocker to the canonical, bounded synthetic qualification.** The amended source closes the v1 audit's admission-chain, external deadline, synthetic memory-reporting and phase import-closure gaps. This decision admits an engineering test of those guards; it does not establish that the actual CUDA/PyG run passes.

Audited [shared_message_quality_v2](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_message_quality_v2/README.md>) against the earlier [v1 audit](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/dense_work_method_v1/SOURCE_AUDIT_v1.md>) and [bounded-supervisor audit](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/dense_work_method_v1/BOUNDED_SUPERVISOR_AUDIT_v1.md>).

- Manifest: `SOURCE_MANIFEST_v1.json` (v2 schema), SHA256 `6de209cd40e748e1a680786e6d1aa1f1230ab71d7995f558a62ade11310b0b98`.
- Bindings: `SOURCE_BINDINGS.json`, SHA256 `6a334b28bd0c4933f399e6acc9888462393caf6e0f5da58ba1ff4ff7a2691cf2`.
- Independently verified 16 manifest files, 61 dependency bindings and 10 Python AST/compile checks without executing their code.
- Exact local recursive import trees match the sealed policy: 10 v2 Python files, 10 engineering files including the legitimate legacy bridge, and 14 mechanics files.
- [Independent receipt](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/dense_work_method_v1/SOURCE_AUDIT_STATIC_RECEIPT_v2.json>).

I performed only stdlib hash/AST/tree metadata work and source inspection. No packet modules, Torch, NumPy, PyG or models were imported; no numerical/scientific run, acquired tensor, label, checkpoint, predictive result or remote connection was accessed. Author stdlib fixture results are sealed source-check metadata; I did not rerun them.

## Model and data semantics remain unchanged

The following files are byte-identical to sealed v1: models.py, counts.py, data.py, PARAMETER_PLAN_v1.json and __init__.py. Their source findings therefore carry forward unchanged:

- The shared message is full P times the mean of separately normalized member states, broadcast to retained private member states.
- Whole detachment stops self and cross-member neighbor-state derivatives together; learned neighbor/root/FFN parameters and the root-state route remain trainable.
- Exact supports packed/serial accumulation with one AdamW update; shared/detached refuse serial recurrent trajectories.
- Fresh seeds 17/29/43, source parameter transfer, keyed paired dropout, float32 raw-logit pooling and serial slice-before-pool behavior are preserved.
- Ordinary matched widths remain 66/65, with the same counts/mismatches and original upstream guards.
- The loader still uses the full acquired training-only graph/targets and exact IDs. It loads full forward+adjoint CSR for differentiable/exact/single arms and only forward for detached. No color CSR or held-out target field is introduced.

Synthetic numerical tests changed only by adding synchronized boundary callbacks and the v2 result schema. Their authored computations and assertions remain the v1 ones. They are still unexecuted in this audit.

## Synthetic evidence chain is now explicit

[synthetic_chain:156](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_message_quality_v2/contract.py:156>) resolves and hashes the exact synthetic request, revalidates that request's source generation and engineering scope, and requires RESULT to occupy the request's prescribed output directory.

It requires the completed inner supervisor's exact expected CLI, interpreter, operation, request and inner-output directory. The completed authorization must be on the corrected sole-A100 route with idle required, zero exit, source-bound route/authorization/logging/wrapper/script and unchanged command/environment/log evidence.

The completed outer TERMINAL is linked to its START and the same request/inner command; it must report zero exit, complete, unchanged request, no timeout/interruption and bounded finite elapsed time. Every inner evidence hash/size must agree with the outer record.

RESULT and its own START must agree with the request/source/live hashes. RESULT, request and real admission must use the same GPU UUID, exact runtime policy and equivalence tolerances. The reported observed runtime must equal the freeze. Synthetic peak memory must satisfy both synthetic and prospective real limits. The exact START/RESULT/artifact-manifest output set and file hashes are checked; copied/mixed output paths or extra artifacts are refused.

Finally, RESULT's source tree must match a fresh sealed-tree check and each recorded imported phase module must match the source allowlist. These checks directly address the previously independent RESULT-versus-supervisor acceptance. In the canonical source-generated path, the exact code is responsible for emitting the actual imported-module list; the audit does not treat a bare passed-check flag as numerical evidence.

## Live outer deadline and CLI binding

[main:258](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_message_quality_v2/qualify.py:258>) bootstraps the request-bound manifest/import guard before importing phase packages. It then validates the request and exact supervised CLI, consumes GNNM_BOUND_START_JSON, and requires the bounded wrapper's cap to equal the request's wall cap.

[outer_start:102](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_message_quality_v2/contract.py:102>) checks the complete argument-list chain, request SHA, actual inner/outer directories, corrected route/repository/phase, session mode, grace and current source hashes. The bounded wrapper source is the independently audited `b096966c1570da0a998dbb9fb464a9ad71490b74d4ba3406b751390637d19a23`.

Cooperative Budget now begins at the outer START time. The independent process-group parent supplies termination of a long/hung inner call. Success remains provisional until its completed outer envelope and inner chain are audited. The documented timing scope distinguishes group deadlines, post-group evidence hashing, final caller-accounted report overhead and unavoidable scheduling/reaping delay; it makes no literal OS absolute-time guarantee.

## Synthetic memory measurement and failure handling

[qualify.py:294](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_message_quality_v2/qualify.py:294>) resets GPU peaks, synchronizes before each boundary, checks wall/disk budget, checks reserved-memory cap, and records boundary peaks. Boundaries include before/after the checks, every named assertion, and defined ordinary/checkpoint/serial training completions.

RESULT records final allocated/reserved peaks, the exact cap, boundary records and the explicit post-operation enforcement scope. A cap breach raises and preserves FAILURE metadata/observed peaks when CUDA is initialized. It is accurately described as peak feasibility checking after operations, not an external hard memory limit.

Real input H2D peak memory is also checked/recorded before variant peaks reset. Other real training/measurement behavior remains the same.

## Import closure and actual runtime integration boundary

[import_guard.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_message_quality_v2/import_guard.py>) checks known manifest/dependency bytes, verifies an exact recursive importable tree, refuses bytecode/extensions/extras/shadow directories/cache directories/symlinks, strips unrelated cwd/script/PYTHONPATH source paths, and installs a source-only finder for the three declared phase packages. Each admitted phase module compiles its freshly hashed source without reading or writing bytecode.

The formerly unbound engineering_entry_bridge_v2.py is now explicitly pinned and allowed, while not required by qualification. No unbound local phase importables remain in these trees.

The installed third-party environment is version-frozen rather than byte-sealed, as the README states. **Root reports the actual frozen PyG is 2.7.0, which may generate and import native SAGE template modules under TMPDIR.** Since TMPDIR is phase-confined, imported_modules rejects such generated files unless policy declares them. This is a concrete integration qualification risk for the genuine synthetic run. It is not evidence of an observed failure here or a claim that source approval includes runtime success.

If that run fails at this guard, preserve the v2 request/output/inner/outer evidence and make a precise versioned template/generator-provenance amendment while keeping the method and tolerances fixed. Do not silently loosen the source allowlist or treat the failed run as completed qualification.

## Prospective path-selection delta

[qualify.py:191](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_message_quality_v2/qualify.py:191>) now records train+validation/5 per sample and selects by its median, then variant ID. This prospectively matches the root design's 300 training updates and 60 validation passes per fit, while retaining raw train/validation/combined samples. It corrects the v1 1:1 engineering-cycle selection caveat without using quality outcomes.

The complete scientific forecast still must charge all 9,000 updates, 1,800 validations, transfers, artifacts/hashes, selected/final replay, trained timings, setup and failed work. Source approval and a passed short profile do not admit that schedule or establish its full resource budget.

## Required next evidence

A genuine source-bound synthetic CUDA run through the canonical bounded/idle wrapper is appropriate. Its completed request/output/artifact/inner/outer/runtime/import/memory chain must pass before real training engineering can be admitted. Complete training engineering, the scientific runner, validation-only exporter and whole-study resource admission remain separate required work.

