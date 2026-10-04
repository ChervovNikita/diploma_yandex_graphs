# Native-only Pubmed continuation control: independent source-v2 review

## Verdict

**PASS, scoped to the disabled source successor and the F06 correction.** The exact candidate manifest is `456a22a28420268e9ae47fd49f6ac322e36856b40a313340da4b31bde954ce7e`; supervisor SHA256 is `7cd4432523e5227575fb58934b4cbc9762a2a47e503904d0cc529bcc1233385d`. No execution, engineering qualification, state donor, scientific fit or replacement admission is granted by this review.

The independent verifier checked 17 payloads (169,412 bytes), all 94 source/scalar input rows (827,432 bytes), 34 predecessor/review preservation rows, all 14 active external source pins and five physical-helper/fix-provenance pins. There are 165 descriptor checks and 42 AST/preservation checks. Candidate Python was parsed as source; no candidate or dependency was imported, compiled or executed. Only local source, JSON, Markdown and diff bytes were read.

## F06 is closed in the declared signal scope

The v1 handler only appended a signal; its last loop read could occur before a late recorded signal, allowing `stop=None` at successful collection. The successor changes exactly four AST elements:

1. The nested handler declares `nonlocal stop` (line 232).
2. After appending, it immediately assigns `stop = stop or "SUPERVISOR_SIGNAL"` (line 234).
3. After restoring prior handlers, the finalizer drains the recorded list and applies the same stop assignment (lines 286–288).
4. The physical receipt copies `received_supervisor_signals=list(received)` (line 291); `terminal=dict(physical,...)` retains the evidence (line 316).

The handler, drain and receipt expression exactly match the separately reviewed census-v3 AST. The drain occurs after handler restoration and before constructing or committing physical evidence. The unchanged guard at line 309 requires physical closure, reap, `stop is None` and child status zero before `collect`. Any signal recorded by the temporary handlers therefore leaves a nonqualifying stop, including a signal arriving after the normal loop's final received-list read. Existing stop reasons remain nonqualifying if they replace or precede the signal reason; the signal list is retained independently.

Removing only these four additions restores the **entire** v1 supervisor AST. Every non-main helper is AST-identical, and the complete `collect` function source is byte-identical. Held `waitid/WNOWAIT` identity, original session/group-only cleanup, five-second cleanup window, guarded physical-before-output collection, exact output inventory/custody, spent-attempt lock, retained `Popen` and flushed `os._exit` path remain unchanged.

This proof concerns signals recorded while the temporary handlers are installed. After restoration, new signals use the prior dispositions. It is not a runtime signal test or a guarantee against ignored prior dispositions, uninterruptible kernel/filesystem failure, process escape or storage failure.

## Numerical work and observer remain preserved

The following payloads are byte-identical to the predecessor: `common.py`, `native_continuation_control.py`, `step_observer.py`, PLAN, both disabled templates, preserved shared4 summary and repeatability assessment. The 14 active dependency-source rows are unchanged. The reference/candidate/public-author native TRAIN functions have identical ASTs after function-name normalization. The fixed original comparator AST remains `608c9d0030b2b2393608a7438f3c25653d33bab909e8700daee16d27e16871c9`.

The substantive control remains one seed-0 freshly warmed native NCNC state: five complete native epochs (180 updates), then two independent complete epoch-6 restores (72 updates), totaling three factories, seven full epochs and **252 updates**. Every warm/continuation optimizer pre-hook binds the same monotone counter and refuses a started count of 252 before increment or Adam execution. The worker and parent both require exactly 252 begun/completed updates, seven begun/completed epochs, 72 pre and 72 post continuation observations, 190 warm and 76 continuation native-neutral observations, and 144 step-hook neutral observations.

The native numerical body, sampler defaults, iterator, TRAIN mask, dropout, Adam and 36 full batches per epoch (1,024 queries, 812-record tail) are unchanged. Complete state clone/restore, independent input clones immediately before each restore, pristine/saved/idle-copy checks and CPU Adam-step object/storage isolation telemetry remain intact. The narrow runtime reads only TRAIN and the authenticated raw feature projection, with one CPU `weights_only` load and no feature cast/normalization. No VALID/TEST, scores, external state file, shared4 facade or prototype is invoked by the active path.

The step observer remains byte-identical to the preserved shared4 diagnostic-v3: read-only pre/post hooks return `None`, take bounded detached CPU snapshots, check generator neutrality and release the 36 retained reference steps. Snapshot, transfer, hash and progress work can change synchronization and scheduling; source identity and RNG neutrality do not establish runtime transparency or identify a causal kernel.

The unchanged elementwise tolerance is `atol=rtol=1.52587890625e-05`, with reference on the right. Exact state/RNG/stream preconditions retain zero tolerance. Numerical mismatches become diagnostic entries and do not shorten either continuation; structural, runtime or resource failures can stop collection. COMPLETE means that the complete engineering control and telemetry were collected. It remains distinct from numerical agreement, engineering qualification or science admission.

## Preserved history and scientific interpretation

The original mistaken v1 PASS and its corrective BLOCKED review are preserved byte-for-byte. The corrective review remains the controlling verdict for source-v1; this review approves only the separately sealed source-v2. The earlier fixed-rule shared4 failure and its summary are unchanged.

The fresh native post-TRAIN state has different weights, parameter hierarchy, flags/RNG history and absence of VALID serving from the failed shared4 post-VALID context. With exact preconditions, a discrepancy establishes a native repeat difference in this control. Agreement is one bounded observation. Neither outcome attributes the old failure to a shared4 kernel, overturns the old failure, changes tolerances, admits a donor state or authorizes replacement scientific fits.

## Remaining release and collection obligations

The four original native qualification/supervision execution receipts are absent locally. Root must authenticate their actual bytes, hashes and required statuses before any release, together with raw-feature authority, runtime/author source, actual TRAIN/features and fresh execution path. The local feature-equivalence scalar receipt hash verified; this reviewer did not read feature or TRAIN arrays, interpreter/runtime binaries or server files.

Any separately authorized release must bind this exact manifest/review, unchanged PLAN/caps/invocation, source/runtime/input pins and the one fresh no-retry scope. The disabled templates and exact-review gate remain unchanged. Before interpreting a collected result, root must inspect physical/custody/resource receipts and the underlying exact-prestate, stream, saved-tree, idle-unit and alias evidence. A failed repeat-control precondition must remain distinct from a numerical discrepancy.

Caps remain 1,800 seconds, 32 GiB sampled host RSS, 70 GiB CUDA allocated and 75 GiB reserved, with setup, warmup, observation and collection included under the original guards. RSS/CUDA observations are sampled, not instantaneous OS bounds; there is no escape sandbox. Tensor payload bounds omit Python/transient overhead and rely on the overall caps. The final persistence/print tail is not continuously monitored. No runtime containment or numerical repeatability claim is established here.
