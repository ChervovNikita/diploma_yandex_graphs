# Independent PENCIL v3 source technical review

The exact candidate receives a **PASS for the source technical scope reviewed**. No concrete blocking source issue was found. This assessment binds manifest `45f84078f57fb788e14302aaedc37a98388ca7a120a8ff52d1a7c0d70a624cea` and seal `a33f3c51e7b539900708435b711b3f89589ecfef78d1d49111b4d4a95a988bd1`. It grants no execution authority, gives no manuscript acceptance assessment, and predicts no scientific quality or successful completion.

## Exact delta and preservation

Independent byte comparisons verify all 58 candidate payloads and all 49 v2 payloads against their manifests. Both manifest/seal bindings match the supplied and locally pinned identities. The candidate has the exact declared payload set, with files at mode 0444 and directories at mode 0555. Source hashes are unchanged after the focused checks.

Only `members_of_session` changes among existing supervisor functions. V3 adds `parse_proc_stat_rss` and `process_identity_with_rss`. After removing those two helpers and substituting the exact v2 sampler, the complete supervisor AST equals v2, including imports, globals, held-child identity, group signaling, cleanup, main, locks, waitid/WNOWAIT, physical receipts, collection and final budget logic. This is stronger evidence than comparing only the named ownership helpers.

`worker.py`, `common.py`, `data_adapter.py`, all 20 pinned native files, the official configuration, local non-weight Bert configuration, dependency binding, and data/runtime authority metadata are byte-identical to v2. PLAN differs only by the v3 execution namespace, RSS protocol disclosure, and v2 failure provenance. The existing scientific fields, workloads, caps, allocator policy, numerical precision and environment are preserved. The disabled release differs from v2 only in its plan digest. These checks establish preservation of source declarations and implementations; they do not inspect actual datasets, checkpoints or quality results.

The retained recipe is three fresh scratch seeds 0, 1, 2, each with all 20 native epochs, batch 1024, accumulation 8, 12 workers, AdamW, bf16/TF32, native 50% positive sampling and global negatives, complete sequential VALID after every epoch, native logit scoring and strict first-tie VALID Hits@50 selection. The worker retains the native loaders and loops; its existing early feature fusion, selective TRAIN/raw/VALID adapter, one DDP rank, pin_memory=False override, contemporary admitted dependencies, selected-only publication and historical TEST disclosure remain adaptations. No exact reproduction claim is inferred.

Caps remain 70 GiB allocated CUDA, 80 GiB reserved CUDA, 64 GiB sampled session RSS, 2 GiB output and 14,400 seconds per fit. `expandable_segments:True` remains a pre-Torch environment requirement. The existing source still charges native work and custody overhead and retains its finite final-publication tail disclosures.

## Sampler repair

At `supervise.py:41`, the new parser validates the PID, page size, comm envelope, required field count, known state, numeric identity/RSS fields and nonnegative values. Its indices correctly map Linux stat field22 starttime to index19 and field24 RSS pages to index21 after comm. It multiplies RSS pages by the runtime page size. Using the final closing parenthesis handles comm text containing spaces and parentheses.

At `supervise.py:58`, identity and RSS are read from the same stat text. At `supervise.py:66`, missing files are omitted, same-session tasks are retained, and changed owned groups remain visible to the unchanged strict group guard. Valid R/S tasks with RSS0 remain in membership with measured RSS zero; no requirement equates non-zombie state to positive or separately observable VmRSS. This removes the concrete fatal branch visible in v2.

The independent synthetic fixture extracts only these three actual source functions into a fake Path/OS namespace. It exercises R0, S0, Z0, positive S RSS, a different runtime page size, a comm containing newline/parentheses, disappearance, session filtering and changed group visibility. Negative/non-numeric RSS, mismatched PID, missing/truncated envelopes, invalid state and invalid page size are rejected. Exactly one stat read occurs per listed task, with no separate status read. No candidate module is imported and no live proc entry is read.

RSS remains a sampled sum of process RSS, with approximate kernel accounting and possible shared-page counting. A single stat read avoids the specific separate-status/separate-state inconsistency; it supplies no instantaneous OS limit or globally atomic snapshot of every task. V3 documents these limits. The fixture verifies source behavior on concrete snapshots; it does not prove real process lifecycle behavior or identify the original failing task.

## Ownership, cleanup and custody

At `supervise.py:79`, held-child identity retains birth/session/group checks. The child is created in a new dedicated session; exit is observed with waitid/WNOWAIT before any final reap. `kill_owned` at line86 checks the held leader identity and signals only its original owned group. The unchanged main and final cleanup preserve the stop path, cleanup deadline, zombie/group closure check, physical receipt before collection and explicit reap. Popen is held through os._exit so an unresolved child's destructor cannot silently reap its session leader. Unresolved cleanup prevents successful adoption and retains the active ownership record. The spent attempt remains consumed; no automatic retry or failed-state donation is introduced.

The source explicitly declares no process escape sandbox. Its session/group supervision and bounded cleanup should be interpreted within that disclosed ownership model. This review uses the exact source and recorded closure metadata; it performs no real lifecycle or signaling check and requests no additional qualification sequence.

At `common.py:100`, the admission gate still requires the exact authorized host/root, external APPROVED release and source/plan digests, exact workload/caps/allocator, independent exact-source PASS, separately admitted dependency inventory and pinned runtime/environment before Torch import. Final custody rehashes source, control, runtime and authorized input files. The disabled candidate has null root authorization and unresolved exact-manifest/review/dependency placeholders. Proposed commands are explicitly disabled and supply no execution client. A technical PASS cannot satisfy the separate root authorization requirement.

## Failure evidence and interpretation

Fourteen locally pinned repair inputs match their recorded hashes. The three copied v2 diagnosis/resource/root-monitor files match their pinned originals byte for byte. Their source manifest, release digest, worker PID and birth identity agree. The physical metadata records `RuntimeError: Owned live RSS observation missing`, owned SIGKILL, exit -9, physical session closure, direct-child reap and unresolved_cleanup=false. The recorded sampled RSS, wall time and final output size are below their respective caps; worker MONITOR_FAILURE is absent. These facts support the sampler assertion as the initiating recorded failure and -9 as the cleanup outcome.

The receipts do not record the failing PID/raw snapshot. DataLoader teardown is consistent with the eight counted TRAIN/VALID traversals and listed descendants, but it remains an attribution hypothesis. An unrecorded generic worker exception is not excluded by these metadata. No failed score/state, epoch quality table, stdout/stderr quality log or manuscript history was consulted. Historical completed resource evidence was considered only as pinned source/control context, without treating it as v3 scientific completion or a state donor.

## Findings and limits

Blocking source findings: none. Required source corrections: none within the reviewed scope. The unresolved failure attribution, sampled accounting, unsandboxed process-escape model, finite publication tail and untested real lifecycle are retained limits rather than invented source failures. No new resource-only run or scientific qualification ladder is proposed. Future release, scheduling and execution remain separate root decisions.

The reproducible checks and machine-readable exact bindings are saved alongside this critique in `FOCUSED_CHECKS.py`, `FOCUSED_CHECKS_RESULTS.json`, `SOURCE_BINDINGS.json`, `FINDINGS.json` and `REVIEW.json`.
