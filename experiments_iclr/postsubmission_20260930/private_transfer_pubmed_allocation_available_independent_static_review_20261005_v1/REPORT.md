# Independent static review of disabled allocation Pubmed input verification

5 October 2026. This review identifies four P2 source defects in the combined disabled input-verification, owned-supervision and client workflow. They concern child cleanup, authoritative wait4 accounting, executable source closure and the inspector's declared soft bound. No input/model/numerical program was run. All root, input-verification and scientific admissions remain absent or disabled.

The reviewed inputs are:

| Packet | Supplied and authenticated pin |
| --- | --- |
| `private_transfer_pubmed_allocation_available_preparation_20261005_v1` | MANIFEST `26d512dbaa5faf97185aa347cb349fc333f652d74819dbfa85ad489db4f88cb0`; SOURCE_MANIFEST `f77ce5821522a4ee3a472c636a0f597ed28e814e36bd8e0152d0b0f0138d3322` |
| `private_transfer_pubmed_allocation_available_supervision_preparation_20261005_v1` | MANIFEST `059cf0b1c6df86967d025f79d938bdccca5c71230bca4f7f1c836ae63efbdc33`; actual `owned_supervisor.py.txt` source `f936dd02dbf3978ca1642667f237a9ef8f81c61997755f1d3074cf51deb4e234` |
| `private_transfer_pubmed_allocation_available_client_preparation_20261005_v1` | MANIFEST `f34f387c64ef7c3003cae47b22213a9065617ab03187cc6d2b2beafca401a356`; actual `execute_once.py` source `a894b5aa57284db2788af4dc6f500befd41520bcde4f686e58241f6bd257af69` |

The supplied supervision “remote source” pin denotes the complete `.py.txt` body, not that packet's separately hashed SOURCE_MANIFEST.json. I authenticated all 10 input-packet, five supervision-packet and three client-packet manifest entries, the four combined source-manifest entries, both source-manifest files and applicable seals. All seven declared input bindings also match, including the preserved inspector and the preserved allocation geometry `execute_once.py` from which the helper originated. No pin mismatch was found.

Five actual new source bodies were parsed as AST data: `available_custody.py`, `extract_available.py`, `inspect_available.py`, complete `owned_supervisor.py.txt` and the new client's `execute_once.py`. Eight independent AST comparisons confirm the preserved inspector helpers and the inherited supervision helpers after only the declared command/signature/scope-metadata adaptations. Source preservation is evidence of lineage, not evidence that inherited supervision is correct.

The work stayed inside `postsubmission_research_20260930` and used saved source/protocol/recipe text, stdlib hashes/JSON and AST inspection only. No target or numerical code was imported or executed; no checker, fixture or test was run. No archive, data, feature, history, checkpoint, score, prediction or outcome payload was opened. No network, remote, MacLink or 18.77 contact occurred. No agents were spawned. Only this new review directory was written; all existing sources and protocols were preserved. INPUT_BINDINGS.json and SOURCE_BINDINGS.json identify the exact reviewed bytes. CHECKS.json records authentication, AST and manual control-flow checks. This is a source audit, not execution or scientific approval.

References below use `input/` for the input packet, `supervision/` for the supervision packet and `client/` for the client packet named above.

## Findings

### F1 — P2: an exception after Popen can abandon the owned child

Locations: `supervision/owned_supervisor.py.txt:75–123,157–159`.

`run()` starts the stage as a direct child with `start_new_session=True`, then performs physical-process reads, START writing, output scans and CURRENT_METADATA writes. There is no `try/finally` or exception handler around the live child that ensures termination and authoritative reaping. The termination/reaping code at lines110–122 is reached only through the normal `failure` branch.

For example, an I/O exception from `output_bytes()` or writing CURRENT_METADATA, or a KeyboardInterrupt while the loop sleeps, exits `run()` before that branch. The outer BaseException handler writes EXECUTION_FAILURE and reraises; it does not terminate or reap the child. Since the child has a new session, the source provides no guarantee that exiting the monitor stops it. The normal900-second/time/RSS/output polling then ceases. This is a concrete exception path in the source, not a claim that such a fault occurred.

Add exception-safe owned-child cleanup that preserves the original failure, signals only the still-owned unreaped child, and obtains or explicitly records terminal accounting before the supervisor exits. Cleanup and failure-receipt errors also need a safe fallback; a failed receipt write must not skip child cleanup. Retain partial files and the no-retry policy.

### F2 — P2: Popen signaling can consume the terminal before wait4

Locations: `supervision/owned_supervisor.py.txt:80–88,112–119`; preserved predecessor remote helper in `private_transfer_pubmed_train_geometry_execution_root_20261005_v1/execute_once.py`.

The supervisor intends `os.wait4(child.pid, ...)` to be the sole reaper, so that final peak RSS and exit status are authoritative. It checks `reap()` and then calls `child.terminate()` or `child.kill()`. Normal POSIX CPython Popen signaling calls `send_signal`, which polls the child to avoid signaling a reused PID; that poll can reap an already exited child through waitpid.

There is therefore a race if the child exits after the explicit nonblocking wait4 probe but before Popen's internal poll. Popen can consume the terminal, leaving the supervisor's local `code` and `usage` unset. The next custom `reap()` then has no child to wait4 and can raise ChildProcessError, losing the authoritative resource usage and the intended stage terminal. The comment that no poll/wait reaps this child does not account for these internal Popen calls. This race was not exercised in this audit, and no prior numerical or geometry result was inspected or invalidated.

Keep one authoritative reaping path. For a direct child whose PID remains reserved by the unreaped kernel relationship, use a signaling path that does not internally poll/reap, handle the exit/signaling race, and preserve wait4 ownership through terminal accounting. This should be repaired together with F1; using Popen polling as a cleanup fallback would retain this accounting problem.

### F3 — P2: unlisted source files are staged and can execute before child authorization

Locations: `client/execute_once.py:54–62`; `supervision/owned_supervisor.py.txt:19–33`; `input/extract_available.py:2–8` and `input/inspect_available.py:3–9`.

The client validates files listed in SOURCE/MANIFEST.json, then separately iterates every regular file in SOURCE and stages it. The second loop does not require membership in the authenticated manifest and does not reject symlinks. The packet manifest itself is not compared with the supplied immutable packet-manifest pin. The remote accepts every relative path whose first component is `source`, and verifies the three files named in the fixed source manifest without rejecting extra files.

As a static counterexample, an added unlisted `json.py` or `tarfile.py` can be staged while all fixed SOURCE_MANIFEST hashes and all existing manifest-listed files remain unchanged. Child Python puts the script's source directory on its import path. The input programs import these standard-library names at module scope, before their call to `available_custody.authorize()`, so an extra module can run outside the reviewed executable closure. No such file was created, imported or tested in this audit; the counterexample follows from the two staging loops and ordinary Python import resolution.

Stage an exact authenticated inventory, bind the packet manifest to its immutable pin where it is used, reject unlisted files and symlinks, and enforce the same exact inventory remotely. Copying only the reviewed executable sources, source manifest and explicitly needed reviewed metadata is also sufficient if that precise inventory is authenticated. A source-manifest hash proves its listed members; it does not by itself exclude extra executable files from the import directory.

### F4 — P2: the inspector does not enforce its declared600-second soft limit

Locations: both disabled input recipes' `bounds`; `input/available_custody.py:74–75`; `input/inspect_available.py:130–158`; `supervision/owned_supervisor.py.txt:142`.

Both stage recipes declare and the authorizer requires `soft_seconds=600` and `hard_seconds=900`. Extraction checks the600-second limit at several points and before its acquisition manifest. The inspector has no elapsed-time measurement or soft-limit check at all. It can therefore finish between600 and900 seconds, write a passed inspection receipt and be accepted by the supervisor, which enforces only the900-second stage limit.

Either enforce the frozen inspection soft limit across authorization, runtime/semantic inspection and receipt production, or explicitly change the source/recipe/review contract before release. The declared fixed bound should not silently be unused in one stage. This finding does not request a longer limit, shortened semantic inspection, partial pool scan or altered population.

## Properties verified in the source

`input/available_custody.py:51–95` checks Linux, exact `anogena-2-0`, project cwd, explicitly empty CUDA_VISIBLE_DEVICES, and exactly the sole declared GPU UUID. It confines source/recipe/evidence paths, rejects traversal and deliberate-path symlinks, authenticates the source manifest and every listed executable source, and requires separate root-review/owned-supervision approved evidence references and their exact hashes. It requires the input-only purpose and false TEST/model/fit/network/retry flags, the four exact archive members, exact archive identity, fixed bounds and exact execution identity. These checks precede archive authentication/extraction or input semantic reads in the ordinary reviewed path. F3 limits the executable-import closure preceding the authorizer.

`input/extract_available.py` authenticates the full resident archive's fixed SHA256 and size, uses streaming `r|gz` traversal, and copies only four exact allowlisted TRAIN/VALID/feature member names into a fresh confined output. It uses no extractall, rejects duplicate/nonregular/oversized required members, creates output files exclusively, verifies per-member size/hash and exact four-member coverage, and preserves failures/partial files. There is no TEST extraction or TEST-value decoding. Whole-archive hashing and gzip traversal still consume archive bytes; “no TEST” here describes member extraction/semantic access, not a claim that compressed skipped-member bytes are never read.

Extraction records positive counts explicitly as pinned contracts, sets input_semantics_inspected=false and training/fit admission false, and leaves measurement to the separate inspector. It does not claim feature exporter semantics. The inspector's original `sha256`, `read_rows`, `load_available` and `inspect_available` helpers are AST-identical, and its four member hashes and declared node population are unchanged.

The inspector authenticates all four file identities before decoding values, requires weights_only=True with CPU mapping and no pickle fallback, finite strided FP32 supplied feature tensors, exact TRAIN/VALID row counts, no duplicate positive facts or self loops and disjoint positive populations. It checks the complete native fixed500 pool's shape, dtype, contiguous layout, anchors, ranges, self loops, own-positive and TRAIN-positive exclusions. Other-VALID-positive collisions and native duplicates are measured and disclosed without editing the pool. No scoring, rank/MRR computation, prediction, model, optimizer, fit or candidate regeneration is present. The supervisor additionally requires the complete[19717,500] feature geometry and complete[2216,500,2] pool in the resulting receipt.

The six-field actual numerical runtime comparison is present in `input/inspect_available.py:145–150` and the supervisor compares the recorded runtime with the inspection release. Its ordering is after `inspect_available()` and thus after feature/pool value access. It is a final receipt-admission check, not a pre-read runtime guard. Root evidence is checked before reads. No completed inspection receipt can be produced by this main path with a version mismatch. A future release that requires actual runtime admission before any values are read must move this comparison earlier; this review does not describe the current ordering as stronger than it is.

On the normal supervision path, the direct Popen child, initial PPID/start identity, exact command/cwd and continued physical identity are checked, with one-second monitoring, monotonic elapsed time, observed RSS/HWM, total owned output and final elapsed/output checks. Linux wait4 peak RSS is converted from KiB to bytes. The child runs with empty CUDA_VISIBLE_DEVICES, fixed CPU library thread environments and explicit phase interpreter/import paths. Per-stage logs, START, current metadata, TERMINAL and final receipts disclose no-retry scope and the newly permitted TRAIN/VALID/feature inspection. The hard time/resource limits are monitored acceptance bounds with sampling and termination grace; the implementation does not provide an instantaneous kernel RSS/output limit. F1/F2 are defects in its failure/terminal path, not a rejection of direct-child ownership itself.

The supervisor checks staged source/release hashes, both enabled input-only recipe scopes and bounds, exact execution/inspection receipt identities, root/independent/supervision review hashes and a root-approved supervision admission before stages. It checks acquisition file coverage/source/recipe/member relationships after extraction, and complete inspection geometry/runtime/source/recipe relationships after inspection. Its normal completed receipt follows two successful `run()` returns and unchanged staged bytes. Actual value content and successful terminals were not inspected in this audit.

The new client pins the exact executable source manifest and complete remote source, requires the exact local release identity, concrete admission and a reviewed client hash, nonempty byte-authenticated review evidence and actual staged release/review files. It writes LOCAL_INTENT exclusively before its sole SSH call, uses only the allocation route on port2222 with existing key authentication, BatchMode, strict host checks and no host-key update, and contains no 18.77/MacLink route. Its remote command uses isolated normal Python and the exact pinned remote body. The timeout records terminal status unknown; the persisted intent prevents automatic relaunch. No private key contents were read by this review or by the client's file code.

The client stores only the compact acquisition, available-inspection and execution receipts; raw inputs remain remote in the proposed workflow. Its completion interpretation depends on the pinned remote body producing its final completed line after successful stages. The supervisor source provides that normal-path sequencing, while F1/F2 must be repaired for its failure/terminal accounting and F3 for source closure. The client has no actual release/admission files in this preparation packet, so source alone cannot make contact.

## Missing future prerequisites and present admission

The newly added disabled client means a launcher source now exists. It has not been independently admitted or released. The original absence of an admitted root launcher is a remaining release requirement, not a claim that no launcher source was reviewed.

Before possible allocation input verification, the four source defects need repair in new preserved packets and focused re-review. Actual root review, source/client/supervision approvals, exact reviewed release files, root supervision admission, fresh local intent/output identity and a concrete bound payload remain absent. The two current input recipes retain all three source/inspection/bounds approval booleans false, all TEST/model/fit/network/retry flags false and both root evidence lists empty. No new root or independent admission was fabricated here.

Resident archive/runtime presence, actual authenticated extraction, semantic inspection results, measured terminal accounting and feature-proof applicability remain unobserved in this audit. The prospective feature receipt correctly leaves exporter semantics and feature/pool training authority incomplete. Source inspection cannot bind those actual future receipts or release training.

The original scientific decisions, original30 promotion, complete39 descriptive diagnostics, candidate/native model qualification, evaluator/replay/cost admission and fitting gates remain unchanged and disabled. This review authorizes no data verification, network call, scientific computation, source adoption, fit, publication, acceptance or novelty claim.
