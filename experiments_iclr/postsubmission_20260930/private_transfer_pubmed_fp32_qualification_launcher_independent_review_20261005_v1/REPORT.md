# Independent static review of disabled Pubmed TRAIN-only FP32 qualification launcher v1

5 October 2026. One actionable P2 source defect was found: the final inclusive soft-time check precedes serialization and flushing of the completed response. That tail can cross the declared 3600-second bound and still return successful completion. No additional concrete defect was identified in the reviewed workflow. This is source review only, with no numerical, scientific, release or execution admission.

## Exact identity and permitted inspection

Target: `private_transfer_pubmed_fp32_qualification_launch_preparation_20261005_v1`.

| Artifact | SHA256 |
| --- | --- |
| MANIFEST.json | `344f387ac75716cc16b1623b221c0267f5a49611bdd327b6fd7e423ceba0de08` |
| SEAL.json | `730efd03973754071c599cb248b17299275f4296e25db42b1c5691cb93f6d4b5` |
| execute_once.py | `1ad3b863a05184f0954c8bfc2b8427c25d0437145ec048a9b715b81ca2012eae` |
| owned_supervisor.py.txt | `e5c2a205acbae223367d38c3cfddc33e8d9ba369516d344b817dbda171540cc5` |
| Frozen numerical SOURCE_MANIFEST.json | `ba897c6588c0ac5d7d2e8473e1582b34c67986c23f265f0588f61c55c747aa6f` |
| Frozen qualify_training_step.py | `38fcec87c05d744135797546c819eb4e203e4a58c0273796ef7f60676194198e` |

All supplied pins and all authenticated hashes/sizes match. I authenticated all 11 launcher packet entries, all 39 frozen numerical-port packet entries, all five prior independent source-review entries and applicable seal links. Forty-two of the launcher's 44 declared input bindings were authenticated. The old Citeseer qualification REPORT, known to contain numerical outcome summaries, and the unnecessary additional TRAIN geometry adoption report were not accessed. The exact TRAIN adoption and permitted compact execution cost receipts suffice for this source scope. No deferred raw proof, geometry result, model result or input data reference was followed. INPUT_BINDINGS.json records 74 unique saved permitted files.

The inspection used local exact prepared source/protocol/disabled-schema/review text, necessary saved compact authority/runtime/input/cost metadata, stdlib hashes/JSON and AST operations only. Twenty-two frozen numerical sources and two actual control sources were AST-parsed without import or execution. No target/model/numerical function, test, checker or fixture ran. No dataset, history, FREEZE, checkpoint, feature/model/logit, score, prediction or raw outcome payload or credential was opened. No allocation/18.77/MacLink contact, staging, admission, source/schema/predecessor/canonical/publisher edit or agent spawn occurred. Only this separate review directory was written. Withdrawn access remains withdrawn.

## F1 — P2: final completion serialization can escape the inclusive soft bound

`owned_supervisor.py.txt:214–215` checks the workflow elapsed time and owned-output size, then evaluates json.dumps for the final completed=True message and prints it with flush=True. There is no time check afterward.

A source-level counterexample is a successful qualification whose final check passes just below 3600 seconds, followed by JSON serialization or a blocked stdout flush that crosses 3600. The remote body then exits zero. The client waits for the whole SSH result and, at lines59–63, accepts the last completed=True line because returncode is zero. The earlier terminal post-print check at remote lines179–184 and the pre/post receipt-file checks at209–214 do not cover this later completion print. Thus the declared end-to-end soft acceptance bound does not include all final success-response serialization/flush work. This path was inferred from source and was never executed.

Repair: recheck the inclusive elapsed bound after the final print/flush inside the existing failure/reraise region, and make any crossing preserve failure evidence and return nonzero. The client already rejects nonzero remote status before reading a completed response, so any early success text remains unadmitted. Preserve source bodies, tolerances, resource caps and the one-attempt identity.

## Staging, authority and TRAIN-only scope

The client authenticates the same numerical source-manifest bytes it parses and stages, requires the exact distinct ordered 22-program whitelist, authenticates each staged byte string and stages only those sources plus their manifest and exact control/evidence files. It does not iterate arbitrary source-directory contents. The remote requires exact unique inventory equality before root creation, rejects traversal/unknown/duplicate paths, authenticates every prepared payload byte/hash/size, the source manifest and all program members, and fixed authority metadata. This closes the executable/import directory to the reviewed source set.

Actual adopted input/feature, CPU package identity, TRAIN geometry and numerical-source evidence is present. The remote fixes the acquisition manifest to `d4550b3183bd6fc51a9db46dd823a37e187a60c5b8c83d859233cddb8da33f7c`, input adoption to `82cda6a1e9821bc43ccee9e46f336093dda40a64434523766d5f62b17230f186`, CPU receipt to `fffa724f10e423830145cb4cec03ca9e72b0f8562a05a4b8c59e5f856380856e`, and TRAIN geometry adoption to `08b00806f46a206d1760bee231ffadd5d34e52d5131fdeecb228cd716fe322ab`. The saved CPU runtime equals the concrete job runtime, the CPU receipt's four declared file identities equal acquisition metadata, and input adoption binds that exact acquisition/inspection and prior identical feature-value proof. TRAIN adoption is explicitly complete589 seed0/cycle0 geometry with training_admitted=false.

The frozen qualifier calls custody.authorize before runtime/model/input work. Runtime compares the actual six package/CUDA identities and one visible GPU. Its only load_inputs call has include_valid=False. The loader authenticates and opens TRAIN and features only; it returns before VALID-positive parsing or the negative VALID pool load. The full acquisition manifest's four roles remain metadata. This operation does not require a prior GPU/model/derivative qualification result or full historical VALID-generator authority. Those would misstate its purpose and loaded values. The generic runtime/negative-role names have explicit honest CPU-package/input and TRAIN-negative-only scope fields.

The qualifier still requires all six root evidence roles and the constant-adjacency repair authorization before computation. The disabled operation template retains false source/runtime/feature/negative/owned approvals and false hard-bound confirmation until an actual root release; the narrow already adopted TRAIN geometry role is true. Its owned-supervision evidence is empty and does not bind an admission containing job_sha256, so this template creates no circular job/admission hash. Future evidence can use independent source/root resource review bytes that do not themselves contain a circular job binding; the separate admission can then bind the final job hash.

## Sole dispatch and direct-child ownership

The client requires the exact fresh local execution root, actual root/independent-client approval, pinned control sources and concrete job/root/launcher review hashes. LOCAL_INTENT is opened exclusively at49 before the sole SSH call at52. Concurrent attempts cannot both create it. SSH uses the sole allocation login, named key path, strict host checking and a 4500-second transport timeout. Key contents are never read. Timeout retains intent and explicitly reports unknown terminal status with no relaunch. A nonzero remote exit is rejected before completed-text parsing. No retry path or 18.77/MacLink route is present.

The remote creates one new-session Popen child with exact normal interpreter/script/job/output argv and repository cwd. It observes PPID/start_ticks/pgid/sid/argv/cwd and retains the unreaped direct-child PID until sole os.wait4 accounting. Signals use raw os.kill after that same reaper probe. No child.poll/wait/terminate/kill/send_signal/communicate or group signaling exists. Exception handling and nested finally retain the reviewed TERM/grace/KILL/blocking-wait4 cleanup for monitor/metadata/CUDA-accounting errors and a first interrupt. Receipt errors after child cleanup cannot abandon that owned child.

Five helper/nested function ASTs — sha, write, output_bytes, reap and signal_direct — match the repaired input-v3 supervisor. The physical helper differs only by adding pgid/sid observations, which are used in command/session checks. The new CUDA query is bounded to 20 seconds and counts the owned child PID only. These are monitored acceptance bounds with sampling/grace; they do not claim instantaneous kernel quotas or authority to signal unrelated processes.

## Frozen qualification and final numerical/resource checks

The complete 22-program numerical source manifest and qualifier bytes match the previously independently reviewed and root-adopted port. Eighteen concrete job science/scope fields match the predecessor template, including architectures, seeds/factor seed, episode0, outer64/inner256, two reachable histories plus one stale discarded commit, source/program pins and all tolerances. The source still checks native Adam parameters/moments, complete direct+mixed derivative coordinates, recomputed/stale serving, RNG/state/mode guards and all three fixed architectures. No numerical body, tolerance, selector, sampler, schedule, fitting policy or original30/full39 scientific gate is changed.

The proposed caps agree across source/release/bounds metadata: soft3600/hard4200 seconds, 16 GiB RSS, 20 GiB CUDA, 64 MiB total remote owned files and transport4500 seconds. The run loop samples RSS/output/time/owned CUDA and fails with cleanup. Terminal wait4 supplies authoritative Linux peak RSS including the child's late RESULT serialization. The source-reported allocated and reserved peaks are checked for all three exact arms at remote lines200–206, with integer 0<=allocated<=reserved<=20 GiB. The final owned CUDA sample peak and wait4 RSS are checked at207. The qualifier synchronizes and captures each arm's actual CUDA peaks; its late JSON writes contain metadata rather than model state.

Successful child RESULT alone is insufficient: zero owned terminal/no failure/no signals, exact PID/source/job/input/runtime/tolerance identities, full589/37676 coverage, all arm/history/commit/state guards and unchanged staged bytes are required. All staged/control/log/result/receipt files are counted by output_bytes. Time and output are checked before compact receipt creation and after both qualification/execution receipt files are written. The single finding above concerns the remaining final stdout serialization/flush tail.

## Disposition

FINDINGS.json preserves F1 as an open source-level P2 defect. No additional concrete source defect was found. CHECKS.json records exact authentication, syntax, preserved helper fingerprints, scopes, source closure and one-attempt facts. Actual new qualification, capacity, numerical correctness, root/resource release and successful owned terminal remain unobserved by this review. Execution, fitting and scientific admission are false. The existing actual input/runtime/feature/TRAIN geometry authorities need no redundant qualification or receipt for this operation.
