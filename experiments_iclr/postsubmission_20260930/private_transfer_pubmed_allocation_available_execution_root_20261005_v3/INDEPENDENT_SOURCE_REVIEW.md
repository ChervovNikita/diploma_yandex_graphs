# Independent static re-review of the disabled Pubmed CPU input workflow, v3

5 October 2026. The four P2 findings in the preserved v1 workflow review are resolved at source level in the complete v3 workflow reviewed here. I found no additional concrete source defect in this focused re-review. This finding disposition does not create root input-verification admission, numerical qualification, fitting permission or a scientific verdict.

## Reviewed bytes and method

| Actual packet/source | Authenticated SHA256 |
| --- | --- |
| Data v3 MANIFEST.json | `c9af4fc06cd97c63fcffebcac558b6d2c87a4ca347cb6905689aa47c0abbba42` |
| Data v3 SOURCE_MANIFEST.json | `1cac73d3ad26f8a6291cbec17695858f81942bdfb538ab86c0cd51b03eb13d9f` |
| Supervision v3 MANIFEST.json | `0fa4a323fcbc17a8145bb16544ded0513b1bebec55ff546fe5597b69f72cca44` |
| Actual supervision v3 owned_supervisor.py.txt | `35d051a63f86d19fd65b16f69a0a70be8c21d9b5ca476e5283ba2e2c569035a7` |
| Client v3 MANIFEST.json | `85d699800e0b7ffc68f10468e5f6aef462312f22d681e3cd5d09d8e88051c92a` |
| Actual client v3 execute_once.py | `a6d605a8fdec412ba71fcaf101d27f84fa3c981c12b5326789ea46cf4554d1e3` |
| Preserved independent v1 REPORT.md | `cf60f43f3c8a68eecab4e84f18511e91287767219d81dbc6027609f4d49dd6bd` |

The three reviewed directories are `private_transfer_pubmed_allocation_available_preparation_20261005_v3`, `private_transfer_pubmed_allocation_available_supervision_preparation_20261005_v3` and `private_transfer_pubmed_allocation_available_client_preparation_20261005_v3`. References below abbreviate these as data, supervision and client.

I independently authenticated all supplied pins, all 10 data-packet, six supervision-packet and seven client-packet manifest entries, their declared executable source members and applicable seal links. All 27 declared v3 input bindings match: three data, four supervision and twenty client bindings. I also authenticated the preserved v1/v2 source-packet manifests and their members/seal links, the prior independent review manifest/members/seal, and the original saved inspector/helper provenance used in the comparisons. No mismatch was found. INPUT_BINDINGS.json and SOURCE_BINDINGS.json identify exact paths, byte counts, roles and hashes.

Five actual v3 source bodies were AST-parsed. Thirteen independent AST comparisons matched. These are source comparisons, not execution tests. The audit used local source/protocol/recipe/review text, stdlib hashing/JSON and AST inspection only, within `postsubmission_research_20260930`. No target/numerical import or execution, checker, fixture, test, staging, data/features/history/checkpoint/score/prediction/outcome access, network/server contact, MacLink or 18.77 access occurred. No agent was spawned. Only this separate review directory was written. All prior packets, source files, target schemas and protocols were preserved.

## Four finding dispositions

| Preserved v1 finding | v3 source disposition | Actual implementation |
| --- | --- | --- |
| F1, P2: live child abandoned on monitor exception | Resolved | Live observation/I/O is covered by BaseException handling and nested finally cleanup that signals and reaps a still-live direct child. |
| F2, P2: Popen signaling can consume wait4 accounting | Resolved | Signals use raw os.kill; one custom wait4 function is the sole reaper. |
| F3, P2: unlisted executable source staging | Resolved | Client stages only the exact authenticated three-source tuple plus its manifest; remote requires that exact source inventory and exact root metadata inventory. |
| F4, P2: ignored 600-second inspection soft bound | Resolved | Input checks and scalar receipt production are checked; final owned stage admission includes terminal writing/printing and rejects late stages. |

### F1: exception-safe live-child cleanup

In supervision `owned_supervisor.py.txt:101–142`, the physical-process observation, START write, monitoring, output scan, CURRENT_METADATA write and monitoring sleep are inside a try/except BaseException/finally region. The exception is retained as `pending_error` and marks a supervisor failure. The finally block checks whether the custom reaper already obtained a terminal; otherwise it sends TERM, allows a short grace interval, then the nested finally sends KILL and blocking wait4 if still required.

This covers the v1 paths where metadata I/O or a first KeyboardInterrupt could exit the monitor without stopping/reaping its new-session child. An interruption in graceful cleanup also enters the inner finally. The original pending exception is reraised after cleanup. A later terminal/output receipt I/O error may still prevent a receipt from being written, but occurs after the owned child has been dealt with; it no longer bypasses live-child cleanup. Partial outputs and no-retry behavior remain intact. No exception path was executed in this audit.

### F2: sole authoritative wait4 reaping

Supervision lines86–100 retain `os.wait4(child.pid, ...)` as the sole consuming wait. On terminal collection, it records exit code and Linux rusage and sets `child.returncode` directly. `signal_direct` first probes through that same reaper, then uses raw `os.kill`, handling ProcessLookupError without invoking Popen polling.

No `child.poll`, `wait`, `terminate`, `kill`, `send_signal` or `communicate` call exists in the actual v3 AST. Thus the v1 Popen internal-poll/waitpid race is removed. A child that exits between the custom probe and the raw signal remains an unreaped direct child, reserving its PID until this wait4 path consumes it. The initial/continued PPID, start-tick, command and cwd observations remain present. Authoritative Linux wait4 peak RSS is still converted from KiB to bytes. Static inspection does not supply an actual terminal or measured resource result.

### F3: exact source and import-directory staging closure

Client `execute_once.py:55–75` reads the executable source-manifest byte string, checks its fixed SHA256, parses those same bytes, and requires exactly the distinct ordered tuple `available_custody.py`, `extract_available.py`, `inspect_available.py` under the exact supplied v3 disabled source-manifest identity. Every file is confined/nonsymlinked, authenticated by bytes/hash and checked again using the actual staged byte string. The staged manifest is that same authenticated byte string.

Only four `source/` records are created: those three programs and SOURCE_MANIFEST.json. The v1 SOURCE directory iteration is gone. The broader packet manifest is no longer the staging selector, so requiring an additional packet-manifest check for staging would add nothing to this executable selection. The actual packet manifest was independently authenticated by this review.

Supervision lines19–40 reject duplicate paths, traversal, unexpected nesting and extra source/root metadata names, and require exact inventory equality before creating the fresh root. The remote then checks the source-manifest hash, exact three-source path list and each source member's bytes/hash. The unlisted `json.py`/`tarfile.py` v1 counterexample cannot enter the child import directory through either staging pass. Exact client/source/supervisor constants and the disabled admission template's source hashes match actual v3 bytes. No staging was performed.

### F4: final 600-second soft admission includes receipt production

Data `inspect_available.py:135–167` starts elapsed-time accounting before authorization. It validates the actual six-field numerical runtime and fixes CPU threads before semantic value loading. It checks the declared soft bound before the values and after complete semantic inspection. After preparing and writing the scalar receipt, it checks the bound again at lines166–167. Crossing 600 seconds during receipt production raises and leaves that receipt unadmitted.

The inspector's stdout/exit tail is also covered by the supervisor's inclusive stage timer. Supervision lines143–157 reject elapsed above 600 before producing the stage terminal, then measure again after terminal writing and stdout printing. A crossing during terminal production sets a failure and writes POST_TERMINAL_BOUND_FAILURE with `stage_terminal_unadmitted=true`. That path raises rather than returning a successful stage. The outer workflow cannot emit its completed result after that failure, and the client rejects a nonzero remote result.

A geometry receipt or early-written terminal can therefore remain on disk after a late stage. The source explicitly preserves it as an unadmitted artifact; its geometry flag alone is not the workflow's admission decision. The passing workflow requires both stage calls to return after the final soft checks, then validates complete receipts/source/input/runtime relationships before final completion. This resolves the declared soft-admission issue including scalar and terminal receipt production. It changes no population, pool, feature, semantic check, threshold or scientific policy.

## Preserved input checks and workflow limits

The extractor is byte-identical to v1. The original inspector's `sha256`, `read_rows`, `load_available` and `inspect_available` helper bodies remain AST-identical. The allocation inspector server guard is preserved. The complete `available_custody.py` AST differs from v1 only in the fresh execution identity. Candidate/model/metric/fit code is not introduced by these repairs.

The existing source authorizer still checks Linux, exact project cwd and allocation hostname, sole UUID, explicitly empty CUDA visibility, confined source/recipe/evidence paths, reviewed source closure and separate approved root/owned-supervision evidence before ordinary input contents. Extraction still authenticates the resident archive and copies only the four exact regular allowlisted original TRAIN/VALID/feature members, with exclusive fresh files, exact member hashes/sizes and complete coverage. It performs no extractall or TEST extraction/semantic decoding. Full archive hashing and gzip traversal consume archive bytes without granting TEST-value access.

The complete feature, positive-population and fixed 500 pool checks remain as reviewed in v1: safe weights-only CPU feature loading, finite strided FP32 features, native row counts/disjointness/duplicate-positive/self-loop checks, full pool shape/dtype/layout/anchor/range/own-positive/TRAIN-positive checks, and preserved disclosed duplicates/other-VALID collisions. Extraction counts remain explicit pinned contracts rather than new measurements. Feature exporter semantics and training authority remain incomplete in the proposed receipts. Runtime admission now precedes semantic input reads; the v1 late-order limitation is removed.

The fixed 600/900-second, 2 GiB RSS and 128 MiB owned-output contract is unchanged. Normal monitoring retains one-second sampling and termination grace, followed by final resource/time checks. This is a monitored admission contract rather than an instantaneous kernel RSS/output quota. The repairs preserve physical direct-child ownership and do not infer actual performance or resource measurements.

Independent comparisons preserve the client's pre-staging admission block and its release/transport/intent/compact-receipt suffix. Its full main body equals repaired v2 after the declared v3 source-manifest scope literal update. The supervisor's complete v2 AST differs only by the fresh execution identity and the explicit final soft-admission statements. Physical/output helpers and the wait4 reaper body are unchanged. CHECKS.json records both fingerprints for all thirteen comparisons.

The client still requires an exact fresh release identity, actual root admission, its exact independently reviewed hash, nonempty byte-authenticated review evidence and concrete release files. Its exclusive LOCAL_INTENT precedes the sole allocation SSH call. The 1900-second timeout preserves terminal status unknown and the one-attempt intent; no relaunch path exists. Only compact acquisition/inspection/execution receipts are returned by the proposed source. Raw inputs stay on the allocation. The route excludes 18.77 and MacLink.

Both input recipes retain false source-review/input-inspection/bounds confirmation, empty unapproved root/owned evidence, and false TEST/model/fit/network/retry flags. The client admission template retains false root/client approval and empty review evidence. Their program/source pins match the actual v3 bytes; all numerical semantic/bound recipe fields are unchanged from v1. No actual release, source adoption, staging or run admission was created by this audit.

Actual allocation archive/runtime presence, truthful approved release evidence, fresh attempt/output custody, extracted member bytes, semantic measurements and successful owned terminals remain future execution evidence. Those absent facts are distinct from the four repaired source defects. The original scientific gates and all model/fitting admissions remain unchanged. This report makes no numerical, predictive, acceptance or novelty claim.
