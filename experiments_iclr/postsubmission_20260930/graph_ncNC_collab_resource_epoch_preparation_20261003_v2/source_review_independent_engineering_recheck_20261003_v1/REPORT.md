# Independent source recheck — preserved resource preparation v2

## Scope and result

Reviewed manifest SHA256: `0dd69361c92852316bb9ddc5004f338a1575b36dd7480ffa05cd4f7ffdd767e0`, packet `graph_ncNC_collab_resource_epoch_preparation_20261003_v2` (19 sealed payloads, nine Python modules).

This focused engineering recheck traces the repair to v1 finding F1, final status/receipt publication, and the new 32-file sparse/scatter binary custody route. **The v1 final-accounting pass-gate defect is repaired on the traced production paths. I identified no new substantive source blocker in this recheck.** This is a source conclusion, not runtime admission, successful GPU qualification, feasibility or a scientific verdict.

I independently checked all 19 v2 and all 15 preserved v1 payload SHA256 hashes and byte counts: no mismatches. I also rechecked the prototype manifest and four imported portable module hashes. `gpu_parity.py` and `train_only_data.py` are byte-identical to v1. I read the new helper and fault-injection source fully, compared every changed module/document against v1, and read numbered production gate sections. I parsed saved JSON metadata to compare the binary inventories. I did not execute/import any reviewed module, test, source checker, prototype, sampler or numerical code; access data/labels/cache/checkpoint payloads or study outcomes; run remote commands; install anything; or change original files. These two new review documents are the only files written by this recheck.

The prior NCNC/context exposure and boundaries disclosed in the v1 source review remain applicable. This is a continuation of that engineering review, not a fresh blind paper review. The complete native graph/data/sampler/all31/twin/work trace is preserved in the [v1 review](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_ncNC_collab_resource_epoch_preparation_20261003_v1/source_review_independent_engineering_20261003_v1/REPORT.md>) and its machine-readable record. No study outcomes were reopened.

## F1 repair trace

| Gate | Exact v2 source trace |
| --- | --- |
| Receipt completes final synchronization | `runtime.py:120–131` first calls `self.synchronize()` and only then constructs the receipt containing `final_cuda_synchronization_completed=True`. An exception before return cannot produce a valid receipt. |
| Required accounting fields | `qualification_status.py:14–32` requires the timing/call/work maps and enumeration list; finite nonnegative timing/memory/map values; nested-inclusive/profiling/final-sync flags; a recorded explicit synchronization; and the opaque-library boundary statement. Missing, nonfinite or unsynchronized accounting raises. |
| Twin success commit | `resource_epoch.py:117–140` requires all 17 batches, obtains final RNG/state metadata, checks the matched RNG schedule, serializes and writes the internal state, then calls `commit_twin`. `qualification_status.py:50–63` starts with checks/accounting false, obtains and validates the receipt, and only then assigns COMPLETE plus true flags. Receipt failure records FAILED and reraises. |
| Twin runtime/accounting error | `resource_epoch.py:142–149` records the concrete error, keeps accounting false, obtains diagnostic accounting if possible, records any secondary error, and reraises. It cannot return a successful candidate. Family work/check gates at lines 151–154 require both accounted complete twins. |
| Top-level numerical isolation | `run.py:57–68` gives the numerical stage a separate `stage_progress` dictionary. Its status/check/accounting fields cannot directly overwrite the top-level qualification. The stage Boolean is retained as `stage_checks_passed` evidence. |
| Final process and setup accounting | `run.py:69–76` requires a fresh validated final process receipt and validates the stored pre-reset setup receipt (or uses the final receipt for GPU parity), then calls `commit_qualification`. Its helper (`qualification_status.py:80–103`) requires the exact stage-specific candidate status, completed stage checks, no prior failures, both required top-level accounting receipts and, for resource execution, both valid twin accounting receipts. Only afterward are final PASS and true flags assigned. |
| Late error invalidation | `run.py:78–85` calls `invalidate_qualification` for caught runtime/required-accounting errors. `qualification_status.py:66–77` clears top-level checks/accounting, invalidates every constituent twin's status/checks and retains prior completed-work evidence. Diagnostic retry receipt success cannot remove the failure list. A commit rejects any earlier failure. |
| Exit status | `run.py:100` uses `qualification_succeeded`, which requires the known stage, exact stage-specific pass status, checks/accounting both true and no failures (`qualification_status.py:106–111`). A previously true Boolean cannot alone yield exit zero. |
| GPU certificate reuse | `guards.py:127–139` requires the v2 schema, exact package/prototype/data/RNG/full-scale bindings, the same final pass predicate and valid setup/final synchronized accounting. A v1 or accounting-failed certificate cannot admit resource execution. |

`error_chain` (`qualification_status.py:35–41`) preserves primary/secondary cause/context conditions with cycle protection. A secondary failed diagnostic synchronization adds failure evidence. No graph/batch/candidate reduction or successful retry rescue was introduced.

The fake-meter fault-injection source covers eight declared cases: exact-status success, twin sync exception, missing twin accounting, invalid final GPU accounting, family failure after both twins commit, refusal of diagnostic rescue, nonfinite/unsynchronized accounting, and retained primary/secondary errors (`test_qualification_status.py:34–107`). `STDLIB_FAULT_INJECTION.json` binds its reported eight passes to the exact helper/test source hashes. I read that receipt and source; I did not rerun it. These are helper/dictionary tests, not live CUDA failure injection or an end-to-end numerical run. Production integration was assessed by the source trace above.

## Final metadata publication

`run.py:89–100` writes the complete receipt exclusively to `QUALIFICATION_PENDING.json`, closes it through `write_json`, and renames it to the final `QUALIFICATION.json`. A caught write/rename exception invalidates in-memory status and results in nonzero exit. A pending file is explicitly not an admissible qualification receipt. This improves the v1 partial-final-write boundary.

Preflight still occurs before output creation and outside the runtime receipt try (`run.py:20`, `guards.py:75–145`). Root must retain launcher output for a refused/malformed/source-invalid admission. Metadata publication failure may also leave no readable final receipt. The reported total ends before publication, so it excludes that reporting write. These limits are now stated in README/accounting documentation; timed internal state writes remain inside their measured phases. The publication path does not promise fsync/crash-durable storage; ordinary complete write and rename are the implemented boundary.

## Binary continuity repair

The saved root live-runtime receipt is SHA `5bf8d386bbcbfdfbad75e4a71e5b9619204fa2cf48880113f913708e1feb1668`; exact sampler source remains SHA `c04beecc5331144a2e10fdc3c66fbd8b4ac495f9dbdeb1649ae3cf047237b2f9`, with root-observed function SHA `0f26dd305f8c4443d0231a9ae09f591bd741d671c3e9ddb68f0debc07218112b`. Those two saved files are now external pinned bindings.

I parsed `BINDINGS.json`, `ADMISSION_TEMPLATE.json` and the live receipt. All 32 binary paths are unique, and both the sealed observed list and template list exactly equal the live metadata's path/byte/hash list. The sampler function pin also equals live metadata. The template remains `PREPARATION_ONLY_NOT_ADMITTED`.

`guards.py:104–106` requires root admission's binary list to equal the sealed observed list and checks every pinned file's bytes/hash before numerical imports. At runtime, `runtime.py:26–39` verifies imported Python source paths/hashes, rechecks every binary file, resolves the admitted binary paths/package directories, requires actually loaded extensions in those directories to be admitted, and requires a loaded extension from each sparse/scatter package directory. `runtime.py:50–59` carries all binary pins plus the sorted loaded extension paths into runtime identity and compares that complete identity to GPU parity during resource execution.

This closes the specific v1 omission of the observed sparse/scatter binary inventory. It is a continuity check for the 32 named on-disk files and source-observable loaded paths. It does not require all 32 CPU/CUDA variants to be loaded, attest every Torch/CUDA/driver/transitive shared library, freeze the whole environment, or hash mapped in-memory library bytes. I did not independently visit the host or read/execute the binaries. Actual imported-path behavior and kernel correctness remain runtime facts. A missing or changed binary causes failure; the code adds no installation/fallback path.

## Preserved semantics and remaining boundaries

The unchanged GPU parity/data modules retain the previous audit of native default negative sampling, ordered raw TRAIN graph, one full native batch, evaluation/training all31 gradients, fixed-pt absences, factor finiteness and RNG identity. Resource changes occur at receipt/status boundaries; the 17-batch work map, graph/candidate coverage, all-node xlin paths, shared encoder, private factor/norm/beta, detached recursive scorer with active dropout, post-clamp mean control, native loss/Adam, stream/RNG matching and mean raw-logit serving remain unchanged in the complete diffs. The immutable prototype sources have the same hashes.

Source and saved stdlib tests cannot replace admitted GPU parity or two actual finite complete native resource epochs. Native helper failures, arithmetic mismatch, memory failure or incomplete scope must be retained as concrete engineering failures. Final process CUDA allocator peaks reflect the last per-twin reset; saved setup and per-twin receipts preserve their own boundaries. Host RSS remains process cumulative, and nested inclusive times are not disjoint phases.

Root still owns the external v2 admission, explicit engineering RNG, interpreter/runtime/device scheduling and eventual full-scale runs. No predictive fitting, selection, quality comparison, manuscript conclusion, novelty or NCNC redistribution clearance follows from this recheck.
