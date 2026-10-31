# Independent review of control fitter v2 repairs

**Verdict:** no confirmed blocker in the two repaired source paths. Source verification passes; numerical/runtime qualification, new-host execution and scientific/report admission remain unestablished.

Reviewed packet manifest: `17c4ec1209e9b718055a52b4b0377689289b45527b16ac87a7acb03eff2c8b2b`. Its seal bindings and all **18 payload files** match. All **17 v1 payload files** remain sealed and unchanged. All **10 target Python files** parse and compile without execution. AST comparison confirms protocol/fitter semantics outside `validate_admission` and `_Recorder.charge` remain identical; both fixed 21-cell plans remain identical.

## Qualification identities

Full admission now requires the qualification assertion to contain well-formed packet-manifest, prepared-graph, compact TRAIN and compact validation SHA256 identities, each exactly matching full admission. The existing cell/runtime/passed-check/receipt identity gates remain present. Admission validation precedes scientific imports; actual input hashes are checked later against admission.

Independent synthetic metadata checks accept an otherwise complete matching assertion for each of the 21 fixed cells, reject **252** wrong/missing/malformed qualification identities, and reject **21** changed-environment admissions carrying an older qualification environment. These are synthetic gate checks, not authentic qualification receipts. Root must still authenticate receipts and verify their actual source/input/runtime/host provenance.

## Synchronization handling

The actual repaired contextmanager was extracted from AST and evaluated with stdlib-only mock synchronization. Six cases pass: success, body-only failure, end-sync-only failure, dual body/end-sync failure, interruption plus end-sync failure, and beginning-sync failure. End synchronization errors are raised; dual failures record both and retain the body error as explicit cause/context. A beginning-sync failure prevents body entry and propagates before an interval is recorded; the enclosing fitter failure path remains responsible for attempt closure.

No Torch/PyG/native scientific source was imported or executed. No CUDA synchronization or other device operation ran. These checks establish the Python control-flow repair, not asynchronous device correctness or numerical compatibility.

## Scope and handoff

The source still retains selected Photo local **model and Adam state** restoration, a fresh global selector, strict earliest-tie validation-NLL selection, fixed schedules/widths/seeds, compact MIMO correspondence, no final-label interface and report-ineligible outputs. This review does not rerun their numerical qualification, promote results or recalculate original scores.

The separate 18.77 portability assessment is in `PORTABILITY.md`. No speculative launcher, network operation, Git sync or compute was performed. Target sources were not edited.
