# Corrective independent source review: BLOCKED

Exact native-control candidate manifest: `28c9cd5232123655ba3588ee6ca8b358133d2afa189c07a1fef4505ef998e7a3`.

This packet supersedes the preserved v1 source-review PASS for release purposes. V1 incorrectly accepted the supervisor without checking its late recorded-signal path. Both the candidate and v1 review remain byte-identical, sealed records of that audit mistake.

## Blocking finding F06

In `supervise.py`, `interrupted()` only appends the signal number to `received` (231–232). The only reader of that list is the monitoring loop (248). If a signal is recorded after that check but before the normal child reap (258) and handler restoration (284), no subsequent drain sets `stop`. Physical/collection success tests `stop is None` (305), so the interrupted attempt can still be emitted as `COMPLETE_NATIVE_CONTROL_ONLY` (311–315). The static recorded-signal name sites are exactly 227, 232 and 248. Handler restoration occurs before all final collection and custody; copied census-v2 helper provenance does not close this main-flow defect.

A separate source successor must make recorded interruption immediately affect stop, drain/record captured signals at restoration/finalization and guard success against them, followed by exact independent review. No candidate amendment or native execution is authorized here. Root has been informed that v1 PASS must not be used for release.

The other source conclusions remain bounded: original native body/comparator/clone/restore and TRAIN-only access, 252-update ceiling and conditional scientific question are supported by source. This remains a fresh native post-TRAIN context, with different weights/hierarchy/flags/RNG history and absent VALID from failed shared4 post-VALID context; it is not a fully matched causal proof. A valid future native discrepancy would support variance only in that control under exact preconditions; agreement cannot establish a shared4 cause, overturn the old failure or admit science.

Only local source/scalar metadata and stdlib AST/JSON/hash analysis were used. No target imports, compilation, numerical execution, dataset/state/score/array/binary reads, signal/process probes, server access, staging, launch, run admission or subagents occurred.
