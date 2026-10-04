# V2 narrow final parent budget repair

V1 remains sealed and disabled. V2 changes only the final parent portion of `supervise.py/main` plus source-check/provenance documentation. All worker, common/data gate, metrics, objective, reduction, runtime profile, caps and disabled release/plan/staging bytes are identical. The unused execution-root-v1 name stays bound so the existing root client can use it.

The parent checks inclusive elapsed and observed output after `collect()` finishes all source/runtime/data reauthentication. It checks again after terminal, inventory/hash and supervisor-custody publication. A second-check exception or overrun turns any late success into failure, updates terminal/custody links, and returns a failing status. Receipts retain the elapsed/output observations at the last completed checkpoint, or the failed-check error and elapsed observation if scanning itself fails.

The last checkpoint cannot cover arbitrary later storage/stdio work. Finite final status/telemetry terminal-custody hash/write/fsync republication, exceptional failure republication/traceback, and final stdio print/flush/os._exit remain explicitly unmeasured tails. No recursive/infinite check-write loop or impossible whole-future instantaneous bound is claimed. Owned child-session RSS is sampled; parent RSS after child closure is unsampled. The original physical receipt records child-session closure before metadata collection and is retained unchanged.

Source-only AST/byte/hash checks. No law QA rerun, numerical/source execution, server/data/model access or review-scope expansion. A new independent exact v2 source review and root release are required.
