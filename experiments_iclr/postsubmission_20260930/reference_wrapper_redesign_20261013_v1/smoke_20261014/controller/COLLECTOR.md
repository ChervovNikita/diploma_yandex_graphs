# Short-test results collector

`collect_results.py` uses the Python standard library only. It reads explicit
project-contained evidence roots and a full 167-case manifest, or complementary
manifests whose union has exactly 167 unique case IDs. It writes one fresh report
directory. It does not import callbacks, PyTorch or datasets, read checkpoint
payloads, execute training, or score the heldout data.

```sh
python -B controller/collect_results.py \
  --project-root PROJECT_REPOSITORY \
  --manifests ACTUAL_FULL_SMOKE_MANIFEST.json \
  --results-roots NODE_QUEUE_OUTPUT NONNODE_QUEUE_OUTPUT PREFLIGHT_OUTPUT PAIR_OUTPUT \
  --output FRESH_RESULTS_REPORT
```

Only pass existing directories. A single parent evidence directory can replace
the individual queue roots. Use `--warmup-roots` when a seeded warmup directory
is not named as a warmup/preflight. Use `--pair-roots` if collecting children
before their parent pair record/barrier has been copied. `--require-final`
returns exit status 1 when any declared case lacks an eligible terminal pass or
a supplied receipt cannot be parsed. It does not require pair probes to pass:
pair outcomes remain a separate admission condition for the launch plan.

## Selection and history

- Every expected case gets a matrix row, including missing cases.
- Global seed must be bound before factory construction and match the frozen
  descriptor. Procedure and complete recipe must match; callback SHA must match
  when the descriptor supplies one. Source descriptor conflicts are rejected.
- The latest eligible terminal attempt is selected, including failures. Earlier
  passes never hide a later failure.
- Unseeded preflight, warmup, pair children and unexpected descriptors remain
  evidence and costs, but cannot fill matrix rows.
- Identical raw `RESULT.json` copies are one attempt; all copy paths remain.
- Dispatcher launch UTC orders attempts where available. Otherwise the earliest
  available evidence-copy filesystem timestamp is an explicitly documented
  fallback. Preserving original timestamps when copying improves ordering.
- Legacy coordinate-replay failures remain `failed`. A diagnostic marks the
  particular case where outputs/loss/RNG passed but the model-state coordinate
  check failed. Only a newly executed amended test can supply a new pass.
- Each selected row exposes the controller receipt and replay policy version;
  legacy records do not claim the newly added immediate loaded-state equality
  check. Both policies can coexist honestly in an incremental report.

## Output

| File | Contents |
|---|---|
| `RESULTS.json` | Counts, full attempt receipt metadata, selection/exclusions, source provenance, checkpoint/replay checks, policy versions, actual work and queue observations |
| `CASE_CHECKS.csv` | All expected IDs; finite loss, gradients, parameter identity, checkpoint/replay, actual optimizer calls, CUDA peaks, RSS and wall times |
| `ATTEMPTS.csv` | All unique terminal case attempts, failures, warmups, pair children and source/policy receipts |
| `STEP_WORK.csv` | Every recorded train callback, including replay; native operation counts and each optimizer owner's state clocks |
| `PAIR_MEMORY.csv` | Separate physical joint-memory samples and child outcomes |
| `REPORT.md` | Concise coverage, actual charged work and remaining evidence |

The separate functional-completion diagnostic requires all four callback rows,
real completed transitions for every optimizer owner, finite loss/gradient
receipts, complete official VALID evidence, source parameter identity,
checkpoint and reload evidence, and global RNG restoration evidence. When the
immediate exact loaded-state check is supplied it must pass. Legacy records
carry a visible `not_exact_restoration_qualified` flag. Old omitted VALID/work
receipts remain unknown; no status-based inference fills them. A strict
post-update floating replay failure remains `failed` and prevents final matrix
success, even if its actual update/save/reload operations completed.

Actual optimizer observer totals include failed/replayed work and all four I4
owners. Missing totals remain unknown. Recorded step-only sums are lower bounds.
Dispatcher failures without terminal receipts are retained with unknown work.
Subtotals for warmups/pair children overlap the excluded-attempt total and must
not be added. Worker wall-time sums are not elapsed queue time.

Queue observations use actual completed seeded-passing case durations. A
same-callback-hash/procedure median, observed minimum and maximum can provide an
explicit short-case proxy for pending cases; unmatched cases remain unknown.
Dataset sizes can vary within a callback family. No exact remaining wall ETA or
full-training ETA is fabricated from these short tests.

## Scope

Source-horizon and main tracks reuse the same initial update path only where
initial operator/optimizer/schedule semantics are identical. These receipts do
not qualify full epochs, stopping, later transitions, convergence, distributed
worker lifecycle, mature-training memory or all 2,480 planned fits. No full
campaign fit, predictive-quality result or scientific GO is implied. Original
paper datasets are rejected by the collector manifest reader.
