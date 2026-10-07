# Closed Citeseer cohort reader source

This v2 packet makes a narrow output-directory correction to the v1 server reader. V1 and the prior scientific interpretation packet remain unchanged. Invocation is disabled; this successor has not been staged or run on the server, and no predictive payload has been opened to prepare it. Selector, metric, custody and mathematical functions are identical to v1.

## Gate and execution

The parent can activate a separate copy of `CONFIG_DISABLED.json`, retain the reader hash, and select initialization, growth, or both complete families. The default request is both. The requested family set fixes the output, and `output_relative` must match:

| Requested families | Fixed phase-relative output |
| --- | --- |
| initialization only (21 banks) | `citeseer_initialization21_closed_cohort_readout_root_20261007_v2` |
| growth only (15 banks) | `citeseer_growth15_closed_cohort_readout_root_20261007_v2` |
| initialization and growth (36 banks) | `citeseer_initialization21_growth15_combined36_closed_cohort_readout_root_20261007_v2` |

Each output must be fresh. Reading initialization21 can therefore proceed as soon as that family closes without blocking a later growth15 read or combined36 read. The combined read requires both complete families and supplies growth-versus-retained-warm comparisons.

Use the existing authenticated singleton route and pinned server Python. The reader verifies `anogena-2-0` and the sole GPU UUID before changing to the repository. CUDA is hidden before importing torch; only existing saved logits are deserialized, with CPU mapping and `weights_only=True`.

Before any HISTORY, logit or checkpoint file opens, the reader checks every requested owner's actual COMPLETE/BLOCK_FREEZE, original PID/start/config receipt, physical absence, full fixed roster, clean child EXIT receipts, exact source/job/plan/input identities and source-written 60-cycle FREEZE endpoints. It checks retained warm and paid-admission metadata using the existing owner contracts. Missing completion, a live/reused PID, failure, changed bytes, missing cell or mismatched endpoint refuses the whole read. There is one physical inventory observation and no polling.

The gate reads FREEZE as metadata without inspecting or exporting selected MRR. It opens no input dataset payload or state file. Source and job files, source manifests, acquisition manifest, qualification gates and completion receipts are metadata/source custody. After the whole gate succeeds, it hashes selected checkpoint bytes without deserializing weights and reads all requested HISTORY/logit banks. Results are written into a fresh readout directory only after every bank has been validated.

## Readout

- `NORMALIZED.json`: exact source-selected rounded4 MRR/Hit10, all member metrics at that same bank, raw CPU rank diagnostics, all 12 VALID exposures and maximum tie cycles, selected artifact hashes and measured per-row costs.
- `PAIRED_SUMMARIES.json`: equal-seed values, mean, sample SD, range, sign counts and exploratory df2 t intervals for prespecified paired MRR/Hit10 deltas; mean/minimum member competence and pool minus mean member quality. Growth versus retained warm identity is included only when both complete families are requested.
- `QUERY_RANKS_AND_COMMON_ERRORS.json`: all 227 query ranks, all member ranks, all-member top1 errors and shared negative-slot competitors. Query and negative-slot indices are zero-based and inherit the exact frozen input identities and order.
- `QUERY_FLOWS.json` and `.csv`: paired wrong-to-correct, correct-to-wrong, wrong-to-wrong and correct-to-correct counts, net repairs and reciprocal-rank deltas on all queries, baseline all-member-error queries and baseline common-negative-slot queries. The full fixed query population is always included.
- `METADATA_GATE.json` and `MANIFEST.json`: completion/input custody and readout file hashes.

The saved served mean tensors define pool ranks. No alternative aggregation, individual member selector, new checkpoint, new inference or query bootstrap is introduced. Source rounded4 HISTORY metrics remain authoritative; raw CPU reductions are diagnostics checked within rounded4 tolerance. Selection is verified as the first strict rounded4 maximum, including later tied exposures.

The combined36 read reopens the original source-selected banks. `METADATA_GATE.json` records `source_selected_bank_reuse`, including any earlier individual-family readout manifests present and their 21/15 bank counts. This presence observation is a reuse disclosure, not another completion gate or an audit of prior results. Each reopened bank's current checkpoint pointer and artifact hashes remain in `NORMALIZED.json`. Reopening banks adds no fits, optimizer seeds, independent evidence, VALID evaluations or selector exposures, and does not create a new selection.

## Custody and interpretation limits

Warm1 retains unknown process exit under the explicit artifact-custody amendment. Warm0 and warm2 retain their actual clean terminal receipts. Donors are charged once in listed physical study costs and again per row for a standalone warm-plus-fit comparison. Growth fit inclusive time already contains that fit's calibration. Qualification runs and complete-cycle cost runs are reported separately. The sum of listed source inclusive times does not measure earlier failed/orphan owner or operational recovery wall time; that missing cost is retained as unknown.

Growth FREEZE does not contain a selected checkpoint hash field. The reader compares its saved logits' checkpoint pointer with the actual selected checkpoint bytes and source-selected metrics. Neither source writes a logit hash into FREEZE: this readout records the first reader hash and does not describe it as a prior seal.

Three optimizer seeds on one graph/split describe conditional seed dispersion. Queries are not independent graph replications; VALID selection optimism remains. Feature covariance uses graph-derived H. Added growth capacity is matched to added single rank8 capacity, not total parameters. Copied independent controls share a common donor and pooled selector. Completion alone supports no positive-effect, novelty or ensemble-necessity claim. TEST remains closed.

## Verification performed

Local AST compilation and static source/schema checks only. The successor check compares every function except the output-handling `main` with v1 and verifies all are unchanged, verifies the three output mappings are unique, and verifies the original v1 seal. The inherited checks inspect actual saved-score keys, rounded metric/strict selector source, both HISTORY schemas, differing FREEZE endpoints, qualification metadata omission, owner completion/terminal schemas and activated config bindings. No check executes the reader, imports torch, inspects prediction values or contacts the server. Runtime closure and tensor verification remain deferred to the disabled reader's future authorized invocation.
