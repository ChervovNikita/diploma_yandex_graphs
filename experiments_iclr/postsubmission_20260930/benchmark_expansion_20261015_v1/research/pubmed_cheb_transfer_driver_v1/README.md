# PubMed/Cheb transfer: fit, selected VALID collection and analysis

This package completes the frozen second-setting protocol with the original
fit runner, a finite selected-checkpoint collector and a saved-prediction
analysis. It grants **no publication, qualification, resource or scientific
execution admission**. Both admission templates remain closed.

## Frozen scope and reused authorities

There are exactly **24 cold training trajectories and 18 reported banks**:

| Per optimizer cohort | New training trajectories | Reported banks |
| --- | ---: | --- |
| Native independently trained bodies 0, 1, 2, 3 | 4 | M1 = exact body 0; I4 = mean of the four own-selected exports |
| Original G0 shared model | 1 | G0 |
| Uniform, structured and permuted input residuals | 3 | uniform, structured, permuted |
| Total over seeds 101, 202, 303 | 24 | 18 |

Native initialization offsets are 0, 100003, 200003, 300007. M1 is counted
once as I4 body 0. There is no separate M1 fit or common-epoch I4 reselection.
The unchanged original `runtime/run_pilot.py` builder produces the four
native bodies and G0; the frozen residual composition produces the other
three descriptors. `SPEC.json` pins their sources, actual provider metadata,
data authorities, source split 0 and source split seed 1941488137.

The native matched controller is unchanged: minimum 2000 epochs, patience
100, cap 10000, whole-VALID accuracy, first strict maximum, no checkpoint 0.
The original PubMed/Cheb preset, optimizer groups, full-graph operations and
native reduction are retained. Residual variants use the frozen rank-8,
zero-up portable residual at native `lin1`, shared/native F updates and
private W updates. Uniform uses the identical F object for W. Structure
uses the frozen within-class fractional quartiles of incoming native degree;
the permutation control moves whole responsibility vectors within class.
See the pinned composition package for the full contract.

The already completed actual CPU qualifications are reused, with exact
planned-descriptor hash joins for all three variants. They include fresh
restores and three full VALID calls per variant. This package launches no
replacement CPU qualification. In local source checks, only the generated
composition extension path and `recipe._repository_root` are normalized to
their canonical allocation paths: macOS resolves allocation `/home` through
`/System/Volumes/Data`. Actual CPU hashes still have to match exactly.

PubMed was chosen for source preparation and has been encountered during
development. It is not a pristine confirmation setting. The five original
excluded datasets are not used here.

## Files and lifecycle

- `study.py`: original-builder descriptors, strict terminal and export gates,
  and pure Python node diagnostics.
- `driver.py`: allocation guard, source/data preflight, root admission and
  one-child serial ownership. It reuses the existing pilot lock, receipt,
  peer PID/start-tick and process helpers, and the unchanged fit runner.
- `collect.py`: one genuine selected shared-bank restore and four native
  evaluation forwards, with TRAIN callbacks and optimizer steps blocked.
- `analyze.py`: complete saved-prediction analysis; no model construction or
  native forward. It uses the original saved-export I4 aggregator.
- `SPEC.json`: frozen source protocol and completed CPU receipt pins.
- `*_ADMISSION_TEMPLATE.json`: closed examples; root must create actual
  reviewed admissions outside the source package.
- `check_source.py`, `SOURCE_CHECKS.json`, `MANIFEST.json`: deterministic
  source/JSON/synthetic scalar verification and file closure.

Default `preflight` launches nothing. `fits` requires the actual package and
SPEC hashes, approved publication commit, original repository virtualenv,
actual lifecycle/timing evidence receipts, finite root wall/memory budgets,
the dedicated allocation and a reviewed one-worker slot. No fit budget is
derived from a source estimate. CUDA lifecycle and isolated production
cost/timing review remain outstanding dependencies.

Only after fit admission are the 24 job files materialized in a fresh sibling
of the prospective study output `results/pubmed_cheb_structural_transfer_v1`.
The controller starts one fit child at a time, preserves each child log and
cost receipt, and durably writes its actual PID/start ticks immediately after
spawn. It stops after failure, timeout or interruption, without retry/resume.
Unstarted jobs remain visible. Unknown work is recorded as unknown, not zero.

Collection requires successful terminal metadata for all 24 fits, their
ordered outcomes, selected checkpoint/export receipts and closed TEST
status. It starts exactly 12 serial collectors: G0 and the three residual
banks for each cohort. Every collector gates metadata before importing
torch and verifies its own fit/runtime/checkpoint authority. The original
CUDA descriptor is retained for restore identity. Each collector records
construction, four forward attempts/completions, zero TRAIN/optimizer work,
close status, wall time and CUDA/host peaks.

The collector takes the full-graph mean **before** masking VALID, preserving
the native serving operation. It saves only ordered VALID IDs, labels and
four member logits. Reconstructed pooled logits are a sanity check against
the own-selected saved pool, with rtol 1e-4 / atol 1e-5. A mismatch is
disclosed as a warning; it never changes scoring authority or causes a
retry. The original saved selected pool remains authoritative.

Analysis requires all 24 successful fits and all 12 successful collections
before importing torch or releasing quality. It uses the same unchanged
collection admission and fresh output separate from both fit and collection
trees. It retains 24 unique fit costs, 12 collection costs and three saved
I4 aggregation/reader costs separately; concurrent peaks and walls are not
summed as resource requirements. A failed gate produces `GATE.json` without
a partial-family quality summary. No selector, calibration or new pool is
introduced, and TEST remains closed throughout.

## Diagnostics and limits

VALID class counts are **[72, 1919, 1952]**: class 0 is only 72/3943 = 1.83%.
Each of the 18 rows includes pooled and member accuracy/NLL, per-class and
macro accuracy/NLL, member mean/worst, own-selected epochs, ties, lost
correct alternatives, all-member errors and strict common false rivals.
All-member wrong predictions are distinguished from a common rival that
strictly beats the true class in every member.

Uniform-per-node CE/NLL is recomputed from saved FP32 logits with stable
Python float64 arithmetic for every bank; original saved FP32 native or I4
CE is also retained. Native first-argmax ties are preserved. Per-class and
macro metrics are descriptive and cannot alter the whole-VALID selector.

There are 14 paired comparisons per seed (42 total), including each
residual against M1/I4/G0, G0 against M1/I4, structured against
uniform/permuted and permuted against uniform. Signed contrasts, three-seed
means/sample SDs, per-class accuracy/NLL, repairs/harms, acquired/lost correct
alternatives and common-rival repairs are retained in `RESULT.json`.
Accuracy changes must equal (repairs - harms)/3943. `REPORT.md` gives a compact
table; the JSON retains all signed diagnostics and original cost receipts.

These are three paired optimizer seeds on one encountered transductive
graph and source split. Nodes, classes and routes are not independent graph
replications. The output supports no TEST, graph-generalization or novelty
claim. No full-campaign GO is granted.

## Invocation after root review

Run source checks locally from this folder:

```sh
python3 -B check_source.py
```

On the dedicated allocation, use the original repository `.venv`. The
following variables are placeholders for root-reviewed, published paths;
the templates themselves cannot authorize execution:

```sh
TRANSFER_ROOT=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
TRANSFER_PACKAGE="$TRANSFER_ROOT/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/pubmed_cheb_transfer_driver_v1"
"$TRANSFER_ROOT/.venv/bin/python" -B "$TRANSFER_PACKAGE/driver.py" --project-root "$TRANSFER_ROOT"
```

After root supplies an approved fit admission and its actual hash, append
`--mode fits --admission <absolute-fit-admission> --admission-sha256 <sha256>`.
After every fit terminates successfully, root may bind the actual END and
OUTCOMES hashes plus fresh collection/analysis output paths in a collection
admission, then run `--mode collect` with that admission and hash.

For the final saved-only pass, use `CUDA_VISIBLE_DEVICES=''` and
`--mode analyze` with the **same unchanged collection admission** and hash.
Root owns publication, actual numerical qualification, evidence review,
resource admission and every execution decision.
