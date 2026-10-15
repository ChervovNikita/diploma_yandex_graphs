# Cost, convergence and expanded-campaign readiness

This note uses saved observations and source metadata. It changes no frozen
pilot, 27-cell roster, 270 source-reference fits or 2,480-fit proposal. The
expanded benchmark has no final admitted count or completion ETA. Current
scientific execution is limited to the authorized allocation pilots; future
three-GPU deployment still requires authorization/runtime admission and the
large study requires the user's GO.

## Measured so far

| Evidence | Observation | What it supports |
|---|---|---|
| MolBACE GIN/dropout0.5, seed101, native body0; observations02–06 | Complete TRAIN has1,210 graphs,38 optimizer maps/epoch, no skipped TRAIN batches; whole VALID has151 graphs. The fit stopped at149 completed epochs by the frozen patience rule. | A real complete-epoch fit and stopping path for this particular source preset/protocol. |
| Same native trajectory | Saved epoch snapshots are4.386s at epoch14,4.314s at67,4.187s at108,4.179s at126 and4.345s at149: approximately4.3s/epoch. Observation02 reports199,229,440 bytes PyTorch peak reserved CUDA and about1.06GB peak RSS. | Local sampled training+VALID/checkpoint throughput and own allocator/RSS observations. The sampled minimum/maximum are **not** future timing bounds or confidence intervals. CUDA context/external occupancy are not contained in the allocator number. |
| Observation07, base GNNM | One early epoch snapshot:10.245s at epoch15. Native149-epoch fit is terminal; no complete paired trial is available in that snapshot. | An early same-preset wrapper timing observation. It is not stable throughput, a method effect, or evidence that wrappers stop after149 epochs. |
| Observation07, selected-state analysis | One native selected-checkpoint VALID analysis completed,5 native forwards, zero training steps; saved-export sanity passed. | The analysis path executed for one own fit. It does not validate every candidate/replica or establish a predictive gain. |

The queue snapshots retain an older-worker reservation. These timings are not
an exclusive-GPU benchmark. Full per-fit `RESULT.json`, epoch curves and
per-invocation `COSTS.json` remain the authorities for final accounting; the
local observations are compact snapshots, not complete timing distributions.

For orientation only,149×4.3s is about641s of epoch work at that observed rate.
This multiplication excludes construction, final export/analysis and queue
effects and is **not an ETA for the remaining23 fits**. Their stopping epochs,
route work and auxiliary overhead are still unknown. Under the current
minimum100/patience50 implementation,149 is the earliest patience stop if
there are no eligible improvements from epoch100 onward; the cap remains1000.
That is a protocol epoch-count property, not a wall-time bound or proof of
optimization convergence.

## Dataset size is a workload descriptor, not a stopwatch

The retained OGB paper reports41,127 MolHIV graphs (average25.5 nodes/27.5 edges)
and1,513 MolBACE graphs (average34.1 nodes/36.9 edges). These are published
whole-dataset statistics, not newly audited counts of every current provider
role. The actual MolBACE TRAIN/VALID counts above come from executed receipts.
Before assigning MolHIV cost, record its authenticated scaffold role sizes,
actual batches/skips, molecular size distribution, complete VALID cadence and
complete-epoch timings for native and representative shared objectives.

The source Collab loader requires235,868 nodes×128 features and complete
VALID60,084 positive/100,000 negative queries. NCN/NCNC costs also depend on
full-graph encoding, target-edge masking, overlap/completion sampling and
query batches. BUDDY adds year-filtered topology, hashes/query features and
cache construction. Query counts and graph counts are different work units.
Full source link epochs, VALID passes and cache costs must be timed before
those jobs are admitted to an executable campaign.

For PolyFormer, one update is one full-TRAIN epoch, but width, polynomial order,
attention/feedforward dimensions and activation checkpointing matter. A
within-route contrastive objective can add quadratic TRAIN-pair work even when
the backbone graph is unchanged. Node count alone cannot predict its duration
or peak memory. No MolHIV/Collab/PolyFormer wall-time bound is justified by the
MolBACE observations. In particular, do not scale4.3s by molecule/node/edge
ratios or assume that four routes cost exactly four native epochs.

## Counts and three-GPU placement

The unchanged proposal contains:

| Track/task | Frozen cold fits |
|---|---:|
| Source native M1:8 node+16 molecular+3 link presets, ten cohorts each |270|
| Main node:80 cohorts×(4 native bodies+5 shared arms) |720|
| Main molecular:160 cohorts×(4 native bodies+4 shared arms) |1,280|
| Main link:30 cohorts×(4 native bodies+3 shared arms) |210|
| **Total** |**2,480**|

M1 exactly reuses native body0 within a main cohort. Four own selected native
states create I4 without another training fit. Source-horizon M1 remains a
separate cold reference; it cannot be replaced by an extended/pilot trajectory
with different stopping or loader diagnostics.

An eventual expanded count must enumerate each explicit dataset/backbone/
preset/cohort and eligible method, then add actual independent bodies and
source-reference obligations. Do not multiply a union of every method by
every task when their targets/interfaces differ, and do not silently discard
unsupported or expensive hypotheses. Register those dispositions and the
needed port/data/compute separately. No count in this note supersedes the
frozen270/2,480 artifacts.

The old793/785/902 fit assignment is a declared placement using unmeasured
relative route/epoch weights. It is not balanced GPU-hours. A later placement
amendment must use actual complete-epoch/cold-start/VALID/export/cache costs,
source horizons, and a disclosed range of possible stopping epochs. Keep
source settings and paired cohorts fixed when changing only placement. Sum
four native bodies' actual work; do not treat an I4 logical bank as one cheap
fit or assume four identical stopping epochs. Do not divide estimated worker
time by three or six without measured concurrency behavior.

## Finite stages that retain the references

These are **options for a later freeze**, not new registered jobs:

1. Finish and analyze the current24 fits:3 trials×(4 own native bodies+base
   GNNM+alignment/variance+input residual+their combination). Preserve every
   arm and seed. Assess route competence/error overlap alongside pooled
   quality; descriptive geometry alone cannot promote a method.
2. A complete four-arm transfer panel on one fixed MolHIV source preset and
   one different MolBACE backbone would be24 fits per new cell,48 additional
   fits **if** those two cells and the same three-trial roster were frozen.
   First obtain honest full-epoch timing/resource admission. This explores
   both dataset and backbone transfer without an optimizer grid.
3. A narrower confirmation cell with one frozen candidate still retains4
   own native bodies+base GNNM+candidate:6 fits/trial,18 for three trials.
   A prospectively frozen cross-task node or link candidate can use that same
   reference structure with meaningful native targets/reductions. It is not
   evidence that the molecular auxiliary transfers unchanged to other tasks.

Outcome-guided screening remains exploratory. Any later confirmation uses a
new frozen question/controls and clearly separated evidence. All-reference
source checks remain in the benchmark. Three pilot seeds on one scaffold split
do not establish benchmark-wide uncertainty or an independent TEST gain.

Compute limits should change launch timing, admitted concurrency or the request
for more GPUs. They must not silently shrink source batches/topology, shorten
the minimum horizon, replace an independent ensemble, or erase a scientific
hypothesis because it is costly. Exact chunking/checkpointing can reduce memory
when it preserves the stated objective; an approximation needs its own explicit
protocol. Record a blocked expensive test and ask for resources when justified.

## Remaining admission gates

- **Convergence and resume:** exercise a fresh-process completed-epoch resume;
  add a versioned resume-finalization guard so an already patience-stopped
  recovery checkpoint finalizes without another epoch. Preserve frozen v1.
  Retain selected/completed epochs, late curves, first ties, cap/stop reasons
  and partial/replayed costs; a patience stop is a declared rule, not a global
  convergence guarantee.
- **Native source reference:** retain native horizons/diagnostic order and
  fidelity flags. Current pilots use matched validation-only stopping. The
  no-TEST source-reference adaptation omits the native TEST iterator's RNG
  effect and cannot claim exact whole-paper trajectory reproduction.
- **Link worker:** finish actual epoch/patience/selection/state binding and
  immutable fit dispatch, including distinct source100-epoch and extended
  policies. Qualify complete query/negative laws and cache custody. Reuse the
  source-specific link gap document rather than a molecular minibatch loop.
- **Selected-state TEST serving:** implement the authorized final evaluator,
  native input topology, full official queries and raw-logit/domain alignment.
  TEST stays closed during current pilots and cannot influence arm selection.
  Statistical endpoints consume authenticated banks without executing models.
- **Runtime and own-fit authorities:** bind all eligible factories, source/data
  receipts, own I4 selectors and exact member0 reuse; preserve failures and
  forbid automatic survivor-only retries. Commit a source/runtime snapshot
  before a final campaign manifest and human GO.
- **Three-GPU/two-slot admission:** verify authorized hosts/GPU identities and
  environments before deployment. At most two owned GPU jobs/processes per
  GPU, subject to measured joint usage and headroom. Model-reconstructing
  analysis also consumes a slot. Include external occupancy; sum VRAM per GPU
  and shared RAM/disk across both77 GPUs. Measure mature optimizer, variable
  batch/query and cache peaks, not only short smoke allocator peaks. Use
  atomic claims/leases and owned detached supervisors; never stop other users'
  jobs or alter mounts/settings. Two slots are a ceiling, not guaranteed speedup.

## Sources

`pilot_20261015_molbace_v1/OBSERVATION_02.json` through `_07.json`, its frozen
`SPEC.json`/`FREEZE.json`; `reference_wrapper_redesign_20261013_v1/SUMMARY.json`
and `ROSTER.json`; retained `references/ogb_paper_text.txt` dataset tables;
`smoke_20261014/link_port/callbacks.py` source guards and its
`FULL_RUN_GAPS.md`; the new runtime's `bookkeeping.py`/`API.md`.

Only source/metadata were read and this note/receipt written. No model, tensor
dataset, server operation, new fit, plan edit, publication or final-count
promise was made.
