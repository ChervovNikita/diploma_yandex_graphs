# Resource budget and workload coverage — draft v1

**The 6,460-fit roster is a very large proposal. It is unchanged here and is
not approved to launch. Its full completion time remains unknown.** This
package uses saved metadata only. It launches no model or worker, reads no
tensor, accesses no TEST data, and changes no scheduler or source method.

Original paper experiments remain excluded. Boundary GNNM is retained as a
baseline in all 27 new native presets. M1 reuses I4/body0, so there are no extra
matched M1 fits. Four physical native fits plus a no-training export form I4.

## What the proposed training means

The roster contains 270 separate native source-reference fits, 1,080 matched
independent native bodies and 5,110 shared fits. There are also 270 I4 exports.
The source-reference and extended matched controllers remain distinct.

| Dataset | Fits | Optimizer maps per native epoch | Source caps + matched minimums | All caps |
| --- | ---: | ---: | ---: | ---: |
| CiteSeer | 480 | 1 | 960,000 | 4,640,000 |
| Coauthor-CS | 480 | 1 | 960,000 | 4,640,000 |
| Coauthor-Physics | 480 | 1 | 960,000 | 4,640,000 |
| PubMed | 480 | 1 | 960,000 | 4,640,000 |
| MolBACE | 1,920 | 38 | 7,296,000 | 70,224,000 |
| MolHIV | 1,920 | 1,029 | 197,568,000 | 1,901,592,000 |
| Collab, NCN/NCNC/BUDDY | 700 | 17 / 17 / 1,585 | 38,822,000 | 373,649,000 |
| **Total** | **6,460** | — | **247,526,000** | **2,364,025,000** |

An optimizer map means one source-sized optimizer transition. **These counts
are neither FLOPs nor an ETA.** A shared map includes four routes and possibly
additional gradient collections; activation checkpointing can add recomputation.
Construction, evaluation, export, failures and recovery add further work.

The first scenario places every source reference at its source cap and each
matched fit at its minimum: 2,000 node epochs or 100 molecular/link epochs.
It is not an exact earliest-stop lower bound: node source references can stop
before 2,000 epochs under native patience250, while matched fits can continue
past their minimum. The matched-only minimum obligation is 237,211,000 maps.
The all-caps count uses 10,000 matched node epochs and 1,000 matched
molecular/link epochs, conditional on no repeated or recovered work.

MolHIV has 27.079 times as many optimizer maps per epoch as MolBACE at the
same batch32. That is an update-count ratio, not a measured runtime ratio.
BUDDY has many smaller mixed-query batches with different operators, so its
1,585 maps cannot be compared to NCNC's 17 as equal-cost operations.

## What has actually been measured

- **MolBACE24:** 24 completed fits, 3,680 epochs and 139,840 maps;
  27,912.87 seconds summed worker invocation wall and 28,105.46 seconds pilot
  elapsed time. These are different quantities. The GPU also hosted an older
  worker; no isolated/pair throughput or mature resource envelope follows.
- **Private BN follow-up:** three additional completed fits, 464 epochs and
  17,632 maps; 4,740.97 seconds summed invocation wall. These are outside
  MolBACE24 and outside the prospective roster, rather than prepaid target fits.
  Patience completion does not establish scientific convergence.
- **Current NCNC CUDA lifecycle:** 181.32 seconds supervisor wall / 180.20
  seconds child wall, two full epochs / 34 maps / four whole VALID calls /
  three constructions / two selected mean exports. Peak allocated 12.67 GB,
  reserved 26.70 GB, sampled child process GPU 27.23 GB. All three memory
  measures are retained separately. No scientific convergence or general
  method-memory dominance is inferred.
- **Current molecular v4 CPU lifecycle:** 45.76 seconds, two complete MolBACE
  epochs / 76 maps / four whole VALID calls / three constructions / two mean
  exports. A selected checkpoint is 23,159,524 bytes; its selected VALID export
  is 66,156 bytes. The 140,209,507-byte engineering footprint includes snapshots
  and is not a full-campaign disk peak. CPU lifecycle success is not GPU timing.
- **Current BUDDY seed1 production CPU gate:** 111.36 seconds supervisor wall,
  two maps / three whole VALID calls / two constructions / 32 forwards.
  This includes hydration, construction, restore and validation. It is not a
  complete epoch or GPU update-time coefficient.
- **Old full smoke checks:** preserve all 167 selected statuses, 94 passed /
  73 failed, and all 200 attempts/costs. The 73 floating replay tolerance
  failures are not relabeled. These short source/shape resource observations
  remain useful without repeating a scientific fit or all 27 presets.

`MEASURED_COSTS.json` retains the exact scopes and scalar costs. Current
molecular evidence is pinned through
`qualification/MOLECULAR_RUNTIME_V4_QUALIFICATION_09.json`. Per-preset legacy
shape observations are in `LEGACY_SMOKE_RESOURCE_SHAPES.json`; they do not
establish current-runtime mature RAM/cache/VRAM/disk bounds. Old simultaneous
`independent4` peaks are excluded from the separate-body maxima.

### A conditional time prior for only 60 fits

The existing catalogue supports same-setting priors for 40 native bodies,
10 G0 and 10 geometry fits on **MolBACE/GIN/dropout0.5 only**. Source-reference
fits and the other 6,400 proposed fits are excluded. New BN metadata is kept
separately; old residual/combination runtimes do not transfer to portable v2.

| Assumed common completed horizon | Conditional summed invocation wall |
| --- | ---: |
| 100 epochs | 10.78–11.02 h; empirical-median prior 10.93 h |
| 149 epochs | 16.07–16.42 h; empirical-median prior 16.29 h |
| 1,000 epochs | 107.84–110.17 h; empirical-median prior 109.35 h |

These calculations require the old per-epoch rates, amortized overhead,
hardware, cache and contention to persist despite changed runtime/reporting
generation and new seeds. The range uses empirical extrema, not confidence
limits or guaranteed bounds. A common100epoch horizon is a work illustration,
not an assertion that the stopping controller stops there. No division by
three GPUs or assumed two-jobs speedup is made. MolHIV and BUDDY complete-fit
GPU runtime remain unknown. This small slice is not a campaign ETA.

## Smallest missing resource cover

Reuse the passed NCNC lifecycle and current node/molecular gates. Attach
resource observations to necessary remaining lifecycle/operator branches:

1. A current MolHIV whole-epoch GPU representative with full VALID and
   checkpoint/export observation. GIN-VN is a candidate heavy branch; first
   establish graph-batch and operator dominance. Dataset size alone does not
   prove memory dominance over BACE, GCN or every method.
2. The current BUDDY compact production-provider whole-epoch GPU lifecycle,
   including cache/hydration growth, selected export and checkpoint peak.
3. Missing mature process/cache/disk high-water marks, reusing existing
   checkpoint-size and sampled process receipts. Allocation-wide cache and
   headroom include peers and are not process-owned RSS.
4. If two jobs per GPU are wanted, a small reciprocal compatible-pair cover
   after single-process admission. It must measure actual throughput and
   process/cache/disk overlap. A serial one-worker first release is valid.

This is a set-cover recommendation, not a new qualification matrix or an
execution plan. No duplicate scientific fits, per-seed resource gates or
native tests to manufacture an ETA are proposed. Existing 19 readiness
obligations remain acceptance obligations rather than19newruns; the three
new passed qualification receipts narrow their still-open scope.

## Recommendation for approval in stages

Retain the full roster. After the finite readiness and resource gates close,
ask the user to approve a first frozen cohort across **all 27 presets and all
declared recipes**: 646 fits (27 source references + 108 native bodies + 511
shared fits) and 27 I4 exports. The remaining nine cohorts are 5,814 fits and
243 exports, subject to the user's further approval and measured budget.

This recommendation changes neither methods nor heldout rules. Staging does
not authorize outcome-driven pruning or turn partial results into a completed
confirmatory campaign. Neither stage is launched or authorized by this package.
18.77 portability and two-worker concurrency remain unqualified.

## Rebuild and provenance

Run `python3 resource_budget_draft_v1/build_budget.py` from the expansion
folder. It uses only the Python standard library and saved metadata/source
text. `INPUTS.json` pins every read by bytes/SHA256, and `BUILD_RECEIPT.json`
pins generated outputs and the builder. No raw tensor is opened; the three
remote BN3 RESULT authorities were already byte/SHA verified when their
compact metadata extraction was saved. Root owns staging, Git, publication
and any future native admission.
