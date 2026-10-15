# Fixed-weight analysis confirms sensitivity to normalization assignment

All three completed private-BN models were evaluated on their original 151
MolBACE VALID graphs, using their already selected checkpoints. Each route's
own TRAIN statistics produced better AUC than tying all four saved banks, and
better AUC than the average score of the three cyclic mismatches, in every seed.
This result supports preserving route-specific running state in this setting.
It does not establish superiority over independent ensembles or a new method.

| Seed | Own bank AUC (%) | Tied bank AUC (%) | Mean of three shifted-bank AUC scores (%) |
| --- | ---: | ---: | ---: |
| 101 | 74.84 | 56.41 | 70.51 |
| 202 | 73.00 | 59.19 | 50.39 |
| 303 | 73.00 | 61.58 | 53.03 |
| Mean | **73.61** | **59.06** | **57.98** |

The mean own-minus-tied difference is 14.55 percentage points. However, the
first cyclic shift improves seed101's AUC from 74.84 to 77.22. The correct claim
is therefore about the three-seed averages and the prespecified average of all
shifts, not that every incorrect assignment is worse. All 15 conditions are
retained in `ALL_SCORES.csv`.

Proper losses add a qualification. Mean BCE is 1.086 with own banks, 1.250 with
tied banks and 1.775 across mismatches. Tying slightly improves BCE for seeds202
and303 even though it harms their ranking. Own-bank mean Brier is 0.298 versus
0.398 tied and 0.524 across shifts. This intervention does not solve the
calibration weakness identified in the earlier selected-VALID analysis.

## Interpretation and limits

Learned weights, checkpoints, examples and aggregation were held fixed. The
tied control averages running means and variances separately; it preserves
the saved histories' common clock but is neither the original shared EMA nor
a pooled-population variance. The cyclic shifts preserve the bank set while
changing its assignment. These controls show sensitivity to bank correspondence
at these selected weights. They do not identify how much of the original
private-BN improvement came from EMA time scale, call order or checkpoint
selection. Comparing this tied score with a differently selected G0 checkpoint
would confound those effects.

This is development analysis on one small scaffold validation split, following
observation of a promising normalization result. It supplies no independent
TEST estimate, significance claim or basis for choosing a shifted bank. The
existing prediction policy and the 6,460-fit proposal remain unchanged. The
ongoing nine-fit pilot separately tests whether normalization makes geometry
and private corrections more useful; this analysis does not substitute for it.

## Work and provenance

Three CPU constructions, 15 complete VALID calls, 300 native route forwards,
zero TRAIN calls, zero optimizer updates, zero TEST access. Elapsed supervisor
time was 30.82 seconds; maximum child peak RSS was 628,108 KiB. All own-bank
AUCs exactly matched their saved exports; maximum logit differences were below
0.000002. Learned parameters were unchanged and banks restored. No extra
parity forward or numerical retry was performed.

The first launch preflight used the wrong argument names for the existing
resource sampler and stopped before any model construction. Its record is
preserved in `PREFLIGHT_ATTEMPT01.json`; correcting the argument names did not
change any scientific setting. `PLAN.md` and `INPUTS.json` were frozen before
native execution. Source/checkpoint hashes and all outcomes are in the fetched
reports; raw logits remain in the allocation repository.
