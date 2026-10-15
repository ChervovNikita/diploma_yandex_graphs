# Private BN state and individual route quality

The three selected checkpoints were reconstructed successfully. Across all
three seeds, the private-BN model had better mean member AUC than base GNNM,
but lower mean member AUC than the four independent native models.

| Method | Mean member AUC (%) | Pooled AUC (%) | Pool minus mean member (points) |
| --- | ---: | ---: | ---: |
| Base GNNM | 62.01 | 67.78 | 5.77 |
| GNNM with private BN state | 72.63 | 73.61 | 0.99 |
| Four independent models | 73.41 | 74.85 | 1.44 |

The private-BN changes in mean member AUC relative to base GNNM were +17.55,
+6.20 and +8.11 points for seeds 101, 202 and 303. The remaining member-quality
gaps to I4 were 0.44, 1.26 and 0.64 points. Thus improved individual rankings
accompany the pooled improvement. This decomposition does not establish that
the buffers alone caused the change. The methods selected different epochs.

The probability-loss result remains unfavorable. Private BN gave mean member
BCE 1.122 and pooled BCE 1.086. Seed 303 had pooled AUC 73.00% but BCE 1.854
and only 43.71% accuracy at raw logit zero. A good ordering of positive and
negative molecules can coexist with poorly placed prediction magnitudes.
Calibration, classification at a fixed threshold, and AUC are different targets.
No calibration rule or alternative serving rule was fitted by this analysis.

## Consequence for the research design

The earlier geometry/correction factorial used shared BN running state. Its
negative interaction is a result for that explicit setting. It is insufficient
to reject the combination under private BN. A useful next bounded study would
compare geometry, correction and both with private BN, retaining the completed
private-BN model as the matched no-ingredient control. This would require nine
new fits across the same three seeds. It should use the same native model,
data roles, optimizer, convergence rule and coefficients, with no tuning.

That possible study is not launched or added to the 6,460-fit roster here.
It needs an explicit implementation, frozen inputs and training plan before
execution. Private normalization is known prior practice. Success would first
support a better baseline and an interaction claim, not novelty by itself.

## Evidence and limits

- Each checkpoint contributes four predictions for all 151 VALID molecules.
- The analysis used 3 model constructions, 60 route forwards and 0 updates.
- All reconstructions matched their saved selected export under the existing
  diagnostic tolerance. No warning, retry, selection or checkpoint change occurred.
- The three subprocesses took 21.70 seconds including startup. Source, input
  hashes, per-seed scores, geometry and error-overlap summaries are retained.
- TEST remained closed. These are selected VALID diagnostics on one scaffold
  split, not independent confirmation or evidence of universal transfer.
- Repeated routes and molecules do not increase the number of training seeds.

The reproducible compact summary is in `SUMMARY.json` and `summarize.py`.
Raw predictions and hidden states stay inside the allocation repository.
