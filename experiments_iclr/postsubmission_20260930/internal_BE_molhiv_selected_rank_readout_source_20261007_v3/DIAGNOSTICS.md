# V3 scope and qualification boundary

This is a separate disabled successor. V2 remains frozen. The inherited `PROTOCOL.md` retains its closure, terminal-custody, original-checkpoint and single-pass serving contracts; this addendum describes the new analysis. Actual current family closure and numerical/runtime qualification remain root controlled.

## Preserved execution contract

- `protocol.py` is byte-identical to v2. The collector's executable AST is unchanged; only its descriptive docstring differs.
- No additional model inference: the complete family still has at most 2,079 member calls. Hashing, sorting, pair comparisons and diagnostics reuse the same full raw logits and are charged in the analysis/cost stage.
- Deployment remains the original float32 **mean raw logit**. Sigmoid is applied to that mean solely for proper-score/calibration summaries; probability averaging, calibration fitting, threshold selection and checkpoint reselection are absent.
- O pair identity and masks are frozen before any non-O endpoint serving. Failed/unavailable fits and collections remain explicit. Any analysis error prevents complete 18 success.
- V3 cohort archives have a new exact schema. They cannot be replaced by v2 cohorts or regenerated after candidates are inspected. A new root-bound readout release must bind the v3 manifest.

## PR and calibration definitions

Every member and served pool receives average precision (the main PR summary), separately named trapezoidal PR area, Brier, BCE, class-specific BCE, fixed ten-bin reliability summaries, ECE, maximum occupied-bin gap, signed mean probability minus prevalence and its absolute magnitude. Logit mean, population SD and range give descriptive scale context.

PR ranks the original finite raw logits and groups exactly tied scores together. **AP is the step precision-recall integral.** Trapezoidal PR begins at recall 0/precision 1 and linearly interpolates successive grouped points; it is a separate convention and can differ substantially under ties. An all-tied balanced population has AP 0.5 and trapezoidal area 0.75 under this convention. Do not present those as the same statistic or use the latter's optimistic tie interpolation as evidence of screening quality.

Probabilities use a stable float64 sigmoid, with no score clipping or calibrator. Calibration bins are equal-width intervals on the positive-event probability: lower-inclusive, upper-exclusive except that the final bin includes probability 1. Empty bins report null means/gaps. ECE is the count-weighted absolute difference between mean positive-event probability and observed positive frequency; it is descriptive and depends on the fixed binning. These are not independent calibration estimates or a fitted calibration slope/intercept.

All six existing seed-paired contrasts retain ROC and add served PR/calibration/proper-score differences, plus member mean/worst competence. Higher AP/PR/ROC and lower Brier/BCE/ECE are favorable. Signed marginal error has no automatic improvement sign. Every contrast retains three optimizer-seed values, sample SD and the existing exploratory df2 interval. There is no molecule, pair, bin or member iid significance analysis. Added diagnostics do not replace the frozen practical ROC decision or authorize policy/recipe changes.

## O identity and attribution

Each baseline archives positive and negative original-ID vectors, their original positions, O pool win/tie/loss states, and the strict common-inversion mask. Its SHA256 covers a versioned ASCII header, positive/negative counts and all ordered `(positive ID, negative ID)` pairs as signed little-endian int64, in positive-major row order. Candidate IDs/labels, cohort domains and this identity are checked before comparisons. Only compact counts and hashes may be mirrored.

On this fixed O common-inversion cohort, every condition reports strict member-win coverage, weak tie-or-win coverage, member win/tie histograms and the joint table of strict coverage × pool loss/tie/win. Newly covered pool wins, newly covered pool ties and newly covered pool losses are separate. `strict_coverage_acquired_but_pool_loss` is the strictly pooling-lost count; `strict_coverage_acquired_not_fully_served` also includes ties. Ties earn half ROC credit. Uncovered pool wins/ties remain explicit rounding-boundary events rather than being forced to zero. Empty O cohorts produce null rates. Cohort net credit is normalized both within the cohort and by the whole pair population.

Exact same-index member pair-state transitions over the full population distinguish unchanged member order from acquired ranking coverage. O/I/P/G matching slots are labeled separately from the arbitrary slot match to independent4. If every member's full pair order is unchanged while pool order changes, relative positive scales or other rank-preserving score transformations can account for that change; the result does not demonstrate acquired member ordering. Changed member order or acquired pair coverage still does not prove causal graph-evidence specialization. Coverage is an unavailable pair oracle, not deployable screening performance.

## Synthetic fixture

`qualification_fixture.py` contains artificial-array checks for grouped PR ties/interpolation, Brier/BCE/reliability, stable extreme scores, ordered-ID hashing, strict coverage yielding only pool ties, scale-only pool repair with unchanged member pair order, empty cohorts, finite full-population validation and three-seed availability. It creates no authority documents, invokes no model or role loader and has zero serving calls.

Preparation parsed/compiled these sources only. The fixture is saved **unexecuted** for root to run separately in an authorized numerical environment. Even a passing synthetic fixture would qualify these helper semantics, not scientific serving, current data/checkpoint custody, calibration under new scaffolds or model superiority. Whole 18 actual closure, terminal custody, intended-runtime representative restoration/serving and an enabled readout release remain required.
