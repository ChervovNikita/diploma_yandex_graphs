# Inactive complete neighborhood-distribution reader

This packet prepares one complete reader for the frozen SAGE neighborhood-distribution pilot. It does not launch a fit, inspect a partial result, construct a native model, load a checkpoint payload, or access TEST. The root owns scientific execution, interpretation and publication. Source preparation used standard-library source, JSON, hash and synthetic-text checks only.

## Fixed admission order

The invocation first verifies the literal authorized host `anogena-2-0`, the sole GPU UUID, the repository scope, the reader files and the pinned research files in `INPUT_BINDINGS.json`. The input bindings include the current production `FREEZE.json` SHA `c3b82223f50cb4a62d8ab7d8736b03854aab439408ad8640f25af110f45ab04b` and its five-arm, TRAIN-only actual qualification. The launch receipt must identify that freeze, hostname and GPU.

Both the new and original owner must close successfully, with direct-child wait, child/CUDA absence, whole-family completion, scientific success and no TEST. Before reading complete outcome records, their immutable header reader must admit exactly 15 new banks/24 optimizer fits and 21 original banks/39 original fits. The complete records then must match the owner hashes, metadata boundary, exact ordered roster, configurations, native sources, extension, descriptor support, seeds, registered parameter counts and operation counts. All 24 canonical new selected-checkpoint paths and byte sizes are admitted and hashed without loading them.

Fifteen new archives and fifteen immutable original-reference archives are admitted together before NumPy is imported. Every one of the 30 archives then passes its exact schema, dtype, shape, ordered VALID identity, finite-output and archived-probability/error consistency checks before any scoring. The ordered VALID role is 5274 rows, with the fixed ID/label hashes. The reused original reference banks have 33 selected fits; the full original 39-fit context remains in custody and cost reporting.

The numerical runtime is intentionally unqualified by these source checks. The root may invoke it only after successful whole-family closure. No partial seed, class or calibration endpoint can choose a continuation.

## Three prediction authorities

1. **Raw corrected or original:** the new `raw_logits` are corrected scores S. Original archives preserve their ordinary native scores. Integer member and pooled decisions use the archived float32 error flags. Stable FP64 logits produce NLL separately. Raw Brier uses the archived probability mean. The original reported metrics remain visible alongside stable NLL and any FP64 pool-decision discrepancies.
2. **Exact native at corrected-selected state:** the new archives contain unchanged native Z, its archived float32 member/pool probabilities and errors, at the exact same checkpoint selected by corrected VALID accuracy. These readouts do not independently reselect a native checkpoint. Same-state corrected-minus-native diagnostics describe the correction at that state. Comparisons with original separately trained banks include learning, stopping and selection.
3. **Fixed calibrated:** a positive scalar temperature is fitted independently for each fresh bank/readout/fold. The stitched held-fold FP64 probability mean determines calibrated pooled errors, NLL and Brier. Raw archived member top-class flags remain the count authority: positive temperature cannot acquire a new member alternative. Actual FP64 calibrated-member argmax discrepancies caused by representation or ties are separately recorded and never promoted into acquisition evidence.

The immutable joined-reader helper supplies member competence, coverage, lost alternatives, aggregation-only correctness, served alternatives, repairs/harms, new/removed coverage, repair/harm causes, four-state transitions, per-class measurements and three-seed paired summaries. Every count preserves `pooled_correct = coverage - lost_correct_alternatives + aggregation_only_correct`. The adapter adds Brier and common strict-rival cohorts without modifying that helper.

## All fixed contrasts and prospective gates

The report retains 45 raw contrasts: all 20 directed fresh-versus-fresh contrasts plus each of the five fresh arms versus all five original references. It also retains the same 45 exact-native contexts, five corrected-minus-native same-state contrasts, all 20 fresh calibrated contrasts for each of corrected and native readouts, and ten calibration-minus-raw same-state contrasts. Each contrast includes quality, member competence, count decompositions, repairs/harms and per-class outcomes.

The primary is `shared4_private_quartile`. Its raw served-accuracy mean must gain at least 0.2 pp over the capable fresh I4, at least 0.1 pp over the other three fresh controls, and at least 0.2 pp over original shared, ordinary I4 and factorized I4. Every required contrast must be nonnegative at all three seeds and positive at least twice. Original plain M1 banks are context, not substitutes for capable fresh controls.

The additional whole-quality development clue requires, against every fresh control, calibrated candidate-minus-control NLL at most 0.02 mean and at most 0.05 at every seed, plus calibrated candidate-minus-control accuracy nonnegative at every seed. A failed calibration endpoint fails that protection; it is retained. Raw NLL and its deteriorations remain in the contrasts even when calibration passes. The reader implements the root's fixed thresholds; it creates no new gate, veto, grid or continuation.

## Fixed scalar calibration and costs

The calibrator is the exact pinned prior `operators.py` implementation: one scalar `log_T=0`, common across the actual members, `log_softmax(log_p/exp(log_T))` per member followed by probability-mean pooling. Each fold receives exactly 500 full-batch CPU FP64 Adam updates with lr 0.01, betas (0.9, 0.999), eps 1e-8 and zero decay, using complement-row NLL only. The endpoint is the final update, with no selector, retry, temperature grid or final all-VALID refit.

The existing CPU local Generator seed 11709 randperm and permutation-position modulo five law supplies the same label-free folds to all arms. All 30 fresh native/corrected readouts attempt all five folds: **150 scalar fits/75000 requested scalar updates**, separate from the **24 native optimizer/body fits** and their 51 route trajectories per complete roster forward. No native forward or gradient is added by calibration.

Every fit retains elapsed time, finite status, scalar state, fitted temperature, the prior fixed trace and endpoint metadata when returned. A failure retains its error and any returned endpoint record. If the prior fitter fails internally before returning, its completed update count/state/trace are unavailable and explicitly marked unknown; the reader does not infer zero cost. Fit-time sums are nested within the calibration wall interval, which also includes conversion/preparation/OOF archive encoding, so these costs are not additive. Acquisition, selected serving, support preparation, richer I4 descriptor work, native counters and calibration work have separate entries. Reused 33-fit cost and full original 39-fit context are also not additive.

The fold assignment and ordered IDs are hashed. Each of the 30 OOF archives retains all raw-row IDs, folds, calibrated member log probabilities and pool log probabilities. Failed fold outputs remain NaN in these compressed NPZ archives and have explicit failure records; JSON never permits nonfinite numbers.

## Complete deterministic delivery

All JSON/NPZ outputs are prepared, encoded and size checked before the first write. Every admitted selected archive and new checkpoint is rehashed before delivery. All destinations must be fresh, with distinct names. The complete report is unbounded and contains complete source/owner custody. Small JSON documents must be below 2,000,000 bytes; fixed-order row chunks split before 1,800,000 bytes. An unexpectedly oversized individual row fails explicitly before writing rather than dropping detail.

`COMPLETE_ANALYSIS_SUMMARY.json` provides the frozen decision, calibration cost/failure summary, input/freeze/complete hashes and receipts for every full report, JSON partition and OOF NPZ. `SAGE_POLICY_SUMMARY.json` is a compact prospective gate summary. Separate chunks retain arm aggregates/details, calibration fits, all contrast categories, checkpoint custody and archive custody. The complete original source context lives in the full report and cannot overflow a small aggregate file. See `OUTPUT_SCHEMA.json` for names and rosters. Synthetic text checks demonstrate deterministic delivery when aggregate content exceeds the previous 2 MB limit.

## Root invocation after closure

From the literal authorized allocation, after checking the sealed manifest and both whole-family closures:

```sh
python3 experiments_iclr/postsubmission_20260930/native_neighborhood_distribution_complete_reader_source_20261010_v1/analysis.py \
  --report experiments_iclr/postsubmission_20260930/native_neighborhood_distribution_pilot_root_20261010_v1/complete_analysis_v1/COMPLETE_REPORT.json
```

The default research root is the source folder's parent and must equal the fixed authorized repository phase. This source adds no owner, launcher, retry or alternate lifecycle. A failed custody gate or failed output preparation does not authorize a replacement run or partial interpretation.

## Interpretation limits

The whole VALID role already selected every checkpoint. Post-selection calibration folds therefore remain encountered development, not fresh validation or whole-pipeline cross-fitting. Three optimizer seeds on one graph do not replicate graphs or splits; paired intervals and sign flips are descriptive. A positive protected screen requires a prospectively fixed reserved calibration role and unused whole-pipeline confirmation before broader claims. PNA, FSW, neighborhood statistics, quantiles, residual classifiers and temperature calibration are established ingredients. This packet supplies no novelty, injectivity, graph-expressivity, portable-gain or manuscript-acceptance claim, and it does not reopen the failed retrieval or recurrent-decoder rules.

Checkpoint SHA receipts are closure-time snapshots of canonical owner-bound paths and byte sizes, not earlier stored checkpoint SHA certificates. Source-level checks do not validate archived numerical payloads, fitted temperatures, runtime resource use or any scientific outcome.
