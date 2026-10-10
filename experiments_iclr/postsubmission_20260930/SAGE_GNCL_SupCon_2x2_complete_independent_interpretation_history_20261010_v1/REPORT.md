# Complete SAGE GNCL / SupCon factorial: independent interpretation and history

## Scope and decision

Root explicitly admitted the complete factorial numerical outputs on 10 October 2026 after the reader closed at 20:22:52 UTC. This note interprets the 14 compact analysis JSON files plus the two reader execution receipts in the verified 16-file collection. It reads statistics only. Source, freeze, model states, serving, selectors and original paper scores remain unchanged. No remote operations, model or raw-array imports, native replay, or TEST access occurred for this interpretation.

**All three exact learning rules fail both the frozen raw-accuracy screen and complete confidence protection. Neither the predeclared raw priority nor the joint priority selects a confirmation candidate.** This closes the declared coefficients and temperature on the encountered WikiCS SAGE development setting; it is not a claim that committee supervision, SupCon, shared ensembles, or SAGE improvement in general is futile.

The roster is complete: 18 fresh banks / 27 native fits; 15 admitted original banks / 33 selected fits; 60 joined selected fit records. All 165 fixed scalar calibration attempts succeeded, with 82,500 completed scalar updates before comparison. These are three paired seeds on the same 5,274 encountered VALID nodes, not 15,822 independent observations and not unused complete-pipeline confirmation. One defensible SAGE improvement would have sufficed; cross-backbone transfer is not a requirement.

The frozen rules are A = 0.5 own CE + 0.5 CE of the served probability mean; B = own CE + 0.05 within-route SupCon; AB combines the three terms. SupCon uses all 580 TRAIN nodes, cosine temperature 0.2, normalization epsilon 1e-12, same-label nonself positives, and the positive average outside the log. The native SAGE depth 2 / width 128 / dropout 0.2, AdamW 0.001 / decay 0, maximum 1,000 updates / patience 300, and first strict probability-mean VALID accuracy maximum were retained. Seeds are 7301, 7403, 7507.

## Complete pooled and member outcomes

Accuracy columns are percentages; NLL and Brier are losses. Member quality is diagnostic. A weaker member score cannot independently veto a worthwhile pooled result.

| Procedure | Raw pool accuracy | Raw NLL | Raw Brier | Calibrated pool accuracy | Calibrated NLL | Mean member accuracy | Worst member accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original shared own | 80.571356 | 1.001493 | 0.311764 | 80.596638 | 0.682065 | 78.384528 | 77.765137 |
| A: committee CE | 80.565036 | 1.209042 | 0.318861 | 80.577677 | 0.700515 | 78.220200 | 77.600809 |
| B: own CE + SupCon | 80.653520 | 1.109375 | 0.314719 | 80.710403 | 0.698390 | 78.537795 | 78.100114 |
| AB | 80.590317 | 1.240180 | 0.318442 | 80.672481 | 0.712013 | 78.281823 | 77.556567 |
| Original ordinary M1 | 79.319934 | 0.754213 | 0.323495 | 79.319934 | 0.696492 | 79.319934 | 79.319934 |
| Original factorized M1 | 79.130325 | 0.778711 | 0.326257 | 79.130325 | 0.707176 | 79.130325 | 79.130325 |
| Original ordinary I4 | 80.021489 | 0.695762 | 0.303533 | 80.021489 | 0.661398 | 79.176147 | 78.934395 |
| Original factorized I4 | 80.084692 | 0.710204 | 0.305418 | 80.091013 | 0.659891 | 79.210909 | 78.757426 |
| Ordinary M1 + SupCon | 79.319934 | 0.755486 | 0.324795 | 79.319934 | 0.697557 | 79.319934 | 79.319934 |
| Factorized M1 + SupCon | 79.180887 | 0.784757 | 0.327446 | 79.180887 | 0.708172 | 79.180887 | 79.180887 |
| Genuine factorized I4 + SupCon | 80.046770 | 0.712149 | 0.306035 | 80.021489 | 0.662006 | 79.223549 | 78.795348 |

The member improvement in B is real: mean member accuracy rises from 78.384528% to 78.537795% (+0.153268 pp), and worst member accuracy rises from 77.765137% to 78.100114% (+0.334977 pp). Its served raw gain is only +0.082164 pp. A and AB reduce mean member accuracy, but the decisive failure for all cells is the complete pooled screen below, not that diagnostic.

### Frozen accuracy screen

Every required contrast needs at least +0.2 pp mean raw gain, nonnegative differences at all three seeds, and at least two positive seeds. AB additionally needs at least +0.1 pp over A and B without a negative seed. All new cells pass their required single / I4 raw comparisons, but every cell fails against original shared own. AB also fails both incremental cell comparisons: +0.025281 pp over A with a negative seed, and −0.063203 pp versus B with two negative seeds.

| Cell | Raw gain vs original shared (pp) | Seed gains (pp) | Descriptive 95% t interval, df 2 (pp) | Exact two-sided sign-flip p | Raw screen | Confidence screen |
| --- | --- | --- | --- | --- | --- | --- |
| A: committee CE | -0.006320 | +0.075844, -0.189609, +0.094805 | [-0.401338, +0.388697] | 1.0 | Fail | Fail |
| B: own CE + SupCon | +0.082164 | -0.056883, +0.227531, +0.075844 | [-0.271359, +0.435688] | 0.5 | Fail | Fail |
| AB | +0.018961 | +0.075844, +0.000000, -0.018961 | [-0.105658, +0.143580] | 1.0 | Fail | Fail |

The intervals and sign flips describe three encountered paired runs; they establish neither statistical significance nor a reliable population uncertainty statement. The frozen worthwhile-effect threshold and seed signs were not relaxed after seeing results. B being highest among these cells supplies no qualification and does not override AB / B / A predeclared priority.

### Equal-policy confidence protection

The same five label-free folds and final-step bank-global temperature policy were applied to every bank. Protection requires calibrated NLL deterioration no greater than 0.02 on average and 0.05 at every seed, and nonnegative calibrated accuracy differences at every seed, versus all frozen required references.

| Cell | Reference | Mean calibrated ΔNLL | Seed ΔNLL | Seed calibrated Δaccuracy (pp) | Protection |
| --- | --- | --- | --- | --- | --- |
| A: committee CE | Original shared own | +0.018450 | +0.037406, +0.010346, +0.007598 | +0.018961, -0.018961, -0.056883 | Fail |
| A: committee CE | Original factorized M1 | -0.006661 | +0.005492, -0.011176, -0.014300 | +1.308305, +1.213500, +1.820250 | Pass |
| A: committee CE | Original factorized I4 | +0.040625 | +0.045275, +0.045191, +0.031407 | +0.436102, +0.720516, +0.303375 | Fail |
| A: committee CE | Original ordinary I4 | +0.039117 | +0.045775, +0.042065, +0.029512 | +0.284414, +0.985969, +0.398180 | Fail |
| B: own CE + SupCon | Original shared own | +0.016325 | +0.003878, +0.014727, +0.030370 | -0.056883, +0.322336, +0.075844 | Fail |
| B: own CE + SupCon | Factorized M1 + SupCon | -0.009782 | -0.033857, -0.009194, +0.013705 | +1.175578, +1.516875, +1.896094 | Pass |
| B: own CE + SupCon | Genuine factorized I4 + SupCon | +0.036384 | +0.009533, +0.047217, +0.052402 | +0.474024, +1.118695, +0.474024 | Fail |
| B: own CE + SupCon | Original factorized I4 | +0.038499 | +0.011747, +0.049573, +0.054179 | +0.360258, +1.061813, +0.436102 | Fail |
| B: own CE + SupCon | Original ordinary I4 | +0.036992 | +0.012247, +0.046446, +0.052283 | +0.208570, +1.327266, +0.530906 | Fail |
| AB | Original shared own | +0.029947 | +0.041567, +0.035747, +0.012528 | +0.170648, +0.056883, +0.000000 | Fail |
| AB | Factorized M1 + SupCon | +0.003840 | +0.003833, +0.011826, -0.004137 | +1.403110, +1.251422, +1.820250 | Pass |
| AB | Genuine factorized I4 + SupCon | +0.050006 | +0.047222, +0.068237, +0.034560 | +0.701555, +0.853242, +0.398180 | Fail |
| AB | Original factorized I4 | +0.052122 | +0.049437, +0.070593, +0.036337 | +0.587789, +0.796359, +0.360258 | Fail |
| AB | Original ordinary I4 | +0.050615 | +0.049937, +0.067466, +0.034441 | +0.436102, +1.061813, +0.455063 | Fail |
| AB | A: committee CE | +0.011497 | +0.004162, +0.025401, +0.004930 | +0.151688, +0.075844, +0.056883 | Pass |
| AB | B: own CE + SupCon | +0.013623 | +0.037690, +0.021020, -0.017842 | +0.227531, -0.265453, -0.075844 | Fail |

Against original shared, A and B satisfy the numerical NLL bounds but fail calibrated accuracy preservation; AB preserves calibrated accuracy but worsens calibrated NLL by +0.029947, above 0.02. Against I4, all cells fail at least the mean NLL bound; B and AB also exceed 0.05 at some seeds. Raw NLL and Brier also worsen versus original shared for each new cell. Calibration materially lowers each procedure's own losses but does not turn this complete factorial into improved joint quality. No raw favorable I4 comparison is erased by confidence failure.

## Exact repairs, harms and alternatives

For each selected bank, let C be the number of nodes with at least one correct member, L the covered nodes lost in averaging, and G the pooled-correct nodes with no correct member. Then pooled correct = C − L + G. The archived float32 error flags are the raw count authority; reported FP64 raw/member-argmax disagreement counts are zero. The old shared bank has one aggregation-only correct readout, while A, B and AB have zero. That one readout must be retained in every count decomposition.

All counts below are summed across three dependent readouts of the same development nodes. Every row contrasts complete independently fitted / stopped / selected procedures against original shared own.

| Readout | Cell | Repairs | Harms | Δ pooled correct | Δ coverage C | Δ pooling losses L | Δ aggregation-only G | Δ member-correct readouts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | A: committee CE | 330 | 331 | -1 | 11 | 11 | -1 | -104 |
| raw | B: own CE + SupCon | 280 | 267 | 13 | -31 | -45 | -1 | 97 |
| raw | AB | 353 | 350 | 3 | -4 | -8 | -1 | -65 |
| calibrated | A: committee CE | 318 | 321 | -3 | 11 | 13 | -1 | -104 |
| calibrated | B: own CE + SupCon | 276 | 258 | 18 | -31 | -50 | -1 | 97 |
| calibrated | AB | 348 | 336 | 12 | -4 | -17 | -1 | -65 |

Thus A yields +11 coverage − 11 extra losses − 1 aggregation-only = −1 pooled correct. B yields −31 coverage +45 fewer losses −1 aggregation-only = +13. AB yields −4 coverage +8 fewer losses −1 aggregation-only = +3. B improves member-correct readouts by 97 while union coverage shrinks by 31: stronger members can become more redundant. Its smaller L count supports fewer available alternatives lost in the final procedure, but does not isolate a pure averaging intervention at a fixed state.

The raw per-seed repairs / harms are A: 137 / 133, 99 / 109, 94 / 89; B: 10 / 13, 141 / 129, 129 / 125; AB: 133 / 129, 135 / 135, 85 / 86. Gains are 4 / −10 / 5, −3 / 12 / 4, and 4 / 0 / −1 correct readouts respectively.

| Cell | New coverage | Removed coverage | New coverage served | New coverage lost | Repairs with newly available alternative | Repairs while both banks cover | Harms with available alternative | Harms after coverage removed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: committee CE | 315 | 304 | 60 | 255 | 59 | 271 | 279 | 52 |
| B: own CE + SupCon | 214 | 245 | 45 | 169 | 44 | 236 | 217 | 50 |
| AB | 313 | 317 | 65 | 248 | 64 | 289 | 292 | 58 |

A serves 60 of 315 new covered occurrences and loses 255; B serves 45 of 214 and loses 169; AB serves 65 of 313 and loses 248. The served-new-coverage count includes the original aggregation-only correct case, so actual new-coverage repairs are 59, 44 and 64. Existing-member-alternative-served means both procedure coverages are true; it does not certify the same route/vector, or exclusive pooling causality. These changes include learning, stopping and selection, with different selected steps:

| Procedure | Seed 7301 selected step | Seed 7403 selected step | Seed 7507 selected step |
| --- | --- | --- | --- |
| Original shared own | 78 | 318 | 165 |
| A: committee CE | 487 | 331 | 168 |
| B: own CE + SupCon | 78 | 479 | 603 |
| AB | 282 | 900 | 165 |

### Original strict-rival cohorts

These cohorts are frozen from the original raw banks: a common incorrect class strictly beats the target for every original member. Entries are corrected readouts as **raw / calibrated** on that fixed cohort; denominators sum the three seed readouts. They are not newly chosen candidate failure sets.

| Original defining bank | Cohort readouts | Original shared | A | B | AB |
| --- | --- | --- | --- | --- | --- |
| Original ordinary M1 | 3272 | 802 / 799 | 845 / 855 | 840 / 835 | 858 / 854 |
| Original factorized M1 | 3302 | 824 / 826 | 868 / 876 | 867 / 864 | 869 / 854 |
| Original ordinary I4 | 2245 | 250 / 251 | 301 / 299 | 290 / 286 | 298 / 289 |
| Original factorized I4 | 2262 | 236 / 236 | 266 / 264 | 258 / 257 | 267 / 255 |
| Original shared own | 1955 | 0 / 0 | 44 / 42 | 33 / 34 | 44 / 41 |

On the original shared cohort of 1,955 readouts, A corrects 44 raw readouts (28 / 5 / 11 by seed), B 33 (0 / 14 / 19), and AB 44 (23 / 16 / 5). Calibration gives 42, 34 and 41 respectively. These are real repairs of original failures. They are offset by full-roster harms and cannot establish a complete-rule success.

At each new cell's own current raw strict-rival set, the served pool corrects zero: A 0 / 1,929, B 0 / 1,975, AB 0 / 1,958. This describes the same selected bank where every member strictly favors a shared wrong rival. It does **not** mean training fixed none of the original wrong cases. The fixed original cohort answers that historical question and shows nonzero cross-procedure repairs. No calibration-only reader can acquire a member alternative; the positive temperature preserves member top classes.

## Interaction and complementary repairs

The accuracy interaction is AB − A − B + original shared. It is descriptive for complete independently selected procedures, not a fixed-state ingredient intervention or an alternative superiority gate.

| Readout | Mean accuracy interaction (pp) | Seed interactions (pp) | A-only repairs | A-only kept by AB | B-only repairs | B-only kept by AB | Shared A/B repairs | Shared kept by AB | New AB repairs where A/B wrong | AB harms where base/A/B correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | -0.056883 | +0.056883, -0.037922, -0.189609 | 199 | 120 | 149 | 31 | 131 | 94 | 108 | 132 |
| calibrated | -0.018961 | +0.208570, -0.246492, -0.018961 | 191 | 114 | 149 | 33 | 127 | 92 | 109 | 126 |

The raw interaction is −0.056883 pp with a descriptive interval [−0.365749, +0.251983] pp and exact sign-flip p = 0.75. Calibrated interaction is −0.018961 pp, interval [−0.584180, +0.546258] pp, p = 0.75. There is no positive accuracy synergy claim. Negative raw NLL / Brier interactions (−0.076744 / −0.003374) describe less than additive deterioration; AB's actual losses still exceed original shared and B.

Raw AB retains 245 of the 479 repairs available in the A/B union and loses 234; it adds 108 repairs absent from both A and B. Only 31 of 149 B-only repairs survive. Its 132 harms on nodes where original shared, A and B were all correct are a direct counterweight to that complementarity. Calibrated AB retains 239 of 467 union repairs, adds 109 new repairs, and has 126 such harms. Neither repair overlap nor a retrospective oracle combination is a deployable or frozen-selected rule; no post hoc member count, class rule, or arm composition is selected here.

### Calibration alone at each fixed native selected state

This is the one comparison that does hold the native selected bank fixed. Every member top class and coverage count remains fixed; only the equal-policy OOF temperature readout changes.

| Procedure | Repairs | Harms | Net pooled correct | Accuracy change (pp) | NLL change |
| --- | --- | --- | --- | --- | --- |
| A: committee CE | 64 | 62 | 2 | +0.012641 | -0.508527 |
| B: own CE + SupCon | 50 | 41 | 9 | +0.056883 | -0.410985 |
| AB | 55 | 42 | 13 | +0.082164 | -0.528167 |
| Original shared own | 54 | 50 | 4 | +0.025281 | -0.319428 |

These temperature changes may improve loss and change averaging decisions, but they do not create member alternatives or undo the raw gate. The OOF folds follow VALID-selected native banks, so these are encountered development estimates, not unused confirmation.

## Favorable I4 comparisons and accumulated history

All A / B / AB raw required I4 contrasts are favorable at every seed. They remain in the record alongside complete-rule failures. The old shared procedure already beats the original ordinary and factorized I4 banks; it also beats the fresh SupCon I4. Therefore, beating I4 with a new ingredient cannot by itself identify that ingredient as an improvement over shared own. The same distinction applies to the capable factorized single comparison. The original shared mean raw advantage is +0.549867 pp over original ordinary I4, +0.486664 pp over original factorized I4, and +0.524586 pp over fresh SupCon I4 (aggregate means, not an added qualification comparison).

| Candidate | Genuine I4 reference | Mean raw gain (pp) | Raw repairs | Raw harms | Mean calibrated ΔNLL |
| --- | --- | --- | --- | --- | --- |
| A: committee CE | Original ordinary I4 | +0.543547 | 723 | 637 | +0.039117 |
| A: committee CE | Original factorized I4 | +0.480344 | 685 | 609 | +0.040625 |
| A: committee CE | Genuine factorized I4 + SupCon | +0.518266 | 691 | 609 | +0.038509 |
| B: own CE + SupCon | Original ordinary I4 | +0.632031 | 727 | 627 | +0.036992 |
| B: own CE + SupCon | Original factorized I4 | +0.568828 | 681 | 591 | +0.038499 |
| B: own CE + SupCon | Genuine factorized I4 + SupCon | +0.606750 | 693 | 597 | +0.036384 |
| AB | Original ordinary I4 | +0.568828 | 741 | 651 | +0.050615 |
| AB | Original factorized I4 | +0.505625 | 692 | 612 | +0.052122 |
| AB | Genuine factorized I4 + SupCon | +0.543547 | 700 | 614 | +0.050006 |

The favorable pooled advantage can coexist with weaker mean members and greater coverage: pooled utility remains decisive. Confidence tradeoffs remain explicit. These three seeds, this encountered development set, and the supplied independently optimized/selected I4 definition set the scope; no broader superiority or novelty claim follows.

The earlier complete neighborhood-distribution recipe remains closed under its own frozen criteria, as recorded in `native_neighborhood_distribution_complete_closure_history_note_20261010_v1`. Its favorable I4 contrast and original-case cross-procedure repairs were preserved despite its inadequate incremental shared gain and confidence failure. The present supervision study is a separate completed development experiment. It does not rescue that recipe, reopen its grid, or authorize ranking and recombining completed methods retrospectively. Original paper scores remain unchanged.

## Costs and completion

The six fresh families preserve 12,078 executed native optimizer updates across the 27 fits and 436.099248 s summed acquisition time (nested timing, not a new wall-time estimate). All full-roster native operation, graph-trajectory, SupCon Gram and selected-serving cost records remain in the source JSON and evidence extract. The 165 successful scalar fits add 225.764207 s summed fit time within 228.659902 s calibration wall time, no native model fits or forwards, no retries and no final refit. Reused original costs retain their original group / selected-fit provenance and are not relabeled as fresh work.

## Next implications — three only

1. Close the exact A / B / AB coefficients and SupCon temperature under their unchanged frozen rules. No cell earns accuracy-only or joint-quality confirmation priority, a TEST opening, or a Coauthor CS launch; no coefficient, temperature, projector, optimizer or class-subset rescue grid follows this result.
2. Preserve the original shared-versus-genuine-I4 pooled advantage as an existing SAGE clue, together with its confidence limits. The new ingredients did not establish a worthwhile incremental shared improvement. Member weakening is not an independent veto, and favorable I4 differences are not discarded.
3. Treat lost alternatives and complementary repairs as unresolved mechanism evidence. They authorize no post hoc member count or arm combination. A distinct future learning rule would need a prospectively frozen complete served-pool comparison with capable same-objective M1 / genuine I4 and the unchanged shared baseline before any unused confirmation; one robust SAGE result remains sufficient.

## Bindings and limitations

`READ_BINDINGS.json` identifies the exact admitted compact files, receipts, freeze and DECISION read for this note. `EVIDENCE_EXTRACT.json` retains exact unrounded statistics for the full arm roster, raw / calibrated paired contrasts, original cohorts, interactions, same-state calibration and frozen decisions. It contains report statistics, not model states, raw prediction arrays, masks or TEST data. `MANIFEST.json` hashes this note and its companion artifacts. Reader/source qualification and custody remain the previously admitted v2 review; this interpretation adds no parity, proof replay, exposure audit or launch authorization.
