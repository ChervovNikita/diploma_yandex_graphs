# Complete Tolokers private sheaf pilot readout

Both predeclared shared recipes fail the unchanged quality gate. Vanilla underperforms the genuine independent ensemble and its fixed member 0 at all three seeds. The identity-centered prior improves the shared model, chiefly at 7409 and 8501, but does not recover independent-ensemble quality. The useful descriptive diagnosis is a vanilla member-learning deficit, followed by insufficient complementary predictions in the centered bank. This closed pilot supplies no basis for advancement under its frozen rule.

All 21 records closed before opening. The unchanged audited V2 reader processed every member and all official TRAIN/VALID rows, with zero new fits, forwards or predictions. Original source, thresholds, selected scores and checkpoints were unchanged.

## Four frozen comparisons

Seed order is 7409, 8501, 9607. Values below are candidate minus comparator; positive AUROC is better and positive NLL is worse. AUROC is in its original fraction units.

| Contrast | VALID AUROC differences | Mean | Sample SD | Descriptive df2 95% interval |
|---|---|---:|---:|---|
| vanilla_pool_minus_independent_pool | -0.012885, -0.029476, -0.012588 | -0.018316 | 0.009665 | [-0.042326, 0.005693] |
| vanilla_pool_minus_independent_member0 | -0.003375, -0.019677, -0.009166 | -0.010739 | 0.008264 | [-0.031268, 0.009789] |
| centered_pool_minus_independent_pool | -0.008631, -0.005889, -0.012541 | -0.009020 | 0.003343 | [-0.017325, -0.000715] |
| centered_pool_minus_independent_member0 | 0.000878, 0.003910, -0.009119 | -0.001443 | 0.006818 | [-0.018380, 0.015493] |

| Contrast | VALID NLL differences | Mean NLL difference | Mean accuracy difference | Mean Brier difference |
|---|---|---:|---:|---:|
| vanilla_pool_minus_independent_pool | 0.011509, 0.036294, 0.007715 | 0.018506 | -0.013610 | 0.007145 |
| vanilla_pool_minus_independent_member0 | 0.005623, 0.028421, 0.006426 | 0.013490 | -0.014291 | 0.005167 |
| centered_pool_minus_independent_pool | 0.003582, 0.003814, 0.012308 | 0.006568 | -0.003062 | 0.002609 |
| centered_pool_minus_independent_member0 | -0.002304, -0.004059, 0.011019 | 0.001552 | -0.003743 | 0.000630 |

The gate requires every seed AUROC gain to be positive, mean gain at least 0.003, and mean VALID NLL difference at most 0 for both comparisons of each recipe. All three aggregate checks fail in each of the four contrasts. Centered versus member 0 has two positive AUROC cells; the negative 9607 cell makes its mean negative. Those positive cells remain in the complete report.

Centering changes pooled VALID AUROC from vanilla by [0.004254, 0.023587, 0.000048] and NLL by [-0.007926, -0.032480, +0.004593]. It repairs the large 8501 deficit, but worsens 9607 NLL and accuracy. Selecting a favorable recipe/seed or treating the centered-versus-pool interval as confirmatory would change the frozen interpretation. These intervals describe three optimizer seeds on one already-used split, with VALID checkpoint selection; they do not establish graph-population uncertainty or statistical superiority.

## Every member and learning

All four VALID AUROCs and own selected epochs are listed in member order 0–3. Shared members inherit one pooled selected epoch; independent members have their own native selections. Full TRAIN/VALID AUROC, NLL, accuracy and Brier for all 90 member/pool rows, plus all complete histories, remain in the reader artifacts.

| Recipe/seed | Member 0–3 VALID AUROC | Selected epochs 0–3 | Mean TRAIN eval NLL | Mean / worst VALID AUROC |
|---|---|---|---:|---|
| vanilla/7409 | 0.844225, 0.842738, 0.842125, 0.843332 | 264, 264, 264, 264 | 0.374956 | 0.843105 / 0.842125 |
| centered/7409 | 0.848317, 0.847907, 0.848175, 0.848009 | 439, 439, 439, 439 | 0.344977 | 0.848102 / 0.847907 |
| independent/7409 | 0.847274, 0.850145, 0.851268, 0.844976 | 182, 246, 400, 477 | 0.364297 | 0.848416 / 0.844976 |
| vanilla/8501 | 0.823078, 0.826583, 0.826318, 0.828806 | 159, 159, 159, 159 | 0.400973 | 0.826196 / 0.823078 |
| centered/8501 | 0.850053, 0.850104, 0.850041, 0.849629 | 307, 307, 307, 307 | 0.354497 | 0.849957 / 0.849629 |
| independent/8501 | 0.846135, 0.845924, 0.852255, 0.843343 | 358, 279, 423, 334 | 0.355072 | 0.846914 / 0.843343 |
| vanilla/9607 | 0.838973, 0.841151, 0.842661, 0.841242 | 473, 473, 473, 473 | 0.366318 | 0.841007 / 0.838973 |
| centered/9607 | 0.842113, 0.842426, 0.842025, 0.842149 | 177, 177, 177, 177 | 0.371809 | 0.842178 / 0.842025 |
| independent/9607 | 0.851328, 0.843626, 0.850311, 0.849052 | 499, 238, 297, 198 | 0.360808 | 0.848579 / 0.843626 |

Vanilla mean TRAIN eval-NLL exceeds independent mean by [0.010658, 0.045902, 0.005510]; mean VALID member AUROC is lower by [0.005311, 0.020718, 0.007573]. This supports an actual learning deficit, particularly 8501, rather than an explanation based solely on an ineffective pooling rule. All members learn, but they do not match the independent members across this panel.

Centering moves those TRAIN NLL differences to [-0.019321, -0.000574, +0.011001] and VALID member AUROC differences to [-0.000314, +0.003043, -0.006401]. Its 7409/8501 own quality is close to or better than the independent mean; 9607 remains deficient. The prediction diagnostics cannot distinguish a capacity limit, shared optimization, prior effects or selection-policy effects. Shared active parameter count 16,594 versus independent total 57,160 is relevant context, not a causal capacity test.

## Complete error coverage and ranks

VALID contains 2939 rows, including 641 positive and 2298 negative rows. The reader visits all 1, 473, 018 positive-negative pairs per bank, including ties. Pairwise rank disagreement ranges below use all six within-bank member pairs. Oracle any-member-correct coverage is an argmax diagnostic and is not an AUROC score.

| Recipe/seed | All 4 wrong | Any correct | Pool wrong | Correct coverage lost by pool | Unique-correct rows 0–3 | Member rank disagreement range |
|---|---:|---:|---:|---:|---|---|
| vanilla/7409 | 528 | 2411 | 548 | 20 | 3, 2, 3, 7 | 1.571128–2.928002% |
| centered/7409 | 521 | 2418 | 531 | 10 | 1, 2, 0, 2 | 0.480646–0.619544% |
| independent/7409 | 449 | 2490 | 532 | 83 | 25, 6, 10, 22 | 6.376433–9.208102% |
| vanilla/8501 | 604 | 2335 | 617 | 13 | 1, 1, 3, 0 | 1.284641–2.800034% |
| centered/8501 | 522 | 2417 | 540 | 18 | 2, 3, 6, 1 | 0.569375–0.939568% |
| independent/8501 | 429 | 2510 | 532 | 103 | 22, 15, 14, 17 | 7.592100–9.420523% |
| vanilla/9607 | 461 | 2478 | 557 | 96 | 17, 3, 30, 26 | 1.873772–3.768182% |
| centered/9607 | 550 | 2389 | 558 | 8 | 4, 1, 1, 0 | 0.396940–0.630542% |
| independent/9607 | 463 | 2476 | 538 | 75 | 9, 11, 23, 10 | 5.844260–7.526113% |

Centering yields only 5/12/6 uniquely correct rows across the four members, compared with 63/68/53 for independent bodies. Centered pairwise rank disagreement is 0.397–0.940%, versus 5.844–9.421% independently. Pool-minus-own-mean VALID AUROC is only [0.000050, 0.000089, 0.000031] centered, compared with [0.008368, 0.009020, 0.006171] independently. The fixed probability mean does exploit useful independent ranking differences; there is little such ensemble benefit available in the centered predictions.

The centered pool loses 10/18/8 rows of any-member-correct coverage; independent pooling loses 83/103/75. This does not establish that centered pooling is the dominant failure: centered members offer less oracle coverage and have larger common-error intersections. Vanilla 9607 is a localized exception: it has 2478 any-correct rows, slightly above independent 2476, yet loses 96 of those rows versus 75 independently and has worse pool accuracy. Its mean member AUROC is still weaker, so that argmax coverage example cannot rescue the primary AUROC result.

| Recipe/seed | Pool rescues member 0–3 | Pool harms member 0–3 | Pool wrong mean NLL | Pool wrong mean true-class probability |
|---|---|---|---:|---:|
| vanilla/7409 | 7, 7, 9, 10 | 7, 4, 5, 10 | 1.161995 | 0.333801 |
| centered/7409 | 4, 4, 2, 2 | 2, 6, 3, 4 | 1.309709 | 0.299807 |
| independent/7409 | 45, 25, 28, 36 | 41, 22, 21, 25 | 1.254504 | 0.310507 |
| vanilla/8501 | 1, 8, 3, 6 | 6, 4, 8, 3 | 1.324772 | 0.291785 |
| centered/8501 | 5, 2, 1, 4 | 6, 6, 9, 3 | 1.215437 | 0.327223 |
| independent/8501 | 35, 33, 42, 42 | 47, 27, 31, 36 | 1.229332 | 0.316620 |
| vanilla/9607 | 38, 18, 26, 20 | 30, 21, 39, 35 | 1.158423 | 0.347223 |
| centered/9607 | 3, 2, 3, 3 | 5, 2, 2, 2 | 1.290180 | 0.298959 |
| independent/9607 | 20, 39, 33, 23 | 18, 26, 35, 23 | 1.263393 | 0.307275 |

All correct/wrong/rescued/harmed confidence and NLL subsets, correct-member-count histograms, error overlaps and rank/tie counts remain in ANALYSIS.json for both TRAIN and VALID. These subsets differ between models; their conditional means are descriptive and are not matched causal effects. No threshold, calibration, reweighting or selected-example rescue was fitted.

## Actual costs and custody

| Recipe/seed | Complete fit and fresh serving seconds | Active parameters | CUDA allocated peak GiB | Serving residency |
|---|---:|---:|---:|---|
| vanilla/7409 | 3404.82 | 16594 | 14.420 | four persistent cache sets |
| vanilla/8501 | 3285.63 | 16594 | 14.418 | four persistent cache sets |
| vanilla/9607 | 2461.17 | 16594 | 14.418 | four persistent cache sets |
| centered/7409 | 2431.98 | 16594 | 14.420 | four persistent cache sets |
| centered/8501 | 2580.56 | 16594 | 14.418 | four persistent cache sets |
| centered/9607 | 1973.46 | 16594 | 14.419 | four persistent cache sets |
| independent/7409 | 4432.63 | 57160 | 11.813 | one native body streamed |
| independent/8501 | 2611.42 | 57160 | 11.813 | one native body streamed |
| independent/9607 | 2428.63 | 57160 | 11.813 | one native body streamed |

Actual per-bank fit/serving totals are 9151.62 s vanilla, 6986.00 s centered and 9472.68 s for the genuine independent references. The references are physically fitted once and fully charged when reused. Inclusive original 18 owner wall is 18631.99 s and centered 3 owner wall 6993.10 s; retain these outer scopes without adding their nested cell times again. Owner CPU user/system totals are 18583.55/78.54 s and 6969.25/26.34 s. Physical sampled GPU high-water is 17, 431, 527, 424 bytes for both owners; child RSS peaks are 1, 541, 210, 112 and 1, 521, 037, 312 bytes.

These are actual unequal early-stopping trajectories and differing serving residency. Vanilla has no uniform wall advantage over independent fitting; centered uses fewer observed fit seconds but fails quality. Smaller parameter storage does not imply smaller native-cache memory or a useful efficiency advantage. The CPU reader itself took 12.51 s wall, 9.84/0.74 s user/system and 437, 866, 496 bytes child RSS; its complete outputs were4.38 MB, and no raw checkpoint/logit file was downloaded.

At 17:56:46 UTC, bound boot24c315a7-3c08-471f-b550-b9a3e1faf75d showed both exact original and centered parent/worker PIDs absent, both parent/worker groups empty and their CUDA contexts absent. Both exact terminal records report complete=true, exit0 and successful reaping. Custody was rechecked before invocation. The exact original18 COMPLETE/origins, centered closure/records and all nine serving references plus 18 histories passed the unchanged reader provenance checks. Other workloads were untouched.

## Decision and existing hypotheses

The predeclared centered prior already tests the specific objective-accounting concern: averaging private likelihood gradients by 1/4 while applying unscaled zero-centered fast decay. Its learning repair at two seeds is compatible with that concern, but its complete negative gate shows that identity centering at the fixed prior is insufficient for the requested predictive-quality advantage. Repeating this prior adjustment or choosing another coefficient after these outcomes would be a new tuned study, not a rescue of this fixed pilot.

For a future distinct explanation, the frozen tied-four-path/dropout-budget and additional-parameter/joint-transport controls remain the relevant discriminators between shared learning/capacity and complementary native models. Authentic BSNN, structured latent and source/frame controls remain required for stronger persistent-mode or functional-geometry claims. This readout runs none of them and does not infer cause from factor norms, raw map disagreement or output diversity. The fixed two-recipe quality direction should be recorded as negative; any next study needs its own prospective hypothesis and admission. Root owns publication and subsequent decisions.

Artifacts: complete_analysis/ANALYSIS.json is the unchanged full reader report; EVERY_MEMBER_QUALITY.csv contains every member/pool metric; REPORT.md is the reader summary. READOUT_SUMMARY.json extracts the complete contrasts and all nine VALID banks for review. Release, closure, terminal, execution and fetch metadata retain exact provenance. No existing source or top-level note was modified and no commit was made.
