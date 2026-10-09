# Closed molecular candidate interpretation

Recommend **no advancement of fixed candidate I**. It loses served ROC AUC to independent4 and single in every paired seed. The whole-family frozen verdict remains **unavailable**, because P_7307 failed; no complete-family success or frozen gate pass is claimed. Root owns the final decision and publication.

The saved reader restored/served all 17 available cells successfully: 1,947/1,947 member calls, no collection errors, 38.246941 seconds. All 18 declared cells are accounted for. P_7307 timed out in the original fit (exit −15), was never served, and was neither shortened nor retried. The reader intentionally exited 1 after writing analysis/cost with `complete18_readout=false`; its wrapper was reaped without timeout. This is successful collection of available evidence within an incomplete scientific family.

## Fixed practical comparisons

The adoption fixes I, λ=0.5, 100 epochs, seeds 7101/7203/7307, official scaffold TRAIN/development roles and strict-first maximum full-development selection. Ordinary independent4 retains each body’s own selected state. Serving uses the source FP32 mean **raw logits**; sigmoid is used only for proper-score/calibration summaries. No probability averaging, new calibrator, reselection or score-parity gate was introduced.

| AUC contrast | Seeds 7101,7203,7307 (pp) | Mean (pp) |
| --- | --- | --- |
| I_minus_independent4 | -1.954, -3.564, -2.542 | -2.687 |
| I_minus_single | -0.135, -2.050, -1.133 | -1.106 |
| I_minus_O | +0.663, -1.537, +0.219 | -0.219 |
| I_minus_G | +1.218, -1.057, -0.323 | -0.054 |
| I_minus_P | +0.131, +0.537, missing | unavailable |
| P_minus_G | +1.087, -1.595, missing | unavailable |

The complete primary loss is −2.687 pp and the required single loss −1.106 pp. I−O is mixed with mean −0.219 pp; I−G is mixed with mean −0.054 pp. Neither supports useful restriction benefit. Missing third I−P/P−G pairs remain null, with no two-seed means. Complete negative candidate/reference contrasts justify withholding investment in this fixed recipe; they do not replace the unavailable whole-family formal decision. No favorable policy, seed or diagnostic substitutes for the frozen primary endpoint.

## Members and realized pooling

| Seed | Model | Member mean | Worst | Best | Pool AUC | Pool−mean (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| 7101 | I | 0.808112 | 0.797821 | 0.822032 | 0.820681 | +1.257 |
| 7101 | independent4 | 0.821733 | 0.811820 | 0.833655 | 0.840220 | +1.849 |
| 7203 | I | 0.810439 | 0.802775 | 0.816089 | 0.814656 | +0.422 |
| 7203 | independent4 | 0.828413 | 0.823015 | 0.834105 | 0.850293 | +2.188 |
| 7307 | I | 0.808548 | 0.805877 | 0.812916 | 0.819157 | +1.061 |
| 7307 | independent4 | 0.827112 | 0.815008 | 0.832210 | 0.844577 | +1.747 |

I member-mean AUC is lower than independent4 by 1.362,1.797,1.856 pp, and lower than single in every seed; even I’s best member is below single in every seed. I gains AUC over its member mean. This is realized pooling lift, whose size also depends on member strength and score scales. The lift is smaller than independent4 in every seed, and I pool is below its best member in seeds 7101/7203. Weaker members and limited pooled ranking quality both appear descriptively. These full-population scores cannot causally divide representation weakness, dependence, optimization or score-scale effects, or quantify error correlation.

## Common-inversion repair versus full-population harm

Every seed evaluates all 326,592 positive-negative pairs from 81 positives and 4,032 negatives; no pair sampling occurred. O-only cohort identities were frozen before non-O serving. Common inversions are pairs with no strict O-member win. Acquired strict coverage means at least one I member now ranks the positive above the negative; it is a pair oracle, not proof of new causal graph evidence.

| Seed | O common inversions | New strict coverage | Covered/pool win | Covered/pool loss | All-pair repairs | All-pair harms | Net I−O (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7101 | 40621 | 13960 | 5476 | 8484 | 13873 | 11709 | +0.663 |
| 7203 | 37190 | 11663 | 6503 | 5160 | 14637 | 19658 | -1.537 |
| 7307 | 41026 | 23223 | 16939 | 6284 | 27084 | 26370 | +0.219 |

Common-inversion pool repairs are real, yet 8,484/5,160/6,284 newly covered pairs still lose under mean-logit serving. Across the full population, repaired rankings are offset by 11,709/19,658/26,370 newly harmed rankings; seed 7203 has net −5,021 pair credits. All shown pool transitions are strict wins/losses; the reader retains half credit for ties. Common-inversion gains therefore cannot establish overall benefit. The full-population net reproduces the saved mixed I−O AUC contrast, rather than promoting the favorable subset.

Matching member orders change in all seeds (the saved all-orders-preserved flag is false), so some member ranking acquisition is observed. This excludes describing the intervention as merely preserving every member order while changing pool scale. Relative logit scales can still affect which member dominates a pair’s mean margin; the diagnostic does not identify scale, graph evidence or correlation as the cause. No scale normalization or calibration rescue is performed.

## PR, probability and calibration evidence

Versus independent4, I average precision loses in all seeds (−8.011,−9.088,−14.032 pp; mean −10.377 pp); trapezoidal PR area also loses in all (mean −10.519 pp). Versus single, AP is mixed with mean −2.913 pp; versus O it is mixed with mean −1.531 pp. AP and trapezoidal PR area use different interpolation and are not interchangeable; the positive prevalence is 1.969%. Their exploratory values do not replace ROC AUC.

I’s served BCE is lower than independent4 in all seeds (mean Δ −0.008288), while Brier is worse in all (mean Δ +0.001111). Positive-event BCE improves but negative-event BCE worsens in all three seeds against independent4. The probability losses therefore favor different outcomes. ECE’s mean change is −0.002444 but ten-bin ECE is descriptive, and occupied-bin MCE is sensitive to small bins. Mean-probability/prevalence agreement describes marginal calibration only. The mean-logit BCE Jensen gap can arise from convex averaging and cannot establish member competence or useful diversity. No independently validated calibration claim follows.

## Costs and interpretation limits

Original family inclusive wall time is 187621.665497s; parent terminal scope is 187621.773610s. Failed P_7307 retains 32390.328045s and all recorded CPU/RSS costs. The readout adds its own saved scope: 38.246941s, CPU user/system 22.996795/3.776471s, CUDA peaks 102620160/132120576 bytes, process RSS high-water 1092055040 bytes. The wrapper scope is 39.571617s. Nested fit/cell/family/parent/readout-wrapper timings are not summed or relabeled; no efficiency benefit is established.

There are three optimizer seeds on one fixed selected development split, no independent TEST evidence. Reused molecules, pair comparisons and members are not independent repetitions. Saved df2 t intervals are fragile exploratory summaries, not confirmatory significance or graph/population generalization. Preserve the failed control, all policies, all member/PR/calibration/transition details and original costs. Do not advance I, average missing-control survivors, tune λ/horizon or substitute another policy. This report used local saved JSON only, with no server/raw model/array access, forward, refit or original-paper recalculation.

`DECISION_INPUTS.json` binds ANALYSIS/COST/COLLECTION, adoption, O-cohort/role custody, execution receipt and original molecular closure/ledger/terminal artifacts by exact SHA256 and paths. It records compact decision fields and all 18 original cell costs; full rows/bins remain in the bound primary artifacts.
