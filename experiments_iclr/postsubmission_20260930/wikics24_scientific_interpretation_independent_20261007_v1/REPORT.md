# Independent interpretation of the closed WikiCS 24-cell family

Evidence: the closed accuracy report and selected-state `PER_CELL`, `PER_SEED`, `PAIRED_REPAIRS`, and `ERROR_SUMMARY` artifacts. All eight arms and three seeds are complete. These are outcomes on the **5,274-node split0 merged validation + stopping development population that selected checkpoints**, on one graph. All comparisons below are descriptive; percentage-point differences are abbreviated pp. Seed-node totals count repeated instances, not independent observations.

## What changed

Equal-seed means; error counts are mean counts per 5,274-node seed block. “Coverage” means at least one member predicts truth; it is an unavailable oracle, not a deployable score. Pool benefit is served accuracy minus mean member accuracy.

| Arm | Member mean % | Served % | Pool benefit pp | Coverage % | All members wrong | Pool harm |
|---|---:|---:|---:|---:|---:|---:|
| single | 81.475 | 81.475 | 0 | 81.475 | 977.0 | 0 |
| single + contrast | 81.576 | 81.576 | 0 | 81.576 | 971.7 | 0 |
| independent4 | 81.440 | 82.044 | 0.604 | 85.754 | 751.3 | 195.7 |
| independent4 + contrast | 81.034 | 81.993 | 0.959 | 85.830 | 747.3 | 202.3 |
| unit factors | 81.102 | 81.128 | 0.025 | 81.355 | 983.3 | 12.0 |
| initialized factors | 79.903 | 80.540 | 0.637 | 84.022 | 842.7 | 183.7 |
| unit factors + contrast | 81.583 | 81.633 | 0.051 | 81.816 | 959.0 | 9.7 |
| initialized factors + contrast | 79.884 | 80.717 | 0.833 | 84.104 | 838.3 | 178.7 |

**Unit factors remain close to redundant predictors.** Contrast improves their pool by 0.506 pp, with positive differences in all three seeds (0.531, 0.588, 0.398 pp). Member mean improves 0.480 pp; pool benefit increases only 0.025 pp. The members have narrow accuracy spreads, and almost all their all-member errors are common wrong-competitor errors. The observed gain is mainly stronger members with little extra benefit from their aggregation. This does not identify which loss term supplied that strength.

**Rademacher initialization buys coverage at a substantial competence cost.** Relative to unit factors, initialized factors increase coverage 2.667 pp and reduce all-member-wrong counts by 140.7 nodes per seed, but member mean loses 1.199 pp and served accuracy loses 0.588 pp in all three seeds. Pool harm rises from 12.0 to 183.7 nodes per seed. The pool-minus-member gain is real as a score difference, yet it cannot offset weaker predictors. Against ordinary independent4, initialized factors have both weaker members and less coverage. The same pattern persists with contrast: init + contrast trails unit + contrast by 0.916 pp and independent4 by 1.327 pp.

**The primary contrast mostly redistributes errors.** Init + contrast minus init is +0.177 pp overall, with seed effects +0.284, +0.417, -0.171 pp. There are 349 repairs and 321 harms (net 28) across 15,822 seed-node instances. Member mean changes -0.019 pp, but this hides damage: five of twelve matched member scores decrease; the mean worst-member accuracy falls 79.465% to 78.795%, and mean within-bank accuracy spread rises 0.847 to 1.656 pp. Pooled NLL changes +0.00491 (worse), with seed signs mixed. There is no supported claim of a reliable competence-preserving primary improvement.

On the frozen baseline common-competitor cohort, 315/2,503 instances acquire at least one correct candidate member, but only 128 are correctly served; pooling loses the correct evidence on the other 187. Common-competitor support within that cohort falls 2,503 to 2,174, while 316 instances outside it enter candidate common-competitor support. The full-population count therefore falls only 13, from 2,503 to 2,490. The 128 repairs demonstrate some task-relevant changes, but neither they nor the favorable cohort NLL mean (-0.1883) erase the new errors elsewhere. Cohort NLL improvement is not uniform across seeds.

**The ordinary controls preserve the negative evidence.** Independent4 has the best mean served accuracy. Adding contrast lowers its member mean 0.406 pp and pool accuracy 0.051 pp, while increasing pool-minus-member benefit: a larger ensemble benefit can accompany a worse ensemble. Single + contrast has a small +0.101 pp mean gain with signs -/-/+, so the unit-arm result is not a universal contrastive effect. Ordinary independent4 uses four individually selected checkpoints; the contrastive untied arm and BE arms use joint selection. Their comparison is not an isolated loss intervention.

## The shared-error barrier

Every cell has zero rescues where all current members are wrong. Almost every all-member-wrong instance admits the same nontruth class strictly above truth in every member: for init, 2,503/2,528; for init + contrast, 2,490/2,515. If each member assigns that class a greater probability than truth, any convex combination of the unchanged member probabilities does too. Reweighting existing predictions therefore cannot repair this support. Reducing unanimous argmax errors alone is insufficient: members can disagree about the top class while retaining a common class above truth. Corrective training must change class rankings or introduce new evidence. The available summaries contain no neighborhood, attention, or intervention evidence establishing a graph-specific mechanism.

## Two precise next hypotheses

1. **The unit-arm gain can survive removal of residual member repulsion.** Test prospectively fixed alignment-only and residual-only objectives against the already defined combined objective and plain unit reference, retaining identical two-view own supervision and selection opportunity. Support would be alignment-only retaining the member and served gains while residual repulsion adds no reduction in common-competitor support; the reverse result would falsify the hypothesis. The present tables only motivate this test, because latent separation and term attribution were not measured. Closest established ideas are supervised contrastive/stochastic-view alignment, GRACE-style graph-view consistency, and CDLG-style cross-channel contrast. The current two-term proxy is not an exact reproduction of these methods.

2. **Small asymmetric first-factor perturbations can preserve competence while adding useful predictive coverage.** Compare one prospectively fixed near-unit first-input-factor initialization to the unit and Rademacher references, with the same loss, architecture, horizon, and selection rule. Require improvements in served accuracy and common-competitor support while protecting mean and worst-member competence; increased coverage or latent distance alone does not count. This directly tests the tradeoff observed between unit and Rademacher endpoints; no intermediate initialization has been tested. Learned rank-one factors and first-adapter initialization are already BatchEnsemble/TabM territory. DICE's label-preserving diversity and FoRDE's functional diversity are close objective-level precedents if a subsequent method targets predictive margins rather than representation separation. This is an ablation direction, not a novelty claim.

The primary contrast is the complete alignment + residual package; it isolates neither term, GNCL, nor internal versus boundary factor placement. All these scores share development selection optimism, three optimizer seeds, one official split, and one graph. They establish no TEST result, generalization gain, causal mechanism, graph-wide superiority, or efficiency advantage. No training, remote access, checkpoint reselection, recalibration, or source modification was performed for this interpretation.

## Source anchors

- `Wiki24_analysis_after_closure_execution_root_20261007_v1/readout/REPORT.md`.
- `Wiki24_selected_analysis_after_closure_execution_root_20261007_v1/predictions/compact/{PER_CELL,PER_SEED,PAIRED_REPAIRS,ERROR_SUMMARY}.json`.
- `learnable_internal_be_contrastive_multitask_suite_20261007_v6/{PROTOCOL.md,ATTRIBUTION.md,factors.py,objectives.py,configs/wikics.json}`; v6 protocol records the WikiCS v4 validity and unchanged model/loss/factor bytes. The v4 `models.py` confirms the arm dispatch and first-factor initialization.

Established-method names above follow the retained source protocol and general method identities; no new paper review or exact prior-art clearance was performed.
