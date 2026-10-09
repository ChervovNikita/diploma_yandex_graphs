# Contrastive learning in the selected shared-backbone representations

10 October 2026. Completed descriptive analysis of all 15 declared endpoints. This is an author-side scientific interpretation, not a fresh manuscript review. Original scores are unchanged.

## What was measured

Each selected model supplied four pre-classifier vectors per node, each with 512 coordinates, on the same 5,274 WikiCS development nodes. P is the plain shared ensemble, A the within-member class alignment objective, R the class-centered contrast between members, C their combination, and S canonical supervised contrast. All three optimizer seeds and all five conditions were retained. The analysis used cached representations, logits and the corresponding selected classifiers. It fitted nothing and did not access TEST.

The class cosine gap is the mean same-class cosine minus the mean different-class cosine within each member. A larger gap describes more class separation in these coordinates. The residual cosine compares different members' representations of the same node after removing each member's class mean. A smaller value describes more angular separation. Neither quantity is accuracy or a guarantee of useful complementary decisions.

| Seed | Condition | Selected epoch | Stage | Class cosine gap | Residual member cosine | Head-visible pair RMS | Class-margin pair RMS |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
|6101|P|130|global|.484000|.999761|.044538|.122137|
|6101|A|68|local|.501936|.999673|.040387|.087530|
|6101|R|132|global|.467034|.999826|.035341|.101258|
|6101|C|58|local|.485696|.999709|.029255|.068443|
|6101|S|58|local|.488689|.999720|.029153|.066136|
|6203|P|127|global|.474426|.999686|.043610|.109476|
|6203|A|60|local|.480271|.999676|.032451|.069837|
|6203|R|148|global|.461815|.999794|.050225|.159648|
|6203|C|131|global|.497443|.999798|.038009|.102386|
|6203|S|1073|global|.658325|.994111|.378165|1.351525|
|6307|P|66|local|.440974|.999668|.034034|.077598|
|6307|A|53|local|.482781|.999693|.027618|.061885|
|6307|R|133|global|.478707|.999845|.037549|.109923|
|6307|C|63|local|.505928|.999581|.042060|.099794|
|6307|S|50|local|.478495|.999653|.032603|.074941|

RMS quantities are in the corresponding classifier score coordinates. They are not probabilities. Full precision and definitions remain in `result/AGGREGATE.json` and the bound source.

## Scientific conclusion

A and C increase the within-member class cosine gap in all three seeds. However, R is more aligned than P in all three selected endpoints, and C changes between-member alignment inconsistently. These objectives changed some class geometry without producing stable useful member differentiation. The previously collected complete C-minus-P accuracy contrast remains -0.0695 percentage points on average. This diagnostic does not replace or recalculate it.

S/6203 has a lower residual cosine and larger differences visible through its classifier. Its previously recorded correct-member coverage remains 4,308, pooling harms rise from 5 to 28, the served prediction loses 23 correct nodes, and NLL worsens by 1.70170 nats. Thus separation can reach the classifier and still fail to improve the ensemble. A hidden-nullspace-only explanation is insufficient for this endpoint.

Selected epochs and stages differ. TRAIN-view representations, their original centering values and optimization trajectories were not cached. These endpoint observations cannot establish failed auxiliary optimization or a causal account of training. Hidden angles and null-energy fractions depend on the representation coordinates. Class-margin differences use the actual corresponding classifier and remain subject to the stated selection limits.

## Consequence for the user's renewed proposal

The backbone can remain shared and trainable while member-specific BatchEnsemble factors produce different intermediate representations. Within a member, same-label or same-object positives and different-label negatives can improve discrimination. Across members, forcing the same node's representations apart can instead encourage member identity or arbitrary coordinate differences. Repelling every pair of nodes also conflicts with same-class positives.

Retain contrastive learning as a possible ingredient, with an explicit useful prediction task for each member. The separate persistent masked-context proposal supplies different missing-feature contexts, masked classification and feature reconstruction. It is a literature-derived, unimplemented hypothesis, not a measured remedy. Its masking-only and objective-matched single/independent controls are required to attribute a future gain. An auxiliary-only representation head cannot by itself establish factual prediction diversity.

Do not repeat the completed basic contrastive conditions with a strength grid merely to seek a positive seed. Read the complete frozen Q/K comparison first. If unsupported, the already reviewed independent native local-scorer initializer remains a finite diagnostic of starting asymmetry. It is known initialization practice and requires no novelty claim.

## Custody and cost

Inputs: `result/AGGREGATE.json`, the unchanged source/configuration identified inside it, `owner/PHYSICAL_TERMINAL.json`, and the existing complete prediction-flow report `Wiki15_functional_diversity_analysis_20261008_v1/REPORT.md`. The independent source and existing-context scientific assessments are preserved separately.

All 15 endpoints completed. The worker used 31.822366 seconds wall time, 16.157048 seconds user CPU, 18.829310 seconds system CPU and a peak RSS of 878,628,864 bytes. Its child was directly waited, exited successfully and was absent with no owned CUDA rows. No forward, fit, raw-array transfer or TEST access occurred. Transfer and root interpretation costs are not included in those worker figures.
