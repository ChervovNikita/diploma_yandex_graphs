# What weakens the shared Amazon ensemble

The complete paired selected VALID banks were analyzed: three splits, 6,123 nodes per predictor per split, and 44,514 denominator-rich metric rows. All rows are retained as exact CSV shards. No training, model forward, hidden export or TEST/control-label scoring was performed. Native VALID accuracy reproduces the completed original comparison. This is descriptive development on checkpoints already selected using VALID; overlapping splits and nodes are not independent replication.

## Overall comparison

| Quantity, mean of three split summaries | Shared four | Independent four |
| --- | ---: | ---: |
| Mean individual-member accuracy | 52.2402% | 52.5859% |
| Native pooled accuracy | 52.4035% | 53.1765% |
| Accuracy added by pooling over mean member | +0.1633 pp | +0.5907 pp |
| Mean six-pair error-indicator correlation | 0.9292 | 0.7304 |
| Nodes where all four members are incorrect | 44.5479% | 35.0536% |
| Mean predicted-class confidence on wrong pooled answers | 92.5966% | 81.0860% |
| Native NLL | 6.1930 | 3.8033 |
| Native Brier loss | 0.883883 | 0.773148 |

The shared accuracy deficit is -0.7730 pp. Its exact descriptive split is -0.3457 pp from lower mean member accuracy and -0.4274 pp from lower pooling benefit. Each decomposition is retained per split; this does not identify causal effects of tying parameters, which also changes optimization, capacity, selection and initialization.

Probability pooling reduces mean member Brier by 0.018067 for shared members and 0.117366 for independent members. The 0.110735 pooled Brier gap equals 0.011436 worse mean member Brier plus 0.099299 less averaging reduction. This exact ambiguity identity diagnoses the observed gap; increasing arbitrary prediction/embedding spread need not improve accuracy or generalization.

## Selection headroom and class/degree patterns

All members are wrong on 93.60% of the shared native errors, versus 74.86% of independent errors (means of per-split conditional fractions). Selecting a member's class cannot repair such nodes. A convex probability combination can sometimes recover a correct class even if no member chooses it, and nonlinear score/hidden readouts or graph correction can leave the member hull. These counts therefore do not prove a lack of usable score or hidden information. No hull oracle was executed.

Shared pooling loses a correct individual-member answer on [189.0, 206.0, 165.0] nodes across the three splits; independent pooling does so on [736.0, 703.0, 723.0]. Cases where pooling succeeds despite every member being wrong are retained: shared [0.0, 0.0, 0.0], independent [0.0, 0.0, 0.0]. A label-informed hypothetical selector is not a learnable predictor or an attainable dense-hull bound.

The mean shared-minus-independent class accuracy gaps, classes0–4, are −2.46, −0.89, −0.28, +3.30, −0.38 pp. The class3 advantage and its smaller support are retained rather than converting the overall deficit into a uniform-loss claim. For degree3–5,6–10,11–20,>20, the mean gaps are −0.56,−0.69,−0.85,−5.27 pp. Degree>20 has only166–183 VALID nodes per split; its large deficit is exploratory. Degree0 and1–2 strata are empty. Degree alone does not establish a graph-neighborhood mechanism. Full confusions, support and per-split contrasts remain available.

## Decision for further method development

The primary observed weakness is correlated mistakes and reduced ensemble benefit, alongside slightly weaker members and very confident errors. The next training hypothesis should seek useful member complementarity while retaining individual strength. Forced hidden orthogonality alone cannot establish this, and ordinary negative-correlation/diversity losses are existing methods that require attribution and strong controls. The running endpoint-separated private-transfer study is one separate training hypothesis; it has no adopted predictive result yet.

The already fixed saved-logit screen asks whether graph-local error context and calibration can improve available predictions. It compares the complete processed shared bank with equally processed independent banks and a capable processed single, with full-information stacking and posterior-projection controls. A win over only the weaker native shared pool would be insufficient. The corrected solver passed five affected checks and the fixed screen was explicitly relaunched in a new output; no aggregation outcome is yet adopted. No outcome-dependent extra grid or hidden extraction was added.

Before proposing a new confirmation study, require that the candidate improves quality against both strong references, identify whether its benefit comes from member quality, averaging or calibration, preserve unsuccessful arms, and freeze its final recipe before heldout scoring. These diagnostics support a research direction, not novelty clearance, general superiority or an acceptance verdict. Original manuscript scores remain unchanged.

## Costs and evidence

Source V1's constant-column correlation defect was corrected in preserved V2 and independently audited. Actual numeric description took 0.882s after 0.297s custody verification; peak RSS 416,231,424 bytes. Hashing loaded15 selected-logit artifacts and nine VALID panels. CSV packaging/SSH are extra costs in the root terminal; no GPU computation occurred. Full original JSON/CSV remain on the server; local exact shards cover every row and bind the original aggregate hash.

- [Compact findings](SUMMARY.json)
- [Exact metric metadata and full-shard inventory](../amazon_polynormer_valid_error_analysis_root_preparation_20261005_v1/observation/amazon_polynormer_valid_error_analysis_root_preparation_20261005_v1/EXACT_METRIC_COLLECTION/AGGREGATE_METADATA.json)
- [Fixed source plan](../amazon_polynormer_valid_error_analysis_source_preparation_20261005_v2/PLAN.json)
