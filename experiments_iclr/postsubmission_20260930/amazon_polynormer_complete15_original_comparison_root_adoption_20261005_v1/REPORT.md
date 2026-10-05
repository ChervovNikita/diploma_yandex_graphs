# Complete original Amazon/Polynormer comparison

All15 fixed fits and all nine family records were evaluated after successful whole-cohort closure. The corrected evaluation exited0 in293.16seconds; every selected model replay had maximum logit difference exactly0. Its peak observed RSS was2.51GB, CUDA allocated43.89GB and reserved45.47GB. This includes45 complete-model evaluation forwards and30 discarded replay updates, not new scientific training. The first CPU/GPU release mismatch and its failed35.06second attempt remain preserved.

## Reserved TRAIN-control outcome

| Family | Mean accuracy | Mean NLL | Mean Brier |
| --- | ---: | ---: | ---: |
| Native member-0 single | 52.7203% | 5.1956 | 0.88284 |
| Shared four | 52.9243% | 5.9592 | 0.87288 |
| Independent four | 54.0533% | 3.6883 | 0.75987 |

The shared bank trails independent four on all three control splits: accuracy differences are−0.7752,−1.2245 and−1.3872points, mean−1.1290points. Its NLL is worse by1.9444,2.2592 and2.6091nats, mean+2.2709. Those data support no superiority claim. All families fit TRAIN-FIT almost perfectly; their much poorer control accuracy and large probability loss require competent regularization/calibration comparators for any later fusion. This observation does not identify a causal sharing failure or prove that aggregation can fix it.

## Validation and interpretation

Mean validation accuracy is52.6920% single,52.4035% shared and53.1765% independent. Validation selected the checkpoints and remains development data. The full raw per-split outcomes are retained inRESULTS_SUMMARY.json and the authenticated originalRESULT.json; no favorable subset or changed selector is used.

TRAIN-control labels were excluded from training and checkpoint selection by the fixed contract. They are now consumed for this original comparison and cannot be silently promoted into a fresh confirmation endpoint for a later modified pipeline. Three overlapping splits of one graph do not establish performance on independent graphs. No TEST labels, hidden states or new fusion outcomes were opened. Original manuscript scores are unchanged.

## Next decision

Proceed only with the small separately specified retrospective VALID aggregation-development comparison and strong processed independent/single references. Its constants were prepared before fusion outcomes. An improvement over this weaker shared baseline alone cannot support the requested new-method superiority. A future confirmation needs its own frozen final-label contract; this result supplies no novelty clearance or acceptance verdict.
