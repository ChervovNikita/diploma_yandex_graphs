# What the complete molecular pilot says about member quality

These are selected-VALID diagnostics for all three paired seeds of the completed 24-fit MolBACE/GIN pilot. They do not change any stored fit score. The same 151 graphs (130 positive, 21 negative) are reused across seeds; neither the four routes nor the 2,730 graph pairs per seed are independent repetitions. TEST remains closed.

## Member quality and averaging

| Bank | Mean member AUC, % | Served AUC, % | Pool gain over mean member, points | Pairs reversed by every member, mean | Pairs reversed by pool despite a correct majority, mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent four | 73.41 | 74.85 | +1.44 | 341.3 | 13.0 |
| Base GNNM | 62.01 | 67.78 | +5.77 | 226.0 | 73.3 |
| Alignment + variance | 64.24 | 71.17 | +6.93 | 195.7 | 25.7 |
| Private input correction | 66.07 | 71.76 | +5.69 | 167.3 | 42.0 |
| Both ingredients | 61.00 | 69.28 | +8.28 | 123.7 | 57.7 |

Averaging improves every bank in every seed relative to its own mean member AUC. The independent members nevertheless start much stronger: 73.41 versus 62.01 for base GNNM. Private correction improves mean member AUC to 66.07, still 7.34 points below independent members. This is direct evidence of a member-quality problem in this setting. It does not prove that the shared parameterization itself causes it.

The combination has the lowest count of pairs reversed by every route, yet mean member AUC falls to 61.00 and its pool is weaker than either individual ingredient. Pair coverage is therefore an inadequate objective by itself. Its decrease does not establish that a feasible learned combiner can recover the missing performance.

The private correction improves member AUC in each of the three seeds. Alignment/variance improves mean member AUC only in the first seed; its other pooled improvements can coexist with almost unchanged or slightly worse members. The separate ingredient controls remain necessary.

## Fixed readout diagnostic

| Bank | Probability-average minus logit-average AUC, points: seed101 /202 /303 | Mean logit BCE | Mean probability BCE |
| --- | ---: | ---: | ---: |
| Independent four | +0.476 / -0.330 / +0.000 | 0.8902 | 0.8202 |
| Base GNNM | +0.366 / +0.403 / +2.930 | 0.7193 | 0.6419 |
| Alignment + variance | -0.037 / +0.183 / -2.894 | 0.5505 | 0.5521 |
| Private input correction | +0.403 / +1.758 / -2.674 | 0.9408 | 0.6380 |
| Both ingredients | -0.989 / +1.172 / +2.930 | 0.8817 | 0.7073 |

Probability averaging gives base GNNM a positive AUC change in all three seeds (mean +1.23 points), but its mean AUC remains below the independent ensemble under either fixed rule. The other candidate readout changes are inconsistent. These are posthoc fixed-readout diagnostics on the same validation data used for checkpoint selection; they do not select a replacement serving rule. Both alternatives and their BCE tradeoffs are retained.

## Research decision

Finish the separate normalization-state control, preserve matched member-quality measurements, and keep private correction, geometry and their interaction distinct in the benchmark. Do not promote a diversity-only or aggregation-only explanation from this pilot. The all-route common-failure and majority-cancellation counts identify different weaknesses, but neither is a realizable oracle accuracy or an independent significance test.

All per-seed member metrics, reconstructed-versus-original scores, exact graph-pair counts, repair/harm comparisons, and descriptive geometry are in RESULT.json. No new model, forward, optimizer step, fitted aggregator, checkpoint selection or TEST access occurred. The CPU analysis took5.70s; peak RSS 500.2MiB. Reconstruction warnings: 0.
