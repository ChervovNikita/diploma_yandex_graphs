# WikiCS fixed6: focused history closure

2026-10-10. The whole six-cell run is closed. Root admitted the original direct wait/output custody, CUDA absence, hash-bound CPU checkpoint metadata extraction and agreement with each original 1100-epoch trace's earliest strict maximum. This note reads only the completed scalar metadata and an existing completed historical scalar table. No new scoring or inference occurred.

## Primary decision: relation feedback fails

| Seed | F accuracy % | J accuracy % | J−F pp |
|---|---:|---:|---:|
| 6101 | 81.759578 | 81.740612 | -0.018966 |
| 6203 | 81.513083 | 81.551003 | +0.037920 |
| 6307 | 81.569964 | 81.569964 | +0.000000 |
| Mean | 81.614208 | 81.620526 | +0.006318 |

The frozen continuation rule requires mean gain **>0.2 pp and all three paired gains positive**. Both requirements fail: one loss, one small gain, one tie. **No relation-feedback continuation.** The selected epochs differ, so this is the admitted comparison of each arm's own original selected endpoint.

## Member competence at all six selected endpoints

F denotes the independently initialized native scorer; J adds the relation feedback. All six records have `selected_global=true`; selection values were read from the original state. The original global-selection route and epochs are retained below; no additional stage label is supplied by the compact metadata.

| Seed | Arm | Epoch | Pool accuracy % | Mean member % | Worst member % | Pool−mean member pp |
|---|---|---:|---:|---:|---:|---:|
| 6101 | F | 132 | 81.759578 | 81.489381 | 81.304514 | +0.270197 |
| 6101 | J | 123 | 81.740612 | 81.551005 | 81.361395 | +0.189607 |
| 6203 | F | 130 | 81.513083 | 81.456201 | 81.266588 | +0.056882 |
| 6203 | J | 126 | 81.551003 | 81.437239 | 81.152827 | +0.113764 |
| 6307 | F | 125 | 81.569964 | 81.484641 | 81.399316 | +0.085323 |
| 6307 | J | 124 | 81.569964 | 81.399317 | 81.285554 | +0.170647 |

Across seeds, F mean/worst member accuracy is **81.476741/81.323473%**; J is **81.462520/81.266592%**. J's mean member accuracy changes by +0.061624/−0.018962/−0.085324 pp: relation feedback supplies no consistent competence gain. Pool−mean member accuracy is a descriptive lift, not a repair, coverage or pooling-loss count. All four per-member accuracies are retained in `DECISION.json`.

The new metadata contains **no NLL or coverage**. It cannot establish acquisition of correct alternatives, repair overlap, common-rival counts, lost alternatives or a proper-score improvement.

## Initializer-only decision: pending admission

The copied historical F is contextual evidence until its exact source, recipe, selector, route, endpoint and terminal anchor are admitted for this comparison.

| Seed | Old copied F accuracy % | New F accuracy % | Descriptive new−old pp | Old F NLL |
|---|---:|---:|---:|---:|
| 6101 | 81.266591 | 81.759578 | +0.492987 | 1.110346 |
| 6203 | 81.475161 | 81.513083 | +0.037922 | 1.068690 |
| 6307 | 81.266591 | 81.569964 | +0.303373 | 1.000630 |
| Mean | 81.336114 | 81.614208 | +0.278094 | 1.059889 |

The apparent **+0.278094 pp** mean is descriptive and pending admission; it is **not an established causal initializer gain**. New F's mean member accuracy exceeds old F in all three descriptive rows, with mean +0.165908 pp. Its worst member loses at seed6203, so universal competence preservation is unsupported. The existing old F coverage is 4318/4317/4311 with 32/20/25 pool harms and zero all-member-wrong rescues; new F has no admitted counterpart here. Root's later parity and NLL/coverage readout is required before promotion.

## Distinct addition to the broad history

This is the second closure after private CMCL18. CMCL established that its fixed package acquired alternatives at competence/serving cost. WikiCS fixed6 establishes a failed relation-feedback increment and leaves a separate initializer-only qualification. The descriptive increase in new F's mean member competence motivates exact reference admission: **standard native scorer asymmetry may behave differently from the old Rademacher starts that bought coverage at competence cost**. All initialization mechanisms should not be collapsed into one negative ingredient. This remains a qualification priority, without a success, novelty or new-run claim.

Earlier synthesis and seals are preserved. No typed-reference/candidate comparison is opened by this closure. No raw arrays, checkpoints, models, datasets or source were read; no source, root state or original scores were changed.
