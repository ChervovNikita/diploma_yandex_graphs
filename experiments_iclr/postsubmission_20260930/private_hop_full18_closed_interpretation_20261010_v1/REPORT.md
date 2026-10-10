# Complete private-hop study

All 18 declared fits and the finite stored-logit comparison closed successfully. The comparison introduced zero model forwards. Original scores and TEST remain unchanged.

The experiment tested whether different incomplete graph contexts teach the four routes useful alternatives while keeping the shared backbone trained on factual full-graph classification. At inference every route uses the complete graph. Full-input auxiliary, common-view, all-block, and factorized-single controls test alternative explanations.

| Condition | Mean VALID accuracy (%) | NLL | Mean member accuracy (%) | Worst member accuracy (%) |
|---|---:|---:|---:|---:|
| shared4_own | 91.045155 | 0.286731 | 91.038813 | 90.977507 |
| private_missinghop | 90.892948 | 0.284649 | 90.804160 | 90.656181 |
| full_aux | 91.036699 | 0.289057 | 91.000761 | 90.960595 |
| common_nonfull | 90.876036 | 0.297289 | 90.821072 | 90.690005 |
| allblock_missinghop | 91.002875 | 0.286984 | 90.958481 | 90.850668 |
| factor1_allview | 90.613902 | 0.332589 | 90.613902 | 90.613902 |

## Scientific decision

Private missing-hop supervision loses final accuracy at every seed: −0.076104, −0.050736 and −0.329782 percentage points. Mean member quality also falls at every seed; the worst-member loss reaches 0.659564 points. Its mean NLL improves by 0.002082 but worsens at the third seed. This does not satisfy the intended joint competence and final-quality improvement. The exact recipe is closed.

It does acquire 45/45/38 newly covered correct alternatives. Pooling serves 30/31/34 and loses 15/14/4. It also loses 28/27/48 previously covered correct alternatives. Final repairs/harms are 37/40, 32/34 and 43/56. At the third seed coverage itself shrinks. More diverse graph supervision therefore produced some useful alternatives but did not preserve enough existing competence. A larger diversity proxy would not establish success.

Full-input auxiliary supervision nearly ties mean accuracy (−0.008456 points) and worsens NLL at every seed. Common incomplete views lose 0.169119 mean points. All-block missing-hop is mixed and loses 0.042280 mean points. The all-view factorized single loses 0.431253 mean points versus shared own. These controls do not support a private-gradient protection claim or ensemble-specific accuracy improvement.

The stronger factorized independent reference remains required. Its prior mean accuracy is 90.825300%, but this family's fresh own baseline is 91.045155%; selecting the best repeated own run would bias a superiority claim. Those banks were acquired in different fixed studies. The current results are paired within this family, on an encountered development split, and establish no unused confirmation.

The private-Adam normalization control separately weakens all four members and fails its frozen competence/NLL gate. Its proposed combination with private-hop remains disabled. No expensive combined study is justified by these results.

## Uncertainty and actual cost

All fixed paired contrasts, sample seed SDs and exploratory 95% t intervals with two degrees of freedom are in SUMMARY.json. They describe three seeds on one development split after checkpoint selection; they do not provide independent node/graph uncertainty or account for repeated research decisions. Full18 worker time totals 6975.969s; finite owner time totals 7011.153s. Actual per-cell training, serving, memory, overlap and selectors are retained. These inclusive measurements are not isolated throughput benchmarks.

## Next action

Prioritize a mechanism that preserves competent factual predictions while changing how useful alternatives affect later graph computation. The separately prepared live route exchange is still an untested hypothesis. Revisit known-method ancestry and capable references before any paper claim. Preserve the entire failed family and do not rebrand it as positive.
