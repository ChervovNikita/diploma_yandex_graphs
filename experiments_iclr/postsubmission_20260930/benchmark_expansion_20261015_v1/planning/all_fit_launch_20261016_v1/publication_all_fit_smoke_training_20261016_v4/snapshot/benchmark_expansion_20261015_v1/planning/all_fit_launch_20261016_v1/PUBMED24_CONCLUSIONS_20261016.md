# Completed PubMed/Cheb pilot

## What was tested

The pilot compared the original shared GNNM baseline (G0) with three residual variants. Uniform gives every route the same training responsibility; structured uses the planned graph-based responsibilities; permuted is the control for whether the proposed structure matters. Each of three paired cohorts also trained four native models. M1 reuses native body 0; I4 averages the four separately selected native models.

All 24 fits, 12 selected-route collectors and 18 saved score banks are complete and audited. Each collector performed four VALID forwards and zero TRAIN callbacks/optimizer steps. Copied evidence bytes, frozen jobs, own-selected exports and I4 member identities match their receipts. This audit read JSON and hashed existing files; it did not deserialize tensors, train models or access TEST.

## Result

**The proposed structured residual fails the frozen primary criterion.** It improves on uniform responsibilities in all three cohorts, but loses to the permuted control in all three. All three residual variants also have lower mean accuracy than the original G0 baseline.

| Bank | Seed 101 accuracy % | Seed 202 accuracy % | Seed 303 accuracy % | Mean ± sample SD % | Mean saved CE |
|---|---:|---:|---:|---:|---:|
| M1 | 91.0221 | 91.4025 | 91.3771 | 91.2672 ± 0.2127 | 0.26654 |
| I4 | 91.7322 | 90.9460 | 91.9351 | 91.5377 ± 0.5224 | 0.25031 |
| G0 | 91.6815 | 92.4677 | 92.2140 | 92.1211 ± 0.4013 | 0.23461 |
| uniform | 90.5148 | 90.8445 | 90.6163 | 90.6586 ± 0.1689 | 0.27278 |
| structured | 90.9713 | 91.2503 | 91.0474 | 91.0897 ± 0.1442 | 0.26796 |
| permuted | 91.1742 | 91.5800 | 91.4278 | 91.3940 ± 0.2050 | 0.27295 |

The frozen primary rule required positive mean accuracy differences for both structured−uniform and structured−permuted. It did not use a significance threshold.

| Paired contrast | Seed 101 pp | Seed 202 pp | Seed 303 pp | Mean pp | Descriptive 95% t interval pp |
|---|---:|---:|---:|---:|---:|
| structured − uniform | +0.4565 | +0.4058 | +0.4311 | +0.4311 | [+0.3681, +0.4941] |
| structured − permuted | -0.2029 | -0.3297 | -0.3804 | -0.3043 | [-0.5315, -0.0772] |
| G0 − M1 | +0.6594 | +1.0652 | +0.8369 | +0.8538 | [+0.3485, +1.3592] |
| G0 − I4 | -0.0507 | +1.5217 | +0.2790 | +0.5833 | [-1.4765, +2.6432] |
| structured − G0 | -0.7101 | -1.2173 | -1.1666 | -1.0314 | [-1.7253, -0.3374] |
| permuted − G0 | -0.5072 | -0.8877 | -0.7862 | -0.7270 | [-1.2164, -0.2377] |
| uniform − G0 | -1.1666 | -1.6231 | -1.5978 | -1.4625 | [-2.0998, -0.8252] |

G0 exceeds M1 in all three cohorts and I4 in two. Its positive mean against I4 has a wide interval including zero. This is useful evidence that the original shared baseline is competitive in this setting; it does not establish a successful new residual mechanism or broad superiority.

The intervals describe three optimizer cohorts on the same exposed graph/split, with seed and host coupled. They do not measure uncertainty across graphs. Exact two-sided sign-flip tests have only eight patterns and cannot produce p<0.25 here; none of the 14 preserved contrasts is significant, before or after Holm correction. The normality assumptions behind the t intervals cannot be assessed with three cohorts. VALID has class counts 72/1,919/1,952; overall accuracy cannot establish minority-class robustness.

## Training and cost

All 24 fits completed 2,099 epochs/updates under the frozen minimum 2,000, patience 100, cap 10,000 controller. All stopped by patience; none reached the cap. Own selected epochs range from 94 to 1,885. The total is 50,376 completed optimizer updates. This supports completion of the planned validation-plateau rule; it is not a proof of globally converged optimization.

Summed fit invocation time is 9.240 hours; summed owner-observed child time is 9.269 hours. These are cumulative across three GPU queues, not elapsed calendar time. The 12 collectors add 56.33 seconds. All 12 collector sanity statuses are complete, with no recorded warning.

| Cohort/host | Sum fit child hours | Max fit CUDA allocated GiB | Max fit CUDA reserved GiB | Max own fit RSS GiB |
|---|---:|---:|---:|---:|
| 101/anogena-2-0 | 3.950 | 2.679 | 2.906 | 1.000 |
| 202/olig | 2.662 | 2.679 | 2.906 | 1.017 |
| 303/olig | 2.657 | 2.679 | 2.906 | 1.024 |

These are observed own-fit resource counters and wall costs, not controlled throughput comparisons. The allocation and server51 run different physical GPUs, and seed is coupled with host. Native member, shared G0 and residual wall times are retained individually in the audit.

## Research conclusion

The structured assignment contains a repeatable advantage over uniform assignment on these paired cohorts, but the stronger permuted control rules out interpreting that advantage as evidence for the intended graph-specific mechanism. The residual intervention as tested does not improve the original shared baseline. Retain this hypothesis as unsuccessful under its frozen criterion.

Keep G0 as the comparison baseline. Do not use these development results to alter the frozen 6,460-fit campaign, change its scoring plan, or select new settings after observing VALID. Per-class/member/error-repair conclusions require the separate preserved-bank analysis; they are not inferred from overall accuracy here. Original-paper scores and experiments are untouched.

## Evidence

- Metadata audit: `PUBMED24_COMPLETED_METADATA_AUDIT_20261016.json`; SHA256 `e72952dc637960487d7a1d64e46c8f978ec8d2ad9d236f118e1ed2a949d5264f`.
- Full 14 contrasts, raw per-cohort deltas, uncertainty and costs: `PUBMED24_PAIRED_CONCLUSIONS_20261016.json`.
- Full original fits, checkpoints, selected exports and collectors remain on their training hosts under the exact PubMed queue result paths recorded in the audit.
