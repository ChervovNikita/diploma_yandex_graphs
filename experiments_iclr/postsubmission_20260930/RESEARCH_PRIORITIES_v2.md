# Research priorities after submission

## Purpose

Develop a method that can decide how much sharing is useful on a new task, then test that decision against strong alternatives. The existing benchmark scores remain fixed. The current phase has no new predictive gain or acceptance recommendation.

## Three experiments with distinct purposes

1. **Choose a private block using training gradients.** The original model shares every propagation block. The proposed rule selects one block to give members separate weights, using recurring differences between their training gradients. PPI tests the locked choice against all three equal-capacity placements; MolHIV tests transfer to bond-aware graph classification. The crucial comparison is with rejected placements and size-matched alternatives, rather than with a larger ordinary ensemble alone.
2. **Keep member graph states narrow.** A common graph state handles most computation, while small member paths retain separate relational updates. The specific hypothesis is that feeding their mean update back into the common state helps predictions. The four centering/feedback cells and packed narrow ensembles distinguish that mechanism from ordinary small branches. Stored parameters, actual selected-checkpoint latency, training cost and memory must be reported separately.
3. **Privatize only selected directions.** Instead of copying an entire block, the spectral branch learns member corrections in fixed training-derived directions. Selected, random and raw-gradient directions start from the same trained checkpoint, with zero correction coefficients and fresh identical optimizers. Full-label controls remain essential: the existing local audit favored raw gradients at all three full-label seeds. That audit does not measure predictive quality.

## Closest comparator

The official KAIST viewer now provides Kim's2023 thesis. Its graph-layer BatchEnsemble factors and probability averaging are verified. A declared SAGE adaptation has passed the synthetic CPU check. Its new studies will record both probability and raw-logit readouts from identical member predictions, so a pooling difference cannot be attributed to factor placement. This is a prior-method adaptation; no exact author implementation was recovered.

## Evidence required before changing the paper

| Review concern | Required new evidence |
| --- | --- |
| Local, post hoc results | Training-only candidate rule followed by independent task confirmation; no confirmation-test-guided revisions. |
| Weak competitors | Initially matched tied/untied models, equal-capacity placements, size-matched single/ordinary ensembles, packed execution and the declared all-layer-factor comparator. |
| Storage presented as deployment efficiency | Measurements from actual selected checkpoints on the same device and complete graph batches. Parameter counts alone do not establish speed or memory gains. |
| Late checkpoint selection | Complete learning curves and a budget-limit audit. Comparisons that remain improving at their training ceiling cannot establish a converged advantage. |
| Uncertainty | Paired effects across seeds, explicit split/corpus dependence, and a stronger confirmation seed budget. Three-seed exploration is a screen, not evidence of statistical significance. |
| Unverifiable results | Shareable source/data/runtime locks, all selection traces, checkpoints, member logits and independent numerical replay. A setup test is not a fresh training reproduction. |
| Thin mechanism | The proposed training-only rule must predict held-out behavior; centering/feedback must beat simpler branches; selected spectral directions must beat equal-rank random/raw controls. |

## Execution boundary

All work stays inside the authorized local project and original-server repository. No new GPU job has started. The latest recorded query found all seven visible A10080GB GPUs occupied. One idle GPU and, for another server, its exact authorized repository path and connection alias are still needed. Current CPU checks establish implementation readiness only.

## Review boundary

Run the requested fresh review workflow after a meaningful manuscript revision supported by new predictive results. Preserve every verdict. Repeated reviews of unchanged evidence cannot establish acceptance. If a direction fails its fixed comparison, record the failure and change the scientific hypothesis prospectively rather than selecting a favorable story retrospectively.
