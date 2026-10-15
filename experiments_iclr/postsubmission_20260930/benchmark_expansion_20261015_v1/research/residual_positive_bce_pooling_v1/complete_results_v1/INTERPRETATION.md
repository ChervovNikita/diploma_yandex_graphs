# A fixed probability mean does not repair the residual bank

The complete saved-prediction comparison covers all 11 methods across three seeds, using the same selected checkpoints and full 151-molecule VALID population. It adds no training, checkpoint loading, model forward, calibration or TEST scoring. Every bank receives both fixed readouts; neither is selected as a new method.

| Selected bank | Mean-logit AUC (%) | Probability-mean AUC (%) | Mean-logit BCE | Probability-mean BCE |
| --- | ---: | ---: | ---: | ---: |
| Residual-only with private BN | 74.518 | 74.493 | 1.952035 | 1.950576 |
| Genuine native I4 | 74.847 | 74.896 | 0.890168 | 0.820155 |

The residual bank's positive-class BCE falls only from 2.235294 to 2.233583. I4's falls from 0.967880 to 0.884886. The residual-minus-I4 positive-class excess therefore **increases from 1.267414 to 1.348697** when both get probability averaging. It increases in each seed. This is a comparative result: the residual model's absolute loss decreases slightly, while I4 benefits more. The native task remains AUC, whose mean deficit grows from 0.330 to 0.403 percentage points under the two matched readouts.

This rules out the tested fixed averaging change as a repair for the current bank. It does not rule out every learned readout or identify the training cause. Combined with the earlier common-ranking-error analysis, it supports prioritizing useful changes to member predictions rather than presuming that replacing the mean will solve the problem. The running structural protocol and prepared PubMed transfer stay unchanged. A new trained probability-mixture objective remains a separate hypothesis; this readout-only result neither validates nor rejects it.

These are descriptive results on the same small, heavily exposed checkpoint-selection VALID split. The 130 positives and 21 negatives are shared across all seeds. No new heldout advantage, calibrated uncertainty or novelty claim follows. The full report preserves every method, both class losses and all signed seed differences. No original paper score or served prediction was replaced.

The CPU analysis took 3.642 seconds, or 4.054 seconds under its supervisor, with sampled peak RSS 359,493,632 bytes. It reused 39 saved exports and 27 saved route collections, with zero new GPU workers. Stable log-sigmoid/logsumexp evaluation avoids clipping confident errors. RESULT.json is the exact 85,863-byte allocation result (SHA256 23670669cd82f3dbb4cead8f6ed1d9c2188b67810727aa1d69d6d0b022c79253); POOLING_DIAGNOSTIC_28.json records execution.
