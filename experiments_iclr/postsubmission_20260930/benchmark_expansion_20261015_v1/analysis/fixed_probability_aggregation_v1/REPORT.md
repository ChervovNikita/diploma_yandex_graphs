# Fixed mean-probability serving diagnostic

Completed MolBACE seed101 checkpoints, evaluated on all151 native VALID graphs. This VALID set already selected the checkpoints; this is posthoc development evidence. Native M1 is a singleton sanity case.

| Bank | Mean-logit AUC | Mean-probability AUC | AUC change, points | Mean-logit BCE | Mean-probability BCE | BCE change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.73846154 | 0.73846154 | +0.000000 | 0.92068817 | 0.92068817 | +0.00000000 |
| base | 0.62783883 | 0.63150183 | +0.366300 | 0.49848030 | 0.50847476 | +0.00999446 |
| C1 | 0.71062271 | 0.71025641 | -0.036630 | 0.55420762 | 0.55775548 | +0.00354786 |
| C4 | 0.67728938 | 0.68131868 | +0.402930 | 0.54666300 | 0.55850485 | +0.01184185 |

Mean probabilities raised AUC for base and C4 and lowered it slightly for C1. Observed-label BCE increased for all three four-route banks. M1 agreed exactly under both rules. No readout was chosen or promoted; the original mean-logit baseline and pilot scores remain unchanged.

Every saved graph ID/order, target including NaN support, selected epoch, job hash and checkpoint/export receipt matched. The native OGB evaluator computed AUC. BCE used observed labels with FP64 logsigmoid/logsumexp, with no clipping or fit. Saved probabilities passed a direct arithmetic-mean check. PREDICTIONS_VALID.pt retains both fixed rules and exact graph IDs/labels; its allocation path/hash is in RESULT.json.

The later completed I4 bank still needs this same fixed control; it is pending here. Do not interpret this single seed on checkpoint-selection VALID as independent accuracy evidence.

Cost: 0 model constructions, new forwards, updates, selector changes, fits or TEST access; two CPU threads, CUDA hidden; full process wall 6.783658s; measured arithmetic loop 0.082664s; CPU user/system 2.881418/1.034848s; peak RSS 499,984 KiB.
