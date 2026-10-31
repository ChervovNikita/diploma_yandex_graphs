# Fixed-basis audit on unused PPI training graphs

The fixed D/E-selected, random and raw-gradient bases were evaluated on previously unused training graphs09 and18, at the unchanged P-only epoch25 checkpoint. This is a local gradient-denoising score. It is not accuracy, a loss reduction, an AdamW update, or independent-task confirmation.

| Cell | Selected score | Random score | Raw-gradient score |
| --- | ---: | ---: | ---: |
| seed0_label10 | -1.18996e-10 | -3.3263e-08 | -7.88604e-07 |
| seed0_label100 | 0.000608431 | 3.13639e-05 | 0.00426067 |
| seed1_label10 | 0 | 0 | 0 |
| seed1_label100 | 7.63078e-07 | 8.41298e-08 | 0.000232796 |
| seed2_label10 | 1.00765e-06 | -1.12913e-07 | -0.000257566 |
| seed2_label100 | 0.000160832 | 9.66456e-06 | 0.0011398 |

With10%labels, selected scores exceed both controls in two nonzero-rank cells; the zero-rank cell ties exactly. One selected sum is slightly negative. With100%labels, selected scores exceed random but lose to raw-gradient in all three seeds. These mixed outcomes support no universal selected-subspace advantage.

All six raw-gradient records passed exact reconstruction and a separate label-free replay process. The two F2 graph units supply no useful confidence interval or p-value. Seeds and nested masks share this corpus. Selected ranks, bases and criterion remain unchanged; any future method change needs a new study identity and independent assessment.

Decision: retain a bounded label-scarcity hypothesis for predictive testing, keep full-label controls and adverse findings, and prioritize the common/narrow architecture and broad task comparisons while GPU resources are pending. No manuscript claim is promoted from this audit.
