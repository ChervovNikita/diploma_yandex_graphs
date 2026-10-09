# Completed stronger BSNN baseline

Three complete fresh fits of the fixed authentic Cayley d2/f32/L4 model. Scores come from fresh reconstruction of each selected model. This report opens no incomplete GNNM family.

| Seed | Epochs | Selected epoch | TRAIN AUROC | VALID AUROC | VALID NLL | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 7409 | 500 | 439 | 0.763505 | 0.747556 | 0.452905 | 3133.7 |
| 8501 | 500 | 478 | 0.767892 | 0.750166 | 0.452552 | 2994.6 |
| 9607 | 500 | 478 | 0.762643 | 0.746727 | 0.452245 | 2998.6 |

Mean VALID AUROC: 0.748149; sample SD: 0.001795.

The intervals describe three optimizer seeds on an already used development split. They do not establish graph-level generalization or a sharing mechanism. General NSD and native BSNN differ in geometry, conditioning, normalization, objective and stochastic serving. No shared-model result is compared here.

All failures and earlier weaker configurations remain in the ledger. Checkpoints and raw predictions remain server-only.
