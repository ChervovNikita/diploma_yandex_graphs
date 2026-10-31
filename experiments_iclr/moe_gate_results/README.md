# Frozen GNNM logit gate diagnostic

This is a one-split diagnostic, not a new graph mixture-of-experts method. The four GNNM members and their shared SAGE backbone remain frozen. Only weights used to combine their raw logits are fitted.

## Inputs and protocol

- Dataset: Roman Empire, official split 0, 22,662 nodes, 18 classes.
- Expert checkpoint: `experiments_iclr/results/checkpoints/roman-empire_SAGE_gnnm_split0.pt`, selected at optimization step 2,260. It has four members, five SAGE layers, and 512 hidden coordinates. SHA256: `072b0e53196c519dcc32efaa019020e9fceb27ad3dd7c2025d30705fa5371c12`.
- Labels used to fit gates: 11,331 training nodes. Validation: 5,665 nodes. Final test: 5,666 nodes. The split masks are disjoint.
- The checkpoint's four raw logits per node were exported on CPU with dropout disabled. Export took 12.19 seconds. The cache has shape `22,662 × 4 × 18` and is stored in `moe_gate_results/roman_empire_sage_gnnm_split0_all_logits.npz`.
- All three aggregation rules receive exactly those same frozen logits. Uniform averaging has zero learned parameters. The global gate learns four member scores `alpha_m` and converts them to nonnegative weights summing to one using softmax. The node gate learns four `alpha_m` and four `beta_m` parameters. For node `i`, its member score is `alpha_m + beta_m * u_im`, where `u_im` is the member's softmax confidence, defined as `1 - entropy(p_im)/log(18)` and standardized using training nodes only. Both gates combine **raw logits**, not probabilities.
- The gates minimize multiclass cross-entropy on training nodes using Adam, learning rate 0.02, no weight decay, and at most 1,000 epochs. A checkpoint is selected by maximum validation accuracy, breaking ties by lower validation cross-entropy. Training stops after 100 epochs without a validation improvement. Both gate checkpoints are fixed before any test labels are evaluated.

The frozen GNNM checkpoint had already been selected using the same validation nodes. This reuse limits how much should be inferred from this pilot. The gate is also fitted on logits from training nodes that the experts themselves nearly memorized, rather than on held-out predictions for those nodes.

## Result

| Aggregation | Learned parameters | Best gate epoch | Validation accuracy | Test accuracy | Test cross-entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform mean logits | 0 | — | 89.2674% | 89.2340% | 0.74276 |
| Global member weights | 4 | 2 | 89.3027% | 89.2164% | 0.74318 |
| Entropy-conditioned node weights | 8 | 0 | 89.2674% | 89.2340% | 0.74276 |

The node gate's selected checkpoint is its initialization, which gives every member weight 0.25. The global gate changes weights only slightly, to approximately `[0.2549, 0.2549, 0.2354, 0.2548]`, and loses 0.0176 percentage points of test accuracy. The frozen uniform predictor reaches 99.9824% training accuracy with cross-entropy 0.000385. This leaves little training error from which a small gate could learn which member to trust on unseen nodes. These data do not show an advantage for a node-dependent gate. They say nothing about its behavior on other splits, backbones, or an out-of-fold stacking protocol.

The gate's confidence input also shifts between masks. Mean normalized member confidence is approximately 0.9997 on training nodes and 0.977–0.978 on validation and test nodes. Training confidence has standard deviation about 0.004, compared with about 0.062–0.065 on validation and test. This is consistent with an in-sample stacking problem: the gate sees nearly certain expert predictions while fitting, then less certain predictions at evaluation.

CPU gate fitting took 0.76 seconds for the global gate and 0.80 seconds for the node gate. The cache export, not gate optimization, was the main incremental cost. No GPU was used for this diagnostic.

## Reproduce

From the repository root, with its existing local environment:

```bash
.venv/bin/python experiments_iclr/moe_gate.py export
.venv/bin/python experiments_iclr/moe_gate.py fit
```

The machine-readable output is `moe_gate_results/roman_empire_sage_gnnm_split0_gate.json`. All paths and generated files stay inside this repository.
