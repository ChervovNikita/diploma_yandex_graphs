# Conditional projector prototype: protocol and scope

`conditional_projector.py` is an unmeasured exploratory extension of GNNM. It is separate from the ICLR submission. The code has passed a small CPU self-test but has not completed a full-graph gate fit. No accuracy or speed claim is available yet.

## What runs

Let `P_in,m` and `P_out,m` be the four frozen BatchEnsemble input and output projector functions from a trained Roman Empire SAGE GNNM checkpoint. A small gate gives each node `i` four nonnegative weights `g_im` summing to one. The prototype computes:

```text
node i input:       h_i^0 = GELU(sum_m g_im P_in,m(x_i))
one graph pass:     H^L = T_theta(A, H^0)
node i output:      z_i = sum_m g_im P_out,m(LayerNorm(H_i^L))
```

The same `g_i` is used at input and output. The complete projector **outputs** are mixed. Mixing input and output factor vectors separately would introduce cross-member products and would not equal this convex mixture of projector functions. All BatchEnsemble matrices, factors, and backbone parameters stay frozen. At gate training and inference, backbone dropout is disabled. The node gate uses the node's original features, standardized from gate-fit nodes, to set `g_i`. Its matched static control learns one weight vector for every node. Both use one graph propagation pass, while original GNNM averages logits from four separate passes.

## Exact distinction from a posthoc logit gate

A posthoc gate over frozen member logits has `z_v^post = sum_m g_vm z_vm(A,X)`. With the member logits held fixed, changing the gate at a different node `u` cannot change `z_v^post`, so `partial z_v^post / partial g_um = 0` for `u != v`.

The conditional projector changes node `u`'s representation **before** message passing. When `u` sends a message to `v`, changing `g_u` can change `v`'s final logit. The derivative `partial z_v / partial g_um` can therefore be nonzero for `u != v`. This is a possibility, not a guarantee for every trained parameter setting. The CPU self-test constructs a graph where this cross-node derivative is nonzero, checks that the corresponding posthoc derivative is zero, verifies exact recovery of each original member under one-hot weights, and confirms that only gate parameters receive gradients.

## Planned one-split test

- Source: the frozen SAGE GNNM checkpoint from Roman Empire official split 0, with four members, five layers, and width 512.
- The 5,665 original validation nodes are deterministically partitioned with seed 2027 into 2,832 gate-fit nodes and 2,833 gate-selection nodes. Original training labels are not used to fit the gate. The initial member is chosen using only gate-fit labels.
- Fit the 4-parameter static gate and the 1,204-parameter node gate on the same gate-fit labels using cross-entropy. Select each checkpoint on gate-selection accuracy, breaking ties by lower cross-entropy. Do not inspect test labels until both checkpoints are fixed.
- Compare the two single-pass gates directly. Report original four-path uniform GNNM, the selected pure member, and matched BASE as descriptive references. The gates use additional labeled nodes that BASE and original GNNM did not use for gradient training, so those reference comparisons do not have matched supervision.
- The frozen GNNM checkpoint was already selected using the full original validation split. Splitting those nodes afterward does not remove this selection dependence. A stronger study would reserve gate-fit labels before expert training or use out-of-fold expert predictions.
- Report trainable and frozen parameter counts, validation and test accuracy/cross-entropy, and measured single-pass versus four-pass latency on the same GPU. No result should be promoted to a method contribution from one split alone.

## Prior art and novelty limit

- [GNN-FiLM](https://proceedings.mlr.press/v119/brockschmidt20a.html) uses target-node representations to modulate graph messages inside one pass.
- [GMoE](https://arxiv.org/abs/2304.02806), [Mowst](https://arxiv.org/abs/2311.05185), and [Node-MoE](https://arxiv.org/abs/2406.03464) use node-dependent expert selection or graph filters.
- [Moscat](https://openreview.net/forum?id=swwAPbtU07) fits a posthoc gate to GNN expert outputs using held-out labeled nodes.
- [MoLE-GNN](https://doi.org/10.1007/978-3-032-37657-2_38) combines parameter-efficient graph adapters and expert assignment. Its published implementation runs separate graph experts and gates their outputs, rather than blending these BE boundary projectors before one backbone pass.

The exact boundary-projector placement and single propagation pass distinguish this prototype from a frozen logit gate. The broader ideas of node-conditioned graph computation and parameter-efficient expert mixing already exist. Novelty remains uncertain and no paper claim is justified without a matched, multi-split comparison.
