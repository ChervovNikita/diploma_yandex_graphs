# TRAIN semantics of route private BN

15 October 2026. Source-level interpretation of the frozen MolBACE GIN implementation, including the geometry and input-correction composition. This is a mathematical statement under native PyTorch BN semantics, not a numerical equivalence gate or a new empirical result. Exact inspected sources and line scopes are pinned in `SOURCE_PINS.json`.

## Lemma and assumptions

Fix an ingredient cell, initial learned parameters, Adam state, eligible TRAIN batches and route order, and the RNG state supplying dropout and sampling. Assume ordinary finite TRAIN forwards with each native BN module in training mode, unchanged affine parameters and BN configuration, and no loss or controller that reads running statistics as an input. Changing only ownership of `running_mean`, `running_var` and `num_batches_tracked` from one shared bank to route-owned banks does **not change the mathematical TRAIN outputs, task/auxiliary losses, parameter gradients or Adam transition**.

For a channel's current activation batch, TRAIN BN computes

```text
y_i = gamma * (x_i - mean_batch(x)) / sqrt(var_batch(x) + eps) + beta.
```

The stored running moments are absent from this expression. TRAIN backward uses the current input and saved batch normalization statistics, not the running moments used at eval. Thus corresponding layers produce the same activations and derivatives; identical dropout draws preserve the argument through the network and all four routes. The unchanged loss and optimizer then give the same update. Induction extends this to every common optimizer step. Running-state updates may differ throughout.

This is conditional on matched mathematical inputs/states, not a claim of bitwise equality, determinism across separate CUDA executions, or equality of checkpoints selected at different epochs.

## Actual source checks

| Potential exception | Finding in this implementation |
| --- | --- |
| BN configuration and counters | Native GIN constructs ordinary `BatchNorm1d` without overriding momentum, tracking, affine or eps: PyTorch defaults apply, including momentum 0.1 and tracked state. Current non-virtual GIN uses its MLP and layer BN; attention-pool and virtual-node BN are not instantiated. No custom read of the three running-state fields enters the forward or loss. `num_batches_tracked` changes the state clock, not this TRAIN output. Even `momentum=None` would change cumulative running-average weights, not ordinary TRAIN normalization; it is not used here. |
| Mode changes during an update | Both native and molecular `train_update` call `model.train()` before any route. The molecular update computes four live TRAIN routes, own loss and optional geometry, then one backward/Adam step. There is no eval-mode branch inside that update. Validation calls `model.eval()` under `no_grad` between epochs; the next update restores TRAIN mode. |
| Private-state implementation | V3 clones detached buffers, preserves native configuration and shared affine Parameters, binds a route's buffer objects for the original call, and restores canonical objects afterward. It does not copy values into an outstanding graph's shared buffers. Its guards reject bypass/nested contexts. It adds no learned parameters, optimizer entries or native forward, and contains no random draw or seed change. |
| Hidden or auxiliary state | Boundary captures are current-route logits/preheads and reset per route. Geometry is stateless, consumes live TRAIN preheads and labels, and has no queue, teacher or running-state input. The composition preserves the original model, optimizer and ingredient update. No activation checkpoint/recomputed BN or synchronized BN is used; SyncBN is explicitly rejected by the bank manager. |
| Sampling and RNG | Matched pilots use a persistent private TRAIN generator and a separate VALID generator, with zero loader workers. Native dropout draws occur in the unchanged route order. Boundary signs and the optional input correction initialize with isolated generators; BN attachment only clones/registers buffers. No ownership-dependent RNG consumption was found. The available shuffled TRAIN-eval diagnostic consumes loader RNG, but `evaluate_TRAIN_each_epoch=false` in this pilot. |

The manager leaves untracked BN unchanged; that case supplies no route-owned running-state mechanism. Output-dependent nonfinite failures, different initial/optimizer/RNG states, custom running-state losses, eval-mode normalization during optimization, or a changed implementation fall outside the lemma. No such extra training path was found in the inspected successful-path source.

## Selection and attribution

The runner trains an epoch, validates, observes the selector, and saves state. Adam settings are fixed; validation does not change the learning rate, restore selected weights between epochs, or steer sampling. The selected state and RNG are restored only after the training loop; resume restores the matching saved optimizer and RNG state. Consequently, under the assumptions above, **selection changes which point is returned and stopping changes the available trajectory prefix; neither changes the earlier common TRAIN prefix**.

The completed private-BN selected-VALID gain can therefore reflect route-matched eval normalization, its different EMA clock/order weighting, and selection of a different epoch under that eval map. It does not establish that BN ownership improved TRAIN gradients or learned representations at a matched update. Improved eval member AUC or eval prehead geometry is compatible with identical mathematical TRAIN learning.

In the nine-fit factorial, geometry and input correction can change TRAIN objectives/activations and their running-state histories. **At fixed geometry/correction, BN ownership is an eval-state and selection/stopping factor under this source contract.** A changed selected-VALID geometry-by-correction interaction can arise from that changed eval map and selected epochs; it is not evidence that BN repaired a TRAIN gradient interaction. Attribute pooled AUC/BCE contrasts to the complete declared procedures, preserve epoch/cost differences, and distinguish them from same-weight buffer correspondence. Existing bank-assignment controls establish correspondence sensitivity at their selected weights, not the causal partition of the full gain.

No source was modified, and no model, tensor, dataset, remote call, numerical comparison, new paper read or Git operation was used for this note.
