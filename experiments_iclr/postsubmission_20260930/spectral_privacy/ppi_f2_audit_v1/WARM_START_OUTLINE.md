# Minimal later warm-start spectral training comparison

This is an outline, not an executable final-training protocol or a runner change. The fixed F2 audit is the current task. A later coordinator decision must freeze a separate final-training study before training or validation comparisons.

For each seed0/1/2 and label10/100 cell, use four arms:

| Arm | Initial common state | Private correction |
| --- | --- | --- |
| tied | Exact cloned original P-only epoch25 source | rank0 in all three targets |
| selected | Same exact cloned source | Saved selected U; zero A |
| random | Same exact cloned source | Existing matching-rank random U bytes; zero A |
| raw | Same exact cloned source | Existing matching-rank raw centered-gradient U bytes; zero A |

Clone a plain common checkpoint independently for every arm. Preserve original source ABI, neighbor bias, all root/normalization/FFN/boundary weights, M4/h128/three SAGE blocks, and native mean-member loss. Freeze U throughout final training. Initialize centered private A to zero. Source-forward identity and common tensor hashes must agree at the start. A rank0 selected cell makes the three correction arms functionally tied; keep and report that cell without forcing capacity.

Create a fresh AdamW optimizer in every arm, with all moments and steps reset. Do not load the P25 optimizer history. Freeze identical optimizer hyperparameters, epoch/graph-step budget, all20 original training graph IDs and masks, graph-order/RNG streams, precision, checkpoint cadence, and runtime handling for every arm. Training remains on the observed training-label intervention. This differs from final cells initialized at canonical epoch0, so it needs its own study identity and cannot reuse those scores.

Before any execution, the coordinator should fix the remaining budget and checkpoint-selection rule, validation-access policy, seeds, comparisons, reporting, and GPU availability gate. Resetting AdamW equalizes initialization; it does not make Adam's trajectory follow the projected-SGD identity. F2 proxy results must not choose arm-specific hyperparameters, ranks, stop times, or favorable cells. Any method revision after F2 needs a separate prospective study rather than a revised claim for this one.

Charge the common P-only probe, D/E selection, F2 audit if performed, final fitting, and exact replay costs explicitly. Record trainable parameters, basis/contrast buffers, optimizer memory, activations, graph propagation, latency, and total wall time separately. No propagation or activation saving follows from these neighbor-weight corrections. Matching random/raw ranks is required for capacity comparison.

Final output should keep per-cell arm results and paired differences visible, report rank0 cells, and account for the shared graph identities and nested label masks. Accuracy or AdamW-benefit conclusions require those separate training outcomes and the frozen evaluation protocol; the present F2 audit provides neither.
