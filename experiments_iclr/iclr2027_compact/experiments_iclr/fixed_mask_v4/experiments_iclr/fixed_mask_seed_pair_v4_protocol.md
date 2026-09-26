# Roman Empire fixed-mask width-512 SAGE replication: disclosed post hoc v4 protocol

This is a fresh six-run diagnostic on the favorable Roman Empire setting. Earlier
v1 and v2 attempts stopped before training due numerical and RNG gates. V3
stopped after three completed arms because seed-1 untied initial CUDA logits
had a maximum absolute difference of 1.049041748046875e-5 against a frozen
1e-5 bound. No V3 partial test scores enter V4 or the paper. The failed V3
artifacts and log are preserved; V4 starts in a separate result root and
trains all six arms from scratch. This entire study is post hoc and does not
estimate across-graph variation.

## Frozen architecture and selection

Roman Empire corrected NPZ and official mask 0; optimization seeds 0, 1, 2;
variants `gnnm` and initially matched `untied_backbone` in that order per seed.
Both use four members, five residual SAGE layers, width 512, LayerNorm,
dropout 0.2, default BatchEnsemble initialization, AdamW learning rate 3e-5,
zero weight decay, and mean four-member cross-entropy. Untied propagation has
four deep copies of the same initial residual stack; boundary projectors and
output normalization remain shared. Train each arm independently for at most
5,000 steps. Validate pooled accuracy at step 1 and every ten steps, select
the earliest strict maximum, and stop after 300 validation-counted steps
without improvement. Restore that checkpoint and score the official test mask
once. Report all three paired test differences and their selected steps and
validation scores regardless of sign. No variant or seed is selected by test.

## Identical initial state with calibrated finite-precision gate

The graph loader must reproduce the exact original DGL-processed edge array,
including edge order; features, labels, masks, and train/validation/test
partition sizes are pinned. Canonical initial parameter tensors and CPU/CUDA
training RNG state hashes must match exactly across paired arms. The CPU smoke
requires exact initial logits on a small graph and a valid training step.

Before V4 training, a no-label/no-score calibration on the same CUDA runtime
ran six evaluation forwards of each initial tied and untied model for each of
the three seeds, with the full Roman graph. Its maximum absolute difference
between repeated forwards of the *same model and same state* was
1.0967254638671875e-5. The V4 initial-pair and independent replay tolerance
is defined as twice that observed same-state variation,
2.193450927734375e-5. The complete calibration rows and source are frozen as
a referenced evidence file. This tolerance is a numerical reproducibility
gate, not a claim that all floating-point logits or hard decisions are exactly
equal. Initial decision differences are reported descriptively. The
calibration also found cross-variant repeated-forward maxima up to
1.1920928955078125e-5, below the chosen gate. The source manifest pins this
calibration, the runner, verifier, data, and runtime before any V4 training.

The independent completion gate requires exactly six complete arms, artifact
hashes, validation traces, matched initial state/RNG, selected checkpoint
replay, saved official-mask predictions, and recomputed validation/test scores.
An incomplete or failed arm invalidates the V4 result; it is not silently
replaced. Full selected predictions and checkpoints are author evidence.
