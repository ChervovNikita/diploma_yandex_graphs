# Roman Empire 1000-epoch budget sensitivity, post hoc protocol v1

This study was selected after seeing the original 300-epoch depth-2 and
depth-5 results, and frozen before opening the complete 40-cell depth-grid
scores. It probes whether the fixed 300-epoch budget limits the depth
comparison. Run every predeclared cell: residual depths 2 and 5, optimization
seeds 0, 1, and 2, and tied GNNM versus initially matched untied propagation
at each depth and seed. All 12 arms must be reported together after an
independent checkpoint replay; no result may be used to choose another
budget, depth, or seed.

Use the corrected public Roman Empire NPZ (`data/roman_empire.npz`, SHA-256
`a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428`),
official mask 0, the symmetrized/coalesced PyG graph with exactly one
explicit self-loop per node, width 128, four members, LayerNorm, dropout
0.2, AdamW learning rate 0.001, zero weight decay, and mean member
cross-entropy. Relative to each archived 300-epoch counterpart, change
only the training budget to exactly 1000 full-batch epochs. Validate every
epoch, select the checkpoint by pooled validation accuracy, then lower
pooled validation cross-entropy, then earliest epoch. Restore that
checkpoint before scoring test. There is no early stop or hyperparameter
search. The original 300-epoch study and this 1000-epoch study are
separate complete runs from the same seed-defined initial conditions.

Require identical canonical initial parameters and post-construction
CPU/CUDA/Python/NumPy RNG states in every tied/untied pair, and CUDA initial
member logits within 1e-5. Test labels are used only after restoring the
validation-selected checkpoint. This is transductive node classification.

Pin each runner, original `models.py`, this joint protocol, raw NPZ,
parsed tensors, and mask in a source manifest before training. Each arm
must produce exactly 1000 validation trace rows, initial audit, selected
checkpoint, member validation/test logits, exact IDs/labels, and artifact
hashes. Require all 12 arms and independent CUDA selected-checkpoint replay
before reporting any paired score or comparing with the 300-epoch results.

Scope: Budget sensitivity on one Roman mask and three optimizer seeds.
This does not establish a generally optimal training duration or isolate
all differences from the original width-512 controls.
