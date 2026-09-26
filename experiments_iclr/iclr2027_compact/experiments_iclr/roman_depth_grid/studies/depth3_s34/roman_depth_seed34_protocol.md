# Roman Empire depth-grid optimization-seed extension, post hoc protocol v1

This extension was frozen after inspecting the earlier depth-2 and depth-5
Roman scores, but before running or inspecting any depth-3/4 score. It adds
optimization seeds 3 and 4 to every fixed depth 2, 3, 4, and 5. Both tied
GNNM and initially matched untied propagation must run at every depth and
seed. All 16 extension arms must be reported together with the 24 already
frozen/run depth-2/5 and depth-3/4 seed-0/1/2 cells, for the complete
40-cell grid. These are optimizer-seed repetitions on one published split,
not independent graph splits. No arm, depth, or seed may be selected based
on test results.

Use the corrected public Roman Empire NPZ (`data/roman_empire.npz`, SHA-256
`a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428`),
official mask 0, and the same self-loop graph: symmetrize/coalesce raw
edges with PyG and add exactly one explicit self-loop per node. For each
depth, use width 128, four members, LayerNorm, dropout 0.2, AdamW learning
rate 0.001, zero weight decay, 300 full-batch epochs, mean member
cross-entropy, and every-epoch joint checkpoint selection by pooled
validation accuracy, then lower pooled validation cross-entropy, then
earliest epoch. Restore the checkpoint before scoring test. No early stop
or hyperparameter search.

Require identical canonical initial parameters and post-construction
CPU/CUDA/Python/NumPy RNG states for the tied/untied pair, plus CUDA initial
member logits within 1e-5. Test labels are used only after restoring the
validation-selected checkpoint. This is transductive node classification.

Pin each runner, original `models.py`, this protocol, raw NPZ, parsed
tensors, and mask in a source manifest before training. Each arm must
produce 300 validation trace rows, initial audit, selected checkpoint,
member test logits, exact test IDs/labels, and artifact hashes. Require
all 16 extension arms and independent selected-checkpoint replay for all
four depths before reporting the full grid.

Scope: The complete depth 2–5 pattern describes the fixed width-128 setting
on one Roman mask. It does not establish a general depth law or explain
discrepancies from earlier Roman controls with several other differences.
