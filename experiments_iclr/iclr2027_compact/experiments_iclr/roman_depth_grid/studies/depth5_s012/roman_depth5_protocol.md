# Roman Empire five-layer depth sensitivity, post hoc protocol v1

This study was chosen after inspecting earlier Roman results, including test
scores from the two-layer self-loop study. It is a retrospective sensitivity
analysis. It is not a prospective test or a basis for choosing additional
reported variants.

Use the corrected public Roman Empire NPZ (`data/roman_empire.npz`, SHA-256
`a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428`),
official mask 0, and optimization seeds 0, 1, 2. Symmetrize/coalesce raw
edges with PyG and add exactly one explicit self-loop per node. Relative to
the archived `roman_selfloop.py` study, change only the number of residual
SAGE blocks from two to five. Keep width 128, four members, LayerNorm,
dropout 0.2, AdamW learning rate 0.001, zero weight decay, 300 full-batch
epochs, mean member cross-entropy, and every-epoch joint checkpoint
selection by pooled validation accuracy, then lower pooled validation
cross-entropy, then earliest epoch. Restore the checkpoint before scoring
test. No early stop or hyperparameter search.

The two fixed arms are tied boundary-factor GNNM and initially matched
untied propagation. Require identical canonical initial parameters and
post-construction CPU/CUDA/Python/NumPy RNG states, and CUDA initial member
logits within 1e-5. Test labels are used only after restoring the
validation-selected checkpoint. This is transductive node classification.

Pin the runner, original `models.py`, this protocol, raw NPZ, parsed tensors,
and mask in a source manifest before training. Each arm must produce 300
validation trace rows, initial audit, selected checkpoint, member test
logits, exact test IDs/labels, and artifact hashes. Require all six arms
and an independent selected-checkpoint replay before reporting any paired
score. Keep this study separate from the two-layer self-loop results and
all other experiments.

Scope: A result here shows sensitivity to depth under this fixed setting.
It does not isolate the source of discrepancies from earlier five-layer
controls, which also differ in width, training budget, optimizer, checkpoint
rule, split handling, and software edge conventions.
