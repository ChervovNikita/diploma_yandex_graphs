# Roman Empire intermediate-depth sweep, post hoc frozen protocol v1

This sweep was selected after inspecting earlier two-layer and five-layer Roman
results, including test scores. It is a retrospective mapping of depth
sensitivity, not a prospective confirmation. Both intermediate depths,
three and four residual SAGE blocks, must be run and reported together,
regardless of outcome. No further depth or other model variant is to be
selected using these test results.

For each depth, use the corrected public Roman Empire NPZ
(`data/roman_empire.npz`, SHA-256
`a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428`),
official mask 0, and optimization seeds 0, 1, 2. Symmetrize/coalesce raw edges
with PyG and add exactly one explicit self-loop per node. Keep the archived
two/five-layer setting: width 128, four members, LayerNorm, dropout 0.2,
AdamW learning rate 0.001, zero weight decay, 300 full-batch epochs, mean
member cross-entropy, and every-epoch joint checkpoint selection by pooled
validation accuracy, then lower pooled validation cross-entropy, then
earliest epoch. Restore the checkpoint before scoring test. No early stop or
hyperparameter search. Relative to the archived two-layer self-loop study,
only residual-block count differs; relative to the five-layer study, again
only depth differs.

The two fixed arms per depth are tied boundary-factor GNNM and initially
matched untied propagation. Require identical canonical initial parameters
and post-construction CPU/CUDA/Python/NumPy RNG states, and CUDA initial
member logits within 1e-5. Test labels are used only after restoring the
validation-selected checkpoint. This is transductive node classification.

Pin each runner, original `models.py`, this joint protocol, raw NPZ, parsed
tensors, and mask in a source manifest before training. Each arm must produce
300 validation trace rows, initial audit, selected checkpoint, member test
logits, exact test IDs/labels, and artifact hashes. Require all 12 arms,
three seeds for both depths, plus independent selected-checkpoint replay
before reporting any paired score. Keep this sweep separate from all earlier
Roman studies and external data studies.

Scope: The full depth 2–5 pattern describes this fixed setting on one
published mask with three optimization seeds. It does not identify the cause
of discrepancies from earlier selected Roman controls, which also differ in
width, optimization, budget, checkpointing, and software edge conventions.
