# Roman Empire self-loop sensitivity, post hoc protocol v1

This study was chosen after inspecting the earlier Roman bridge and its
test results. It is a post hoc sensitivity analysis. The source, data, masks,
three arms, seeds, and verifier are frozen here before any self-loop run.

Use the corrected public Roman Empire NPZ (`data/roman_empire.npz`, SHA-256
`a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428`),
official mask 0, and optimization seeds 0, 1, 2. Symmetrize/coalesce raw
edges with PyG and add exactly one explicit self-loop per node. This is the
**only intended change** from the separately archived no-loop Roman bridge.
It aligns the existence of self-loops with the original selected Roman
controls, but still differs from them in depth, width, learning rate,
training budget, checkpoint rule, mask selection, and PyG/DGL edge handling.

Keep the OGB-300 schedule: original repository `TABMModel` SAGE, four members,
two residual layers, width 128, LayerNorm, dropout 0.2, AdamW learning rate
0.001, zero weight decay, 300 full-batch epochs, mean member cross-entropy,
and every-epoch joint checkpoint selection by pooled validation accuracy,
then lower pooled validation cross-entropy, then earliest epoch. Restore the
checkpoint before scoring test. No early stop or hyperparameter search.

The three fixed arms are tied boundary-factor GNNM, initially matched
untied propagation, and tied GNNM with identity-initialized in-layer factors
around the four linear maps in each SAGE block. Require identical canonical
initial parameters and post-construction CPU/CUDA/Python/NumPy RNG states,
and CUDA initial member logits within 1e-5 (same-model A100 scatter drift
was 2–3e-6). The independent CPU smoke must show exact tied/untied initial
logits. Test labels are used only after restoring the validation-selected
checkpoint. This is transductive node classification.

Pin the runner, original `models.py`, this protocol, raw NPZ, parsed tensors,
and mask in a source manifest before training. Each arm must produce 300
validation trace rows, initial audit, selected checkpoint, member test
logits, exact test IDs/labels, and artifact hashes. Require all nine arms
and a fresh selected-checkpoint replay before reporting any paired score.
Run and result folders are separate from the no-loop bridge and the
prospective external WikiCS/Actor study.
