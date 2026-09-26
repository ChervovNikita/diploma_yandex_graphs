# Roman Empire parameter-matched narrow UNTIED control

This is a six-cell resource-sensitivity control, designed after the
complete Roman24 test audit. It compares a narrower four-stack UNTIED
model against the immutable width-128 TIED cell at the same depth and
optimization seed. Both use Roman Empire official mask 0. It does not
match the initial function, since changing width changes learned matrix
shapes and initialization. It cannot isolate a pure architecture cause.

## Width selection, fixed before narrow training

For each depth, instantiate the frozen model class at every integer
width from 32 through 128. Count all trainable parameters, including
projectors, graph stacks, LayerNorm and output layers. Select the
UNTIED width minimizing absolute parameter-count difference from the
existing TIED width-128 model. On a tie choose the smaller width. No
accuracy or score enters this rule. The full 97-candidate table per
depth is in `ROMAN_NARROW_UNTIED_WIDTH_LOCK.json` (SHA-256
`49df5c4706694914250f08513ffbb30bee5a78d6ae1368a2f083383d461ad9e9`).

| Depth | TIED width | TIED parameters | Selected UNTIED width | UNTIED parameters | Difference |
|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 208,960 | 68 | 211,600 | +2,640 (+1.26%) |
| 5 | 128 | 456,640 | 65 | 451,924 | −4,716 (−1.03%) |

## Frozen six-cell matrix

Run depths 2 and 5, seeds 0/1/2, one narrow UNTIED arm at its locked
depth-specific width. Use the frozen Roman24 official mask-0 data and
65,854 symmetrized directed edges without added loops; four boundary
projectors; the same residual SAGE graph-stack class, dropout 0.2,
member-mean training cross-entropy, AdamW learning rate 0.001, zero
weight decay, and 1,000 full-batch epochs. Reset Python, NumPy, CPU and
CUDA RNG states to the seed at the start of each cell. This matches the
Roman24 UNTIED training recipe except for width. No tuning or early
stopping is allowed.

Save each epoch's pooled validation accuracy and cross-entropy. Select
the checkpoint by highest validation accuracy, then lowest validation
cross-entropy, then earliest epoch. The training model and optimizer see
training and validation node IDs/labels, not test IDs/labels. Before
training, freeze the width lock, source files, library versions, six-cell
matrix, original Roman24 source/validation/test-audit hashes, and the six
TIED128 baseline result/checkpoint/test-result hashes. All six initial
models must pass seed-reset, finite-logit, parameter-count, and repeated
initial-function checks on CUDA.

After all six 1,000-epoch cells finish, independently replay every initial
state, 1,000-row trace, selected checkpoint, and selected validation
logits. Write a complete six-cell validation lock before any narrow test
score is opened. Then score each official test cell once. A separate
CUDA audit must replay all six selected checkpoints, test scores and
class decisions, and compare the held-out accuracy to the immutable
same-depth/same-seed TIED128 score. Report all six paired signs and the
parameter counts. Repeated seeds share one graph and one mask, so results
are descriptive; they cannot support an across-graph generalization claim.
