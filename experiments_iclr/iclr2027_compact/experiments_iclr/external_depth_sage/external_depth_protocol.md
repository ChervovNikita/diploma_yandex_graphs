# External SAGE depth grid, frozen protocol v1

This grid is a post hoc extension chosen after observing the Roman Empire SAGE
2-vs-5-depth interaction. The present WikiCS and Actor depth-5 outcomes are
unobserved at freeze time. We will report every predeclared cell, including
unfavorable results. No result from this grid will be treated as a prospectively
registered discovery or as independent graph-split replication.

## Fixed grid

Datasets: WikiCS and Actor/Film, their official published split 0 only.
Optimization seeds: 0, 1, 2. Residual SAGE depths: 2 and 5. Arms: tied GNNM
and untied propagation. There are 2 x 2 x 2 x 3 = 24 complete runs.
Published training labels alone enter the full-batch loss; feature and edge
structure for all nodes may be used transductively. Edges are undirected with
no explicitly added self-loops. Every cell uses original `models.TABMModel`
SAGE, four members, width 128, LayerNorm, dropout 0.2, AdamW learning rate
0.001 and weight decay zero. At a given dataset/depth/seed, the untied arm is
constructed from exact copies of the tied residual stack; the input/output
BatchEnsemble maps and output normalization remain shared. Initial state and
post-construction Python, NumPy, CPU and CUDA RNG states must match, and
initial member logits must differ by at most 1e-5 maximum absolute error.
This threshold was fixed from the earlier v3 A100 CUDA smoke results and
addresses floating-point reduction drift, not different parameter starts.

Training lasts exactly 300 epochs for every run, no early stopping. The loss
is the mean cross-entropy of the four member heads. At every epoch, evaluate
pooled raw member logits on the validation nodes. Select the highest
validation accuracy, then lowest pooled validation cross-entropy, then the
earliest epoch. Restore the selected joint checkpoint before test scoring.
No hyperparameter choice or arm selection may use test labels.

## Audit and reporting gates

Before launch, source, protocol, raw data bytes, parsed tensors and software
versions are pinned in each dataset/depth manifest. The data loader and source
are direct copies of the previous external SAGE study except parameterized
depth, narrowed arms, protocol ID and separate result paths. Run CPU
preflight and GPU initialization smoke for all four dataset/depth pairs before
opening test outcomes. Existing SAGE result directories are untouched.

Every cell must provide a 300-row validation trace, initialization audit,
selected checkpoint, selected member logits for validation and test,
official indices/labels, parameter count and hashes. An independent audit on
the Mac must check manifest/source lock, official split/labels, arithmetic of
all 24 selected scores and checkpoint replay where feasible. If a cell fails,
exclude the grid from the paper rather than report only favorable cells.
Report each paired tied-minus-untied validation and test difference, both
means by dataset/depth and their signs. This study cannot establish a
universal depth law because it fixes one split and three optimization seeds.
