# Roman Empire mask-0 optimizer-history diagnostic, V3 numerical preflight

The V1 Roman design and source were frozen after earlier Roman mask-0
studies, including a favorable width-512 result, and before test outcomes
from the 48-cell masks-1–4 study were opened. V1 Roman training was never
started. A separate V1 NORM-SYNC run on Cora failed a strict applied-update
norm gate after FP32 rounding at epoch 510. A training-only diagnostic
reproduced that failure without inspecting held-out outcomes. Roman V2
was frozen after the complete 48-cell Roman study was opened, but before
any Roman24 training or test outcome. V2 CUDA preflight passed depth-2
seed 0, then stopped at depth-2 seed 1 on a too-strict first-step
TIED-versus-virtual AdamW **vector** equality gate. No V2 Roman model
was trained. Training-only diagnostics across all six depth/seed cases
found closely matching gradients, with small FP32 summation differences
amplified at coordinates near Adam's epsilon. V3 changes only that
pretraining numerical equivalence gate; no Roman24 training or test
outcome preceded this source freeze. This remains a post hoc diagnostic
on one graph and one official mask, not an untouched
confirmation or a new optimizer claim. Separate task moments have prior
work, including AdaTask (Yang et al., arXiv:2211.15055v2).

## Complete fixed matrix

Run exactly 24 new cells: Roman Empire official mask 0; residual depths 2
and 5; optimization seeds 0, 1, and 2; arms TIED, UNTIED, SYNC, and
NORM-SYNC. Each uses the same public corrected NPZ, 65,854 symmetrized
directed entries with no added self-loops, width 128, four boundary
BatchEnsemble projectors, LayerNorm, dropout 0.2, AdamW learning rate
0.001, zero weight decay, and exactly 1,000 full-batch epochs. The loss
is the mean of four member training cross-entropies. No tuning or early
stopping is permitted.

TIED shares one residual SAGE graph stack across members. UNTIED starts
with identical copies of that stack and trains each copy separately. SYNC
also keeps separate graph copies and separate AdamW moment tensors, but
after each optimizer step averages corresponding graph parameters and
copies the average to all four stacks. The private graph gradients are
multiplied by four before AdamW, reversing the mean-loss division and
giving each graph copy its own member's full gradient. Shared boundary
projectors and output normalization receive the ordinary mean-loss
gradient. NORM-SYNC uses the same provisional SYNC update direction,
then scales its single global graph-update L2 norm to match a persistent
virtual AdamW reference driven by the mean graph gradient at the current
NORM-SYNC weights. This reference is not the separately trained TIED
trajectory. Each target is formed in float64 and cast once to FP32. The
**applied FP32 parameter difference** must satisfy the original norm gate:
`abs(applied_norm-reference_norm) <= max(1e-8, 1e-5*reference_norm)`.
If the nominal scale misses this gate after rounding, use exactly 48
deterministic bisection steps in the fixed interval ±0.1% around the
nominal scale and choose the evaluated target with least absolute error.
Failure to bracket or satisfy the original gate invalidates the cell;
the tolerance is not relaxed. If the reference norm is zero, NORM-SYNC
applies zero update. If the candidate norm is zero while the reference
norm is positive, or a required value is nonfinite, the cell fails. The
mechanism primitive is copied byte for byte from the separately reviewed
study; `norm_sync_v2.py` is copied byte for byte from its independently
reviewed V2 numerical revision. `tuning.py` is a pinned Roman adapter and
does not reuse the external study's tuning or score selection.

## Gates before training

Freeze the design, exact source and public NPZ SHA-256, runtime, all 24
cells, and numerical tolerances before training. An independent agent
reviews the source. On the assigned CUDA device, for every depth and seed,
check all four arm constructions have identical canonical initial graph
and boundary tensors and Python, NumPy, CPU, and CUDA RNG state. Require
initial member logits within `1e-5` maximum absolute difference and equal
member decisions. Run five matched zero-momentum SGD steps comparing TIED
with SYNC. Their graph parameters and member logits must stay within
`1e-4`, with equal dropout RNG states after every step. For NORM-SYNC,
require that the virtual reference gradient is bitwise equal to the
explicit FP32 mean of all four private member graph gradients. Compare
this mean with the matched TIED accumulated graph gradient: global L2
error must be at most `1e-6 * max(gradient norms) + 1e-8`, and maximum
coordinate error at most `1e-7`. Independently reconstruct each arm's
first AdamW graph update from its own actual gradient in float64, cast
once to FP32, and require maximum coordinate difference at most two
float32 machine epsilons (`2.384185791e-7`). Assert FP32 graph tensors
and initial graph weights of maximum absolute value at most 1. Record
the TIED-versus-virtual first-step update vector L2 and Linf differences
without enforcing a numerically unstable equality between their
different FP32 gradient-accumulation paths. Keep the unchanged applied
global graph-update norm gate and bounded rounding correction against
the reference for five steps, separate private AdamW moment
storages, equal graph weights, and successful collapse to the TIED
inference layout. A failed gate stops the complete study before training.

## Validation lock and test scoring

Training bundles expose only training and validation node IDs and labels.
The loader may read the full public NPZ to construct the published masks
and fingerprints, but the training model and optimizer do not receive
test IDs or labels. Validate the pooled raw logits after every epoch.
Choose the selected checkpoint by greatest validation accuracy, then
lower pooled validation cross-entropy, then earliest epoch. Save the
full 1,000-row trace, initial logits and audit, selected checkpoint, and
selected validation member logits for each cell. Independently replay all
24 selected checkpoints, initial states, validation traces and scores,
norm gates, and collapse checks. Only the complete 24-cell validation
lock permits test scoring. After the lock, restore every selected
checkpoint, score each official test mask once, and independently replay
all 24 test outputs and scores. Report every predeclared depth/seed
contrast, including negative signs.

Both depths and all three seeds reuse one graph and one published mask.
Their differences are descriptive optimizer and depth sensitivity, not
independent graph replication or a general selection rule. The complete
24-cell V3 source, validation, and test replay audit must finish by
2026-09-26 06:00 UTC to be considered for this submission.
