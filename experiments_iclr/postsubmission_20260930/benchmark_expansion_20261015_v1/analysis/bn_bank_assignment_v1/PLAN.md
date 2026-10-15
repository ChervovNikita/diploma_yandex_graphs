# Normalization bank assignment at fixed selected weights

This development diagnostic tests whether the successful private-BN policy
depends on assigning each route its own TRAIN-derived statistics. It uses all
three completed MolBACE private-BN models (seeds 101, 202, 303), their already
selected checkpoints and all 151 native VALID graphs. No model is trained,
checkpoint selected, normalization recalibrated or TEST prediction evaluated.

For each checkpoint evaluate these five fixed conditions:

1. **Own bank:** each route uses its original saved statistics.
2. **Tied bank:** every route uses the arithmetic mean of the four saved running
   means and, separately, the arithmetic mean of their running variances.
3. **Cyclic shifts 1, 2 and 3:** route m uses bank (m + shift) modulo four.

The three shifts preserve the set of saved banks while removing the original
route-to-bank correspondence. All shifts are reported; none is selected as a
better prediction rule. The tied variance is an average of EMA variances, not
the variance of a pooled route population. All learned weights, checkpoint
epochs, labels, graph order, aggregation and BN configuration remain fixed.
Counters are checked equal across routes and unchanged. All five conditions
use average raw logits. The existing selected original export remains intact.

Report AUC, BCE and Brier for every condition/seed, own-minus-tied differences
and own-minus-average-of-three-shift scores. Averaging the three *scores* is a
diagnostic summary, not a new ensemble. No hypothesis test or superiority
claim is warranted by these three development-selected checkpoints. An own-bank
advantage would identify a dependence on bank assignment at these weights;
it would not isolate the original EMA time-scale, route-order or selection
effects, nor establish generalization to another dataset.

Execution is CPU-only on the authorized allocation, two threads, one seed at
a time, at most 300 seconds per seed, no retries. Each condition has five
source VALID batches and four native route forwards: 100 native forwards per
seed, 300 total, three constructions, zero backward/optimizer/TRAIN calls.
The own-bank call also checks reconstruction against the saved export; no
separate parity run is added. Save scalar reports and source/checkpoint hashes
for Git; keep tensors only in the allocation repository. Preserve any failure.
