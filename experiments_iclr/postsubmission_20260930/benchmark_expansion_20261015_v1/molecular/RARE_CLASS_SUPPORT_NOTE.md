# Molecular label support under the native sampler

The original batch size is32 for both molecular datasets. A TRAIN-only audit
of the pinned processed labels gives the following exact population counts.
The batch probabilities use sampling without replacement; they describe a
uniformly shuffled full batch, not an observed training trajectory.

| Native TRAIN population | Graphs | Negative / positive | Both classes in batch | At least two positives |
| --- | ---: | ---: | ---: | ---: |
| ogbg-molbace | 1,210 | 730 / 480 | 100.00% | 100.00% |
| ogbg-molhiv | 32,901 | 31,669 / 1,232 | 70.53% | 33.81% |

For MolHIV,29.47% of full batches are expected to contain no positive molecule,
and36.72% exactly one. Thus only33.81% contain at least two distinct positive
molecules. MolBACE almost always supplies both classes and multiple examples.
No VALID or TEST labels, model predictions or optimizer states were inspected.
The audit took3.38 seconds and constructed no model.

This matters for the precise objectives already implemented. Class alignment
is inactive when a batch has only one class. With one positive molecule,
its other route copies still provide cross-route positive pairs, but they do
not provide a second distinct positive molecule. Positive-class conditional
variance and distinct positive-class donor targets are unsupported in that
batch. The negative class usually remains supported. Existing source records
these cases and preserves the full native supervised loss.

This is a limitation of available auxiliary examples per step. It does not
establish that the objective fails on MolHIV, explain the MolBACE outcome, or
justify changing the source batch sampler. Longer training provides more
random batches but does not change this expected support fraction. We should
report observed support in complete MolHIV runs and keep the same native
batch size for fair comparisons.

A future extension could retain a bounded queue of earlier TRAIN representations
so rare-class molecules have distinct donors across batches. Memory-based
contrastive training is established prior work; such a queue would not itself
be a novelty claim. It introduces stale representations, state-restoration
requirements and another controlled comparison. It is recorded as an untested
possibility, not added to the6,460-fit roster or substituted for its frozen
methods. The existing factual-context comparison remains useful for testing
route information targets, with its own support limitations made explicit.

`TRAIN_SUPPORT_REPORT.json` contains exact input hashes, counts, assumptions
and costs. `train_support_audit.py` reproduces this metadata analysis. Original
hyperparameters, methods, training jobs and paper scores were not changed.
