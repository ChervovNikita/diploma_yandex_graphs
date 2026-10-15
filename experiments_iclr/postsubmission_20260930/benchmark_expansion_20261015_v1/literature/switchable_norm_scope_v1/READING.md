# Prior work relevant to route-specific normalization

## Source and reading scope

Read the abstract of **Slimmable Neural Networks**, arXiv:1812.08928, and the
author repository's complete `SwitchableBatchNorm2d` and `USBatchNorm2d`
definitions in `models/slimmable_ops.py`, commit
`7f0e4ef5f6abb07a054c1722abc6ea6ad77b682d`. The retrieval receipt preserves
URLs, time and hashes. This is a scoped abstract/code inspection, not a claim
to have read the full paper or verified its experiments.

The abstract describes one shared network executable at different widths,
trained with switchable batch normalization. `SwitchableBatchNorm2d` selects
a separate ordinary BN module for each configured width; those modules include
separate affine parameters. The same repository's `USBatchNorm2d` implementation
uses shared affine vectors, slices them for the active width, and passes the
chosen width's own running mean and variance to `functional.batch_norm`.
Thus separate running statistics with shared learned affine parameters already
have a concrete implementation precedent in this family of networks.

The latter code belongs to the repository's universal-width implementation.
This inspection does not establish which precise paper/version introduced each
line. Cite the pinned implementation for that statement rather than attributing
every detail to the 1812.08928 manuscript without checking it.

## Consequence for GNNM

Our current graph intervention switches statistics by ensemble route at fixed
native width. It shares every learned backbone/BN-affine parameter, uses native
TRAIN histories only, and performs no post-hoc VALID recalibration. The setting
and conditioning variable differ, but the basic normalization mechanism has
clear prior art. A private-BN validation gain should be presented as a useful
control and implementation finding, not as the paper's methodological novelty.

The fixed-weight bank-assignment diagnostic makes that control informative:
there is a measurable mismatch cost when route histories are tied or exchanged.
The remaining scientific objective is to improve the learned route predictions
and their complementary errors under competent normalization. The ongoing
geometry/correction factorial study addresses that question. No new benchmark
arm, hyperparameter search or paper claim follows from this reading alone.

Sources: https://arxiv.org/abs/1812.08928 and
https://github.com/JiahuiYu/slimmable_networks/blob/7f0e4ef5f6abb07a054c1722abc6ea6ad77b682d/models/slimmable_ops.py
