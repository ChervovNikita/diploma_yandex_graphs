# Official filtered-Chameleon sharing-position holdout

The exact four-arm recipe and split were frozen before opening any result
from this filtered graph. The motivating low-training-label prediction was
written in `POSITION_PREDICTION_FREEZE.md` before its outcomes. That threshold
was discovered post hoc on earlier graphs and remains exploratory.

The graph, symmetric no-added-self-loop preprocessing, features, and official
mask 0 are the same as `new_graph_depth_protocol.md`. The complete matrix has
three optimizer seeds and four arms: tied, private-first, private-last, and
fully untied propagation. In each partial arm, exactly one of two residual
SAGE blocks is private per member; the other is shared. Private-first and
private-last have equal parameter counts and paired initial member functions.

All arms use four members, width 128, LayerNorm, dropout 0.2, 300 epochs,
AdamW at learning rate 0.001 with zero weight decay, and mean member
cross-entropy on training labels. Pooled validation accuracy chooses the
checkpoint after each epoch, with lower validation cross-entropy and then
earliest epoch as tie-breakers. Test labels are scored only after restoring
that checkpoint. The full-node initial-logit maximum absolute tolerance is
1e-5; selected validation/test CUDA replay allows at most 1e-4 numeric drift
and requires exact member and pooled decisions.

All 12 cells must complete and pass the full verifier. The planned
validation-choice secondary analysis uses both partial arms and three seeds;
it is a two-configuration search, not a new routing method.
