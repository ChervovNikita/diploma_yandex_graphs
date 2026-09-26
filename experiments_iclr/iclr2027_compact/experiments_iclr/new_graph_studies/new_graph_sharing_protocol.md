# Frozen Cora and Chameleon sharing-position holdout

The exact source, graph data, masks, 24 cells, and selection rule are locked
before opening any Cora or Chameleon depth or sharing-position outcomes. The
graph choice is exploratory. The prediction that private-last will exceed
private-first on the mean paired test accuracy of each graph is recorded in
`POSITION_PREDICTION_FREEZE.md`; the training-label threshold that motivated it
was discovered post hoc on four earlier graphs.

Each graph uses the same loader, feature tensors, coalesced and symmetrized
edge index, and split as `new_graph_depth_protocol.md`: Cora Planetoid public
and legacy Chameleon Geom-GCN split 0. No self-loops are explicitly added;
Chameleon's public edge list already has 50 and they are retained. The
published Chameleon split has known duplicate-node/evaluation concerns, so
its result is descriptive and cannot by itself establish a general rule.

For each graph, optimization seeds 0, 1, 2 and four arms are mandatory:
`tied`, `private_first`, `private_last`, and `untied_propagation`. The
four-member `models.TABMModel` has width 128, two residual SAGE blocks,
LayerNorm, dropout 0.2, and the original BatchEnsemble input/output
projectors. The partial arms keep exactly one block private per member:
block 1 in `private_first`, block 2 in `private_last`. They must have equal
parameter counts and identical initial member functions. All arms use the
same canonical tied weights and post-construction CPU/CUDA RNG per seed.

Each run uses 300 epochs, AdamW with learning rate 0.001 and zero weight
decay, full-batch message passing, and mean member cross-entropy on training
labels. Validation is checked after every epoch. The selected checkpoint
maximizes pooled-logit validation accuracy, breaking ties with lower pooled
cross-entropy and then the earliest epoch. Test labels are evaluated only
after restoring that checkpoint. Initial-logit matching requires max absolute
drift at most 1e-5, checked on every node of each graph; selected
validation and test logits are saved for full CUDA replay, allowing at most
1e-4 numeric drift and requiring exact member and pooled decisions.

All 24 cells must pass `verify_new_graph_sharing.py` or the predicted sign is
not evaluated. The unit of replication is an optimization seed on one fixed
split. Convergence cannot be inferred from a 300-epoch selected score alone.
