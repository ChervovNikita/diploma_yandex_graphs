# Prospective holdout prediction for sharing-position tests

Frozen before reading any Cora or Chameleon depth or sharing-position outcome.
The previously observed four-graph pattern is post hoc: private-first had the
higher selected test accuracy on Roman-Empire and ogbn-arxiv, whose fixed train
sets exceed 5,000 nodes, while private-last had the higher selected test
accuracy on WikiCS and Actor, whose fixed train sets have fewer than 5,000
nodes. Training-set size may be a proxy for graph structure, heterophily,
optimization, or another factor; this is only an exploratory prediction.

For Cora Planetoid public split and Chameleon Geom-GCN split 0, each with fewer
than 5,000 training nodes, the predicted sign of mean paired test accuracy
(private-first minus private-last, three optimizer seeds, equal parameters,
300 epochs, validation-selected checkpoint) is negative on both graphs.
The sign of each individual seed is not predicted. The full four-arm study
(tied, private-first, private-last, untied) must be completed and audited on
both graphs before evaluating this prediction. If not completed before the
submission packaging cutoff, this file remains author research evidence only
and no claim about the prediction enters the submission.
