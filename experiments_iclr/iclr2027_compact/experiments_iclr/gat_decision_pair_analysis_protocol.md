# Roman Empire GAT decision analysis

Run after the GAT pair launcher reports success:

    .venv/bin/python experiments_iclr/gat_decision_pair_analysis.py

The script first runs the existing GAT verifier with --complete. It refuses to
read predictions or write results unless the frozen protocol, all ten rows,
checkpoints, and saved prediction files pass that audit. It then checks that
each variant uses the same official held-out node IDs and labels on each mask.

The output is gat_failure_pair_results/gat_decision_pair_analysis.csv, with
one paired row for each of official masks 0 through 4. An adjacent JSON file
records source and input SHA-256 hashes. Run --self-test to check the numerical
identities on a small CPU example without reading or writing study results.

The model supplies four class-logit vectors per test node. The reported pooled
decision averages these four logit vectors and takes the largest coordinate.
Mean member accuracy averages the four separate member decisions. Pooling gain
is pooled accuracy minus mean member accuracy in percentage points. Pair
disagreement averages the fraction of nodes on which each of the six pairs of
members chooses different classes.

A rescue counts one member-node pair where that member is wrong and the pooled
decision is correct. A harm counts one member-node pair where that member is
correct and the pooled decision is wrong. The exact pooling gain in percentage
points is 100 times (rescues minus harms) divided by four times the number of
test nodes. Counts describe decisions, not independent observations.

Cross-entropy uses float64 logsumexp on the saved member logits. For each node,
mean member cross-entropy minus pooled-logit cross-entropy equals the mean
KL divergence from the pooled softmax distribution to each member softmax
distribution. The script checks this identity on every node with a maximum
absolute tolerance of 1e-8 nats. The CSV reports the five-mask metrics
separately and the difference between GNNM and the untied model for each mask.
It also counts nodes where only one variant is correct and where both are
correct or wrong.

This is a post hoc descriptive study on one graph. The failure case and depths
were chosen after archived scores were viewed. The five official masks overlap
in graph nodes, so their outcomes and individual nodes are not independent
replications. These results cannot establish a general advantage for tying on
unseen graphs. They can show exactly which paired decisions changed on this
selected case.
