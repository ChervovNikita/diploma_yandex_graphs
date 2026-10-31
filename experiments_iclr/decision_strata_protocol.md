# Roman Empire SAGE decision strata: analysis protocol

This author note fixes the comparison and output before the pooled ENS test predictions are available. At 06:06 UTC on September 25, 2026, the main control CSV contained 20 of the required 50 rows and no `ens_pooled` row. The analysis is exploratory. It is not evidence for the manuscript unless the complete results have been checked and interpreted separately.

## Inputs and gate

Use exactly the ten variants in `projector_controls.py` on official Roman Empire masks 0 through 4, all with SAGE, five layers, width 512, learning rate 3e-5, a 5,000-step cap, seed equal to mask index, and four members for GNNM and pooled ENS. For both methods, the chosen checkpoint uses pooled validation logits. Require all 50 unique result rows, the selected checkpoint and prediction file for every row, matching settings, and the split-0 verification audit. The pilot or fresh GNNM split-0 result is used solely as directed by that audit. Reject incomplete or inconsistent inputs before writing any result.

The graph, its node labels, and the official test masks come from `data/roman_empire.npz`. Recreate the undirected graph by adding reverse edges, removing duplicate directed pairs, and removing self loops. For each node, count incoming neighbors after that transformation. Count how many of those neighbors have the same true class label. Compare the resulting degree and local homophily with the values saved in **every** selected prediction file. Require the selected node IDs and true labels to match the official test mask, and require saved predictions to match the saved logits. This makes the descriptor and node alignment checks independent of either method's accuracy.

## Fixed groups and measurements

Analyze two separate partitions of each test mask:

1. Graph degree exactly 2 versus graph degree at least 3.
2. Zero same-label neighbors versus at least one same-label neighbor. This is equivalent to local label homophily exactly 0 versus greater than 0 when degree is positive.

The code refuses a test node of degree below 2 because these fixed degree groups would not cover it. A group with no test nodes also fails. No threshold is chosen from model results. The two partitions are not crossed into four joint groups.

Within each group and mask, count test nodes and the four paired correctness outcomes: both pooled methods correct, only GNNM correct, only pooled ENS correct, and neither correct. Calculate each method's pooled accuracy, then GNNM minus ENS accuracy in percentage points. The paired difference also equals 100 times (GNNM-only correct nodes minus ENS-only correct nodes) divided by the group size. For each method, calculate average member accuracy over four members and nodes, pooled accuracy minus that member average, and the mean disagreement rate over the six pairs of member class decisions. Also calculate the fraction of nodes where the two pooled class decisions differ. Pooling means averaging the four member **logits** before taking the class with the largest mean logit, matching the runner.

Report one row per mask and group. For each of the four groups, report the unweighted mean and minimum to maximum range across the five masks for every count and rate. The unweighted mask mean describes the fixed evaluation protocol. It is not a standard error or a confidence interval.

## Interpretation limit

All masks come from one graph, and test nodes can appear in multiple masks. Do not treat the 20 group rows or their node counts as independent observations, compute an independent-node significance test, or infer a population-wide causal effect. Group membership from local label homophily uses the **true labels of neighboring nodes**, including held-out neighbors. It is a retrospective diagnostic only. Neither the model nor an inference-time gate is allowed to use those labels. The analysis can describe where two selected models disagree on this graph; it cannot establish why the difference arose or predict which model would win on another graph.
