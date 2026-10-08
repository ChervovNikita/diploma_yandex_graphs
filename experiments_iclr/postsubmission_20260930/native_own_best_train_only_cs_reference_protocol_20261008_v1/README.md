# Fixed C&S reference budget

Eighteen autoscale correction/smoothing configurations and six smoothing-only configurations form a prospectively fixed validation budget. The reference uses each native model's own selected checkpoint and only TRAIN labels. A common recipe is selected by mean development correctcount across all three seeds; every configuration and cost is retained. No WikiCS author preset is claimed.

The supplied graph removes the native preparation's known self loops. Author normalization and propagation remain unchanged. The score-to-probability map is declared before scoring; zero target probabilities retain infinite NLL. Nothing has been executed and the candidate's original gates remain fixed.
