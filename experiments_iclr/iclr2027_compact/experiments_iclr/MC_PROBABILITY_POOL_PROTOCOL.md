# Additional aggregation of the fixed MC Dropout outputs

This analysis was specified after the five original raw-logit-pooling
outcomes had been inspected, and before computing their probability-pooling
outcomes. It introduces no new training, checkpoint selection, dropout-rate
tuning, draw selection, or test-label-dependent choice.

For each of the five official Roman masks, use all four already recorded
dropout outputs. Convert each draw's class logits to softmax probabilities,
average those probabilities, and choose the class with the largest mean.
This is the Monte Carlo average of the categorical predictive distribution
in Gal and Ghahramani (ICML 2016, equations 5–6). It differs from taking a
softmax after averaging raw logits. Report both aggregation rules for all
five masks, together with the deterministic BASE accuracy from the same
checkpoint. Do not select an aggregation rule by its test score.

The implementation uses float64 log-sum-exp and receives no labels.
Accuracy is calculated afterward against the official test labels. The
source hash and five original prediction hashes are fixed before execution.
The underlying selected weights and fixed dropout draws must pass the
separately frozen CUDA replay before manuscript inclusion. The four-draw,
dropout-0.2 comparison remains an untuned local baseline. It cannot establish
a general ranking of MC Dropout or uncertainty quality.
