# A constraint of the native IMDB output head

This is a source and algebra observation. It is not an observed cause of the completed model errors, a new method or an accuracy result. No reported score is changed. The sealed typed-context protocol retains the native head.

The pinned native SeHGNN model ends its IMDB task head with `LayerNorm(nclass, elementwise_affine=False)` before the Bernoulli loss and sigmoid. The complete benchmark has five labels. Source: `sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/model.py`, SHA256 `0948239c4c4d06dd258cd106fd2fa7b6aaa2a62688fe70dd662413fa1eae532a`, lines168–170 and final forward.

Let the five pre-normalization scores be u. The final scores z subtract their mean and divide by the square root of their population variance plus epsilon. Consequently their sum is zero and their total squared size is at most five. For one component z_j, the other four sum to minus z_j. Cauchy–Schwarz gives a squared size of at least z_j²/4 for those four. Thus the total squared size is at least5z_j²/4, and |z_j| is at most2.

Each native member's sigmoid output therefore lies between sigmoid(−2) and sigmoid(2), approximately0.1192 and0.8808. Averaging member probabilities retains these bounds. Averaging raw normalized logits retains zero sum, so all five final logits cannot be strictly positive at once. This describes the permitted output space. It does not show how often the restriction matters on the benchmark.

## Why pooling equivalence must not be assumed

For an illustrative example, use five members. Each has one score−2 and four scores0.5, with the negative coordinate in a different position. These vectors have zero mean and unit population variance. Their raw-logit average is zero in every position. Their mean sigmoid output is approximately0.5218 in every position. A strict0.5 threshold therefore predicts no positive labels for the raw mean and every label for the probability mean. A positive LayerNorm epsilon changes the numbers slightly and preserves the distinction. This example is not a measured prediction, and uses five members for its construction.

The prospective typed-context bank uses a fresh common raw-logit rule across its conditions. Historical probability-pool scores cannot be identical-protocol references. The direct source audit has already required fresh references.

## Scientific consequence and possible later test

A reconstruction, contrastive loss or upstream factor change may alter the direction of the output vector while preserving this final constraint. Its useful effect still requires complete task evaluation. No shared-backbone impossibility follows, and the same constraint also applies to native single and independent references.

If a completed same-information study leaves a relevant head limitation unresolved, a separate prospective comparison could remove only the final five-logit normalization in the single, shared bank and ordinary independent bank. Keep data, upstream architecture, optimizer, horizon and selection fixed. This would diagnose an output-head restriction. Removing that normalization is a baseline correction, not methodological novelty by itself. No such fit is admitted by this note, and the frozen current studies remain unchanged.
