# Harness preparation result

Prepared the exact-source synthetic harness against preserved curvature-selector v3 manifest `d53d6963b3c66cb59061c4f5be6a81108adf8e9792a3f3cf93daea56fe35ec70` and frozen-constants SHA256 `633ea814bfc82c094ecb2d98699e77258f0f644d13cf17fa7f8d65691cf637dc`. All nine payloads, eight bound runtime sources, seal and constants matched after preparation. Originals were unchanged.

Five grouped stdlib/source-pin checks passed locally, including the previously defective cache branch and native API identity rejection before acquisition. The final launcher was also run using the discovered project Python. Both it and system Python lack NumPy/PyTorch; exit 2 and `UNEXECUTED_MISSING_DEPENDENCIES` are preserved in separate result files. Both harness Python sources compile with stdlib. Numerical cases executed: zero.

Ten implemented numerical cases cover state/checkpoint/Adam/RNG restoration, independent coupled Adam/AMSGrad trial equations, partial-trial discard, rank/D/cap/finite guards, graph/permuted/random control and returned-state identities in FP32/FP64, and the 512-wide Photo head closure. Temporary fixtures use explicit project-local paths. Requirements, exact commands and detailed scope are in `README.md`; input pins are in `SOURCE_PINS.json`.

GNNM's broader objective is predictive accuracy through ensembles, with storage secondary. This harness supplies implementation checks and no accuracy conclusion. Actual native warm-state correctness, live-factor continuation, real memory/cost and predictive quality remain untested. No numerical library, real data/checkpoint/current outcome, server operation, warm update or predictive training run was used in preparing it.
