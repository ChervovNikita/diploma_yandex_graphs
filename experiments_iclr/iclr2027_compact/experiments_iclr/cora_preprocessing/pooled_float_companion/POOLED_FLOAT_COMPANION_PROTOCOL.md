# Exact pooled float32 test logit companion

This companion was prepared **after** the frozen study had completed its
validation selection, allowed test scoring, independent metric replay, and
exact pooled/member class-decision replay. It is a transport and audit
addition only. It does not define or change a model, feature transform,
training cell, hyperparameter candidate, checkpoint, or test allowlist.

Export every allowed selected/default scored checkpoint uniformly. The
source for each file is the `test_pooled_logits` float32 array in that
checkpoint's full author `predictions.npz`, whose SHA-256 is already recorded
in the original final score audit. Save the array in NumPy `.npy` form
without quantization or temperature scaling. The companion manifest links
each new file to the immutable compact bundle manifest, validation lock,
final score audit, exact decision audit, hard-decision export, source score
JSON, and source float prediction archive.

The public verifier checks all allowed keys and exact float32 tensor hashes,
recomputes pooled argmax classes and compares them to the original hard
classes, and independently recomputes test accuracy and cross-entropy from
the frozen public test labels. Cross-entropy is evaluated with a stable
float64 log-sum-exp of the transported float32 logits and must agree with
the original PyTorch score within `1e-5`.

The compact folder still omits trained checkpoints and full per-member
float logits, so it cannot independently rerun a model forward pass or
recover member probabilities. The original complete author evidence and
strict replay audit retain those files.
