# Fixed probability aggregation diagnostic

This is a posthoc exploratory serving diagnostic on already saved route logits
for completed MolBACE seed101 native M1, base, C1 and C4 checkpoints. The same
native VALID set selected those checkpoints. Results are development evidence,
with no independent heldout, significance, novelty or benchmark claim.

Report both fixed rules for every case:

- Original baseline: arithmetic mean raw logits in the stored native dtype.
- Alternative: arithmetic mean sigmoid probabilities with equal fixed weights.

Use the unmodified native OGB evaluator for ROC-AUC. For BCE, retain only
observed binary labels and evaluate log probabilities using FP64 logsigmoid
and logsumexp; no clipping or probability fitting. M1 is a singleton sanity
case where the two rules must agree. All four per-graph ID arrays, label
arrays including NaN support, selected epochs, checkpoint/export receipts and
job hashes must align exactly. The saved route export already passed its
fixed mean-logit sanity check; verify that check and its original tolerance.

No model construction, new forward, optimizer update, selector change, TEST,
calibration fit, best-readout selection or promotion. Keep the mean-logit
baseline and all original pilot scores unchanged. The later complete I4
cohort needs the same fixed control; it is pending here. Do not synthesize
missing route tensors using new forwards.

Fresh output only. Preserve failures, source hashes, source tensor receipts,
both predictions, per-case metrics and actual CPU/wall/RSS costs. No automatic
retry or tolerance adaptation. This diagnostic uses two CPU threads with CUDA
hidden on the guarded one-GPU allocation.
