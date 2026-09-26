# NORM-SYNC V2: bounded correction for finite precision

This is a new 12-cell study. It preserves the V1 model, data, seeds, masks,
optimizer, default hyperparameters, 1,000 epochs, validation checkpoint rule,
and the original graph-step norm tolerance. V1 is immutable. Its Cora seed-0
run stopped before a result file because an FP32 parameter write changed the
applied graph-step norm by about 3.9e-8 when the allowed difference was about
1.27e-8. A separate training-only replay localized the failure to epoch 510.
No held-out accuracy or logits were examined to design V2.

The reference update is still one virtual AdamW graph update fed the mean of
the four full member graph gradients on the current V2 trajectory. Four
private AdamW states still propose the candidate graph update. The shared
boundary and output parameters still receive the ordinary mean-loss update.
The reference is **not** the update of a separately trained TIED model.

For a positive reference norm, let `b` be the current FP32 graph parameter
vector, `d` the candidate graph update, and `s0 = ||r||_2 / ||d||_2` the
nominal scale. Form the target as `float32(float64(b) + float64(d) * s0)`.
Measure the actual norm of the FP32 parameter difference, rather than the
ideal norm before rounding. Keep this target if the original gate passes:

`abs(actual_norm - ||r||_2) <= max(1e-8, 1e-5 * ||r||_2)`.

If the direct cast misses the gate, evaluate scales in the fixed interval
`[0.999*s0, 1.001*s0]`. Require its endpoint norms to bracket `||r||_2`.
Run exactly 48 scalar bisection steps. Each evaluation forms a float64
target, casts it once to FP32, and measures the FP32 applied difference.
Choose the evaluated scale with smallest absolute norm error, breaking ties
by smaller absolute relative change from `s0`, then smaller scale. If no
evaluated scale meets the original gate, mark the cell invalid. Do not relax
the norm tolerance or use a fallback direction. The bisection is justified
because the absolute rounded change of each FP32 coordinate is nondecreasing
with a positive scale along fixed `d`, so the L2 norm is nondecreasing.

For zero reference norm, use zero graph update and record the same two zero
cases as V1. A positive reference norm with zero candidate norm remains an
invalid cell. The four graph stacks must remain bitwise equal after each
update. Log per epoch: reference/candidate/applied norms, relative norm
error, nominal/applied scales, signed relative scale correction, correction
iterations (0 or 48), direct-cast absolute norm error, and the validation
accuracy/CE and checkpoint flag. Summarize how often correction is used.

Run the complete fresh matrix: Cora, WikiCS, Actor, official filtered
Chameleon, seeds 0, 1, 2, one fixed learning rate 0.001 and zero decay.
Audit all 12 validation traces and checkpoints before writing the V2
validation-only lock. Test inference requires that lock, the separate
complete 96-cell mechanism lock, and the separate complete 432-cell global
selection lock. Score all 12 V2 cells once, then replay independently.
The complete study audit cutoff is 06:00 UTC on 26 September 2026.
