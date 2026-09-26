# Post hoc validation-temperature sensitivity (72 selected cells)

This analysis is specified after the original training, validation choice,
and held-out test scoring. It responds to probability-quality concerns; it
is **not** an untouched test confirmation. The same validation nodes used to
select each checkpoint and hyperparameter are reused to fit one scalar
temperature per selected cell. That reuse is disclosed and can favor the
validation objective. No model weights, candidate, or class decision change.

## Coverage and data boundary

Apply the identical procedure to all 72 validation-selected cells: four
graphs, six arms, three optimizer seeds. The original 432-cell selection lock
determines these cells before calibration. A separate validation-only
transport stage contains each original float32 `valid_pooled_logits` array
and each graph's frozen validation indices and labels. The transport
exporter verifies source prediction hashes against the independent final
score audit and validation-index/label hashes against the frozen graph
descriptor. It writes no test logits or test labels. The temperature fit
command reads this validation-only stage and writes a complete lock for all
72 beta values before the test-score command can run.

## Fixed fitting rule

Let `z_i` be the raw pooled validation-logit vector and `y_i` the label.
For each cell, fit positive inverse temperature `beta=1/T` on the closed
interval `[0.01, 100]` by minimizing mean validation negative log likelihood
of `softmax(beta*z_i)`. Do all operations and reductions in float64. The
objective is convex in beta. Its derivative is the mean of
`sum_c softmax(beta*z_i)_c*z_i,c - z_i,y_i`.

- If the derivative at 0.01 is nonnegative, choose beta 0.01.
- Else if the derivative at 100 is nonpositive, choose beta 100.
- Otherwise perform exactly 100 bisection updates on the derivative: a
  negative derivative replaces the lower endpoint, and a nonnegative
  derivative replaces the upper endpoint. Choose their final midpoint.

Record beta, temperature, endpoint derivatives, boundary/interior case,
and pre/post validation NLL for every cell. Re-run a verifier on the full
72-cell validation lock before any scaled test metric is computed.

## Held-out diagnostic

Only after the complete validation lock is saved and verified, apply each
fixed beta to the corresponding original pooled test logits. Compute
accuracy, NLL, multiclass Brier score, and fixed ten-bin equal-width ECE,
all before and after scaling, on every cell. Accuracy and argmax class must
be identical before and after positive scaling. Multiclass Brier is mean
`sum_c(p_c-onehot(y)_c)^2`; ECE uses confidence bins `[0,0.1), ...,
[0.9,1]` and the sample-weighted absolute confidence/accuracy gap.
Summarize each graph/arm by mean and sample standard deviation over its
three paired optimizer seeds and retain all 72 per-cell values. Interpret
ECE as bin-dependent and avoid significance claims from three seeds on one
fixed split per graph.

Reference: Guo, Pleiss, Sun, and Weinberger, *On Calibration of Modern Neural
Networks*, ICML 2017. This is scalar temperature scaling with a bounded,
fully specified one-dimensional solver.
