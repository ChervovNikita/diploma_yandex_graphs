# What the private-normalization result identifies

The private-BN study improves selected validation ranking, but it changes more
than which route owns a buffer. This note makes an existing mechanism in
`catalogue/BN_STATE.md` explicit. It adds no new experiment or literature claim.

## The running-state clock also changes

The shared model updates each tracked running statistic four times for one
logical training batch, once per route. Each private bank updates once. With
constant native momentum a, the shared buffer retains (1-a)^4 of its previous
value after the four calls; a private bank retains (1-a). For the illustrative
case a=0.1, the shared buffer replaces 34.39% of its old information per logical
batch, while each private bank replaces 10%. This calculation is conditional
on the stated momentum and one call per route; it is not a new measurement of
every native layer or architecture.

Even within one shared update, the route contributions are unequal. At a=0.1,
the four normalized weights are 21.20%, 23.55%, 26.17% and 29.08%, in call order.
For fixed route means these are also their stationary weights after complete
cycles. When training changes the means, no stationary interpretation applies.
The same linear update law applies to the per-batch variance estimates stored
by BN. Their weighted mean is not the variance of a pooled route population:
the latter also includes variation between route means.

## What is established and what remains unresolved

For ordinary training-mode BN, the instantaneous forward uses current batch
statistics. Moving its tracked buffers into private banks does not itself add
learned parameters or new training supervision. Evaluation and validation-based
checkpoint selection change. The completed three-fit result compares complete
normalization policies and their selected checkpoints. It does not separately
identify route matching, EMA time scale, order weighting or changed selection.
This is consistent with the earlier stated causal limit and does not invalidate
the measured comparison.

The current nine-fit follow-up keeps the private-state policy fixed while adding
geometry, corrections and their combination. It tests whether those ingredients
work better under that policy. Even a favorable result would not isolate why
private normalization helps.

A later same-checkpoint diagnostic can compare each route's own saved bank
with a common equal-weight bank built from those same four TRAIN-derived banks.
That holds weights, checkpoint and bank histories fixed. Average the running
means and running variances separately and disclose that this is a tied EMA
control, not a reconstructed original shared state or pooled-population
variance. Use no validation recalibration and no tuning of mixture weights.
Such a diagnostic would measure dependence on route matching at the selected
weights; it would still not identify the complete training-policy effect.
No such diagnostic is launched or counted in the current 6,460-fit design.

All scientific source inputs and the arithmetic are in CALCULATION.json.
