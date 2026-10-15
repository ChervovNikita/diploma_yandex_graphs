# Why fixed context assignment can differ from a common target

This note explains the proposed intervention. It is an elementary property of
cross-entropy, not a new theorem or an empirical result.

For one fixed anchor/donor set, cross-entropy has derivative `p-q` with respect
to predicted donor logits. Common uses the target `(qA+qB)/2` in every route;
split gives qA to two routes and qB to two routes. The target sum across four
routes is identical in both methods. The difference is which route receives
which task gradient, rather than an extra average target channel.

If all four routes have identical donor predictions and identical Jacobians
with respect to shared parameters, the summed shared gradient of this
auxiliary is the same at that state. Private-coordinate gradients can still
differ because targets are assigned to specific routes. Once routes differ,
their Jacobians and predictions need not coincide, so shared gradients can
also differ. Native signed BE routes do not generally meet the initial
identical-Jacobian assumption. No unchanged shared trajectory is claimed.

This makes fixed assignment a more discriminating intervention than an
updatewise random assignment with the same expected target. The latter can
mainly add gradient variability; it does not automatically supply persistent
information roles. It also explains why merely increasing latent distances
does not answer the proposed information question.

A zero-start private-residual ensemble starts at the same native function in
all routes and offers a possible future interaction. The ordinary native
constructor is untrained; preserving its initial function is not preserving
an already competent predictor. Distinct private down-projection features can
give different initial up-matrix gradients, but neither this fact nor the
context assignment guarantees better prediction or sustained diversity.

The current finite roster tests context assignment around the base boundary
GNNM. It does not silently add context-by-residual or every-family combinations.
Those would require their own ingredient controls and a later explicit freeze.
