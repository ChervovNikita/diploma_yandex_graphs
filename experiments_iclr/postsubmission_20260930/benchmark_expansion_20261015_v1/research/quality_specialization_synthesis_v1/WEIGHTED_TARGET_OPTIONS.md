# Two future weighting options: conditional-target algebra

15 October 2026. Requested mathematical follow-up only. No new primary reading, implementation, experiment or change to the running pilot.

## Correcting class-dependent weights

Fix x, let `eta=P(Y=1|x)`, and let both positive counterfactual weights `w1(x),w0(x)` be known. For the private loss

```text
a(x) = log[w1(x)/w0(x)]
loss = w_y(x) * BCEWithLogits(z(x)+a(x), y),
```

the derivative of its conditional expected risk is

```text
[w1 eta + w0(1-eta)] * sigmoid(z+a) - w1 eta.
```

For `0<eta<1`, its unique unrestricted optimum satisfies
`sigmoid(z+a)=w1 eta/[w1 eta+w0(1-eta)]`, hence **`z*=logit eta`**. Boundary probabilities are limiting optima. The positive second derivative confirms conditional convexity in z. Thus the offset sign above is correct. Serve the **unshifted** z; apply the offset only inside the weighted private loss. Shared F would still use the unshifted native loss.

The current pilot's observed-label TRAIN table supplies only `w_y(x)`, not necessarily the other class's weight. A future definition must freeze **both class-conditioned structural functions from TRAIN** before optimization. Applying the other class's empirical quartile rule at a structure value absent from that class needs an explicit extension/tie/boundary convention; the current finite membership table does not silently provide one. No VALID/TEST labels are needed or permitted for constructing these functions. With weights in [.5,2.5], offsets are bounded by ±log5. The multiclass analogue is weighted CE on `z_c+log w_c(x)`, requiring every counterfactual class weight; its unrestricted unshifted softmax optimum is likewise the original conditional distribution.

## Simpler option: label-independent structural weights

If each route instead uses positive `w_m(x)` from **pooled TRAIN structural quartiles without labels**, then

```text
E[ w_m(x) BCE(z,Y) | x ] = w_m(x) E[ BCE(z,Y) | x ].
```

The optimum already is `sigmoid(z)=eta`, without an offset. The same applies to multiclass CE. With the existing fractional quartile law `.5+2q`, per-object mean weight one and each route's total TRAIN mass remain exact; **per-class mass generally does not**.

| Future option | Retained advantage | Cost or limitation |
| --- | --- | --- |
| Class-conditioned weights plus offset | Exact observed TRAIN class-mass balance and corrected ideal conditional target | Needs all counterfactual class-weight functions and a frozen extension beyond observed class-specific values. |
| Pooled label-independent weights | Same ideal conditional target directly; simpler fixed construction | Class totals and emphasized class composition differ by route; finite-capacity models can respond to that shift. |

For preserving the original conditional target, pooled label-independent weights are the simpler future hypothesis. That is a simplicity judgment, not an empirical preference. Both reweight the importance of x regions; finite shared/private capacity, regularization, optimization, dropout, BN trajectories and selection can still change ranking and calibration. Shared-F/private-W updates do not gain a global scalar-objective or finite-quality guarantee from these conditional calculations.

Importance weighting and logit adjustment are established ancestry. The retained SADE scope supplies class-offset examples, but its constant prior offsets are not this exact feature-dependent construction. These elementary derivations establish no novelty or graph benefit. Both options remain outside the frozen atom-quartile study; its completed outcome must retain its original interpretation.
