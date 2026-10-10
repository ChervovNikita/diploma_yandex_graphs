# Fixed state symmetries of projected neighborhood quartiles

10 October 2026. These are known quantile reflection and parameterization
identities, without a new theorem or useful-diversity claim. Hold one route's
native hidden states H, logits Z and factual neighborhood records fixed. Use
the frozen normalized direction and centered-quartile definition; no source
or parameter transformation is applied to the live study.

## Sign reversal and compensated score map

Write each nonempty neighborhood's projected scalars in ascending order as
`a_0,...,a_(d-1)`. Negating the direction negates the projections. The ascending
negated values are `-a_(d-1),...,-a_0`. For the declared interpolation at
`(d-1)q`, reflection gives `Q_q(-a) = -Q_(1-q)(a)`; the mean also negates.
Consequently, for the row descriptor

`t = [Q25-mean, Q50-mean, Q75-mean, log(1+d)]`,

direction reversal gives `t' = [-Q75+mean, -Q50+mean, -Q25+mean, log(1+d)] = t R`, where

```text
R = [ 0  0 -1  0
      0 -1  0  0
     -1  0  0  0
      0  0  0  1 ].
```

Since `R^2 = I`, the simultaneous replacement `u_raw' = -u_raw`, `V' = R V`
preserves the residual exactly: `t' V' = t R R V = t V`. Native and corrected
logits, probabilities and decisions are unchanged in exact arithmetic. The
norm floor does not break sign reversal because it uses the same norm for
both signs. Empty descriptors stay zero; degree-one centered quartiles also
stay zero. A common direction consumed by several routes requires compensating
every consuming map, rather than only one route's V.

Tied scalar values do not break this forward identity. They can break a claimed
correspondence between the implementation's chosen subgradients: stable sorts
retain original record order within a tie for both signs, whereas reflected
rank positions can select different tied records. No smooth derivative identity
or equivalent optimization trajectory is asserted at those ties. Floating point
reduction and evaluation order also preclude an unqualified bitwise claim.

## Positive raw direction scale

With normalization `n(u) = u / max(||u||, epsilon)`, positive rescaling by c
gives `n(cu) = n(u)` when both norms are above the floor. Holding H, Z and V
fixed therefore preserves all descriptors and logits. If a norm crosses or
lies below the floor, this scale invariance need not hold.

On a smooth branch away from the floor and rank ties, a function-only loss has
`grad L(cu) = grad L(u) / c`. Equal functions can therefore have different raw
gradient scales and finite parameter updates. SGD steps, Adam moments and its
epsilon, and relative angular step sizes need not transform equivariantly under
raw rescaling. Forward redundancy is not a guarantee of identical learning or
evidence that all associated parameter motion is harmful.

## What angles and hidden distances do not establish

The compensated sign reversal can change an angle with another direction from
theta to pi-minus-theta while leaving this route's predictions fixed. At V=0,
any directions produce the same zero correction. Descriptor differences can
also lie in the score map's nullspace. Large parameter or descriptor separation
therefore does not certify prediction separation, correct alternatives or a
better served pool.

This is a within-route reparameterization with H held fixed. Two different
routes have different H, Z and V; neither opposite nor equal directions imply
the same projected distributions or predictions across those routes. Hidden
distances likewise ignore how the projection and classifier use their
coordinates. The saved classifier-nullspace and compensated-head examples
already show this interface limitation, without establishing arbitrary
whole-network symmetries of the native factorized GNN.

Useful diversity still requires correct predictions that survive serving, net
repairs versus harms, final accuracy and probability quality against the fixed
capable controls. Angles and distances remain descriptive. No numerical run,
outcome, data, checkpoint, remote operation, TEST or new queue contributed to
this note.

## Reused scope

- [Frozen descriptor contract](../native_neighborhood_distribution_source_contract_20261010_v1/CONTRACT.json), SHA256 `8df938530129183a89bcfbc647da4d7fc1e2abef0a01c1e90aebd387b6e3fe88`.
- [Sealed quartile helper](../native_neighborhood_quantile_native_source_20261010_v1/quartiles.py), SHA256 `d55f3518f4a56d7fbf2781758aea1552d26236f2cd83652d1a28c411f9be6ab5`; stable record ordering and interpolation were already source-reviewed.
- [Saved decision-visible contrast](../shared_backbone_decision_visible_contrast_assessment_20261007_v1/NOTE.md), Sections 1 and 2: classifier-nullspace and compensated-coordinate limits.
- [Saved common-scale gauge assessment](../BE_common_scale_gauge_normalization_independent_assessment_20261007_v1/REPORT.md), selected passages in Sections 1 and 3: function preservation and optimizer dependence are separate.
- [Saved projected-quantile conclusions](../private_neighborhood_quantile_quality_hypothesis_20261005_v1/CONCLUSIONS.json): PNA/FSW ancestry and finite-descriptor limitations.
