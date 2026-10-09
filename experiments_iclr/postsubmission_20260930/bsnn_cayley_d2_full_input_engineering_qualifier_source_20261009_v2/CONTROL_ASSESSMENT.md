# Deterministic control for interpreting Cayley d2 competence

**A deterministic bundle/orthogonal d2, width64, L2 control is needed to interpret BSNN competence relative to its geometry.** General-map NSD and the feature MLP remain useful broad competence anchors. They do not distinguish a limitation of the orthogonal/bundle family from a limitation introduced by sampling, KL regularization or the BSNN feature path. This recommendation adds no requirement to the one-update engineering check and no control execution authority.

The chosen authentic BSNN uses a **Cayley probability distribution around Householder-parameterized mean maps**. Its `orth="householder"` setting is the mean-map parameterization. A deterministic `orth="cayley"` NSD configuration would change that parameterization as well as removing stochastic sampling; it is not the closest matched control merely because its option is named Cayley.

## Closest unmodified author control

The original author repository contains `models.disc_models.DiscreteBundleSheafDiffusion`, with `NormConnectionLaplacianBuilder` and deterministic orthogonal maps. A prospective native d2/f32/L2/no-LP-HP configuration with Householder mean maps, the same labels/roles/seeds/scorer/selector and documented starting optimization is the closest authentic deterministic family control. It should remain separately named and costed. General-map `DiscreteGeneralSheafDiffusion` learns unrestricted matrices, which changes the map family.

The native deterministic bundle control is not a pure removal-of-sampling ablation:

- BSNN has a separate `lin_maps` feature path and one distribution learner; its distribution parameters are computed once per full call, then fresh sheaves are sampled at each layer.
- With nonlinear transport, the native deterministic bundle has separate per-layer sheaf and edge-weight learners, uses intermediate prediction features to infer maps and applies layer-dependent map-feature dropout.
- The deterministic model has no concentration/variance head or native KL. Its active parameter count and saved transport/storage costs therefore differ and must be recorded.
- Deterministic inference needs one native call; BSNN serving uses four full calls. Report that actual work difference.

An exact BSNN mean-map-only deterministic control retaining its separate feature path and map sharing would require a new, explicitly declared adaptation: deterministically use the learned mean maps, remove sampled variation and the KL/concentration learning objective, and account for inactive or removed parameters. It must not be called the unchanged author model or silently substituted into the authentic wrapper. Setting a distribution parameter close to a boundary is also an adaptation and is not an exact deterministic limit of the current bounded gamma parameterization.

## Source anchors

All references below are original files pinned through the authentic wrapper's immutable source bindings:

- `models/bayes_disc_models.py`: `BayesBundleSheafDiffusion.__init__`, `forward`, separate `lin_maps`, one learner and per-layer builder calls.
- `models/bayes_laplacian_builders.py`: native frozen SciPy SO sampler, Cayley distribution transform and Householder mean-map composition.
- `models/disc_models.py`: `DiscreteBundleSheafDiffusion`, per-layer deterministic learners and feature-dependent map recomputation.
- `models/laplacian_builders.py` and `models/orthogonal.py`: deterministic connection normalization and orthogonal-map option semantics.

This is a source assessment only. No control was imported, qualified, fitted or scored; no baseline amendment, architecture freeze or comparative opening is authorized by this packet.
