# Independent source assessment of the Wiki15 geometry aggregate

10 October 2026. Source and retained resident metadata only. The reviewed source
packet is `Wiki15_selected_member_geometry_descriptive_source_20261010_v1`.
Its source/config/metadata bytes are bound in `SOURCE_BINDINGS.json`; no source
manifest or seal was present at the reviewed snapshot. This assessment does not
authorize numerical execution or interpret scientific outcomes.

## Conclusion

The geometry, active FactorLinear head fold, class-contrast map, native-metric
row-space projection and cached full-margin summaries match the stated
descriptive task. The selected-state limitations are explicit. One concrete
failure-retention issue remains before a numerical launch: the source's final
receipt does not survive hard external termination. The actual CPU/RSS/time
supervision command is not part of this reviewed packet.

## Formula and binding checks

- `aggregate.py:21–54`: the sum-of-unit-vectors identity subtracts self-pairs.
  Same-class and different-class means weight every ordered distinct-node pair
  equally, rather than giving each class equal weight. Every disclosed class
  must have at least two nodes. Each member's residual subtracts its own whole
  development class mean; six same-node member pairs are retained equally, with
  normalization-floor counts. These are selected eval analogues, as PLAN states,
  not the sampled two-view TRAIN losses.
- `aggregate.py:125–145`: checkpoint epoch, selected local/global mode and core
  identities are checked. The active head uses
  `B_m = diag(s_m) W diag(r_m)` and `h_m @ B_m.T + bias`. This matches the pinned
  FactorLinear law, including bias after output scaling. The pinned collector
  captures the head input before those factors and disables TF32.
- `aggregate.py:148–160`: subtracting the class-row mean gives
  `(I - 11^T/10) B_m`. The residual is mapped through this actual saved map.
  FP64 SVD uses fixed `max(shape) * eps64 * smax`; retained right singular
  vectors span its numerical row space. Projected energy is consequently in the
  native hidden Euclidean metric. Zero head rank and zero residual energy have
  defined handling. A class-centered map has mathematical rank at most nine;
  the conventional tolerance handles the remaining floating-point centering
  residue. PLAN correctly avoids a common-nullspace or whole-network claim.
- `aggregate.py:57–63,161–186`: dispersion is component RMS over the six member
  pairs. All nine truth-versus-wrong-class margins use cached logits, including
  bias and class means. No argmax, accuracy, NLL, Brier or calibration calculation,
  model construction, forward, fit, probe or selection occurs.
- INPUTS fixes all fifteen P/A/R/C/S × 6101/6203/6307 endpoints. Its configured
  hash matches; all fifteen raw/checkpoint bindings match the retained PASS
  resident metadata. Headers report FP32 `[4,5274,512]` representations and
  `[4,5274,10]` logits. This establishes retained metadata custody, not numerical
  tensor content or reconstruction success.
- PLAN discloses selected-development reuse, absent TRAIN views/centers and
  trajectories, stage/epoch confounding, native-coordinate dependence and no
  graph-specific or novelty inference. The actual selections are:

  | Seed | P | A | R | C | S |
  | --- | --- | --- | --- | --- | --- |
  | 6101 | 130/global | 68/local | 132/global | 58/local | 58/local |
  | 6203 | 127/global | 60/local | 148/global | 131/global | 1073/global |
  | 6307 | 66/local | 53/local | 133/global | 63/local | 50/local |

## WG1 — retain partial results across hard termination

Locations: `aggregate.py:111–114,168–204`; `PLAN.md:118–122`.

Completed rows are only in RAM. The 90-second check is only before a cell starts;
one load/geometry/SVD cell, including the last cell, can cross that deadline.
The compact output is written only in `finally`. An external timeout or RSS
termination need not execute `finally`, so it can lose every completed row and
the process CPU/RSS receipt. Admission/provider failures before the `try` also
have no aggregate receipt. Ordinary caught cell exceptions do retain prior rows,
failure details and resource usage; those paths are sound.

Before execution, use a minimal durable per-cell progress record and the existing
finite supervisor's attempt/termination/cost receipt, or show that the already
existing supervisor retains the required completed-cell and cost evidence.
Bind the concrete 100-second timeout and CPU/RSS cap invocation. The reviewed
relay's 120/150-second transport timeouts do not establish those numerical job
limits. No new ownership or policy framework is requested.

Thread intent is correctly set before numerical imports: BLAS/OMP/NumExpr two,
Torch two with one inter-op thread, and no CUDA visibility. Source records peak
RSS but does not cap it. PLAN's one 45.6 MB archive plus one 77–92 MB checkpoint
describes input sizes, not a peak-memory limit: four simultaneous full hidden
FP64 arrays in geometry alone occupy about 346 MB, before other temporaries,
provider overhead and allocator retention. No runtime memory/time measurement
was made in this review.

## FP32 reconstruction qualification

The fixed `gamma_(2*512+6)` absolute-product expression is conservative for the
ordinary normalized FP32 multiply/accumulate model, with the original TF32-off
collector. It is a fidelity guard rather than a score tolerance. Its final
`+tiny32` does not establish a universal bound for arbitrary underflow followed
by large downstream scaling, or for overflowing intermediates. The source does
not test those arithmetic assumptions. The metadata-only review cannot establish
whether the actual head/cache values enter such regimes. State this qualification
if describing the bound as conservative; this is not evidence of an observed
head/cache mismatch and does not call for a tolerance search.

## Scope and verification

Only source text, JSON metadata, hashes, NPY headers and retained checkpoint key
names were read. AST parsing/compilation was done without executing the aggregate.
No numerical provider was imported; no checkpoint or scientific array was opened,
no fixtures/tests or numerical work ran, no server was contacted, and no Q/K
outcome was inspected. Original source, scores, recipes, freezes and seals remain
unchanged. A local metadata command using unavailable `python` exited 127; the
same source-only inspection succeeded with `python3`. Its separated resource
cost was not measured.
