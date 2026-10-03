# Count-aware single comparator implementation preparation

Source only, unqualified and unreleased. This packet implements the frozen
`ncnc_cardinality_single_comparator_source_plan_20261003_v1` without changing
the J/F sources or their definitions. It contains no execution entry point,
root release, model state, dataset read, generated diagnostic mask, fitted
checkpoint, numerical receipt or scientific result.

## Implemented paths

- One exact native width64 encoder and single native completion decoder,
  including the unused fixed-pt `ptlin`; one 270→16→16→1 count head. The
  prospective constructor audits 42772 total/38547 active parameters and a
  4625-parameter head. These checks have not been invoked.
- Both full native left/right depth-zero calls execute with auxiliary
  autograd, including empty-support calls and their second full-node xlin.
  The masked graph determines support before encoder-only edge dropout.
- Symmetric 264-entry usable context plus six candidate-count features;
  every feasible count 0..R. No teacher, target label, split flag or synthetic
  removal marker enters any encoder, context, head or sampler interface.
- FP64 centered full-count logaddexp ESP; full-pattern NLL/R with R0 zeros
  in all-query means. The custom backward saves slot logits/offsets/K and
  recomputes one query's full prefix at a time. The numerator's centering
  gradient cancels the normalizer's centering gradient correctly.
- SHA256 counter draws with stable open-midpoint **log** uniforms. D4 samples
  a complete count and a conditional subset; M4 samples independent bits
  from the same bank's **actual marginals**, sharing slot keys and draw IDs.
  Marginal reverse derivatives treat centered u as independent DP inputs.
- Four detached 1.05*B residual routes, common weight 1, live outer features,
  one shared native nonlinear decoder, per-draw native BCE and mean raw
  logits. The native soft 1.05*q final route is changed as the plan requires.
- Typed complete authenticated TRAIN observation teacher outside the model;
  the exact declared TRAIN tensor hash and full dimensions are required for
  project creation. A separately typed fabricated teacher is for future QA.
- D4-only 17-full-batch epoch helpers, native negative/permutation/drop-last
  semantics, all epoch/mask/support digests, complete TRAIN-only VALID
  scoring, strict official Hits@50 and first strict best tie handling.
- M4 requires a hash-bound D4-selected bank after 100 states. Its helper
  cannot train or select a bank; future root orchestration admits the one
  complete VALID/fixed-diagnostic calculation and verifies checkpoint bytes.
- Frozen diagnostic record/negative/support-custody recipe; no record IDs,
  labels or model values have been generated during preparation.

## Source custody

`FROZEN_PLAN/` preserves all original source-plan bytes, including its manifest
and seal. `cardinality_graph.py` is an unchanged copy of the already qualified
portable graph-operations source. `SOURCE_BINDINGS.json` binds preserved
native source, constructor/loader conventions and authority metadata.
`cardinality_dependencies.py` verifies those pins before any future native
load. Private native source is referenced at its existing location and is not
copied or edited here.

`source_check.py` uses only stdlib parsing/compilation and byte/hash checks.
It does not import Torch/NumPy, implementation modules, project data or native
code. Its PASS denotes syntax and static custody checks only.

## Remaining gates and cost

Every runtime gate in `FROZEN_PLAN/QUALIFICATION_PLAN.json` remains pending:
enumeration R0..8, boundary/extreme laws, first-order gradients/gradcheck,
100000-draw small-fixture frequencies, true marginals/M4 factorization,
teacher/support/gradient separation, actual parameters/native RNG custody,
complete VALID semantics and complete-support resource qualification.

The DP costs O(R²) per query. The current portable implementation launches
Torch DP operations query by query, transfers offsets/key coordinates to the
host, synchronizes scalar checks, and recomputes full auxiliary prefixes.
It also retains the complete scorer/context autograd graph. These operations
and four native decodes are paid work; speed and peak memory are unmeasured.
No smaller supports, count caps, lower precision, changed gradient route,
smaller training/evaluation batch or extra fit is an admitted fallback.

Future engineering must use a separate root release and disposable states.
Only after fabricated and complete-support gates pass may root issue a new
release for the single frozen seed0/100-epoch/lambda1 C64-D4 fit. C64-M4 is a
fixed-bank prediction contrast with finite four-draw noise and no extra fit.
Parameter proximity and equal epochs do not establish equal resource cost.

No global novelty, latent-link calibration, graph-mode benefit, count-model
superiority or same-count expressive-completeness claim follows from this
source packet. Existing J/F and running families are untouched.
