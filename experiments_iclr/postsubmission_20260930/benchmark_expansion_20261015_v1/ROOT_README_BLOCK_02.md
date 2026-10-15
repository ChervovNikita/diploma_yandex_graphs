## Expanded benchmark and representative training — 15 October 2026

The target is improved prediction quality with a shared backbone. Every compatible
setting retains separate paper-reference single-model reproductions, matched
single models, base GNNM and a genuine independent four-model ensemble. Published
backbone settings and LR/WD remain fixed. The2,480-fit predecessor proposal is
preserved below; the broadened campaign's final count is not yet frozen.

**The24-fit molecular pilot is running on the dedicated one-GPU allocation.**
Three paired MolBACE/GIN trials compare base GNNM, private nonlinear input
corrections, class-aware graph embedding supervision, their combination and
four independently trained native models. Every fit uses complete TRAIN/VALID,
cap1000/minimum100/patience50 and its own selected state. TEST stays closed.
Two fits have completed; no complete paired cohort or method advantage is
available in the latest saved observation. Full curves, failures and costs stay
on the server.

New source includes explicit link epoch/selection integration, versioned
resume finalization, a corrected structured-responsibility control and
route-private BatchNorm running state with shared learned weights. The latter
passed actual native GIN TRAIN-batch semantics with zero optimizer steps;
it has no prediction-quality result yet. Cross-task adapter ports and a finite
campaign launcher are being completed. Prepared code, semantic checks and
scientific results are reported separately. No new novelty claim is made.

The old226-contrast exact-test policy has insufficient p-value resolution with
ten trials and Holm correction. Its audit and prospective alternatives are
saved before the large launch. The full comparison matrix remains in scope.

**The giant campaign remains disabled pending the exact design, measured
resource assignments and the user's GO.** Only the dedicated allocation is
currently authorized. No original-paper dataset reruns or score changes occur.

- [Current methods, pilot, code readiness and open work](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/README.md)
- [Frozen24-fit pilot](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/pilot_20261015_molbace_v1/SPEC.json)
- [Reviewed method catalogue](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/catalogue/CATALOGUE.md)
- [Saved literature conclusions](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/literature/SCOPED_CONCLUSIONS.md)
- [Statistical design audit and alternatives](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/statistics_proposal_v1/AMENDMENT_PROPOSAL.md)

The dated sections below record earlier states and proposals.
