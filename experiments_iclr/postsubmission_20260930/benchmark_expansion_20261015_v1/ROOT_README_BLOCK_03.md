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
Three fits have completed in saved observation11; no complete paired cohort
or method advantage is available. Full curves, failures and costs stay
on the server.

New source includes explicit link epoch/selection integration, versioned
resume finalization, a corrected structured-responsibility control and
route-private BatchNorm running state with shared learned weights. The latter
passed actual native GIN TRAIN-batch semantics with zero optimizer steps;
it has no prediction-quality result yet. Portable BE+input and residual-only variants passed real MolBACE and full
CiteSeer CPU update/VALID/checkpoint checks. Mean/global/conditional aggregation
passed real molecular CPU checks under the same own/pool loss. Native link
ports and CUDA/full-runtime qualification remain pending. A finite campaign
launcher is prepared; historical worker dry runs remain blocked without concrete
job bindings and measured profiles. Prepared code, semantic checks and
scientific results are reported separately. No new novelty claim is made.

The old226-contrast exact-test policy has insufficient p-value resolution with
ten trials and Holm correction. Its audit and prospective alternatives are
saved before the large launch. The full comparison matrix remains in scope.

A frozen three-fit route-private BN follow-up waits for the current24 fits to
finish, then reuses their exact base/M1/I4 controls. No BN quality gain has been
measured. A prediction-only analysis watcher waits for all three outcomes.

**The giant campaign remains disabled pending the exact design, measured
resource assignments and the user's GO.** Only the dedicated allocation is
currently authorized. No original-paper dataset reruns or score changes occur.

- [Current methods, pilot, code readiness and open work](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/README.md)
- [Frozen24-fit pilot](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/pilot_20261015_molbace_v1/SPEC.json)
- [Current scientific direction](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/RESEARCH_SYNTHESIS_20261015.md)
- [Reviewed method catalogue](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/catalogue/CATALOGUE.md)
- [Saved literature conclusions](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/literature/SCOPED_CONCLUSIONS.md)
- [Statistical design audit and alternatives](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/statistics_proposal_v1/AMENDMENT_PROPOSAL.md)

The dated sections below record earlier states and proposals.
