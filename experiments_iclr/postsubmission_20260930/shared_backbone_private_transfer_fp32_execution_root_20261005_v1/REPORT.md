# Private-transfer implementation: completed FP32 check

One root-authorized GPU check finished on5October at04:24:41UTC, exit0,
157.90s under external supervision. It used the exact one-GPU host and source
v2, made no fit and read no VALID/TEST values. All transient model states were
discarded. No cap signal or retry occurred. Original paper scores are unchanged.

All three architectures passed: shared four-route GNNM, a single model with
its entire nonlinear predictor privately adaptable, and four complete untied
backbones. Each passed two reachable recomputed Adam-history steps and one
additional discarded stale-commit check. The gate checked every declared shared
gradient coordinate against a separately assembled direct-plus-mixed chain
rule, native Adam parameters/moments, recomputed serving, original donor
serving, dropout/state restoration and initial shared/untied parity.

| Architecture | Shared coordinates per checked step | Largest derivative relative L2 error | Peak CUDA allocation |
| --- | ---: | ---: | ---: |
| Shared four |1,409,026|1.0642e-7|1,381,391,360bytes|
| Capable single |948,225|6.5481e-7|426,987,520bytes|
| Untied four |5,636,104|1.0380e-7|1,608,961,024bytes|

The largest coordinate error divided by its frozen allowed limit was0.1773,
below1. All original pass criteria and numeric tolerances stayed fixed. The
v2 diagnostic-only JSON repair preceded execution and changed no training/math
source or tolerance. The earlier inconclusive FP32 finite differences and failed
native sparse higher-order check remain preserved within their original scopes.

RESULT SHA256:
`8601b1138c3f7ab65cb1525d5d8274a8e14e22731cd060801eec437f6169806d`.
Source manifest SHA256:
`db7102df30491be8809ea4295b9ce0d5c48f3608129f09dcfef74b8f7aa2233f`.
Raw result bytes were authenticated from the remote supervisor before the
compact root summary was assembled. Peak supervised child RSS was1,093,881,856
bytes and observed child CUDA usage was2,195,718,144bytes.

This clears only the declared implementation fixture. It establishes neither
better ranking nor methodological novelty. Complete TRAIN-only cost measurements
for the candidate and strong controls are next; numeric training horizons and
paired quality comparisons remain unfrozen and no predictive fit is admitted.
