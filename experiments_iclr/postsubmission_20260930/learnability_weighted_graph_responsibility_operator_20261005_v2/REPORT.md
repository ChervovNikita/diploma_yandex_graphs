# Torch compatibility repair of the disabled responsibility operator

The preserved V1 CPU attempt with the existing project Torch2.1.2 runtime failed in the outer derivative: creating a constant with a functorch-wrapped tensor's `new_tensor` raises a dispatch-key error. V2 subtracts the Python `math.log(M)` scalar in the pooled log-probability calculation. This is the same mathematical probability-mean loss; it removes the unsupported wrapped-tensor constant creation. Floating precision still belongs to actual qualification.

Only the math import and that constant expression change. The source guard stays false. The exact finite solver, response, coefficients, controls, private-partial/outer-total derivative ownership and post-core recomputation are unchanged. The full method specification remains the sealed V1 REPORT.md; this repair adds no native qualification, scientific run or novelty claim.

The missing-Torch attempt and subsequent functorch failure remain in math qualification V1/V2. The unchanged fixture is rerun under a separate V3 identity; no old result is overwritten.
