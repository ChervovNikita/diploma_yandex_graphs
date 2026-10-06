# WikiCS qualifier v3 engineering COPY amendment

The actual diagnostic shows that repeated native local FP32 forwards themselves fail the original logit criterion. Native/family differences have the same approximately6e-7 RMS scale; same-parameter full-graph FP64 copy differences are approximately2.08e-14. On this runtime, fixed identical-input boundary fused-bias and separate+B arithmetic was exact in FP32. Every earlier source and paid failure remains preserved.

This is a prospective engineering contract change, with no accuracy/label/score choice. The original FP32 rtol1e-5/atol1e-6 criterion, pass booleans and violation counts remain recorded. False results are never called passed.

V3 additionally requires full-graph dropout-off FP64 copy in both native stages and all four neutral routes at fixed rtol1e-10/atol1e-12, using original copied FP32 parameters converted to FP64. FP32 copies must satisfy both fixed maxabs and RMS envelopes:2x maximum pairwise difference among four native repeats, plus2x machine epsilon times max(1,native output scale). The formula, repeats, constants and scales are frozen here before a new qualifier run; no fallback, grid or rescue.

Exact parameter/storage checks, original gradient/Adam moment/parameter/field tolerances and all training/six complete-cost code remain unchanged. Only the COPY block, its constants and stdlib copy import differ. Same900/1200s and24GiB caps. The full-graph reference is engineering numerical evidence, not exact real arithmetic or a universal noise bound.

Source is sealed disabled for root's independent amendment/delta review. No model/tensor/numerical work, held outcomes or launch performed during preparation. Actual prior TRAIN diagnostic metadata was inspected to justify the explicit contract change.
