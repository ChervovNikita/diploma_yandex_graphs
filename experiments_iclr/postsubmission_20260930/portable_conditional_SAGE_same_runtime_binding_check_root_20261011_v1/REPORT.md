# Actual portable binding check in the existing runtime

The exact sealed V1portable package constructed the full Physics E4 path, executed one ordinary sequential own-CE update (four backwards/one AdamW step), saved and strictly restored its model, Adam state and route RNG, then produced finite float32 SELECT logits with shape4x150x5. The direct child exited0 in14.60seconds and is absent. Project model/route/factor imports came from the portable source tree. Scientific selected states, CAL/TEST targets and predictive accuracy/NLL were not used. The constructor reads its ordinary TRAIN and SELECT containers; no selection metric was computed.

Actual CUDA peak is8,072,842,240bytes and lifetime RSS2,793,586,688bytes. Costs overlap unchanged current fits. There is one disposable manual training update, not a scientifically selected model or new predictive result.

This validates the new package's binding/update/checkpoint path in the existing environment. It does not validate a clean installation, external runtime equivalence, author-reference/export/scorer commands or the complete fresh reproduction. The observed SciPy version is1.14.1, so the package's provisional1.11.4 dependency recipe is being corrected in a separate preserved V2 source. No optional engineering repeat is required here.

[Complete receipt](complete_v1/COMPLETE_BINDING_CHECK.json) · [Owner closure](OWNER_END_V1.json).
