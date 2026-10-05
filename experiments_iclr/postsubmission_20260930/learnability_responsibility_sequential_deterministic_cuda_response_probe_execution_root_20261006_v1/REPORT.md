# Strict deterministic response observation

The fixed full-native Amazon probe completed two LIVE responses from the same initial shared parameters and original private parameters. Their complete response packages have zero numeric differences across 269 tensor leaves and 146,102 coordinates; no structure or scalar-metadata mismatch was reported. This is an observed numerical comparison, not a byte serialization comparison or proof of determinism at every state.

The process set CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing Torch and strict deterministic algorithms before CUDA initialization. Native model, response mathematics, FP32 context, graph, labels and original tolerances were unchanged. Backend flags, reversible CPU threads, environment, caller states, RNG and source gates were restored. No host settings changed.

## Cost and evidence

24 native forwards, 16 private gradient constructions, 20 Q-map primals; zero native/map/query VJPs. Worker elapsed 40.238283 seconds, peak CUDA allocation 15,407,968,768 bytes and reservation 17,131,634,688 bytes. External wall time was 41.171870 seconds. Result SHA256: e6fe1a133b13ef9f7197578503c842c4b71205560fbe05b25e8da2c8692b0245. Terminal descriptors authenticate both remote operation/comparison JSONL streams; they remain on the authorized server.

## Interpretation and next action

This changed execution mode gives repeatable responses at the tested initial state. The earlier nondeterministic diagnostic was at returned theta+, so the evidence does not isolate one kernel or prove that all prior variation has the same cause. The original all-six failure remains preserved and failed.

A separately reviewed continuation will run the complete fixed all-six episode and original-phi recommit qualification in this strict mode. The measured two-response comparison does not qualify its shared derivatives, changed states, controls or future fitting. No corrective fit, predictive gain, novelty clearance or manuscript acceptance follows. The contingent small VJP control is not needed before this bounded continuation and is not queued.
