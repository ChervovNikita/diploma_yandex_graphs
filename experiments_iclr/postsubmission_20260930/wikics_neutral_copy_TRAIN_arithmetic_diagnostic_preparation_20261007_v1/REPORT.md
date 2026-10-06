# WikiCS neutral-copy arithmetic diagnostic

Disabled source for root review and one bounded TRAIN-only run. Preserves both paid qualifier failures. No tolerance, trainer, model, weights or protocol changes.

On the full11701-node/442907-edge public graph, run four dropout-off native and neutral-family repetitions in both local/global stages, using FP32 and exact original FP32 parameters converted to FP64. Verify all native parameter bytes including undotted beta, identity factors, distinct private bias rows and no native/family storage alias; verify all parameters unchanged afterward. No backward or optimizer step exists in this program.

Record native and family repeatability; paired and other neutral-route output differences; captured stem/head input/output differences; fixed identical-input native Linear fused bias versus separate no-bias+B and literal wrapper arithmetic; per-boundary FP64 references and full FP32-versus-FP64 trajectories. Original qualifier tolerances and violation counts are reported unchanged. Output is diagnostic evidence, never a method qualification pass.

Same900/1200s, GPU24GiB/RSS32GiB/freeGPU28GiB/output1GiB caps. FP64 is mandatory within those bounds; failure is preserved without fallback. Four cases and44 complete dropout-off forwards plus small boundary calculations. All ephemeral models/state discarded and Python/NumPy/Torch CPU/CUDA RNG restored.

Self review is AST/source only, not independent approval. Root must inspect, bind its evidence and use the existing bounded TRAIN adapter/run_fit/helper. No launch or numerical work performed during preparation. Growth operational preparation follows separately.
