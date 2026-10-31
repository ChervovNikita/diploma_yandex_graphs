# CPU execution of the complete spectral PPI development experiment

This experiment tests whether training-only spectral directions improve predictions on the two official PPI validation graphs. The selected directions are compared with tied weights, equal-rank random directions and equal-rank raw-gradient directions. The preceding local gradient audit is mixed and is not predictive evidence.

## Prospective execution amendment

Version3 preserves version2 unchanged. All seven visible GPUs are occupied. The supervised aggregate query `protocols/RESOURCES_20260930_v6.json` reports256 accessible logical CPUs, load around10 and ample project disk space. This amendment allows the complete study to run on CPU, using one intra-operation and one inter-operation thread per fit, with at most8 concurrent fits. `CUDA_VISIBLE_DEVICES` must be empty and no CUDA context may exist. This decision precedes every predictive fit and validation outcome.

The architecture, exact warm-start tensors, masks, bases, ranks, arms, seeds, label budgets, dropout/graph-order streams, optimizer,100-epoch continuation, checkpoint rule, diagnostics and development gate are unchanged. Every fit uses all20 training graphs and both validation graphs. Seeds0/1/2 and label fractions10%/100% produce24 full trajectories, including the rank-zero cell and all full-label controls. Test graphs are never loaded or scored. No favorable subset can satisfy the development gate.

The original float32 source model uses4 members,3 width128 SAGE blocks and dropout0.2. Only neighbor maps receive fixed output bases and zero centered private coefficients. All arms start from the same epoch25 training-only checkpoint and fresh AdamW with learning rate0.001, epsilon1e-8 and zero weight decay. Complete graph updates and mean-member observed-entry BCE are preserved. Each fit performs2000 updates and saves all101 validation member-logit epochs. Select maximum micro-F1, then minimum pooled BCE, then earliest epoch; epoch0 remains eligible. Record fixed epoch100 outcomes as well.

The frozen continuation gate requires all24 fits and numerical replays. At10% labels, selected directions must exceed each tied/random/raw control by at least0.005 mean micro-F1, have positive differences in both nonzero-rank seeds, no difference below-0.002, and no greater mean pooled BCE. A primary late selected checkpoint with a continuing meaningful improvement blocks promotion under the unchanged late-budget rule. Report all full-label outcomes even if adverse. A successful screen requires independent new-task confirmation; three seeds on one partition do not establish method superiority.

## Costs and reproducibility

Each trajectory retains checkpoints, masks, source/runtime locks, optimizer states, RNG states, every validation-logit epoch, complete selection traces and an artifact manifest. Numerical replay rechecks every saved metric and selected checkpoint and performs a fresh CPU forward. This is checkpoint replay, not an independent training reproduction. Preserve every failure and repair attempt.

CPU fit and evaluation time, actual parameter/buffer/optimizer bytes and descriptive selected-model latency are recorded. CPU allocator peak is unavailable and recorded as null, not zero. Concurrent CPU timing does not establish comparative efficiency, and this study provides no GPU speed or memory evidence. Prior probe/collection/audit costs remain separately recorded. No activation or propagation saving is claimed for full-width spectral corrections. CPU and GPU outcomes cannot be exchanged to select a favorable result.

Before fitting, require the prospective CPU source seal, a fresh24-update actual-training-graph CPU preflight and a matching CPU runtime freeze. Every command uses the authorized repository interpreter, explicit repository environment and logged supervisor. All sources, outputs and temporary/cache paths stay inside the research phase. The8-fit queue dispatches the complete24-fit matrix, replays each completed fit and applies the unchanged gate only after all required records pass. The coordinator preserves its PID, queue state and failures; no unrelated GPU/process inspection is permitted.

See `CPU_EXECUTION_AMENDMENT_v3.json` for original version2 hashes and the execution change. This document and the CPU runner/preparer/summarizer receive separate hashes and study identities. Version2 GPU locks are retained as provenance and cannot certify CPU version3.
