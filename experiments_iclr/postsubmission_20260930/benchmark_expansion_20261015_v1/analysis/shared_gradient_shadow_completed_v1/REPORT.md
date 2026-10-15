# Actual shared updates lowered every measured member loss

## Result

The predefined harmful-interference gate is **closed** for all three MolBACE/GIN seeds. Each selected private-BN model was restored and followed for one disposable complete TRAIN epoch, using the original batches, own-loss objective and Adam update. All 456 member/batch observations had a negative first-order predicted joint change and a negative measured joint loss change under the same dropout view. The shared part after the private update also lowered loss in every observation.

| Seed | Qualifying harm events | Negative gradient-pair dot products | Mean finite private loss change | Mean finite shared-after-private change | Mean finite joint change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 0/152 | 22/228 | -0.000349 | -0.021864 | -0.022213 |
| 202 | 0/152 | 11/228 | -0.000375 | -0.025248 | -0.025624 |
| 303 | 0/152 | 1/228 | -0.000317 | -0.021222 | -0.021540 |

Losses are per-batch native binary cross-entropy. These summaries weight the 38 batch events equally, including the smaller final batch. They are diagnostic summaries, not whole-population risk estimates. Member/batch events share a trajectory and are not independent experimental replicates.

Some pairs of member gradients point in opposing directions. Nevertheless, the actual joint Adam displacement improved every measured member's same-batch loss. A negative pairwise dot product by itself would therefore have been a misleading reason to add gradient surgery here. Private changes account for only a small part of the observed immediate decrease; their scale alone does not identify an under-optimized parameter block, because block dimensions, gradients and sensitivities differ.

## Decision

Do not launch the proposed gradient-projection training comparison on the strength of this audit. Preserve its proposal and the possibility of studying other states if new evidence warrants it. Prioritize the already specified residual-only contrast: native-initialized additive routes with the same private BN law, compared with the completed BE-plus-residual, private-BN, original GNNM, native M1 and genuine I4 controls.

This result narrows a hypothesis; it does not establish why independent ensembles remain better. It covers selected checkpoints at epochs 92/54/116 and their next full TRAIN epoch, rather than cold-start training or every possible setting. Same-batch improvement does not establish generalization, calibration or eventual validation improvement. No validation or test scores were computed by this audit, and no new trained predictor was retained.

## Execution and provenance

All three cases completed. Total process wall time was **186.40 seconds**. Work was 114 temporary Adam updates, 570 reverse collections and 1,380 native route forwards. Every official TRAIN graph appeared exactly once per seed. The replay restored native parameter values, BN buffer values and identities, optimizer state, global RNG and loader state after each observation. Each disposable branch was returned to its original selected state before release. Peak allocated CUDA memory was at most 460,819,968 bytes; peak reserved memory was 660,602,880 bytes. No retry occurred.

`SUMMARY.json` contains the complete original case outcomes; `ANALYSIS.json` contains the scalar reductions. `EVIDENCE_MANIFEST.json` pins the remote raw member rows, batch-gradient summaries, operation ledgers and execution receipts. The full raw records remain in the authorized repository under `results/shared_gradient_shadow_audit_v1`. The immutable source and gate are in `analysis/shared_gradient_shadow_audit_v1`; the earlier intervention proposal remains unchanged.
