# Fresh PENCIL Collab scientific baseline

This is runnable source preparation for three independent fits, seeds **0, 1, 2**, with a fixed native 20-epoch feature-enabled PENCIL recipe. It contains no predictive result or execution client. Release and dependency templates are disabled. The next step is root technical source review and release of these scientific fits using the already adopted resource result.

## Fixed experiment

Use author commit `2d32e29dbed533288d9d758138e07547a0a7d8a9`, the bound official Collab YAML, scratch BERT hidden 512 / 8 layers / 8 heads / intermediate 2048, and early feature fusion. Each seed starts with a fresh model and AdamW optimizer. Keep native sampling, seed order, batch 1024, accumulation 8, 12 workers, bf16, TF32 and all 20 epochs. There is no grid, resource-state continuation, retry, resume or state donation to another fit.

Every epoch evaluates all 60,084 official VALID positives and 100,000 official VALID negatives in their official order. Select **VALID Hits@50** with strict improvement; the first epoch wins a tie. The unchanged native evaluator computes Hits 20/50/100 internally. Only Hits@50 is published or used for selection. Metrics are computed on native logits before export.

Seed labels align with the GNNM development comparison. They do not imply shared random draws, matched neighborhoods, equal compute or identical training supervision across model families. This is an **adapted native PENCIL scientific baseline**, not an exact reproduction of the author environment or published results. No quality or architectural advantage is claimed by this preparation.

## Retained adaptations

The full list is in `PLAN.json`. It includes the authenticated selective TRAIN/raw/VALID adapter, pinned local non-weight BertConfig, one directly supervised NCCL/DDP rank on GPU0, `pin_memory=False` applied to both native-created loaders before their iterators, and the admitted contemporary runtime. Native files and scientific helper implementations remain byte-identical. Native dataset constructors and the real feature-dimension query retain their original seed/RNG ordering.

The harness replaces broad native checkpoint publication with atomic selected-only publication, adds RNG custody and records complete coverage and resource checks. Resource caps are 4 hours, 2 GiB output, 64 GiB sampled owned-session RSS, 70 GiB CUDA allocated and 75 GiB reserved **per fit**. The inherited supervisor holds its child identity until session closure and commits physical evidence before collecting scientific receipts. A shared owned active-fit lock serializes the fits on GPU0; unresolved physical cleanup leaves it in place. GPU1 and its queue are outside this preparation.

## Selected artifacts remain on the server

Each `seedN/run01` retains `SELECTED_FULL_STATE.pt` containing the full DDP model, AdamW state, selected epoch/counters/configuration, and Python, NumPy, Torch CPU/CUDA and loader-generator RNG states. It is recorded for the selected scientific state; this experiment never loads it. The full state remains server-side.

The same selected epoch retains `VALID_POSITIVE_SCORES.npy` and `VALID_NEGATIVE_SCORES.npy`. Native bf16 or float32 logits are converted losslessly to float32 after native metric computation, with an exact round-trip value check and observed native dtype in `SELECTION.json`. This avoids unsupported direct bf16-to-NumPy conversion. The exported dtype is float32; native dtype is explicitly preserved as metadata. `SELECTION.json`, `EPOCH_HISTORY.json`, `FIT.json` and final custody hashes bind the state and both arrays. The supervisor hashes artifacts without deserializing checkpoint or array payloads. Failed or incomplete fits remain ineligible for adoption.

## TEST history

This experiment does not open TEST data, construct a TEST dataset/graph, or score TEST. The official Collab TEST split was already consumed by the frozen NCNC all-25-cell evaluation. The saved root adoption summary is cited by `PLAN.json` and hash-bound in `INPUT_BINDINGS.json`; that summary says successors cannot claim those TEST values were unseen during design. This preparation does not restore a fresh heldout claim or release a subsequent TEST evaluation.

## Runtime estimate and launch specification

The adopted v3 resource observation completed 793 TRAIN batches / 811,404 queries / 100 updates, plus 157 VALID batches / 160,084 queries. TRAIN took 337.68 s and full VALID 39.76 s. Twenty repetitions give 7,548.78 s, about 125.8 minutes per fit, before extra publication work. A practical estimate is **2.2–3 hours per fit**, or **6.5–9 hours for three sequential fits**. Later epochs, seeds and selected-state serialization have not been measured; the 4-hour cap is a ceiling, not an estimate. Exact evidence is copied under `evidence/`.

`PROPOSED_COMMAND.json` specifies the exact environment and three supervisor argv arrays in seed order. The root release must bind this source manifest, plan, exact independent technical review, existing runtime/dependency inventory and the already completed root resource adoption. No new resource-only qualification is requested. Commands are proposals until root release. No server action, native import, fitting, metric computation or graph/score/checkpoint payload read was performed during preparation.

`static_check.py` runs local stdlib AST/hash checks only. `AUTHOR_SOURCE_CHECK.json` records the preparation check. `V3_TO_SCIENTIFIC.diff` and `PROVENANCE.json` expose the harness changes. `MANIFEST.json` and `SEAL.json` bind the prepared closure. Source checks do not establish scientific execution success or numerical equivalence.
