# Independent targeted v2 source recheck

Target: `amazon_ratings_native_warm_study_preparation_20261003_v2`.
Manifest SHA256: `ae1a30748b4b16b3c68c477e85aaa3cbbe033d9018cdaa6075686c949728ef96`.

Both v1 findings are repaired at source level. No new blocker was found in the narrow successor diff. The v1 source and independent review remain preserved, including both original findings. This recheck does not claim actual Torch/CUDA/data/resource/replay passage or authorize numerical launch.

## AW-01: byte identity

The new `byte_identity.tensor_equal` (`byte_identity.py:9–14`) requires actual tensors, identical dtype and shape, then compares detached CPU-contiguous `reshape(-1).view(torch.uint8)` values. Positive and negative floating zero therefore have different compared bytes; numeric equality no longer controls the bitwise guarantee. Flattening before dtype-viewing handles scalar optimizer tensors. Canonical contiguous conversion compares logical tensor bytes rather than storage strides.

Every affected native-token (`native_training.py:84–87`), selected full-logit (`:257`), recursive next-step state (`:223–236`, `:264`) and evaluator saved-ID/VAL-logit (`evaluate_study.py:108–109`) path uses this comparator. NumPy arrays, when applicable, require matching dtype/shape and contiguous bytes; Python float leaves use packed IEEE double bytes. Remaining numeric `torch.equal` calls only assert R/S values equal one and do not emit a bitwise guarantee.

Data-free fixtures cover signed zero, tensor dtype and shape rejection, scalar tensors, unchanged bytes, noncontiguous logical bytes, applicable NumPy arrays and Python float leaves. `prepare_resource.py:24–27` calls the real Torch/NumPy fixtures after admitted runtime setup and before loading the graph. A fixture failure follows existing resource failure handling. This is part of the existing qualification and adds no scientific or launch gate.

The external preparer's local NumPy receipt correctly says `Torch_checks_executed=false` and NumPy2.3.5. It is not root Torch2.7.1/NumPy1.26.4/CUDA evidence. This reviewer read the fixture source and receipt metadata; no numerical fixture was executed independently.

## AW-02: admitted default-Adam scope

Documentation and receipt flags now claim the source-created default Adam `state_dict`, bound constructor defaults and parameter-group options/order. They explicitly decline arbitrary optimizer hooks or complete optimizer `__dict__` custody.

`native_training.adam_constructor_options` (`:64–69`) requires the exact Torch Adam class and records its defaults. Resource and training bindings retain those options; checkpoint state also retains them. Restoration checks the fresh constructor against state and bindings (`:204–213`), and the evaluator checks reconstructed constructor defaults and group ordering before restoring (`evaluate_study.py:98–103`). Recursive next-step replay covers that admitted state scope. The renamed replay flag is consistently produced and required by evaluator closure.

## Preservation and verification

All98 independently hashed descriptors match, including the exact v2 seal, all17 v2 payloads, every source binding, preserved v1 source/review bytes and two external preparer metadata receipts. The v2 inventory is exact. All nine Python sources independently parse and compile. All scientific `DESIGN.json` fields are identical to v1 except the requested checkpoint-scope wording. The complete new byte module and every changed source/document/template/binding diff were inspected; unchanged data preparation, recipes, roles and scientific gates were not reopened as new review work.

`HASH_AND_DIFF_CHECK.json` records verification; `DIFFS.txt` retains the full predecessor-to-successor diff; `INSPECTED_HASHES.json` records custody and review scope; `REVIEW.json` records the two repaired dispositions and limits. The prior review seal remains `006a2722815bf6e85148cdc56469a08e5996428791737ab6606589a29c9a8c17`.

Only project source/metadata reads and stdlib hashing/JSON/AST/compile work were performed. No packet code was imported or run; no Torch/NumPy, dataset, label, checkpoint, predictive outcome, remote code, installation or subagent was used. Required actual official-data/runtime/resource/training/process-exit/evaluation admissions remain unchanged. Scientific competence and later M4/intervention admission remain unresolved.
