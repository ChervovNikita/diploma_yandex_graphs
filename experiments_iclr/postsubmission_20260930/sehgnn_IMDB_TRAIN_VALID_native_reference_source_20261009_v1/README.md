# Literal SeHGNN IMDB development reference

This packet contains executable code for the pinned SeHGNN IMDB reference. It is inactive by default. Root owns staging, runtime review, enabled releases, allocation monitoring, numerical qualification and the five native fits.

`model.py` is the complete byte-exact author model from commit `e92bd37d0b803457339555684f139b4c8f3e160d`. `native_helpers.py` retains six exact required helper bodies. The stock dataset constructor, TEST accuracy diagnostic and TEST export paths are absent. The root-published RoleData seam supplies the three allowed inputs and all ten frozen development descriptors.

## Native computation

- Dense original keyword identity, complete typed graph, native CPU four-hop propagation and all 25 movie feature channels.
- All 12 native movie-returning label channels: TRAIN labels alone, original normalized products, diagonal removal without renormalization.
- Original 512 embedding/hidden width, two feature projection layers, four task layers, semantic fusion, internal task residuals, dropout .5, input/attention dropout zero, no raw residual, final nonaffine LayerNorm(5).
- Five simultaneous Bernoulli logits, BCE, Adam .001 with zero decay, native AMP, batch 10,000, maximum 200 zero-based epochs. Selection is strict full VALID BCE decrease with earliest ties; literal stopping is `epoch - best_epoch > 50`.
- Literal author seeds 1–5 in one process. A single native master GradScaler is constructed before this ordered seed loop and persists across seeds. Per seed, the original seed helper and global NumPy shuffle are consumed before clone/label setup, native TRAIN DataLoader construction and CPU model initialization. The DataLoader has its original default `generator=None`.
- Serving uses a prebuilt list over concatenated TRAIN then VALID, native eval FP32 and original threshold diagnostics plus final sigmoid > .5. No serving DataLoader consumes RNG. Serving snapshots/restores the current streams without reseeding.

Only explicit TRAIN/VALID targets are finite. Other movie targets are NaN. A separate zero-filled TRAIN label-source buffer is an unavailable-label propagation interface; it is never used as negative supervised truth. Unassigned movie features remain available for full transductive propagation. No official TEST identity or truth is opened, scored or exported.

## Root execution sequence

1. Review the sealed source and exact existing runtime. Fill a fresh external copy of `QUALIFICATION_RELEASE.disabled.json` with the packet/protocol hashes, exact Python/provider versions and paths, observed native math flags, runtime environment, actual-schema receipt, all ten role identities, three allowed input identities, fresh output and external resource monitor custody. DGL remains null until checked. Then enable that external release.
2. Execute `runner.py --qualify --release /absolute/qualification-release.json --input-root /absolute/development_inputs --output /absolute/fresh-qualification-output` with the bound interpreter and environment.
3. Adopt the final `COHORT_REPORT.json` only after `COMPLETE.json` and seed1 completion certify success. Bind this receipt into a fresh external copy of `REFERENCE_RELEASE.disabled.json`. Use the exact qualified runtime and same source/input/role identities. Then enable that external release.
4. Execute `runner.py --reference --release /absolute/reference-release.json --input-root /absolute/development_inputs --output /absolute/fresh-reference-output`.

Without a mode flag, the entry prints an inactive record and returns before source/data/provider/model access. An enabled release is refused if any required root field is missing. Source templates and seals are immutable; root fills external release files.

The packet reports the fixed actual-schema receipt hash given by root; its author did not read that receipt, dataset or numerical outcome. Source parsing and provenance checks establish no numerical backbone competence. Native reference seeds 1–5 are representative development fits. The separate heldout confirmation plan still calls for ten paired seeds and its registered contrasts.

## Full-input numerical qualifier

The qualifier uses seed1 and exactly one native full TRAIN minibatch, backward and actual Adam update, with finite logits/unscaled gradients and changed parameter objects. It measures the complete 83,659,532-parameter model. No retry or smaller architecture is supplied.

The selected checkpoint owns the model/buffers, Adam, AMP scalar state, safe primitive RNG state, known-role logits and provenance. `torch.load(weights_only=True)` reconstructs a fresh complete CPU model, ordinary placement, Adam and a separate replay scalar. Parameters, buffers, optimizer, scalar and owned streams must restore exactly. Fresh serving records logit/probability drift and prediction changes. Qualification uses practical maximum absolute logit/probability drift of .001, with no bitwise output gate. Replay preserves the live native master scalar and caller post-fit RNG position.

## Costs and partial failures

The root release must allow at least 32 GiB host RSS and 24 GiB device memory and name root custody of external resource monitoring. These are conservative full-model planning allowances; the driver does not substitute an architecture or enforce a new training schedule. Peak RSS is the process high-water mark, including preprocessing and checkpoint copies. Allocation/device peaks are captured for setup and each seed; cohort peaks are their maxima.

`COHORT_REPORT.json`, per-seed `RESULT.json`, `HISTORY.jsonl`, `COST_EVENTS.jsonl` and completion records are preserved. Cost scopes include provider/import cohort overhead, once-only parsing/hashing, dense keyword identity/graph, full feature propagation, per-seed clone and TRAIN-label propagation, original constructor/placement, native updates, complete VALID selection, checkpoint CPU copies/writes and fresh selected reconstruction/serving. Whole-cohort wall/CPU times are inclusive; nested scope times must not be summed as if disjoint. Failed CUDA timing and cleanup are explicit failures; final cleanup records determine completion.

## Source checks and attribution

`verify_source.py` uses stdlib source reads, AST parsing and hashes. `SOURCE_ONLY_CHECKS.json` records the passed check. No numerical library, model, data or server operation is required for it. These checks are separate from runtime qualification.

`SOURCE_BINDINGS.json` pins exact author, schema and state-helper origins. `LICENSE_SCOPE.json` preserves the earlier locator evidence and attribution. No applicable SeHGNN/HGB redistribution grant was located at these pins, and this packet claims no blanket grant. It implements the requested internal technical reference and preserves original sources and licenses.
