# Structured native single and ordinary independent4 fit driver

Source preparation only. The existing `joint_pattern_capable_single_control_source_20261005_v1` adapter and root's completed fabricated CPU check are authoritative. This new real-graph driver has been parsed/compiled as source only; no data, outcomes, checkpoints, server or numerical modules were accessed during preparation. Full native-batch time/memory and this driver remain unexecuted. The current development cohorts must finish before root chooses any physical comparison. This directory contains exactly the driver, this README and minimal source bindings; it releases no fit and changes no existing queue.

`fit_member(...)` performs one fresh native N64/Adam fit with the existing conditional endpoint teacher/core. `fit_and_pool(...)` performs either one such fit or four independent fits. Every independent member owns its encoder, recursive scalar scorer, target decoder and four-emission structural auxiliary: the ensemble has16 internal density components and four served target trajectories. Loss is computed separately within each model; members are never optimized under one bank mixture loss. This is a strong quality control with the same information and supervision, with substantially different capacity/work from F4; it is not a pure parameter-tying contrast.

Keep native masking, width64, Adam settings, coefficient1, query reductions, 100 epochs,17 full batches of65,536,1,700 updates per fit,1,114,112 processed positive/negative records per epoch and64,940 dropped positive tail records. The CLI's `joint`/`separate` choice selects the existing auxiliary, with no normalization change. Initial structural-gradient weakness is not predictive evidence and supplies no tuning rule.

Seeds come from the completed ordinary recipe's `design_spec.native_member_seed(base,m)=base+5m`, with its existing base set0..4. Base0 means `{0,5,10,15}`; a single uses member0. No new seed search or trained-state donor exists. Independent members run sequentially, fresh from their own seed. If both control invocations are chosen, they require five fresh fits: the single is not borrowed from the ensemble. Their different stochastic schedules do not certify identical realized masks; actual epoch negative/permutation/tail and batch/RNG hashes are retained.

Every member selects its first strict best complete official VALID Hits@50 among100 epoch states, then performs the existing two complete selected-state replays. Serving uses one native scalar raw logit per single; the ensemble averages its four independently selected raw logits. The direct scalar adapter uses `model.encode` and `query_forward(...,emit_auxiliary=False)` on complete TRAIN, with native131,072-query VALID batches. The bank-oriented `pilot_evaluate.score_valid` cannot dispatch this wrapper and is not called. Bound OGB Hits50 and `mean_native_scores` are reused.

The completed ordinary independent4 recipe additionally searched100 synchronized epoch banks and candidate101 formed from individual winners. This driver uses only individual winners and makes one final pool measurement; it has no ensemble checkpoint search. That selector difference must be disclosed in comparisons and saved ordinary ensemble outcomes cannot be relabeled as matched results.

CLI, from the repository's existing qualified runtime, with the authorized single GPU already supplied through `CUDA_VISIBLE_DEVICES`:

```sh
python -B postsubmission_research_20260930/joint_pattern_single_independent4_fit_driver_20261005_v1/fit_driver.py \
  --control single --auxiliary joint --base-seed 0 \
  --data-authority /absolute/path/to/existing/DATA_AUTHORITY.json \
  --runtime-authority /absolute/path/to/existing/RUNTIME_AUTHORITY.json \
  --authorization-reference 'actual prospective root decision' \
  --output /absolute/path/to/a/new/single-output
```

For the independent ensemble use `--control independent4` and a distinct new output. The two authority files must match the existing native byte pins; this is not a replacement data/runtime contract. The existing77 TorchSparse dependency location is `<repo>/.gnnm_runtime/buddy_extra_v1/site` in `PYTHONPATH`; use the qualified interpreter/runtime, without installation. The driver authenticates the ordinary runtime and applies the existing exact-CB deterministic transition before initialization. No old exact-CB release or supervisor is invoked.

Outputs retain acquisition/data hashes, full epoch stream receipts, two rotating journals, selected model/Adam/RNG states, selected logits, phase/attempt costs, inclusive CPU/wall time, GPU memory peaks and complete failures. Stdout exposes epoch counts only. There are102 complete VALID traversals per fit, or408 for independent4; cached selected vectors supply the final pool without extra graph traversals. Setup, hashing, loading, sampling, teacher lookup, ESP/backward, every fit, selection/replay and output work are paid. The pool still requires all four models when serving new queries; no resource advantage is claimed. A failure stops the invocation and preserves partial outputs. There is no automatic retry, resume, donor loading, TEST stage, slot cap or smaller-batch fallback.
