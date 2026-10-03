# Fresh Q02 source review

The numerical work, data scope and CPU-to-CUDA byte-custody design are supported by source inspection. I recommend a newly sealed successor before accepting the current packet as a contract-exact resource profiler. Two measurement/custody defects need correction; this is not evidence of a numerical formula error or a dangerous GPU action. No execution is authorized or performed by this review.

## Findings

**F01 (P2): incomplete inter-recipe cleanup.** `profile_streaming_resource.py:293` leaves the local `factor` referencing the last MemberAffine. `model.to(device)` at line331 transfers that module to CUDA. The delete list at line378 does not remove `factor`, so the following `gc.collect()`, `empty_cache()` and cleanup snapshot (379–382) occur while it remains live. It is released only on function return. The next recipe can inherit cached reserved blocks, contradicting README31. Confine initialization to a helper or end every case CUDA reference before clearing the allocator and recording cleanup. The byte impact has not been measured.

**F02 (P2): stale/missing failure phase.** `Profile.stage` sets `active_stage` at233 and never clears it. `one_recipe` checks memory at278–279 before registering its case at280–283. If the second recipe fails its precondition, the final receipt can name the first recipe’s completed custody stage and contain no failed second case. The exception handler (429–438) supplies a traceback but no current recipe. Register attempted cases/phases before their guards and clear completed stage labels. Stage-entry synchronization/reset/snapshot can also fail before stage registration (228–230).

**F03 (P3): rejected runtime details are absent.** Actual runtime is obtained at144 and rejected by equality at146 before returning to main. Main only retains actual_runtime/torch_local after a successful return (406–408). A mismatch receipt can therefore omit the exact actual runtime and a GPU snapshot despite CUDA already having initialized. Preserve mismatch evidence while continuing to reject data/model work.

## What the source supports

- Complete 24,492-node, 300-feature Amazon graph; full native edge records, K+1 CPU tokens and all-N row powers. Selected-row predictor work is valid for the shown row-local native eval operations.
- Fixed source_defaults then roman_mono, seed17 source-fixed state, M4/B128, all existing R/S including three boundary maps, frozen common state and one CE+R derivative over every private tensor. No optimizer/checkpoint or predictive metric is used.
- Exactly the Q01-pinned streaming engine and original FoRDE primitive: full-X normalization, stopped reference, global complete statistics/bandwidth/R cotangent, one-target reconstruction in forward and backward, and live q-to-private mixed path.
- CPU descriptor construction followed by byte-checked CUDA transfer, with no CUDA sparse recurrence or token recalculation. Device is omitted from the digest while dtype/layout/coalescence/indices/values are retained.
- Real clones, GPU-to-CPU hashes, five batch validations, retained q/logit graphs, streamed recomputation, finite/range diagnostics and source preservation are inside the measured work. Stage peak statistics concern the current process’s PyTorch allocator. Whole-body time includes receipt/runtime/cleanup work; root external elapsed time must cover startup/admission.
- Exact source inventory, declared runtime/settings/A100 UUID, public graph and split0 TRAIN payloads are bound. Raw/VAL/TEST label/checkpoint payloads are never opened by the proposed profiler. Ordinary exceptions stop the family without fallback or retry.

## Correspondence limits

All93 textual descriptors matched: seven Q02 payloads,73 pinned source payloads and13 predecessor metadata files. The Q02 manifest seal matched. Q01’s source SHA exactly matches the streamer bound here. Its existing receipt says all four native CPU B2 cases and eight coefficient cases qualified, with CUDA uninitialized and no unexpected failures. Q01 compared R and every private R-only mixed derivative on tiny fixtures; it does not admit this CE+R real-graph GPU composition. Its tiny native state uses a lin2 bias override absent from Q02.

No real Amazon Q02 execution, GPU equivalence, resource feasibility, trained donor behavior, predictive competence or scientific success is established. The separately specified full-graph B2 oracle is not implemented or executed. Runtime binding covers declared files, not the full linked/transitive dependency closure. Failure self receipts cannot replace externally observed exit/custody, and final payload preservation is reached only in the success path.

`REVIEW.json` preserves source hashes, exact line excerpts, each supported claim, each concern and the requested corrections. `TEXT_CUSTODY_CHECKS.json` preserves the93 standard-library text hash comparisons.
