# Installation-witness mapping repair

This successor repairs one representation mismatch in the full returned-model state comparison. The saved witness uses a plain `dict`; `model.state_dict()` returns `OrderedDict`. V3 rejected their container types before inspecting equal keys and tensors. V4 applies `dict(...)` to both flat state mappings at that one comparison. The tensor dtype, shape and `torch.equal` predicates, recursive comparator, optimizer semantics, selector, scientific settings and all other witness functions remain unchanged.

The original five live custody failures remain recorded. All five arms passed their other custody checks. Independent reconstruction retains a separate failure in `fixed_first_graph_pair`, `head.R`, at maximum absolute difference `1.1920928955078125e-07`. V4 does not repair or waive that failure. Preprocessing and warm repeatability concerns are also outside this patch.

## Contents and execution scope

- `install_witness.py`: the single comparison-expression correction.
- `repair_regression.py`: a future explicit CPU-only callable using saved evidence. Import and CLI status execute no numerical code.
- `SAVED_STATE_CHECK_PLAN.md`: an unexecuted plan for the two omitted native checks on reconstructed preserved intended states.
- `V3_FAILURE_SUMMARY.json`: compact original engineering observations, retaining the independent failure.
- `SOURCE_BINDINGS.json` and `SAVED_EVIDENCE_BINDINGS.json`: exact predecessor and saved archive descriptors.
- `STATIC_CHECK.json`: syntax/AST verification only.

No new warm runner is copied. No native model, preprocessing, optimizer update, selector or predictive evaluation is executed by preparing this packet. A source seal is not a runtime result or launch authorization.

## Bounded saved regression

The parent may explicitly call `repair_regression.run_saved_regression(output)` inside the authorized one-GPU repository process with `CUDA_VISIBLE_DEVICES=''` and the declared `GNNM_SSH_DESTINATION`. The unchanged predecessor checks that route and the sole authorized GPU UUID; the regression uses CPU comparisons only. Output must be an exclusive new project-phase directory. There is no automatic retry.

Before loading, the callable hashes the two saved archives, original engineering receipts and bound source. It loads only `INSTALL_WITNESSES.pt` and `INDEPENDENT_RECONSTRUCTION_HEADS.pt`. It does not load the warm checkpoint or large captured input tensor. Their descriptors remain explicitly marked as descriptor-only evidence for this regression.

For each arm it constructs an evidence-derived `OrderedDict` from the saved actual frozen prototype and that arm's saved intended head slices. It requires the old predicate to reject its representation and the corrected predicate to admit equal keys/tensors. Missing-key, dtype, shape and one-FP32-ULP negative controls must remain rejected. It separately checks the saved independently reconstructed heads and requires the fixed-pair mismatch to remain a failure.

These comparisons concern reconstructed saved evidence. They are not new observations of the original live returned models, whose objects were not saved. A passing regression establishes only the representation repair's bounded behavior; it does not establish complete native qualification, repeatability, or predictive improvement. The omitted mean-logit and live-factor checks remain unexecuted.

## Preparation status

This packet was prepared with stdlib source inspection, syntax compilation and AST comparison only. No numerical package or source module was imported/executed during preparation; no saved tensor/archive was opened locally and no server was accessed. Every predecessor is preserved.
