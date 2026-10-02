# Modern native teacher amendment v1

Status: **source only, unexecuted, not admitted for a scientific run**. This is a new packet. The immutable `round16_derived_roles_amendment_v2` packet is unchanged.

The teacher pair is pinned PolyFormer-Mono on filtered Squirrel and pinned Polynormer-r on Photo. Each graph has three paired families: a native single model, GNNM with four complete trajectories and private input/output factors, and four independent native models of the same width. Every family has the same four prospectively fixed configurations and seeds17/29/43 paired with split indices0/1/2. There are72 full family cells. Models, attention and optimizer schedules are not shortened for final fits.

Primary prediction and every checkpoint/configuration selector retain **softmax of mean raw member logits**. Mean member probabilities are a fixed secondary from the same selected saved logits. Covariance features still center member probabilities about their arithmetic mean. There is no pooling selection grid.

## Interfaces

- `prototype/modern_teacher_adapter.py`: exact native factories, label-free preprocessing, three family controls, native optimizer groups, fit/stage restore, checkpoint replay and fixed probability views.
- `prototype/modern_teacher_driver.py`: future `teacher-qualify`, `teacher-fit`, `select-config`, and `select-study` commands. Fit accepts exactly train and predictor-validation compact packs. Selection requires all12 family cells; study closure verifies all72 and exact cross-family role pairing.
- `prototype/correction_screen_driver.py`: a new copied version that consumes a selected full teacher checkpoint after study closure. Its source correction, scoring, calibration, final reporting and primary pooling retain v2 semantics. Native preprocessing and all four fresh trajectories are charged in cold serving; every warm request runs all trajectories again.
- `prototype/modern_teacher_control_report.py`: final APS/point reports for the18 selected family/graph/seed cells. Final labels are accepted only after all72 teacher source selections and all six primary GNNM correction SCORE_FREEZEs close. It never fits or selects.
- `prototype/backbone_boundary_adapter.py`: four complete native body forwards and shared boundary weights/private R/S/B. `set_boundary_identity_` preserves a warmed native single function after wrapping; `copy_member_function_` clones one warmed boundary member; `clone_warm_native_boundary` copies the native donor before wrapping, preserving its function and charging donor/clone memory. These helpers serve the separately versioned graph-initialization proposal and do not change baseline initialization.

The future CLIs use explicit admission/request paths and exclusive output paths. Templates in `templates/` contain unfilled fingerprints and `execution_authorized=false`; they are not runnable admissions. Root owns environment qualification, admission, model/data access, execution, scientific conclusions and commit/push.

## Qualification limits

Only stdlib source/JSON processing, exact source slicing, byte hashes and AST parsing were performed here. No model/source module was imported or executed; no data, checkpoint, scientific array, GPU, SSH or training call was made. No numerical parity, gradients, compatibility, utility, memory or timing result is claimed. Qualification is prospective and report-ineligible; full native fits are a separate operation. See `QUALIFICATION.md`, `PROTOCOL_AMENDMENT.md`, `COSTS_AND_FAILURES.md`, and `REPORT.md`.
