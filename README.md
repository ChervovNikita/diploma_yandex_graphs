# GNNM: shared-backbone graph ensembles

GNNM learns graph-layer weights shared by several member trajectories. Lightweight private factors give those members different input and output transformations. The original five-dataset benchmark compares prediction quality with single models and independent ensembles. Its reported scores remain unchanged.

## Current paper and evidence

The complete anonymous source draft is [quality draft v3](experiments_iclr/postsubmission_20260930/complete_anonymous_quality_draft_20261011_v3/README.md). It includes the original benchmark, matched Roman Empire analyses and all completed native SAGE and Packed-Ensembles follow-ups. It is uncompiled; pagination has not been verified. This working draft is not asserted to be the final submitted manuscript.

On the encountered WikiCS development graph, endpoint-conditioned shared SAGE reaches81.1971% accuracy versus80.6282% for unchanged sharing and80.8937% for four independently trained models with the same operator. The three-seed paired interval against the independent ensemble includeszero. Independent models retain better NLL and Brier, and receiver-only conditioning explains most of the baseline gain. The parameter-budget packed reference reaches79.7876% with better rawNLL than the endpoint model. These observations support further testing, without establishing unused confirmation or a new primitive.

- [Complete endpoint comparison](experiments_iclr/postsubmission_20260930/native_SAGE_endpoint_value_closed_interpretation_root_20261011_v1/REPORT.md)
- [Complete packed comparison](experiments_iclr/postsubmission_20260930/native_SAGE_PE4_execution_root_20261011_v1/complete_closed_joined_evidence_v1/complete_joined_raw_analysis_v1/COMPLETE_JOINED_RAW_SUMMARY.json)
- [Current research state](experiments_iclr/postsubmission_20260930/RESEARCH_STATE.md) and [research ledger](experiments_iclr/postsubmission_20260930/research_ledger.json)

The original model and training implementations remain in `models.py`, `run_base.py` and `run_common.py`. New research sources, fixed protocols and compact results are under `experiments_iclr/postsubmission_20260930/`. Large datasets, checkpoints and member-logit evidence remain in the authorized server repositories; they are not included in these small Git publications. Exact inventories and push receipts record what is available in Git.

## Next fixed study

The [prospective Physics protocol](experiments_iclr/postsubmission_20260930/native_SAGE_Physics_confirmation_root_20261011_v1/PROSPECTIVE_SCIENTIFIC_PROTOCOL_V1.json) compares five matched shared/single/independent procedures and a source-derived published meanSAGE reference across ten paired masks. TRAIN and checkpoint-selection budgets retain20/30 labels perclass;30 additional labels perclass are reserved for calibration. The remaining nodes form the final scoring population. All selected states and calibration endpoints must close before one complete heldout readout.

The author archive has been verified against a pinned Git blob. Custodial preparation and both full-graph execution checks are complete. The fixed60-bank/90-body scientific study is running on the authorized allocation alongside Penn94. All selected states, prediction archives and60 fixed calibration endpoints must close before one full heldout readout. A scoped history audit found no earlier Physics evaluation, with broader history uncertainty disclosed. Arxiv has affirmative older outcomes and is not classified as untouched. GitHub's fixed provider is unavailable, so a different feature release was not substituted. Penn94 continues under its original fixed programme.

## Research practice

Keep original scores, unsuccessful studies, costs and decisions. Use representative complete tasks, paired controls, declared selection rules and uncertainty. Stored-weight savings do not establish lower computation or peak memory. The current components have prior ancestry; a supported methodological contribution remains a research objective.

Fresh independent Codex manuscript reviews follow substantial scientific progress and a complete anonymous paper revision using the requested review guidance. They are not called after individual implementation steps or small experiments, and no requested verdict is supplied. No acceptance verdict has been established.
