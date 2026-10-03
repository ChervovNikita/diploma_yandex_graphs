# Closed full40 development evaluation preparation

This packet prepares one CPU evaluator for the actual fresh full40 mixed-policy run. It has not been executed or numerically qualified. Compilation, hash custody and stdlib module import wiring are the only preparation checks. Original source, status, ledger and scientific design remain unchanged.

## Root admission and execution

After the complete40 training child has closed and been reaped, root creates an external evaluation freeze from `EVALUATION_FREEZE_TEMPLATE.json`, binds the exact final `STUDY.json` hash/length and a fresh output path, and creates a separate true evaluation release from `EVALUATION_RELEASE_TEMPLATE.json`. The release must bind the final evaluator manifest, exact evaluation freeze, exact study, independent source critic receipt and root observations of full40 closure/reaping. The preparation templates are explicitly disabled.

The evaluator admits exactly DBLP/ACM × five fixed seeds × four policies in source order. All40 must be selected, replayed, unreused and binding verified. It then verifies the study/start/terminal/selection/trace/artifact custody and original latest-tie/patience selector. It deserializes no tensor, reads no development labels and imports no Torch runtime before that complete family admission. Failed or incomplete40 studies produce no comparison; root retains their original complete terminal receipts.

Run in the original qualified Linux/Python3.11.14/Torch2.1.2+cu118 CPU environment, one thread, from the deployed phase directory, with an existing root evaluation directory:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python graph_mixed_block_closed_family_evaluation_preparation_20261003_v1/evaluate.py --freeze /absolute/root/EVALUATION_FREEZE.json --admission /absolute/root/EVALUATION_RELEASE.json --output /absolute/root/EVALUATION_run01.json
```

The output must match the freeze's absolute fresh output path. It is written exclusively only after full primary evaluation and a final source/input hash recheck. No original output is replaced. There are no remote calls, training, retuning, checkpoint choices or heldout-label code paths.

## Primary audit and interpretation

For every selected predictor the evaluator reconstructs the original graph/global_BE model, verifies paired initialization and module-semantic shared/private partition, checks exact checkpoint source/graph/split/backward bindings, restores model/optimizer/OneCycle/member/global RNG state, and independently replays saved member logits. It preserves/checks served FP32 selected NLL and fixed-class F1, the recorded selected descriptives, final trace endpoint and optimizer payload cost.

`frozen_primary_selected_raw_contrasts` contains pool/own minus own/own and pool/pool for served FP32 NLL and macro-F1. `frozen_role_assignment_selected_raw_contrasts` contains the separately frozen own/pool contrast. All five vectors, SD/SE, illustrative t95 and leave-one-block-out means remain visible. Root applies the unchanged prospective scientific design SHA `739160acd4ffb0380348f79496e88624db6163f4a3a93902b043d58eaacc18c5`; this evaluator computes no scientific gate or success decision. Calibration, final-epoch descriptions, ambiguity accounting or a newly selected contrast cannot rescue a primary gate failure.

Uncertainty is conditional descriptive stability on two fixed graphs and overlapping development split blocks. The unchanged core's illustrative t intervals assume independent approximately normal deltas; the overlapping blocks do not establish those assumptions. These are not confirmatory tests, independent dataset sampling, a population-generalization result or a paper acceptance claim. Paid wall/memory receipts retain original whole-process and order-dependent accounting; selected optimizer cost is logical tensor payload, not measured peak memory.

## Calibration and arithmetic

The previously root-qualified `heterogeneous_calibration_evaluation_preparation_20261003_v1/core.py` is imported unchanged. It serves the original FP32 member-logit mean, then converts that served result to FP64 for the fixed inverse-temperature80 solver over `[0.05,20]`, with identity first, frozen ties and recorded identity fallback. Raw FP32 NLL remains distinct from the FP64 calibration reference and calibrated scores.

Fit and evaluation both use each source VAL block, which already selected the checkpoint. Calibration results are in-sample development descriptions, not independent calibrated performance or heldout confirmation. Heldout labels stay closed.

The unchanged known geometric-pool identity converts each individual member to FP64 **before** averaging. Its own-minus-pool NLL equals mean KL(pool || members), checked at technical tolerance1e-10. This is a separate algebraic arithmetic reference and no new theorem; it is not the served FP32 predictor or the FP64-after-served-mean calibration reference.

## Optional native15 secondary table

Default `native_secondary` is null. A separately byte-bound root audit may request the complete15 DBLP native table. Its schema must be `complete15_native_development_audit_v1`; it supplies true `root_observed`, `all15_closed`, `training_child_reaped`, `all15_checkpoint_replays_audited`, `originals_preserved`, `heldout_labels_closed`, dataset `HGB-DBLP`, plus exact descriptors `source_manifest`, `freeze`, `release`, `study`, `graph_schema`. The native-v2 manifest and actual native serial CPU release are pinned in provenance. `cases` contains every seed/arm in source order, each with `seed`, `arm` and descriptors `selection_receipt`, `selected_checkpoint`, `selected_validation_logits`, `training_trace` in its exact original case directory.

The table reuses compact saved VAL logits in original split row order; it checks exact checkpoint admission/graph/split/selector custody and original replay attestations. It **does not perform fresh native-model inference**. GAT/Simple-HGN latest ties and SeHGNN earliest ties, native architectures, preprocessing, batch composition and unequal training budgets are retained. It is a secondary development table, not a matched-recipe causal comparison or a practical gate. Any optional native audit/table failure is reported as `not_admitted`, with no native subset table, and cannot block a valid primary40 evaluation.

## Static verification and disclosure

```sh
python graph_mixed_block_closed_family_evaluation_preparation_20261003_v1/verify_source.py
```

`STATIC_CHECK_RECEIPT.json` is excluded from the sealed payload to avoid a manifest self-reference. It confirms source custody, compilation and stdlib import wiring only. Root still requires a separate source critic before releasing scoring. Historical DBLP CP outcome-summary text and live health-only summaries were supplied by parent messages; no underlying study outcome, dataset, label or fitted-state file was opened during this preparation, and those messages changed no scientific design or contrast.
