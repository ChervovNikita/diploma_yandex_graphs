# Complete factorial route analysis after study termination

## Scope and source decision

Prepare one descriptive selected-VALID route analysis of all eight BN by geometry by correction cells across seeds 101, 202 and 303, plus M1 and genuine I4 context. This is 24 factorial banks and six context banks. Retain all three new arms and all 15 reused factorial controls. The original native bodies are shared context inputs, not extra factorial cells or independent repeats. Leave the scalar AUC/BCE interactions, cell summaries and training-cost report to the existing `study.py`.

**Existing saved predictions cannot directly supply the nine new member analyses.** `runtime/run_fit.py` exports IDs, labels and pooled `raw_logits` in `selected_VALID.pt`; it does not export member logits. `analysis/complete_route_diagnostic_v1/analyze.py` consumes previously collected route tensors and is fixed to the original 24 fits/four shared methods. Its input file has the original 12 shared-bank route exports and all 12 native-body exports. The three BN-only route exports already exist in the pinned BN diagnostic reports. Do not recollect these controls.

The reusable route collector is `analysis/selected_molecular.py`. Source inspection shows that it reconstructs each fit's original adapter from its own JOB/runtime binding, loads the selected state and calls `model.run_member`; the composed extension restores both ingredient binding and registered private BN banks. Its source paths therefore cover the new composition structurally, but its execution on those selected states is not yet verified. The collector constructs a model and performs VALID forwards. **This preparation executes none of that work and does not grant admission.** After complete termination, root may separately admit exactly the nine missing collections. No training, TEST or alternate checkpoint is needed.

`INPUT_AUTHORITY.json` pins the implementation and completed-input metadata used here. No tensor or new-fit result was opened during preparation.

## Complete terminal gate before opening scores or tensors

On the dedicated allocation, verify hostname and sole GPU UUID before project reads, following the existing collector guard. Bind a fresh root-reviewed snapshot of the study output and record SHA256 receipts for its `END.json`, `OUTCOMES.json` and terminal `REPORT.json` before analysis. Then require:

1. The SPEC SHA256 equals the pinned value. Derive the exact nine jobs with the unchanged `study.jobs(spec)` and `study.digest(job)`, checking against all nine frozen `job_sha256` values.
2. `END.status == "complete"`; its nine outcomes exactly equal `OUTCOMES.json`; every expected job appears once, has status `complete` and exit code zero. No admitted, running, not-started, missing or failed job may pass this gate.
3. `REPORT.complete_new_fits == 9`; `all_24_cells` is exactly seeds 101/202/303 crossed with `(n,g,c) in {0,1}^3`, with no duplicates, and all rows are complete. The nine new rows and 15 reused rows must have the identities and origins specified by SPEC. Preserve the original terminal report unchanged.
4. Verify every fit RESULT against the report's authority receipt. For reused cells also require the exact SPEC authority receipt. For new cells require the deterministic `study.fit_directory(output,index,job)`, expected job hash/ID/seed/policy, TEST closed and exact expected runtime binding, including the composed extension hash. Apply the existing `study.scalar_result` role/completeness checks; do not recompute its contrasts here.

If the study stops with failure or missing cells, preserve the terminal outcomes and costs through `study.py`; label this complete-route analysis unavailable. Do not report a favorable subset, impute a cell or initiate a retry. A later formally completed study requires a new immutable terminal snapshot.

## Collection and exact input join

If separately admitted, run the unchanged collector once per new job, sequentially, with its exact fit directory and a fresh output `fit_000` through `fit_008` in the frozen job order:

```text
python -B -u analysis/selected_molecular.py --project-root <verified-project-root> --fit <study.fit_directory(study-output,index,job)> --output <fresh-route-root>/fit_00N
```

Use its existing default diagnostic tolerances; do not tune them or retry numerical differences. Charge every construction, attempted/completed forward and failure. Expected successful coverage is nine four-route banks on all 151 original VALID graphs; retain the actual reported batch/forward counts, time and memory. Do not relabel collection as saved-tensor-only work.

Before aggregation require all nine collection RESULTs and route receipts. Match each collection's `fit_receipt`, job hash/ID, selected checkpoint receipt and saved-export receipt to its terminal fit authority. Check its collector source pin and TEST/role/zero-update fields. Retain `complete_with_sanity_warning` and its discrepancy explicitly; do not silently exclude it or recollect until it passes. A failed or missing collection blocks the complete cohort summary.

Join inputs as follows:

| Inputs | Authority and join |
| --- | --- |
| Twelve shared-BN factorial banks | Match SPEC reused job IDs to `analysis/complete_route_diagnostic_v1/INPUTS.json`; reuse each fit, analysis, ROUTE_VALID and selected-export receipt. |
| Three private-BN no-ingredient banks | Use each pinned BN diagnostic RESULT's `fit_receipt`, `raw_tensor_receipt`, `saved_VALID_export` and checkpoint receipt; match SPEC reused authorities. |
| Nine private-BN ingredient banks | Use the new complete-terminal fit authorities and the nine newly receipted collector outputs; map cells from SPEC methods, never filename or score ordering. |
| M1 and I4 for each seed | Use original INPUTS cohorts: body0 is M1; the mean of the four independently selected native-body exports is I4. Preserve their four distinct selectors. No new native collection is needed. |

For every bank verify receipt hashes, job/selected-state identity, VALID role, TEST closed, finite logits and matching IDs/order/labels across all inputs. Require 151 unique IDs, 130 positives and 21 negatives, and four members for every factorial bank and I4. Use the **saved selected-export pooled logits** as the served prediction authority; member metrics use collected route logits. Retain reconstructed-pool versus saved-export discrepancies instead of replacing recorded predictions. Do not read selected checkpoint weights in the saved-tensor aggregation stage.

## Narrow reuse of existing calculations

Generalize only the bank enumeration and input join in the existing complete-route analyzer. Reuse `selected_molecular.binary_metrics` for member and uniform-graph pooled BCE/AUC. Reuse the complete-route analyzer's positive-minus-negative margin and tie-credit arithmetic for pair diagnostics. No new embedding geometry, probability readout search or model inference belongs in this aggregation stage.

For each seed and bank retain:

- Every member AUC/BCE, mean and **minimum member AUC**, mean and maximum member BCE; served pooled AUC/BCE; pooled-minus-mean-member AUC in percentage points.
- Pooled correct/reversed/tied pairs, all-member reversals, and pool reversals despite a correct member or strict majority. These describe alternatives and cancellation, not an attainable oracle gate.
- Strict pair repairs and harms for each member versus the pool: member margin negative to pool positive, and member positive to pool negative. Count ties separately. Also compare served pools for every existing two-cell contrast from `study.contrast_definitions()`; compare every factorial bank with M1 and I4 context.
- For each paired pool comparison, report strict repairs, strict harms, changed half-credit pairs and exact net AUC difference. With 2,730 positive–negative pairs, credit is 1 for a positive margin, 0.5 for zero and 0 for a negative margin. Verify its mean against the official evaluator. Do not treat pairs or routes as independent repetitions.
- Attach selected/completed epochs, stop reason, convergence flag, invocation wall time, optimizer counts and resource fields from the authoritative terminal fit records. For I4 keep all four fit records. The 30 banks draw on 36 unique training fits: 24 factorial fits and 12 native bodies. Charge each fit once; M1 is reused body0. Record collector work separately from historical training and zero-forward aggregation.

Keep `saved_native_VALID_BCE` distinct from `uniform_graph_BCE`: the former is the unchanged native reported reduction already analyzed by `study.py`; the latter is the uniform graph reduction used for comparable member/pool diagnostics. Do not overwrite one with the other.

Produce one per-seed table for all 30 banks, matched pair-repair/harm rows, and mean/sample-SD summaries only when all three seeds are present. Include source/input receipts and reconstruction warnings. Preserve all signs and negative results. The interpretation should ask whether BN changes member competence, useful pooling or cancellation, and whether geometry/correction add value over their matched private-BN controls. A lower common-error count, wider spread, smaller negative interaction or favorable AUC with worse BCE is not sufficient by itself to promote a method.

## Execution boundary

This is the exact source plan permitted by the assignment, rather than a second scalar reporter or a new analysis framework. Future collector admission and saved-tensor aggregation remain separate actions. No remote call, partial-score inspection, checkpoint/tensor access, native execution, TEST access, Git operation or training occurred during preparation.
