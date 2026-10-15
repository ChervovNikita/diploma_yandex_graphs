# Complete factorial route analyzer

Successor implementation of the unchanged [published plan](../complete_route_analysis_source_v1/PLAN.md). `analyze.py` is one saved-tensor analyzer: 24 factorial banks, M1 and I4 for each of three seeds, 30 per-seed bank rows, and 36 unique historical training fits. It reuses the pinned original source authority, existing collector outputs and `selected_molecular.binary_metrics`; pair arithmetic follows the existing complete-route analyzer. It does not recompute `study.py` scalar interactions.

The nine route collections are prerequisites. Root must separately admit the existing `analysis/selected_molecular.py` collector after the complete study is terminal. This analyzer never starts it, creates models, opens checkpoint weights or performs native forwards. All 15 completed factorial controls and the 12 original native-body exports are reused. Collection status, warnings, time, memory and work remain distinct from historical training costs and zero-forward aggregation.

## Terminal inputs and execution

Root must review and pin the final study `END.json`, `OUTCOMES.json` and `REPORT.json` before execution. The analyzer checks all nine ordered successful outcomes before reading new scores; it then requires exactly 24 complete factorial cells and all 24 route collections before importing tensor tools. Failed or missing inputs produce `GATE.json` with no partial cohort summary. It reads all nine collection metadata outcomes before stopping for a collection failure and retains their charged work. Reconstruction warnings remain in the complete output and do not trigger retry or favorable exclusion.

After resource admission on the dedicated allocation, use fresh project-contained output paths and the exact three terminal hashes:

`--project-root` is repository R. Relative receipt and CLI paths resolve beneath `R/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1`; absolute project-contained receipts retain their paths.

```text
CUDA_VISIBLE_DEVICES='' python -B -u <expansion>/pilot_20261015_bn_geometry_correction_v1/complete_route_analysis_v2/analyze.py \
  --project-root <verified-project-root> --study-output <terminal-study-output> \
  --end-sha256 <reviewed-END-sha256> --outcomes-sha256 <reviewed-OUTCOMES-sha256> \
  --report-sha256 <reviewed-REPORT-sha256> --new-route-root <nine-collection-root> \
  --output <fresh-analysis-output>
```

New collection directories are `fit_000` through `fit_008` in frozen job order. The original [input authority](../complete_route_analysis_source_v1/INPUT_AUTHORITY.json) is hard-pinned by SHA256; it supplies existing control receipts and all nine job definitions. Any source or authority change stops the analysis.

## Outputs and interpretation

`RESULT.json` contains every member AUC/BCE, mean and worst member metrics, saved pooled AUC/BCE, pooling gain, common failures, cancellation, per-member pool repairs/harms, and pair repairs/harms for the existing two-cell contrasts and each factorial bank versus M1/I4. Ties receive half credit. Only complete three-seed mean/sample-SD summaries are produced. `REPORT.md` is the compact 30-bank table.

Saved selected pooled logits retain serving authority. Reconstructed pooled logits supply a disclosed sanity comparison. Native reported BCE and uniform graph BCE remain separately named. All unique fit outcomes and costs, all route collection outcomes/work, and the original full collection receipt are retained without summing concurrent wall times or memory peaks.

These are selected-VALID diagnostics on one scaffold split. Route/pair counts do not enlarge training-seed n. Common-error coverage and cancellation do not establish an attainable gate. AUC gains with worse BCE retain that limitation.

## Source verification

`test_source.py` uses synthetic metadata and four-element arrays only. It verifies the complete-terminal gate, exact factorial coverage/origin, tie-aware repairs/harms and rejection of two-seed summaries. It imports no tensor/model library and opens no study data. It is not runtime qualification or scientific evidence. Real saved-tensor execution has not been performed in preparation.
