# Native diagnostic v2: independent attribution assessment

Engineering source/evidence assessment only. No runtime execution, model/data/checkpoint payload reads, source edits or novelty claims.

## What is attributable

The immutable v2 source manifest is `0e75b301e548704aee01cf42a3bfb8f2af5fe944bbe69ea6b50738d47f74da83`; all 14 payload hashes match. Actual receipt, monitor pins and FIRST_ERROR agree. The diagnostic’s runtime identity matches both the original failed native qualification and the v1 observer receipt.

At sequence **7619**, the first synchronized reported CUDA failure is **CALL_THROW_ERROR in aten.addmm.default**, inside native forward #8 nested under #6. The traceback reaches full-node `x = x + self.xlin(x)` and `F.linear`. Bias is `[64]`, dense input `[16,64]`, weight `[64,64]`, all float32 on cuda:0. The dense dimensions are compatible and nonempty.

The pre-call fence returned successfully: sequence 7617 is BEFORE_SYNC and 7618 is BEGIN. Empty selection/indexing and multiplication returned `[0]`/`[0,64]` and END_SYNCHRONIZED at 7607, 7610 and 7613; the weight transpose ended synchronized at 7616. These show an empty-query context followed by a full-node dense transform. They do **not** show that an empty sparse multiplication caused the failure. No sparse-call failure occurs in the copied tail.

The supported attribution is the **first failure reported during that dense overload under this instrumented schedule**. The exact CUDA kernel/backend and initiating defect are unresolved. Successful fences weigh against simple delayed reporting from the displayed preceding operations, but do not exclude prior silent corruption or context/history effects. Shape metadata cannot establish values, finiteness, strides/alignment, workspace validity or grad-enabled state. The full operator stream and earlier call-stage markers were not copied; the last 16 events cannot establish the complete earlier sparse history.

## What the four disabled controls isolate

The controls template is disabled, has no authorization reference and a placeholder manifest pin. The enabled release authorizes only `exact`. Every control requires a distinct fresh bounded child, model seed 610041, fixture seed 91234 and the same pinned deterministic profile. No control result is asserted here.

| Control | Query/common geometry | What it tests | Remaining context |
|---|---|---|---|
| isolated_depth0 | 1 query; common support `[1,16]`, nnz 0 | Complete native depth-zero eval forward for an isolated query with no common neighbors | Full-node xlin, overlap/SPMM and decoder heads remain; no original recursion/history |
| empty_depth0 | 0 queries; common support `[0,16]`, nnz 0 | Same forward with zero output/query rows | xlin still transforms all 16 nodes; its dense input is not empty |
| isolated_spmm | 1 query; native common support `[1,16]`, nnz 0 | Native overlap then unchanged zero-nnz aggregation, without later decoder heads | Native support check, eval encoder and full-node xlin execute first |
| empty_spmm | 0 queries; native common support `[0,16]`, nnz 0 | Zero sparse output rows compared with the one-row SPMM control | Same dense prefix and native overlap; no original recursive state/history |

This is a useful two-by-two comparison of **one versus zero query rows**, with nnz fixed at zero, and **complete depth-zero forward versus the overlap/SPMM branch**. The SPMM cases are not standalone primitive tests. An encoder/xlin/overlap failure must retain its own attribution. Require the CONTROL_NATIVE_COMMON_SUPPORT marker and entry into native.spmm_add before calling a subsequent error an SPMM-boundary failure.

A successful zero-nnz control means that geometry alone was insufficient to fail in that fresh control context. An empty_spmm failure at its synchronized sparse boundary would establish a separate zero-row sparse reproducer; it would not automatically explain the exact case’s nonempty addmm error. If aggregation controls complete while a depth-zero control fails, its first-error trace must identify the extra forward operation. If all four complete, exact recursion/history remains unresolved. None supplies a standalone `[16,64]` addmm/xlin control or a matched intervention on the exact recursive history.

## Preserved failures and qualification boundary

The original native qualification remains **FAIL**, with the same CUDA invalid-configuration error and nested xlin/F.linear traceback site. Its asynchronous-error warning originally limited localization; synchronized v2 now corroborates the dense call site, without identifying a root cause.

The v1 observer remains a separate **ValueError: Out of range float values are not JSON compliant: inf**. Its traceback fails in strict JSON metadata serialization while intercepting the GCN normalization comparison with Python `inf`, before locating a CUDA failure. Its missing counter 1791 reflects increment-before-failed-write; the retained outer error is 1792. V2’s literal source delta adds only a nonfinite Python-float metadata tag and `math` import. Original overload arguments, driver and limits are unchanged; it does not replace an infinity in numerical computation or repair the native kernel error.

V2 exact remains ERROR_OBSERVED_NO_QUALIFICATION: child exit 1, no bound stop, file custody PASS only. Neither bounded process completion nor a future COMPLETED_DIAGNOSTIC_NO_QUALIFICATION control is native qualification PASS. These controls do not verify full arm coverage, training/backward, optimizer/restore/RNG parity or a complete output oracle. No continuation, qualification promotion or repair is authorized by this assessment.

## Seal

ASSESSMENT.json records the evidence, control interpretation and limits. INPUT_HASH_RECEIPT.json authenticates all source/metadata inputs used. MANIFEST.json seals this new assessment folder. Earlier errors, observer receipts and immutable sources were preserved. No packet was imported or executed by the reviewer.
