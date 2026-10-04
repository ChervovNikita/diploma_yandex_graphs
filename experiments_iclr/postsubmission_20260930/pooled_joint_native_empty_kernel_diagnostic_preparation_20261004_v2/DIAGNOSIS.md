# Evidence and diagnostic question

## V1 diagnostic observer failure and v2 scope

The actual v1 `exact` diagnostic stopped with a JSON observer ValueError on Python `inf`, before the GCN equality comparison executed and before a native CUDA failing operation was located. Worker wall time was 13.845152698457241 seconds; supervisor wall time 15.014877691864967 seconds; child exit1; no process/resource bound failure; file custody passed. Last16events show synchronized `aten.pow_.Scalar` completion at sequence1790, then an outer `CALL_THROW_ERROR` at1792. Sequence1791's attempted metadata row was not serializable. This is an observer failure, not evidence that the power/equality operation, nonfinite tensor value, native linear map or sparse kernel failed.

V2 only makes nonfinite Python metadata floats strict-JSON-safe with explicit type/string tags. It changes no original operation argument, numerical result, error handling, fixture, native source, seed, or profile. The original qualifier evidence below remains the unresolved reason for diagnostics. The actual v1 observer failure and source remain separate preserved artifacts, and the exact-first v2 run would be a new attempt in a fresh process/output, not a continuation or a repaired PASS.

## Original native qualifier evidence, still unresolved

The preserved actual native qualification failed during `direct_native_prediction`; the earlier exact native coordinate check passed. Worker wall time was 28.61870628595352 seconds, supervisor wall time 35.547242160886526 seconds, child exit 1, and no resource bound was breached. There were no data files, fits, VALID evaluations, TEST reads, old checkpoints, or retained engineering state files.

The traceback surfaced `CUDA error: invalid configuration argument` at the right recursive depth-zero call's full-node `xlin` linear operation. CUDA reporting was asynchronous, so this is not proof that linear was the failing kernel: a preceding left recursive operation could have failed earlier. Runtime was the admitted Torch 2.7.1/CUDA 12.6, torch-sparse 0.6.18+pt27cu126, torch-scatter 2.1.2+pt27cu126, deterministic algorithms enabled, TF32/autocast disabled, and CUBLAS workspace `:4096:8`, on the pinned UUID. Source is GraphPKU/NeuralCommonNeighbor commit `11d597013750da17ce7468e344bec756a7af39a4`.

## Source boundary worth testing

The exact sealed fixture has 16 nodes and 128-wide features, with the unchanged positive query list `[[0,1],[1,2],[1,3],[7,8]]`. `direct_native` loops over the full list and `queries[-1:]` in both eval and train modes. The last pair `(7,8)` is isolated. Its depth-one residual groups both have zero query columns. Native recursion nevertheless calls depth zero on each group and retains full-node `xlin`, native overlap, `spmm_add`, and empty output decoding.

A native sparse common tensor of shape `(0,16)` is therefore one possible failure boundary. A tensor with positive row count but zero common nonzeros is a different boundary. Neither is established by the original traceback. Source observation also shows that the actual full-batch/global residual groups are nonempty; a zero-size upstream-reference limitation would require its own disposition and would not establish that the full-batch architecture fails.

## Cases, each independently released in a fresh process

| Case | Preserved native work | Question |
| --- | --- | --- |
| `exact` | Calls the unchanged sealed `direct_native`, including full/isolated, train/eval, recursive empty calls, RNG/parity audits, and C native references | Which synchronized original operation first fails in the actual qualification schedule? |
| `isolated_depth0` | Full native depth-zero forward on `(7,8)`, including full-node transform and empty-common sparse reduction | Does one query with no common support fail? |
| `empty_depth0` | Full native depth-zero forward with query shape `(2,0)`, including full-node transform and empty decode | Does the intact zero-query native forward fail? |
| `isolated_spmm` | Native encoder/outer transform; native overlap creates common shape `(1,16)`, nnz 0; original native `spmm_add` | Is zero common nnz with one row supported? |
| `empty_spmm` | Same original operations; native overlap creates common shape `(0,16)`, nnz 0; original native `spmm_add` | Is a zero-row common sparse reduction supported? |

Every case retains the original previously passed native-coordinate fixture gate before its target path. Controls are eval only and are diagnostic reductions; they do not substitute for qualification or claim equality to the complete train/eval schedule. The first case that fails stops its owned child. No automatic control continuation is present.

## Attribution and limitations

Before each observed overload and native boundary, a successful device fence separates prior errors; after its unchanged call another fence prevents inheritance by a later operation. `BEFORE_SYNC_ERROR_PRIOR_OPERATION` points to prior pending work. `CALL_THROW_ERROR` and `AFTER_SYNC_ERROR` attach the first failing observed call to its native stage and arguments' metadata. If an extension does not expose its internal CUDA kernel through dispatch, the synchronized original extension operation is the narrowest available attribution. The trace does not promise a CUDA kernel symbol.

`TRACE_BOUND_STOP_NO_KERNEL_ATTRIBUTION` and `RESOURCE_BOUND_STOP_NO_KERNEL_ATTRIBUTION` are diagnostic bounds. Log/fsync or observer failures may leave no `FIRST_ERROR.json`; consult worker traceback and receipt. Bound/process completion never gives native PASS. Device peak caps are checked after successful fences, so an individual allocation can precede its cap check. The supervisor applies wall/RSS/output limits and terminates only its owned child.

The worker does no GPU/RNG probe after an observed failure and performs only file/module custody checks while propagating it. It retains no donor state. Instrumentation is fully billed but is not an efficiency measurement. Empty coverage is not dropped, repaired, or analytically replaced. If native empty input is unsupported, an explicit analytical/portable oracle and separately described reference limitation would need a later sealed successor and root decision; no native parity would be claimed for that unsupported call.
