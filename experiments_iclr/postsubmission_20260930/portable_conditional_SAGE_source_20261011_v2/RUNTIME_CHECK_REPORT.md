# Same runtime portable E4 binding result

One root-owned execution completed successfully against the exact V1 portable manifest. It constructed fresh E4 using only portable project dependencies, performed one sequential own-CE update, strictly restored model/Adam/route RNG after deliberate state mutations, and returned finite float32 SELECT logits `[4,150,5]`. It evaluated no predictive metric and opened no CAL/TEST target or current scientific state.

| Recorded item | Value |
|---|---|
| Manual updates | 1 |
| TRAIN backwards and AdamW steps | 4 and 1 |
| Completion time after binding metadata | 13.748793878 seconds |
| Total run time reported by root | 14.60 seconds |
| Peak allocated CUDA memory | 8,072,842,240 bytes, 8.07 decimal GB |
| Process lifetime peak RSS | 2,793,586,688 bytes, 2.79 decimal GB |
| Observed SciPy | 1.14.1 |

The [actual root completion receipt](../portable_conditional_SAGE_same_runtime_binding_check_root_20261011_v1/complete_v1/COMPLETE_BINDING_CHECK.json) is copied without byte changes as [RUNTIME_CHECK_RECEIPT.json](RUNTIME_CHECK_RECEIPT.json). The total 14.60 seconds comes from root's execution report; the JSON records the narrower post-binding-metadata time.

V2 corrects the installation pin from SciPy 1.11.4 to the observed 1.14.1. Its numerical source and every dependency are byte-identical to V1, as recorded in `NUMERICAL_BYTE_EQUIVALENCE.json`. V1 and its seal remain unchanged. This preparation performs no additional runtime check, fit or numerical-parity evaluation.

The result covers only this existing `.venv` binding/update/restoration path. It does not establish a clean installation, external portability, SciPy preparation behavior, full CLI completion, numerical parity or predictive quality. The tested manifest belongs to V1; V2's new documentation manifest was not executed. No complete-workflow verification is claimed.
