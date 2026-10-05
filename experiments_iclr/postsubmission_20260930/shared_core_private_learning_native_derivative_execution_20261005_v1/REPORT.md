# One discarded full-shape CPU NCN derivative qualification

**Result: `FAIL_OR_UNQUALIFIED_NATIVE_DERIVATIVE`. The diagnostic completed; no scientific fit or committed update was made.**

The single root-authorized attempt ran on `anogena-2-0` after verifying exactly one GPU UUID, `GPU-44039938-fd82-41d2-fefd-de71514e2fac`. The numerical child had `CUDA_VISIBLE_DEVICES=""`, two CPU threads and one inter-op thread. Seed and factor seed were both `20261005`. It used the pinned private interpreter and native overlay plus the repository Python 3.11 libraries.

The exact sealed qualification script (`1c0a2017b772972b4f5bb0cfa267c1b6afdd2b631d1d5d19ed6557194c38af27`) and parameter partition (`b374fbcaa0c70331be1a5731107c675990384b3eb7c39a3ffb93ba9d29823dd2`) were staged unchanged. All staged bytes were unchanged after execution. The preparation packet remains preserved.

## Actual geometry and bounds

- Authenticated float32 features: `[3327, 3703]`; width 256; four members.
- Authenticated TRAIN: 3,918 raw rows, 48 self loops removed, 3,870 unique nonself undirected positives.
- Inner and outer queries: 1,024 positives plus 1,024 native negatives each; disjoint positive query IDs. No endpoint separation was claimed.
- The union-masked support retained the full node population and had 3,644 symmetric entries.
- Child inclusive time: **11.4778 seconds**; external supervisor time: **12.1505 seconds**. External hard bound: 360 seconds; no timeout.
- Peak child resident memory: **3,101,868,032 bytes** (about 2.89 GiB).
- Native runtime: PyTorch `2.1.2+cu118`, NumPy `1.26.4`, PyG `2.7.0`, torch_sparse `0.6.18+pt21cu118`, torch_scatter `2.1.2+pt21cu118`.

Only TRAIN/features payloads were opened. No VALID/TEST payload, predictive metric, checkpoint, epoch, persistent training update or scientific fit was produced. The supervisor launched exactly one numerical child and performed no retry.

## Native sparse identity findings

For `Q(Z)=||KZ||²/2`, the native expected `KᵀKd` was nonzero on every active support. Autograd's dense-input adjoint had no second-order graph on all four active supports:

| Native support | Finding |
| --- | --- |
| Encoder graph sparse product | Failed: missing second-order graph |
| Inner positive CN | Failed: missing second-order graph |
| Inner negative CN | Empty; path unexercised |
| Outer positive CN | Failed: missing second-order graph |
| Outer negative CN | Failed: missing second-order graph |

The inner CN coverage gate passed because the positive CN path was active. The active sparse identity failures decisively disqualify the installed native operator for the required exact mixed derivative. This confirms the source risk identified in the reference dense-input `SPMMSum` backward.

## Full mixed-gradient diagnosis

The functional private SGD map was evaluated at the full native shape. Its autograd chain-rule decomposition passed and its mixed pullback was nonzero. That internal algebraic consistency does not repair missing derivative paths.

Both predefined full-network directional FD gates failed:

| Shared direction | Native autograd projection | FD at `2^-8` | FD at `2^-10` |
| --- | ---: | ---: | ---: |
| `encoder.xemb.1.bias` | -2.8751228685841568e-5 | -6.798746526315291e-6 | 1.1129211179738974e-4 |
| `predictor.xlin.ops.0.bias` | -1.0359129087378349e-4 | -4.942520667405859e-6 | -1.4566065903842684e-4 |

The two step estimates are unstable under the predefined convergence check, so nonsmooth crossings and finite-precision effects can contribute. These FD failures leave the full operator unqualified; they do not independently identify which missing derivative branch caused a discrepancy. The sparse quadratic failure supplies the clear implementation diagnosis. FD supplied no update.

## Consequence and preserved evidence

The source barred the committed operator after derivative failure. The shared SGD update, private recomputation at updated shared weights and actual committed-copy raw-logit parity were **not reached**, so they remain unverified by execution.

Original fixture weights/buffers and TRAIN support/features passed their unchanged checks. All candidate states were discarded. The process exit code was 0 because the diagnostic completed and wrote `RESULT.json`; this does not mean the derivative gates passed.

This is a native implementation limitation, not evidence against the scientific shared-backbone/private-adaptation hypothesis. A separately declared constant-adjacency recursive-adjoint wrapper could be considered as an exact repair, but it was neither substituted nor qualified in this attempt. Its equivalence, native forward/first-order parity, sparse quadratic identity and full mixed FD checks would require a separate authorized gate. The current result admits no final Adam recipe or predictive fit.

Evidence: `RESULT.json`, `EXECUTION_RECEIPT.json`, `STAGING_RECEIPT.json`, `PREFLIGHT.json`, `JOB.json`, child logs, transport receipt, and exact staged source/native source copies. `SOURCE_MANIFEST.json` hashes the complete local execution packet. The remote execution packet is at:

`/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/shared_core_private_learning_native_derivative_execution_20261005_v1`
