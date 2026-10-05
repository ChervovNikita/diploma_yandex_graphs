# One discarded CPU recursive-adjoint qualification

**The exact sparse repair passed forward/first-gradient/HVP checks. Full FP32 mixed verification remains unqualified because both predefined FD checks are unstable.**

One authorized numerical child ran on verified `anogena-2-0`, with singleton UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac`, CUDA visibility empty, two CPU threads, one inter-op thread, seed/factor seed `20261005`, and the same authenticated native TRAIN/features/runtime/47-name partition as the preserved native attempt. The full `[3327, 3703]` features, four members, two 1024-positive/1024-negative query batches and 3644-entry full-node support were retained.

Qualification source SHA256: `6509b990821f2fcc148203e7142adb491b5fc4bdddfd94032f092997fde1bdc2`. Wrapper SHA256: `14bf734b74983e91dec78ad8bba1ef7dcd910dc71aa20a9e75da466d76362293`. All staged bytes remained unchanged.

## Results by gate

| Gate | Result |
| --- | --- |
| Full inner training-mode route-logit parity | Bitwise native-equal |
| Full outer evaluation-mode route-logit parity | Bitwise native-equal |
| All 47 inner first-gradient tensors | Bitwise native-equal; maximum error 0 |
| All 47 outer first-gradient tensors | Bitwise native-equal; maximum error 0 |
| Encoder sparse quadratic HVP | Passed; maximum error 0 |
| Inner positive CN quadratic HVP | Passed; maximum error 0 |
| Inner negative CN | Empty and unexercised; no active-path failure claimed |
| Outer positive CN quadratic HVP | Passed; maximum error 0 |
| Outer negative CN quadratic HVP | Passed; maximum error 0 |
| Inner CN coverage | Passed via active positive CN |
| Full autograd chain decomposition | Passed; mixed pullback nonzero |
| FP32 full mixed directional FD checks | Both failed agreement/stability criteria |
| Committed update/recomputation/actual serving parity | Not reached after failed/inconclusive FD |

The expected HVP used saved original native forward products. These results verify the recursive constant-adjacency adjoint on every active support in this fixture and preserve the complete native forward/first-order behavior. They supply no predictive result or method-novelty evidence.

## Unchanged mixed FD findings

| Shared direction | Recursive analytic projection | FD at `2^-8` | FD at `2^-10` |
| --- | ---: | ---: | ---: |
| `encoder.xemb.1.bias` | -2.8693836199522593e-5 | -6.798746526315291e-6 | 1.1129211179738974e-4 |
| `predictor.xlin.ops.0.bias` | -1.0415191067014717e-4 | -4.942520667405859e-6 | -1.4566065903842684e-4 |

The FD estimates are exactly the prior native attempt's estimates, consistent with bitwise first-gradient preservation. The analytic projections changed after restoring higher-order sparse dependencies. The two finite-difference scales remain unstable, so nonsmooth crossings or finite precision may contribute; these checks do not isolate a remaining derivative defect. Their thresholds were preserved and no FD estimate supplied an update. The overall status is `FAIL_OR_UNQUALIFIED_RECURSIVE_ADJOINT_DERIVATIVE`.

A separately authorized float64 full-network analytic mixed-gradient comparison against dense constant adjacency is the planned way to resolve this verification ambiguity. It was not executed or substituted here, and would have explicitly separate precision/verification scope. Current evidence does not admit the complete committed operator, an Adam recipe or a scientific fit.

## Bound, preservation and evidence

The child took **14.6495 seconds**, external supervision **15.3545 seconds**, and peak resident memory was **3,033,030,656 bytes** (about 2.82 GiB). The 360-second external hard bound did not fire. Exit code 0 means the diagnostic completed and recorded its failed/inconclusive gates.

The process-local native alias was restored; the torch_sparse package was not patched. Original fixture weights/buffers and TRAIN support/features passed unchanged checks. The original native result remained byte-identical at SHA256 `470aaa5ba37d7240581836797b98435662b9db778e7e03935b8b7e46d15db9c7`. No VALID/TEST payloads, metrics, checkpoints, persistent training updates, oracle, fallback, retry or scientific fits occurred.

Evidence is preserved in `RESULT.json`, execution/staging/preflight/job receipts, child logs, transport/supervisor source, and exact qualification/native source copies. `SOURCE_MANIFEST.json` hashes the complete local packet. Remote packet:

`/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/shared_core_private_learning_recursive_adjoint_execution_20261005_v1`
