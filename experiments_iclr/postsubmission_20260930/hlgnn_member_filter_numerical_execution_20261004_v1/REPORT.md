# HL-GNN member-filter numerical check

## Result

**PASS: outputs and all input/trainable-parameter gradients agree for the planned fixed-operator comparison.** One bounded CPU invocation ran the unchanged `planned_equivalence.py` against the unchanged member-filter source and the complete pinned author HL-GNN layer module. No source repair was needed.

The 16 direct/factored comparisons cover K=0/4, M=1/3, tied/private hop coefficients and both `powers` and `aggregates` storage paths. The M=1 private flag is intentionally inert, so those cases duplicate the effective M=1 configuration. The separate native M=1 check uses K=15, KI alpha=0.5, dropout=0.3, fresh tied native state and replayed dropout. It checks native operation-order reduction, rather than only algebraic reassociation.

| Check | Observed result |
| --- | --- |
| Assertions | 109 passed: 17 output and 92 full-gradient comparisons |
| Direct/factored output maximum absolute error | 4.44e-16 |
| Full-gradient maximum absolute error | 2.44e-15 |
| Native M=1 output/input/coefficient/weight/bias maximum absolute error | 0 |
| Largest elementwise error divided by its comparison tolerance | 2.93e-6 |
| Child / outer elapsed time | 5.40 / 6.66 seconds |
| Child process peak RSS | 655,405,056 bytes |
| Child / transport exit codes | 0 / 0 |
| CUDA initialized | false |

Outputs use rtol=1e-9 / atol=1e-10; gradients use rtol=1e-8 / atol=1e-10. The computation is CPU float64, on six nodes with input width 5 and hidden width 4. The irregular weighted undirected graph includes an isolated node. Its normalized P times the all-ones vector is approximately `[0.8838, 1.1670, 0.8999, 1.0369, 0.9585, 1.0000]`; the largest departure from one is 0.1670. This tests the propagated constant channel for affine bias on an operator that does not preserve constants.

The factorization cases use nonzero dense biases, private bias offsets, nonidentity input/output factors, arbitrary signed hop coefficients and a fixed differentiable dropout realization. Gradients include input X, hop coefficients, shared weight/bias, private input/output factors and private bias offsets. Native M=1 uses the fresh native KI coefficient initialization; arbitrary signed coefficients are tested in the separate factorization cases. P is fixed: graph-operator/edge-weight gradients were not tested.

## Invocation and custody

The authorized `peptide` project repository ran the pinned RAPIDS Python through the existing confined MacLink transport. The command hid CUDA devices, disabled bytecode/user-site writes, set OMP/MKL/OpenBLAS and Torch intra/inter-op threads to one, and bounded the child by a 75-second wall timeout and 45-second CPU ceiling. Only exact source copies and small synthetic tensors were used. The native layer's exact `utils.py` and `negative_sample.py` import closure was supplied; no negative sampler, trainer, dataset or evaluation helper was called.

The original preparation manifest and all its files were verified locally. The source capsule and admitted runtime files matched before and after execution. The result also records actual imported module/binary origins and SHA256 hashes. CPU tensors were checked at every assertion; CUDA remained uninitialized. CUDA-capable extension libraries were imported as part of the existing package registration, without GPU computation.

`COMMAND.txt`, `REMOTE_EXECUTION.py` and `INVOCATION.json` preserve the exact command and source capsule identity. `TRANSPORT_RECEIPT.json` retains the actual transport output; `REMOTE_INVOCATION_RESULT.json` retains the remote invocation, child stdout/stderr, exit code and remote receipt hash. `STDOUT.txt` and `STDERR.txt` are the child's actual output. `RESULT.json` includes all 109 assertion observations and runtime hashes. `SOURCE_INSPECTION.json` binds the exact original/copied sources. `SUMMARY.json` gives compact result/error/custody fields.

No install, model fit, optimizer update, dataset/checkpoint read, heldout read, GPU computation or predictive-family launch occurred. The source capsule is an execution harness and exact source copy, not a repaired encoder version. Existing source packets remain unchanged.

## Limits and integration recommendation

This demonstrates synthetic tied-parameter/context equivalence of the affine fixed-P computation and its ordinary differentiation. It establishes **no speedup, training-memory scaling, novelty, learned specialization or predictive quality**. The short elapsed time measures this check, not a realistic graph workload. Float32/bf16, large graphs, changing graph contexts and an auxiliary objective are outside this numerical result.

Use native M=1 as the anchor, then a **target-only** member model with the native ranking objective, one shared differentiable embedding and a common graph/dropout context. Compare tied versus unconstrained private hop coefficients with the same pair predictor, candidates and budget; private hops supply actual filter variation whereas feature factors alone do not. Keep the propagated constant channel for bias.

For conditional-objective integration, place joint-versus-independent-side training auxiliaries on that same private-hop model, holding its parameterization, candidates, dropout and responsibility construction fixed. Keep counts and responsibilities out of served pair scores, and preserve legitimate gradients through the shared embedding/powers. These are integration recommendations; no auxiliary has been implemented and no predictive experiment is released.
