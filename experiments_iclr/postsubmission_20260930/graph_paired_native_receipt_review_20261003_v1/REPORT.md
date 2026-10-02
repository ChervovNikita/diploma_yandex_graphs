# Independent native receipt audit

Date: 2026-10-03  
Verdict: **No receipt blocker found. All 63 independent receipt checks passed.**

The supplied receipts support successful native construction, active-precision AD qualification, and installed-output correspondence for this exact frozen Squirrel17 warm state. They establish distinct reported paired constructions under the support and topology controls. Predictive gain, novelty, generalization, continuation behavior, and qualification across additional seeds were not assessed.

## Scope and custody

This audit used only the supplied qualification JSON, original native qualification copy, cost/interface traces, output-custody JSON, and the reviewed v3 `MANIFEST.json` and `BOUND_INPUTS.json` metadata. It used Python's standard library; it did not reopen original numeric inputs/checkpoints, import Torch, execute a model, run native/GPU code, or make remote calls.

- The 171,600-byte qualification file exactly matches the original native copy, including SHA-256 `3417f9e7e0c30e8f502407737d0487ad1768e564155a7ac374b229f4b2969e21`.
- Local original-output copies match the sizes and hashes recorded in custody. All recorded remote output paths belong to the v3 `runs/root01` directory. The 36 cost records exactly match `operation_costs`; the interface trace contains 84 records.
- All 151 reported original-preservation records are identical before/after, marked preserved, and match the reviewed bound-input path/hash map. The seven numeric descriptors were inspected as metadata only. The receipt reports the warm donor unchanged, zero optimizer steps, no continuation, no checkpoint/RNG writes, and no validation/test label or score use.
- The reviewed manifest seal and its script descriptor agree with the prior v3 review: manifest `fba0de7e94eb3974fd5df40cd996bdccb816a0748c6834925c3eca3881a743c7`; script descriptor `3d38e56dd2b87fea3507e9b96ae5987c72283582321d123f58da2ee3df3febb1`.

The receipt does not directly embed its assay script hash. Its v3 schema, custody path, reviewed manifest, and preservation metadata correspond. This is a supplied-receipt audit, without an independent server rehash or reconstruction of execution.

## Native and installed-output correspondence

The reported full output is `[2223, 5]` in canonical node order with exact coverage; TRAIN contains 444 nodes. The identity factor boundary is `stem.S` and `head.R`, each `[1, 256]`, totaling 512 parameters. Model, logits, and gradients are FP32.

All five native/K1/K4 initial comparisons passed with finite values and exactly zero maximum absolute/relative difference. Each of the four arms passed both fresh-warm equality and installed all-member output equality against its independent closure, again with exactly zero maximum differences. The four installation trace receipts exactly match the final qualification receipt.

Active-precision AD passed on the actual closure using `FP32_per_example_CE_FP64_mean_v1`; the original FP32 measurement also would pass, and the receipt reports the original training gradient unchanged. All three directions passed duality (maximum relative error `4.7787e-6`, threshold `2e-4`). Both fixed finite-difference epsilons, `0.001` and `0.0003`, passed logits and CE thresholds for every direction.

## Shared radius, support, and topology

All four arms jointly accepted the first shared trial, with no fallback: `alpha = alpha0 = 4.6637179768264225`. Receipt scalar arithmetic independently reconstructs the radius `0.01 * sqrt(512)`, the direction bound `sqrt(1.25) * ||g||`, and the Armijo bound `1.3553255358390457` from baseline TRAIN CE `1.3553264141082764`. Every member and pooled TRAIN CE passed that bound.

| Arm | Maximum reported relative factor step | Minimum source TRAIN pair RMS | Minimum finite TRAIN pair RMS / alpha |
|---|---:|---:|---:|
| common_only | 0.00894427191 | 0 | 0 |
| train_remasked | 0.00966357735 | 0.000762264302 | 0.000763634539 |
| full_node | 0.00971783017 | 0.000428365689 | 0.000429043611 |
| full_node_permuted | 0.00971120443 | 0.000482646050 | 0.000483886881 |

All relative steps are below `0.01`. All three graph arms pass the source and finite TRAIN separation threshold `1e-6`; common-only correctly has zero separation and is exempt. Partition, band-gradient sum, descent, and tangent-mean certificates pass their tolerances. Each graph tangent Frobenius norm is approximately `0.0216978998 = 0.5 * ||g||`; the common tangent is zero. These checks reconstruct reported scalar certificates, without recomputing raw tensors or Jacobians.

The saved seed-80017 permutation is a bijection of all 2,223 nodes, and its native int64 byte hash was independently reproduced. The receipt asserts `Pi S Pi^T` with features/labels fixed. Full-output tangent Grams are finite, symmetric, and have nonnegative diagonals. Their Frobenius differences are nonzero: full versus remasked `2.9600e-6`, full versus permuted `8.2839e-6`. Signed finite/first-order ratios are approximately `0.996263`, `0.996768`, and `1.002320`; the remasked diagnostic is explicitly cross-support, and every diagnostic is marked non-predictive. These are construction results.

## Costs and counter reconciliation

The paired helper reports one VJP primal, 13 pullbacks, 12 JVPs, six completed sparse products, and 16 candidate forwards (four per arm), with zero extra common diagnostic forwards. Its closure-forward total is 29.

Whole-assay closure forwards reconcile as `1 contract + 16 AD + 29 paired + 16 installation references = 62`, with 62 started/completed and zero failed. Trace call numbers are exactly 1–62. Outside the observed closure, 11 containers completed and represented 38 scheduled member predictors. The nine trace batch events reconcile because the initial identity batch covers three containers/six members; eight later K4 batches cover the remaining eight containers/32 members. Thus successful calls represent 100 member predictors in total.

All 36 native ledger operations and five stdlib stages completed; no observed resource failure was recorded. Reported process wall time is **14.5414 s**; the paired operation is **1.8027 s**. Largest recorded per-operation peaks are **1,313,777,152 allocated bytes (1.22355 GiB)** and **1,568,669,696 reserved bytes (1.46094 GiB)**. Preflight free memory was 52,084,867,072 bytes versus 12,561,940,480 required. Nested loader timing and peak resets prevent interpreting operation sums/maxima as independently observed whole-process cost/peak. RSS is retained as `1018544` in native units.

## Reproducibility

The complete check ledger and hashes are in `AUDIT_CHECKS.json`; `CONCERNS.json` records the empty blocker list and evidence limits. Re-run the audit with:

```sh
/usr/bin/python3 -I -S -B postsubmission_research_20260930/graph_paired_native_receipt_review_20261003_v1/audit_receipt.py
```

The audit writes only its own `AUDIT_CHECKS.json`. Reviewed source packets, source inputs, native receipts, and prior reviews remain untouched.
