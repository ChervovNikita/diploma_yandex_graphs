# FoRDE direct streaming remedy proposal

Use an explicitly selected two-pass direct backend as the next arithmetic remedy for the preserved float32 cancellation failure. It reconstructs the represented full-input gradient for one target at a time, streams member pairs, and computes the loss and its cotangent over the complete logical batch. This removes the large batch and pair broadcasts while retaining the existing source formula. It is the smallest proposed change with a qualified direct reference; its numerical equivalence and actual resource use still require qualification.

This packet is a source-only proposal. It adds no implementation, numerical execution, dataset access, GPU access, optimizer update or predictive result. The sealed source, execution, review, baseline-context and literature-index records are preserved. The numerical evidence below is reused from the prior execution receipt.

## Evidence for the choice

The prior tiny native tests compared the explicit direct coefficient backend with independently differentiated full-X target gradients. All four float32/float64 native cases passed the predeclared tolerances, including mixed derivatives for every one of the tiny model's 26 private R/S tensors. The largest direct mixed-derivative error was 2.38419e-6 in float32 and 1.77636e-15 in float64. These are tiny CPU fixtures, not qualification of a streaming or Amazon implementation. [E1]

The severe fixture has identity P, three identical polynomial rows, and coefficients `q0=8192*d`, `q1=-8192*d`, `q2=0.001*d` for distinct fixed member directions. Direct reconstruction produces squared norms near 1e-6 and finite forces. Float32 Gram evaluation returns finite values but differs from direct by 0.17023945 in normalized similarity, 0.12914085 in squared distance, 0.01165354 in repulsion and 73.030838 in the coefficient derivative. The squared-norm error alone passes its absolute tolerance, so finiteness and a small absolute norm error cannot establish force fidelity. The float64 diagnostic's passage on this fixture does not qualify a native precision switch. [E1, E2]

Direct streaming addresses the particular Gram failure by forming `A_b^T q` before normalization. It does not guarantee all cancellation cases: target slicing can change contraction kernels, and pair accumulation can change reduction order. Those changes need direct value and derivative comparisons under the declared native precision settings.

## Formula retained

For member i, reference j, target b, polynomial order k, node n and feature f, let fixed selected rows be `A[b,k,n]=row_target_b(P^k)[n]` and let q be the selected raw-output derivative with respect to the native mono tokens. The existing row-local native composition obtains q with `create_graph=True`, so q retains its dependence on all enabled private parameters. [S1]

The prospective backend retains these equations without a new coefficient:

```
g[i,b,n,f] = sum_k A[b,k,n] * q[i,b,k,f]
s[i,b] = g[i,b] / sqrt(sum_(n,f) g[i,b,n,f]^2 + 1e-24)
D[i,j,b] = sum_(n,f) (s[i,b,n,f] - stop(s[j,b,n,f]))^2
h[j,b] = stop(median_i D[i,j,b]) / log(M) + 1e-12
R = (1/B) sum_i (logsumexp_(j,b)(-D[i,j,b]/h[j,b]) - log(B))
L = sum_i mean_b CE(logits[i,b], label[b]) + R
```

The bandwidth median removes only the live-member axis. It includes self values and averages both central values for even M. Reference directions are stopped after normalization. Live normalization remains differentiable. h is stopped, and target averaging remains inside each member's logarithm. Self entries remain in the KDE denominator; their distance and first-argument derivative are zero at equality. The selected epsilon is the author README replication setting; the author code's configuration default is a different value. [S1, S2]

For the entire logical batch, define `Z[i,j,b]=-D[i,j,b]/h[j,b]` and `p[i,j,b]=softmax_(j,b)(Z[i,:,:])`. With h stopped, the global distance cotangent is

`C[i,j,b] = dR/dD[i,j,b] = -p[i,j,b]/(B*h[j,b])`.

There is no extra 1/M or extra target averaging in C. The implementation should obtain C by differentiating the existing `repulsion` operation on a small complete D leaf with frozen h. The formula states the mathematical contract; reusing the native logsumexp derivative avoids introducing a separate softmax arithmetic implementation.

## Two pass schedule

The proposed backend name is `direct_streaming_two_pass`. Selection occurs before the call and is recorded in its receipt. One process and one device handle M=4 and the complete logical B=128. Streaming is a storage schedule, with one loss and one optimizer update for that logical batch.

1. Construct the live q and logits once using the existing native selected-token derivative path. Keep their outer model graph. Bind the ordered targets, score channels, P, complete token bank, feature identity, row powers, K/base, dtype, device and precision settings. Keep the parameter snapshot, modes and all fixed inputs unchanged throughout both passes.
2. Inside a custom autograd forward, reconstruct all M full N-by-F gradients for target b from detached q values, normalize them with the source operation, and evaluate all ordered pairs one at a time with explicit difference, square and sum. Retain only complete `D[M,M,B]` and small diagnostics across targets. Do not retain the target's reconstruction or pair graph.
3. Compute h, R and C from the complete D using the source column median and complete logsumexp. Save q values, C and the fixed descriptor for recomputation. The outer custom-autograd result is connected to the live q input.
4. In backward, for each target, recompute its M gradients and normalized directions from a fresh local q leaf. Detach normalized directions for the reference argument. For each pair, use the source difference-square-sum operation and its native VJP to the live normalized direction, weighted by the saved C entry. Accumulate a normalized-direction cotangent in declared reference-index order and immediately discard that pair's graph.
5. Apply one native VJP from the target's normalized directions to its local q leaf. This differentiates the live norm denominator and the A contraction. Place the resulting slice in the complete q cotangent, then free the target graph. Pair VJPs stop at normalized directions, so they do not repeat the A pullback for every pair.
6. Return the complete q cotangent through the original live q graph. Ordinary outer backward combines this mixed private derivative with CE and performs one source optimizer update only after the whole batch completes.

The normalized-direction cotangent in step 4 is mathematically `w[i,b]=sum_j 2*C[i,j,b]*(s[i,b]-stop(s[j,b]))`. The proposal uses native pair VJPs and then the native normalization VJP; it does not replace either with a newly coded normalization Jacobian. The returned q cotangent is evaluated at the original snapshot. Holding that cotangent constant while applying the q-to-parameter VJP is the chain rule for the desired parameter gradient; differentiating C again would introduce an unwanted additional derivative.

A loop that stacks differentiable target distances and calls backward once can retain every large target graph and lose the intended memory reduction. A loop that computes a loss separately per target changes the mixture inside the logarithm. A custom autograd boundary or an equivalent explicitly audited recomputation VJP is therefore required. This proposal covers the mixed score/private derivative needed for one FoRDE step; differentiation of the optimizer step itself is outside its scope.

The first implementation should reconstruct full N-by-F arrays per target, use the same einsum signature with a target slice of length one, and use explicit pair differences. Feature/node tiles, distance identities such as `norm_i+norm_j-2*dot`, symbolic zeroing, sparse support optimizations, adaptive target blocks and multiple devices are separate prospective arithmetic changes. They are not part of this initial contract. The target slice and fixed pair accumulation still require qualification; bitwise identity is not claimed.

## Estimated storage and work

All estimates are decimal bytes from source-declared shapes: N=24,492, F=300, M=4, B=128, float32, and L=11 or 15. They are neither data reads nor measured peaks. The complete model graph for q is retained and must be measured separately. [E1]

| Value allocation | L=11 | L=15 |
| --- | ---: | ---: |
| Complete fixed native tokens | 323.29 MB | 440.86 MB |
| Selected A for logical B | 137.94 MB | 188.10 MB |
| q values | 6.76 MB | 9.22 MB |
| Complete returned q cotangent | 6.76 MB | 9.22 MB |
| One target's M input gradients | 117.56 MB | 117.56 MB |
| One target's M normalized directions | 117.56 MB | 117.56 MB |
| One target's normalized-direction cotangent | 117.56 MB | 117.56 MB |
| One pair's N-by-F array | 29.39 MB | 29.39 MB |
| Complete D or C | 8.192 KB | 8.192 KB |
| Complete h | 2.048 KB | 2.048 KB |
| Prior full-batch input gradients | 15.05 GB | 15.05 GB |
| Prior full pair broadcast | 60.19 GB | 60.19 GB |

The streamed values no longer scale as B times the full input arrays or M squared times a full-batch array. Multiple pair temporaries, squared arrays, saved normalization inputs, backward temporaries, model activations, q's higher-order graph, sparse P, allocator and workspaces remain chargeable. The table is not a peak-memory bound or resource admission.

Retaining the existing descriptor is the smallest initial implementation change. It also retains its Gram, full A hashing/copies, token-bank reconstruction during binding, and row-list plus stacked-A preparation peak. Those costs must be included. A production descriptor optimization should follow a successful streaming qualification, with its own custody witnesses.

Relative to one materialized direct forward and backward, this schedule adds one direct reconstruction pass. Across the logical batch it uses two forward A-to-g contractions plus one A VJP, roughly `3*M*B*N*F*L` multiply-accumulates: 124.15 billion for L=11 or 169.29 billion for L=15. It evaluates `M^2*B` full-input pair reductions in the first pass and repeats the local pair operations for their VJPs in backward. These source-shape counts exclude native q construction, its mixed parameter backward, CE and fixed-row preparation. They do not predict elapsed time. Cold row preparation costs K sparse transposed propagations of B basis columns, with normalized nnz still unmeasured.

## Why QR is deferred

For a complete thin factorization `A_b^T=Q_b R_b` with orthonormal Q, the exact-arithmetic identity `g=Q_b(R_b q)` gives norms and distances from `R_b q` in the full L-dimensional row basis. Retaining all L rows without rank truncation preserves the scientific metric in exact arithmetic. R storage is only 61.95 KB or 115.20 KB for this batch, and compressed member gradients occupy 6.76 MB or 9.22 MB. Per-step kernel arithmetic is much smaller than direct full-input reconstruction after cache preparation. [E1]

That algebra is not a native float32 qualification. If `A_b^T=Q_hat R_hat+E`, a cancellation vector can have tiny `A_b^T q` while `E q` is material relative to it. Orthogonality error also perturbs the metric. Rank deficiency does not make a full Householder QR invalid, but exact-null and near-null vectors are particularly sensitive to these residuals and to the existing 1e-24 stabilizer. Unpivoted QR avoids a permutation contract; a pivoted version would additionally need the full explicit permutation and inverse mapping, without truncation. Neither path has a source-equivalence receipt.

QR requires cold work proportional to `B*N*L^2`, factorization workspace and potentially thin-Q allocation. Preparation for every intended TRAIN target and reuse accounting must be charged. It may become the faster backend if separately qualified; it is not the cheapest established numerically faithful remedy today. The recommended first remedy minimizes new arithmetic mechanisms and implementation scope. No claim of globally minimum runtime or measured feasibility is made.

## Failure and qualification contract

Native float32 remains explicit, with autocast disabled and the multiplication/reduction settings, library versions and device recorded. Float64 is a separately named synthetic diagnostic. No execution may silently switch dtype/backend, clamp norms/distances, add a stabilizer, truncate rows/neighborhoods, remove self entries or replace the KDE formula.

Finite nonnegative norm and distance checks, finite positive h, finite R/C/q cotangents and finite private forces are ordinary numeric validity checks. A zero norm or zero median is valid under the existing epsilon and bandwidth floor. Explicit nonpositive/nonfinite h, other nonfinite arithmetic, custody mismatch or resource exhaustion stops the update and records the failing stage and attempted work. No optimizer state is committed before both passes and the complete backward succeed. Such a failure does not trigger an automatic fallback or a candidate-specific acceptance rule.

Before implementation use, reuse the existing float32 value tolerances (atol 3e-6, rtol 3e-4) and derivative tolerances (atol 1e-4, rtol 3e-3), plus the separately named float64 diagnostic tolerances. Compare streamed values, q cotangents and every actual private R/S derivative with materialized direct and the independent full-X oracle. Include the preserved severe, ordinary, exact-null and all-zero fixtures; dependent and near-dependent rows, norm regimes around the stabilizer, duplicate member directions and tied/floor medians; nonsymmetric and duplicate COO cases; whole-batch versus stored global-cotangent behavior; and any declared final partial batch using its actual B. Self D and self force must remain exactly zero in the preserved exact fixtures. Results must not change tolerances after observation.

A separately authorized resource receipt must cover the full logical B=128, cold and reused A preparation, descriptor validation, both streaming passes, complete q/model mixed backward and all peak allocations. A tensor-only shape rehearsal would not certify native all-private backward feasibility. Real data release, warm competence, SGD learning rate/schedule/duration and validation selector remain unresolved in the existing program; this arithmetic proposal closes none of them and launches no fit.

## Source and reuse record

- **E1:** `forde_graph_cpu_execution_20261003_v1/REPORT.md` and scoped `run01/EXECUTION_RECEIPT.json`, plus `forde_graph_source_style_engineering_20261003_v1/run_oracles.py`. Reused numerical execution, not a new run.
- **E2:** `source_review_forde_graph_gram_v2_20261003_v1/REPORT.md`. Reused Gram identity, conditioning, logical-batch and resource findings.
- **S1:** `forde_graph_source_style_engineering_20261003_v1/forde_graph.py`, especially selected-token q, explicit direct backend, bandwidth and repulsion, with its execution contract and source bindings.
- **S2:** Saved author `forde_train_forde.py:270–320` and `forde_README.md:38–80`, pinned to AaltoPML/FoRDE commit `e8f9d7418be43064ffdc247ba7e62a61149da542`. The v1 preparation report has superseded full-matrix bandwidth and averaging statements; the v2 independent review and engineering source govern this proposal.
- **Literature reuse:** FoRDE is the already retained *Input-gradient space particle inference for neural network ensembles*, arXiv 2306.02775v3. BatchEnsemble conclusions already establish shared weights with private input/output scaling and complete member work. No new paper, primary-paper reread, novelty, full-method reproduction or posterior guarantee is claimed. The old index's log-probability score description is superseded for this control by the saved author-code raw-score correction.

`METHOD_CONCLUSIONS.json`, `PROSPECTIVE_CONTRACT.json`, `RESOURCE_ESTIMATES.json`, `QUALIFICATION_PLAN.json`, `READ_SCOPES.json`, `SOURCE_BINDINGS.json` and `REUSED_CONCLUSIONS.json` preserve the derivation, declared limits and exact custody.
