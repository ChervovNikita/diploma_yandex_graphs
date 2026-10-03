# FoRDE native graph source-style CPU engineering

The minimal successor is `forde_graph_source_style_engineering_20261003_v1`, outside every sealed predecessor. Its source-only APIs integrate the unchanged qualified native v4 PolyFormer with a distinct all-existing-private-R/S setup, ordinary coupled-decay Nesterov SGD ownership and exact operator/token/target/Gram custody. The executed scope is synthetic CPU derivative engineering on 192.168.18.77. No labels, datasets, checkpoints, quality scoring, optimizer updates or predictive fits were executed or read. No environment changes, installs, other-job changes or candidate-specific guard/null were introduced.

The numerical equations remain fixed: raw selected output score; full-X norm denominator `sqrt(norm2+1e-24)`; stopped normalized references; `h[j,b]=stop(median_i D[i,j,b])/log(M)+1e-12` with even central averaging and self; target mean inside log; positive `R_sum=(1/B)sum_i log(sum_j mean_b kernel)`; source CE defined as sum of member means. The labelled CE integration is present but unexecuted. Executed score channels are fixed synthetic scalar definitions, not supervision.

## Actual execution

The ordinary process exited zero in 6.972855 seconds including startup; the runner took 3.469802 seconds. Torch reported 2.7.1; the authorized interpreter SHA256 is `14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2`. CPU threads/inter-op threads were one, matmul precision highest, autocast inactive, distributed inactive and CUDA remained uninitialized. The source transfer verified all 37 native v4 payloads unchanged. Actual code hashes are recorded inside the execution receipt and match local source bytes.

The tiny native model has F=2, K=2, H=4, one native attention/FFN layer, four members and two selected targets. Every existing affine R/S is trainable: 26 named tensors containing 468 scalar entries, including all boundary sites. Common weights/biases, LayerNorm, bias scale and buffers remain frozen after the explicit synthetic bank construction. The bank uses a predeclared sinusoidal member-factor rule; it was not fitted or adapted to results.

| Native fixture | Maximum value error | Maximum Gram mixed-derivative error | Result |
| --- | ---: | ---: | --- |
| float32, nonsymmetric duplicate COO | 2.38419e-7 | 9.53674e-7 | passed stated tolerances |
| float32, identity/repeated eigenvalues | 2.08616e-7 | 7.15256e-7 | passed stated tolerances |
| float64, nonsymmetric duplicate COO | 5.55112e-16 | 1.77636e-15 | diagnostic passed |
| float64, identity/repeated eigenvalues | 5.55112e-16 | 1.77636e-15 | diagnostic passed |

The checks compare selected/full native logits, independently differentiated full-X target pullbacks, normalization, normalized similarities, distances, column bandwidth, kernels, repulsion and mixed derivatives for **every** trainable tensor. Both explicit Gram and explicit materialized-direct coefficient backends are checked against direct full-X autograd. The direct-backend maximum mixed-derivative error was 2.38419e-6 in float32 and 1.77636e-15 in float64. Tolerances were embedded before execution: float32 value absolute3e-6/relative3e-4 and gradient absolute1e-4/relative3e-3; float64 value absolute1e-11/relative1e-8 and gradient absolute1e-9/relative1e-7. No tolerance was relaxed after results.

All native fixtures preserved exact self-zero distances, frozen state bytes, eval modes, empty parameter `.grad` fields and restored member selection. Every off-diagonal direction distance was positive: minima were approximately0.00072361 for nonsymmetric P and0.00106414 for identity P. This qualifies the distinct **synthetic** factor bank, not a real warm checkpoint or graph competence. Swapped target order and modified G were rejected. The descriptor also hashes and validates operator COO indices/values/coalescence flag, complete token bank and row powers; those additional mutation paths were not separately executed as negative witnesses.

## Concrete float32 Gram blocker

A repeated-eigen/collinear fixture uses identity P, three identical selected row powers and coefficient vectors

`q0=8192*d`, `q1=-8192*d`, `q2=0.001*d`,

where each member's predeclared two-dimensional direction d differs. Materializing the represented full-X vector before normalization yields norm2 near1e-6 and finite force. The Gram form's float32 summation loses cancellation accuracy. It completed without throwing, but failed the predeclared force-equivalence tolerance:

| Quantity | Maximum float32 Gram versus direct error |
| --- | ---: |
| normalized similarity | 0.17023945 |
| squared distance | 0.12914085 |
| bandwidth | 0.00436170 |
| kernel | 0.02145123 |
| repulsion | 0.01165354 |
| coefficient mixed derivative | 73.030838 |

Direct R was0.88945425; Gram R was0.87780070. A permissive absolute norm2 tolerance alone would have missed the problem: norm2 error was only1.28987e-7, yet normalization/force differed materially. The float64 diagnostic passed this particular cancellation fixture, with similarity error5.07957e-10 and mixed-derivative error7.91515e-8. It does not authorize a silent dtype switch or certify all cancellation regimes.

Ordinary collinear, exact-null cancellation and all-zero fixtures passed in both dtypes. Exact-null/all-zero norm2, distances and forces were zero, self distances exactly zero and bandwidth was the author's existing positive1e-12 floor. A zero median **is not a zero bandwidth** under this recipe. Deliberately supplied exact-zero bandwidth was explicitly rejected in every fixture; its recorded failure message is preserved. No zero-bandwidth failure was fabricated for cases where the source floor makes the bandwidth positive, and no stabilizer/floor was altered to obtain passage.

Thus generalized native float32 Gram evaluation remains unqualified. The implementation exposes the unchanged full-X direct backend explicitly; it performs no automatic fallback, precision switch, clamp, projection, null or scientific guard. The native tiny direct-versus-Gram checks do not erase the severe-cancellation failure.

## Smallest prospective source-faithful arithmetic remedy

A prospective row-basis evaluation can preserve the same full-X metric without forming `q^T G q`. For each target, let `A` have polynomial rows and compute a full thin factorization `A^T=Q R`. With orthonormal Q, `g_X=Q(Rq)` and `||g_X||²=||Rq||²`; cross-member inner products and distances can be evaluated using `Rq` in the full row basis. This replaces Gram quadratic evaluation with a row-basis contraction of the same algebra.

This QR/R path is **not implemented, adopted or qualified** here. It must retain the full declared row dimension with no rank/eigenvalue truncation, neighborhood truncation or metric projection. Rank-deficient/near-dependent row powers, QR residuals and signed cancellation in Rq can still be amplified by the source's very small normalization epsilon. Householder arithmetic, orientation, deterministic backend, norm regimes, normalized similarities and every mixed-private derivative must qualify against direct full-X pullback before use. A named float64 diagnostic can help identify arithmetic error but cannot silently change the native float32 method. No additional null/stability penalty, energy band or damping is proposed.

## Full Amazon logical-batch feasibility limits

The following are source-informed **formula estimates**, not dataset reads, host measurements or resource admission. The pinned PyG source docstring reports Amazon-ratings N=24,492 and F=300. The independent native warm source design declares transferred K=10 and K=14 recipes, so L=11 or15. Neither recipe, real M4 warm bank nor SGD schedule is adopted by this engineering packet. Take the proposed source-style M=4 and whole logical B=128.

| Float32 value allocation | L=11 | L=15 |
| --- | ---: | ---: |
| complete native tokens | 323.29 MB | 440.86 MB |
| selected row powers A | 137.94 MB | 188.10 MB |
| G only | 61.95 KB | 115.20 KB |
| live q only | 6.76 MB | 9.22 MB |
| materialized full-X gradients | 15.05 GB | 15.05 GB |
| one MxM full-X difference broadcast | 60.19 GB | 60.19 GB |

These decimal-byte counts exclude autograd graphs, normalization outputs, dense predictor activations, sparse P, allocator overhead and QR workspace. The direct backend in this tiny reference uses the MxM broadcast, so its full Amazon memory would be substantially larger than the15.05 GB gradient values alone. It is **not demonstrated feasible** for the native logical batch. Keeping the source score and logical B fixed would require a separately qualified memory implementation, such as explicit pair/target streaming with the complete global kernel cotangent, before a real fit; no such implementation is adopted here.

The current descriptor deliberately retains A for its direct oracle. The row list plus stacked A transiently costs roughly twice A; unlike a production G-only cache this object is not just62/115 KB per batch. Its full-tensor hashing, CPU copies and native token reconstruction during binding also cost memory/time. A QR cache would need approximately one A-sized thin Q or construction workspace while retaining only full R for kernel evaluation, with work proportional to B*N*L² plus sparse row-power propagation. Full TRAIN cache preparation and its amortization must be charged; actual native normalized nnz, target population, dtype settings and hardware peak/time are unmeasured. The source docstring edge count is not the constructed normalized COO nnz receipt.

## Remaining gates

All-private integration, narrow source custody witnesses, single-process whole batch and noncollapsed synthetic factor construction are now concrete. Float32 Gram cancellation is a preserved blocker. Real graph data/operator release, actual M4 warm acquisition/competence, SGD learning-rate/schedule/duration, validation selector, full-B=128 memory/runtime and any QR/streaming/direct production arithmetic remain open. No predictive fitting, full-method superiority, posterior guarantee or methodological novelty follows from this packet. `BLOCKER_CLOSURE.json` maps these results to the six earlier engineering findings.
