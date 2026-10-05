# Operator9 probability projection: numerical amendment V3

## Bound evidence and diagnosis

Root's sealed source V2 diagnostic succeeded without fits, optimizer updates, metric calls, scored outcomes or GPU work. Its bound witness SHA256 is `4a5310a1549d231fc1a2dd65838d39030a986cea1cd1ca8d022034e59efe0700` (19026 bytes). The authoritative native-batch traceback identifies node709, first chunk[0,2048), as an actual unchosen NaN row in old `simplex_qp`. Only its P[4,5], permitted-anchor posterior and numerical diagnostics were supplied to this preparer. No true label was exported.

All15 single-row recomputed old candidates were rejected. The legitimate free[1,2] candidate has weights approximately[.0125,.7617879182059456,.21321208179598564,.0125], zero floor/dual violation, but sum error1.9311219290329973e-12 exceeds the unchanged strict1e-12 candidate tolerance. Its augmented KKT matrix has full rank3 and smallest singular value2.2253707582837687e-6: this face's rejection reflects equality roundoff and conditioning, not cutoff truncation. The nearly agreeing members make normal equations vulnerable to common-mode cancellation; full-face rank loss is a separate issue. Single-row recomputation is diagnostic and can differ in rounding from the authoritative batch.

## Exact problem and repaired arithmetic

The scientific objective remains min ||P^T w-posterior||² with sum(w)=1 and w_m>=.0125. No ridge is introduced. Every nonempty free face is enumerated in the same increasing-size/lexicographic order. Fixed members remain at the floor. For free face F of size k and fixed set I, s=1-|I|*.0125, w0_F=s/k, w0_I=.0125, and w_F=w0_F+U z, where U is an orthonormal Helmert zero-sum basis.

Solve direct probability least squares Bz=d, with B=(P_F-P_ref)^T U and d=posterior-P^T w0, using a small batched FP64 SVD. P_ref is the first free member; centering is algebraically equivalent because 1^T U=0 and makes identical-member differences exactly zero. The retained-subspace Moore-Penrose solution minimizes ||z|| and hence weight norm on the face, since w0_F is orthogonal to U. Faces enforce equality through the parameterization instead of an augmented normal-equation constraint solve.

The old per-row scale max(maxabs(PP^T),maxabs(P posterior),1e-12) remains the comparison/KKT scale; Gram matrices are used only to reproduce that scale. Candidate comparisons use direct residual risk/scale; this differs from the former scaled quadratic by a row-constant and avoids cancellation. Objective ties1e-14, weight-norm ties1e-14, first-face residual ties, floor feasibility1e-12, sum feasibility<1e-12, candidate scaled KKT<1e-10, cleanup and raw/scaled post-check<=1e-10 retain their policies. Gradients are computed directly as P(P^T w-posterior), without Gram subtraction. Stable pool/log-pool code and all serving/fit inputs remain identical.

## Explicit effective-rank policy and limits

SVD retains singular values sigma>tau, tau=64*eps_FP64*max(1,sigma_max). This is numerical effective rank, not exact mathematical rank. For feasible weights the face coordinate norm is below1, so the discarded prediction motion has a local bound approximately2.84e-14. The independent arithmetic review gave conservative local bounds approximately8.1e-14 raw risk,4.1e-13 scaled risk and2.9e-13 gradient residual. These are local numerical bounds, not a claim that every cutoff perturbation is below the frozen1e-14 objective tie tolerance. Affected qualification includes both sides of this cutoff and nonzero-residual weak directions.

## Scope and qualification

Only `probability_hull_qp` is added and existing `projection` is redirected. All other numerical functions/classes, imports, constants and counters are preserved exactly. `custody.py`, `study.py` and `run_development.py` are byte-identical to V2. The generic `simplex_qp` including its ridge/PSD moment path remains intact. Protocol/reference correction, graph masks/folds/anchors, objective/fit/update budgets, scientific configurations and selection gates are unchanged. This arithmetic applies uniformly to operator9 wherever it is called.

The affected-only qualifier authenticates V3, V2 and the actual tiny witness, reconstructs exact unchanged source, and compares the witness with an independent80-digit Decimal analytic two-member constrained optimum. It also checks identical/duplicate minimum-norm optima, active bounds, known near-agreement/cutoff cases, local weak-direction risk bounds, independent direct-risk SLSQP and2051-row native chunk consistency. It constructs no head/calibrator, reads no scientific payload/label, calls no metric, and performs no sparse propagation or optimizer updates. Root must separately bind the passed V2 qualification for unchanged code; no discarded150-update head test is repeated.

This preparer performed only source edits, AST parsing, JSON parsing and hashes. Neither V3 nor the qualifier was numerically executed here. A qualified source still requires separate root admission before any full fixed development run; the failed V2 attempt is retained.
