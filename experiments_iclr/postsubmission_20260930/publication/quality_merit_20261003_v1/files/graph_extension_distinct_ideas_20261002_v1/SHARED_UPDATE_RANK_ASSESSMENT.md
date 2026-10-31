# Shared-W arithmetic from low-rank member update inputs

Root-requested follow-up, 2 October 2026. Saved reports only; no further broad search, new primary read, source implementation, data, rank measurement or execution.

**Decision: no-go for launching a GNNM novelty pilot on the current evidence.** This is a distinct, precisely implementable shared-arithmetic hypothesis, but it is ordinary low-rank multi-right-hand-side multiplication with an error budget. It needs an observed low-rank property of the *modulated active update inputs*, a complete kernel/cost advantage over exact batching/composition, and a closer compression-prior review before it supports a field contribution. The rank property does not follow from shared weights or common raw updates. A realistic sign-factor counterexample removes any uniform exact saving. Keep the idea recorded, not as a rescue experiment.

## Actual delta over the earlier secondary lead

The earlier feature-axis proposal compresses graph transport P C_m B and reconstructs messages, reducing sparse edge channels. This follow-up instead targets **repeated dense application of the same W to a small group of active member residuals**. It need not discard a private graph mode or use a common recurrent state. It reconstructs every member's dense update before its private output factor, graph scatter and activation.

At active node i, let a <= M be the number of member updates legitimately due under the unchanged queue. Stack

    U_i[m,:] = deltaZ_i,m diag(r_m),     U_i in R^(a by h).

Compute a rank-q representation U_i approximately A_i B_i, where A_i is a by q and B_i is q by h. Then

    U_i W approximately A_i (B_i W).

Apply each member's diag(s_m) only after reconstruction, obtaining its own t_i,m. No nonlinear operation may move before reconstruction, and no member delta/state is averaged. If rank(U_i) <= q and the factorization is exact, this is an exact linear identity. Generic full-rank U_i requires q=a for exactness. An independent ensemble of arbitrary W_m cannot reuse this same B_i W transform without additional structure; independent matrices with a common factor/tensor representation could also share work, so “impossible for independent ensembles” is too broad.

The proposed adaptive operation is: choose the smallest q passing the *current error budget* at that node, otherwise use the qualified exact batched transform. This is not rank HPO. The tolerance/cost policy must be fixed before outcomes and must account for every factorization, failed low-rank attempt, fallback and rebuild. A cached basis needs a valid update-input drift rule; stale approximation cannot silently inherit a certificate.

## Error accounting that the maintainer would need

Define E_U = U_i - A_i B_i. For each active member the maximum dense-output error obeys

    ||e_t,m||_infinity <= ||E_U[m]||_infinity ||W||_1 ||s_m||_infinity.

The input residual E_U is available without computing the full U_i W. This bound is conservative, and includes the output factor. The contraction rho_m separately includes *both* r_m and s_m as in DERIVATION.md.

Approximate pushes break the exact Y invariant. Maintain a nonnegative defect bound b_i,m satisfying

    ||Y_i,m - [alpha X_m + (1-alpha) P Z_m V_m]_i||_infinity <= b_i,m.

A compressed push at source i adds at most (1-alpha) P[j,i] ||e_t,m||_infinity to b_j,m for each affected destination. A row-mean edge insertion/deletion repair scales its existing endpoint bound by d_old/d_new and adds the actual endpoint transform error divided by d_new. A forcing-only repair is exact if its learned encoder value is computed exactly. Start with b=0 after an exact bootstrap, or with a separately justified bootstrap bound.

The observed R = sigma(Y)-Z is not the true fixed-point residual after compression. The valid state certificate becomes

    ||Z_m - Z_m_star||_max
      <= (||R_m||_max + max_i b_i,m)/(1-rho_m).

Thus approximation noise cannot be counted as member diversity, and convergence of R alone is insufficient. As b consumes the budget, increase q or recompute a destination's exact current aggregate and channel map to reset b there. A full exact rebuild is an available final fallback. These corrections have real graph/dense cost. The small numerical output tolerance proposed in REPORT.md could force q=a or frequent rebuilding.

This defect accounting is an elementary extension of the invariant/error argument, not an independently reviewed theorem. It needs a full queue/deletion/roundoff qualification before use.

## Realistic full-rank falsifier

Take M=4 and h divisible by 4. Let all raw member residuals be the same row of ones. Choose the r_m to be four orthogonal equal-norm Hadamard sign rows, let s_m be ones, and let W=I. These are permissible finite factor maps; with alpha=0.1 they have rho_m=0.9, so the contraction condition does not remove the example.

The modulated input matrix U has rank four and four equal singular values. The best rank-q Frobenius approximation discards the fraction (4-q)/4 of its squared norm. For q=3 the discarded norm is half the full norm; q=2 loses sqrt(1/2) of the norm. Its output under W=I has the same loss. The declared conservative per-update fidelity bound can therefore force q=4 and offer no dense-transform reduction. Nonlinear clipping or a particular head could reduce downstream sensitivity; neither repairs a universal exact-saving claim.

This is a symbolic counterexample, not a measured model. Random sign input factors make full row rank plausible even when raw updates align; trained factors might instead become coherent. No trained active-update spectrum was inspected. Useful small rank must be demonstrated on the actual factors, queue overlap and update distribution, without selecting favorable nodes or excluding high-rank updates.

### Rademacher calculation supplied by root

For one common raw row x and independent Rademacher input-factor rows r_m, U_m=x elementwise-multiplied by r_m. Its Gram has diagonal D=sum_j x_j^2. For m distinct from n, G_mn=sum_j x_j^2 r_mj r_nj has expectation zero and variance sum_j x_j^4. Define d_eff=D^2/sum_j x_j^4 for nonzero x. The relative offdiagonal standard deviation is 1/sqrt(d_eff). Distributed feature energy therefore makes an approximately diagonal, four-dimensional Gram plausible initially; common raw inputs do not establish a small member rank. This is a moment calculation conditional on independent signs, not a representative trained-spectrum observation. Learned factors, private residuals and their dependencies can later differ. Any future diagnostic must use all predeclared actual TRAIN-role active-update spectra, and distinguish a trained property from this initialization calculation.

This removes an automatic low-rank premise, not all scientific potential for trained adaptive compression. The current scientific no-go is for a new GNNM principle based solely on ordinary factorization; its empirical whole-cost opportunity remains unestablished and could be resource-deferred if a separately qualified spectrum/cost diagnostic justified it. Present GPU capacity was not used to reject the hypothesis.

## Whole cost and exact alternatives

For a active rows, full dense multiplication costs a h^2 MACs. The low-rank path costs roughly q h^2 plus O(a^2 h) Gram construction, O(a^3) small factorization, O(a q h) encoding and reconstruction, private factors, packing and synchronization. For a=4, h=64, q=2, the common W term falls from 16,384 to 8,192 MACs, but the basis/Gram/reconstruction overhead is additional. For q=3 the dense saving is only 4,096 MACs; q=4 adds overhead and should bypass compression. If only one member is active, no rank reduction exists. These are analytic operation counts, not latency measurements.

Required exact controls are the fastest valid common-W stacked GEMM, frozen factor composition V_m=diag(r_m)Wdiag(s_m), cached exact transformed states where valid, and all applicable affine composition from the saved dense-work report. That report already supplies an exact composition across adjacent linear SAGE/FFN maps; it does not commute through an activation, and does not establish low rank of U_i. Apply it fairly before claiming an approximation saves work.

Also compare a once-profiled low-rank factorization of W with its output-error budget: if W is already low rank, that can remove repeated per-node Gram/SVD overhead. A competent joint tensor/factor compressor for independent W_m is a closer shared-arithmetic comparison if its approximation is admissible. The saved task-directed report already attributes Fisher-weighted SVD/output-distortion rank allocation to FACTS/CoRS (arXiv:2609.07155v1), low-rank factor uncertainty/distillation, and task-trained GNN communication. It does not verify this exact active-update assembly; it prevents novelty credit for SVD, adaptive rank or error-budget language alone.

## Representative falsifier and confirmation, strictly conditional

First perform a separately admitted **label-free common-state update replay**, after a competent fresh frozen equilibrium source is available, on complete ogbn-arxiv publication-year prefix changes and seeds 17/29/43. Record every active a, rank selected under one fixed tolerance, discarded bound, exact-bypass fraction, b accumulation, exact repair/rebuild work, input-factor statistics and complete elapsed/peak cost. Compare every reconstructed update with the full exact batched transform. A low-rank common-state matrix alone is not a stream-serving result.

The decisive pre-fit/resource stop is no meaningful whole-update benefit under the fixed output error tolerance. A candidate whose active inputs usually require q=a, whose queue seldom has a>=2, or whose basis/defect/rebuild work costs more than the saved transform fails. Even if q=2 arithmetic looks favorable, the whole p95 freshness and bootstrap-plus-stream thresholds in REPORT.md must hold against the fastest exact batching path. Any invalid defect bound or accumulated numerical discrepancy kills the path. Do not lower accuracy tolerances, change factors or train for rank after these outcomes; that would be a new learned architecture/objective.

A later quality-neutral predictive replay must compare the complete compressed recurrence with the exact maintainer at the same frozen weights/pool, not only dense update error. It must retain near-margin class changes, all solver/approximation failures and every fallback. The source’s full-role target policy remains fixed. Confirmation requires the locked policy on complete ogbn-products mechanical updates and new seeds 47/59/71, with its large state/graph memory charged. Real temporal-event transfer remains a separate task/protocol.

No replay, implementation or launch is authorized by this note. It supplies the concrete shared-arithmetic property that a later evidence packet would need, and a reason it can fail under ordinary private factors. On current evidence there is no demonstrated low-rank distribution, whole-cost advantage or methodological delta beyond known compression; the method/pilot lane stays closed.

## Saved source bindings

- `dense_work_method_v1/REPORT.md`, “Repeated dense work: an exact optimization to keep separate”: exact adjacent-affine composition and fair exact execution.
- `graph_ensemble_gap_skeptic_v1/task_directed_contrast_transport_v1/REPORT.md`, sections 2–5: joint member/feature distortion, rank obstruction, weighted SVD/rank-allocation priors, graph coordinate limits.
- `member_subspace_messages_v1/REPORT.md` and `low_rank_communication_assessment_v1/ASSESSMENT.md`: ordinary global member mixing and per-site versus whole-trajectory approximation.
- `continuous_graph_efficiency_gap_v1/round02_incremental_trajectories/REPORT.md`: private caches, tagged queues, competent independent batching, retained RIPPLE++/NeutronRT/STAG scopes.
- `gine_closest_control_v1/implicit_ensemble_gap_v1/REPORT.md`: member maps versus solver error, IGNN/DEQ/IGNN-Solver scopes, unchanged well-posed predictor.

Hashes and report-read limits are preserved in INPUT_BINDINGS.json. No retained primary paper was reopened for this assessment.
