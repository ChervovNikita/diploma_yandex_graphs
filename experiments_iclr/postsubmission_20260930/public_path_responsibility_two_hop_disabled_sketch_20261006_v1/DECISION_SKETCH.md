# Disabled decision sketch: public paths of length two

**Decision:** nominate one unsigned, fixed two-hop responsibility operator for a future prospective comparison, conditional on one engineering qualification. It is mathematically coherent and avoids sparse linear solves or dense terminal matrices. Runtime, useful restored coverage and higher-order implementation feasibility remain unverified. No source, framework, fit, tuning rule or held gate is changed; no execution is authorized by this sketch.

## Why this operator now

The authorized actual censusV2 found **58.82%–82.34% induced support isolates**, with 35–436 nonloop edges across ten pair graphs. The full **24,492-node / 93,050-undirected-edge** public graph is connected. Missing public paths are observed, not inferred from small S. This supports examining path restoration; it proves no predictive harm, assignment inactivity or alternative-prior improvement. Connected full topology does not imply that two-hop terminal graphs are connected.

## The one nominated operator

For each fixed class-pair terminal set T=Sₐ∪Sᵦ, set U=V\T. Let A be the immutable symmetric, nonnegative **loopless public adjacency**, with its original unit edge weights; retain classifier self-loops and inference. D_U uses the **full public degrees** of interior vertices, including their terminal neighbors. Interior labels are unused, including other S classes and W/R/A/VALID/TEST.

\[
C=A_{TU}D_U^{-1}A_{UT},\qquad
W_D=A_{TT},\qquad W_B=\operatorname{offdiag}(C).
\]

Thus each added weight is Σᵤ∈U AᵢᵤAᵤⱼ/dᵤ: a public path i→u→j through **one** interior vertex. Use every pair row; no sampling, threshold, top-k, learned affinity, extra walk depth or path-length coefficient. The observed full graph has no zero-degree interiors; in the general case a zero-degree vertex contributes zero, rather than receiving a ridge or invented connection.

Let L_D=diag(W_D1)−W_D and L_B=diag(W_B1)−W_B. Fix the permutation-invariant shared bound

\[
s=1+\max_i(W_D\mathbf1)_i+\max_i(W_B\mathbf1)_i,
\qquad L_2=(L_D+L_B)/s.
\]

Both affinities have zero diagonal; Laplacian diagonals retain their degrees. The zero-diagonal affinities need not be PSD; the required PSD property belongs to L₂. L₂ is symmetric PSD, L₂1=0, and ‖L₂‖₂≤2. Its unsigned gradient is γL₂Q, with uniform rows in its nullspace. Retain the original Q costs, entropy, positive row-simplex constraints, member-column masses, eight differentiated steps, γ and conservative step denominator. Q is not multiplied by W as a probability transition. No interior assignment or interior member-column mass is introduced.

**Sparse application:** C Q requires two full-public sparse products with an intermediate U mask and division by fixed degrees. No C or W_B matrix need be materialized. If c=C1 and κ=diag(C), then

\[
L_BQ=(c-\kappa)\odot Q-(CQ-\kappa\odot Q)=c\odot Q-CQ.
\]

Diagonal cancellation is an algebraic device, not permission to retain self-affinity. Static bridge degrees must be computed correctly, using nonnegative bridge contributions; floating-point identities require qualification. L_D uses the current direct sparse edges. Each ten-pair assignment-map evaluation adds **160 full sparse products** for 80 Laplacian applications, plus the small direct-edge work; reverse/adjoint and repeated episode calls add cost. With M=4, each full-public field has 97,968 scalars, but unrolling and the shared/private model derivatives determine actual peak memory. This is O(|E|M) work per bridge application and O(|V|M) field storage, not a whole-episode resource bill.

The graph coefficients are fixed. Differentiation must flow through dense Q and the existing cost/shared/private chains, not through A, degrees or normalization. Sparse backend support for the required higher-order derivatives is unqualified; no detach-Q or approximate-gradient substitution is permitted.

## Why not exact Kron or a resolvent first

| Construction | Paths and guarantees | Sparse implementation/compute boundary |
|---|---|---|
| Exact Kron Λ=L_TT−L_TU L_UU⁻¹L_UT | Exact minimum Dirichlet energy; all terminals connected here. Reduced edges follow paths with only eliminated interiors, not necessarily a full clique. Terminal-only Q mass remains. | Factorization with fill-in, or a grounded solve for each application; extracting/materializing a terminal kernel can be dense. Full derivative and normalization costs require accounting. Uncertified approximate solves cannot inherit exact PSD, row sums or gradients. |
| Fixed exact resolvent R=(I+L₀)⁻¹, W=offdiag(R[T,T]) | Classical symmetric nonnegative PSD full kernel, row-stochastic before restriction. Connected full graph gives positive off-diagonal terminal weights. The restricted kernel is substochastic; using its rebuilt Laplacian preserves uniform-null energy, not mass by multiplying Q. | Full-graph SPD solves for R1_T and R applied to terminal-supported Q, with adjoint work. Its implicit Laplacian is (R[T,T]1)⊙Q−R[T,T]Q and has norm≤2; no diagonal inverse entries are needed for this bound. Numerical solve guarantees and repeated derivative costs remain unmeasured. No residual/depth/tolerance choice is selected. |
| **Nominated two-hop operator** | Restores direct paths and paths through one eliminated vertex only; symmetric nonnegative affinity gives a PSD normalized Laplacian. It may leave isolates or components. It is not exact harmonic reduction. | Two sparse bridge products; no inverse, factorization or dense terminal kernel. Finite algebra and gradients can be fully specified before implementation. Qualification remains required. |

The two-hop bridge is the first term A_TU D_U⁻¹ A_UT in the grounded inverse's walk expansion. Longer interior walks are deliberately absent. This supplies classical ancestry, not novelty or an exact-Kron approximation-error claim after normalization.

## One necessary engineering check

Before a distinct pilot, perform **one paired, source-bound, no-commit full-map qualification** of live and the control below at the intended immutable bank and full native dimensions. Use all ten retained pairs, all eight assignment steps and the actually required shared/private reverse derivatives; discard virtual states. It must report:

- actual newly coupled previously isolated rows and components for each fixed pair; at least one previously isolated row must acquire an off-diagonal bridge in every pair, otherwise this fixed candidate fails its minimal path-restoration premise;
- implemented symmetry, uniform-null action, nonnegative affinity/PSD construction, declared norm bound and existing positivity/marginal/derivative checks at unchanged qualification tolerances;
- whole-call wall time, peak process/device memory, sparse-operation/adjoint counts and incremental cost against the direct-only map, within resource limits declared before the check.

Nonzero coverage is a minimal premise, not evidence of adequate coverage or quality. If coverage, native derivative equivalence or the declared resource budget fails, this candidate is **NO-GO**. Do not silently add walk depth, tune weights, replace gradients or launch a pilot. This check is a requirement, not an executed result.

## One necessary causal control

Use the frozen within-S-class permutation restricted to each T; call it P. Keep **direct edges unchanged** and permute only the bridge Laplacian:

\[
L_{\rm control}=(L_D+P^\top L_BP)/s.
\]

The same s applies because the bridge degree multiset is unchanged. Both arms use the same γ, 2γ denominator term, costs, labels, finite solver, gradient path, sparse-product count and model/serving protocol. The control preserves terminal target roles and the bridge's weights/spectrum while breaking their alignment with responsibility rows. Combined direct-plus-bridge spectra need not match; that is an effect of bridge placement. Apply direct sparse work separately in both arms so an optimization shortcut does not change their operation budgets.

This comparison asks whether **placement of added public bridges** helps beyond the same amount of permuted bridge coupling. Qualification must establish that the fixed permutation actually changes the bridge operator; if PᵀL_BP=L_B, the control is uninformative and no new permutation is selected to rescue it. It does not compare target orientation: **orientation is off in both arms**. It is also not the existing graph-free comparison, which removes 2γ from the step denominator. A class-preserving whole-operator permutation would additionally move the original direct edges and would not isolate the added bridge placement as closely.

## Scope and credit

Current seven-arm training, original scores and held gates stay fixed. Any later distinct pilot requires a prospectively frozen protocol after engineering qualification; this sketch supplies neither execution authority nor a new score/selection rule.

Credit the retained [Zhu/Ghahramani/Lafferty ICML2003](https://cdn.aaai.org/ICML/2003/ICML03-118.pdf) harmonic methods and [Dörfler/Bullo arXiv1102.2950v1](https://arxiv.org/pdf/1102.2950) §2.1/Lemma2.1/Theorem3.4. Classical diffusion/transport context is retained in [Singer/Wu arXiv1102.0075v1](https://arxiv.org/pdf/1102.0075) §3 Eqs3.1–3.8 and the already scoped resolvent priors. ReNode and L2R remain close reweighting context; their scopes/access limits are unchanged. No new broad search, primary document, author-code read, fit, tuning, held-score access, predictor-capacity or novelty claim is added.
