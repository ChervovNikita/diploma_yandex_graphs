# Proposed theory statement: class imbalance and centered graph energy

**Status:** analytic proposition for discussion, separate from empirical support and from root adoption. No novelty or improvement claim. It uses standard Laplacian spectral bounds and orthogonal projection, with harmonic/Kron and connection-Laplacian ancestry explicitly credited below.

## Proposition

Let L be a symmetric PSD Laplacian of a connected terminal graph on n≥2 vertices, with λ₂(L)>0. It may be the exactly normalized label-free Kron Laplacian specified in the sealed decision. Let t∈{−1,+1}ⁿ encode the two permitted terminal target roles, J=diag(t), d=Σᵢtᵢ=nₐ−nᵦ, and ρ=d/n. Let Q∈Rⁿˣᴹ satisfy the original row and member-column masses, Q1_M=1_n and 1_nᵀQ=(n/M)1_Mᵀ. Write U=1_n1_Mᵀ/M. Then

\[
\operatorname{tr}\big[(Q-U)^\top J L J(Q-U)\big]
\;\ge\;\lambda_2(L)\rho^2\,\|Q-U\|_F^2.
\]

For d≠0, uniform Q is the unique feasible zero-energy assignment. With coefficient γ>0, this graph quadratic has curvature at least γλ₂(L)ρ² on the original row/column tangent space. Positivity is unnecessary for the inequality; positive simplex rows are still required by the responsibility objective and finite solver.

## Proof

Put H=J(Q−U). Column mass implies tᵀH=1_nᵀ(Q−U)=0. For each column h of H, define unit vectors e=1_n/√n and s=t/√n. Since sᵀh=0 and eᵀs=ρ,

\[
|e^\top h|^2=|(P_{s^\perp}e)^\top h|^2
\le(1-\rho^2)\|h\|^2.
\]

Consequently ‖P_{1⊥}h‖²≥ρ²‖h‖². The Laplacian spectral inequality gives hᵀLh≥λ₂(L)‖P_{1⊥}h‖². Sum over member columns and use ‖H‖_F=‖Q−U‖_F. Applying the same argument to feasible perturbations proves the curvature statement. If d≠0, zero energy therefore implies Q=U. ∎

## What this says and what remains empirical

The ambient spectrum of J L J equals that of L, while its intersection with the original mass-constrained tangent space changes. The bound makes the interaction with class imbalance explicit. It is a lower bound, not an exact constrained eigenvalue or a claim that oriented energy is stronger. For comparison, ordinary unsigned energy on the same column-balanced deviations has lower bound λ₂(L)‖Q−U‖²; the oriented bound includes ρ²≤1.

If d=0, class-only rows Qᵢ=u+tᵢv can have zero oriented energy, with Σₘvₘ=0 and |vₘ|<u=1/M. If the graph is disconnected, the connected-graph proposition does not apply: constants v_c on each component need only satisfy Σ_c(nₐ,c−nᵦ,c)v_c=0. The supplied class counts do not establish connectivity or λ₂. No value of this bound is asserted for the current pilot.

The proposition gives no reason that canonical member corrections are smooth, that this prior improves served quality, or that finite eight-step assignments minimize the objective. The actual graph-free control also changes the step denominator by removing 2γ. Neither a regularizer identity nor this conditional bound isolates empirical graph usefulness. No experiment, solver change or hyperparameter is proposed here.

## Attribution

- **Dirichlet/Harmonic:** Zhu, Ghahramani & Lafferty, ICML 2003, §2 Eqs2–5, [official primary](https://cdn.aaai.org/ICML/2003/ICML03-118.pdf). Their harmonic extension supplies the classical graph-energy context; the member-column constraint here is different.
- **Kron:** Dörfler & Bullo, arXiv:1102.2950v1, §2.1 Eq2.1/Lemma2.1 and Theorem3.4, [primary](https://arxiv.org/pdf/1102.2950). Their Schur complement and edge/path theorem supply the reduced Laplacian. The disconnected/singleton conventions and the inequality above are explicit derivations in our notes.
- **Transport:** Singer & Wu, arXiv:1102.0075v1, §3 Eqs3.1–3.4, [primary](https://arxiv.org/pdf/1102.0075). Their weighted orthogonal edge maps give transport ancestry; tᵢtⱼI_M is our elementary pure-gauge specialization.
- **Signed graphs:** Kunegis et al., SDM 2010, DOI10.1137/1.9781611972801.49 is identified bibliographically. Its body was unavailable (403). It cannot supply a completed close-prior method comparison or novelty clearance.

The argument above is a proved conditional statement using standard machinery. Its role as a methodological contribution remains unassessed against close priors, including graph meta-reweighting and class-oriented allocation. Exact source bytes, selected passages and the limits of those comparisons are retained in PRIMARY_CITATIONS.json and the sealed source packet.
