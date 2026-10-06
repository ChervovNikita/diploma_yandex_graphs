# Public paths and target-oriented responsibility: scoped decision

**Decision.** Label-free Dirichlet/Kron reduction is a mathematically coherent way to retain public paths between the existing supervised responsibility rows. Centered target orientation is a separate, coherent change to the responsibility prior. Both are attributed optimization hypotheses; neither establishes improvement, novelty, a new predictor class, or additional tangent rank. Keep the current pilot and its score gates unchanged. This packet specifies boundaries and literature credit only; it contains no implementation, experiment recipe, grid, fit, or score.

## 1. What motivates the question

The pinned native `_sparse_pairs` first removes every edge leaving inner S, then restricts to the two classes of each pair. It therefore discards paths through other public nodes. Root-supplied audited counts are V=24,492, S=2,449 (10.0%); class counts are 631/906/578/231/103. Pair support ranges from 334 rows (1.36% of V) to 1,537 (6.28%). These establish sparse **row support**, not low connectivity, isolation, or a weak graph prior. The full ten-pair connectivity census remains pending.

Root-supplied training summaries say finite response differs from first-order utility and Q differs from uniform. Close aggregate losses for graph-free, permuted and stop-Q controls establish neither pointwise equality nor graph inactivity. No underlying label, checkpoint, prediction, scientific result file, or held-score payload was read for this decision.

## 2. Exact label-free public-path energy

Use the original symmetric, nonnegative, off-diagonal public adjacency A to form the loopless combinatorial L₀=D−A **for responsibility regularization**. Preserve the classifier graph, its self-loops and inference. For pair (a,b), terminals are T=Sₐ∪Sᵦ and interiors are U=V\T. Interior labels provide no boundary values: this includes other S classes and W/R/A/VALID/TEST. The original permitted S labels still define the existing probe and cost.

Work separately in each public connected component containing terminals. If its interior is nonempty,

\[
\Lambda=L_{TT}-L_{TU}L_{UU}^{-1}L_{UT},\qquad
X_U=-L_{UU}^{-1}L_{UT}Q_T,
\]

\[
\min_{X_U}\operatorname{tr}([Q_T;X_U]^\top L_0[Q_T;X_U])
=\operatorname{tr}(Q_T^\top\Lambda Q_T).
\]

Here L blocks are from that component. The grounded L_UU is invertible when the connected component has at least one terminal. The conventions are explicit:

- **No terminals:** omit the component from responsibility energy. Its minimum is zero and its constant interior field is nonunique. Introduce no label, boundary value, or mass there.
- **One terminal:** the reduced Laplacian is zero. Retain that terminal's cost, entropy and participation in global terminal balance; topology supplies no coupling.
- **At least two terminals:** Kron preserves their connectivity. A reduced edge exists precisely when its endpoints have an original path whose intermediate vertices are all eliminated interiors. This does not make every connected terminal component a clique; the boundary neighbors of an interior connected component form a clique.
- **Empty interior:** use L_TT directly. Introduce no cross-component connection, ridge, pseudoinverse, or teleportation convention.

Retain every terminal row. Set s=1+maxᵢΛᵢᵢ and L_eff=Λ/s across the pair. It is a symmetric PSD Laplacian, L_eff 1=0, and ‖L_eff‖₂≤2. Retain the objective

\[
\langle a,Q\rangle+\tau\sum_{i,m}Q_{im}\log(MQ_{im})
+\frac\gamma2\operatorname{tr}(Q^\top L_{\rm eff}Q),
\quad Q_{im}>0,\quad Q\mathbf1_M=\mathbf1_{|T|},\quad
\mathbf1_{|T|}^\top Q=(|T|/M)\mathbf1_M^\top.
\]

Mass constraints concern terminals only. Unsigned harmonic extension is nonnegative row-stochastic mixing of terminal Q, hence preserves positive simplex rows in supported interiors. It does not impose balanced interior member columns. The classifier does not receive this extension.

Dense fill-in and linear solves can be expensive; resource feasibility is unmeasured. Approximate solves cannot silently inherit exact symmetry, PSD, mass, or norm guarantees. The existing eight differentiated assignment steps remain a finite training map, not an exact optimizer.

**Precisely defined alternative, not a selected recipe:** R_ρ=(I+ρL₀)⁻¹, ρ>0, is a symmetric nonnegative row-stochastic full-graph resolvent. Set W=offdiag(R_ρ[T,T]), form diag(W1)−W, and normalize as above. Restrict **after** diffusion. R_ρ[T,T] itself is generally substochastic, so multiplying Q by it does not preserve mass. Components remain uncoupled. This is classical diffusion; no ρ or new comparison is selected here.

## 3. Centered target orientation

Write d=zₐ−zᵦ, tᵢ=+1 for target a and −1 for target b, and bᵢ=softplus(−tᵢd). Then ∂bᵢ/∂d=−tᵢσ(−tᵢd). For any nonzero finite Δd, sign(Δbᵢ)=−tᵢ sign(Δd). A smooth canonical correction need not yield smooth target-specific usefulness. Conversely, monotonicity does not prove that canonical member corrections, response costs, or responsibilities are graph-smooth. Node/member susceptibility, private Jacobians, nonlinear finite updates, per-pair RMS normalization and balance remain consequential.

Let u=1/M, U have every row u, ΔQ=Q−U, and J=diag(t). Define h=JΔQ, B=J L_eff J and

\[
E_{\rm oriented}(Q)=\frac\gamma2\operatorname{tr}(\Delta Q^\top B\Delta Q),
\qquad \nabla_QE_{\rm oriented}=\gamma B(Q-U).
\]

B is PSD and has the same spectral norm as L_eff because J is orthogonal. **Centering is necessary:** B1 generally differs from zero, so tr(QᵀBQ) would penalize uniform Q. The original row/column tangent projection and norm-based positivity/curvature bounds can be retained in real arithmetic. No floating-point or implementation equivalence is certified.

This is an O(1) coordinate transport of responsibility deviations. Equivalently its edge energy is Σᵢ<ⱼwᵢⱼ‖ΔQᵢ−tᵢtⱼΔQⱼ‖². Edge transports tᵢtⱼI_M are a pure gauge: their signs multiply to +1 around each cycle. This situates the algebra within signed graphs and connection/transport Laplacians. It is not a theorem transporting the actual private-gradient covectors, which also contain susceptibility and Jacobians, nor a general heterophily principle.

### The energy gauge does not equate the whole constrained problem

Under Qᵢ=u+tᵢhᵢ, constraints and other terms become

\[
\sum_mh_{im}=0,\qquad\sum_it_ih_{im}=0,\qquad u+t_ih_{im}>0,
\]

with cost ⟨Ja,h⟩ plus a constant and entropy Σ(u+tᵢhᵢₘ)log(M(u+tᵢhᵢₘ)). Thus balance becomes label-weighted and the domain depends on orientation. Keeping the original Q cost, entropy and feasible set changes the prior. For M=4, reflection Q→2u−Q can be negative and is not a simplex automorphism. For M=2 the row reflection resembles a member swap, but global balance still becomes label-weighted. A global pair-name swap flips both t and d and leaves the energy unchanged; no numerical class ordering is preferred.

If the reduced terminal graph is connected, zero energy requires hᵢ=v. Member-column balance then requires (nₐ−nᵦ)v=0. Unequal counts force v=0; balanced counts can permit class-only assignments u±v with Σv=0 and |vₘ|<u. For M=4 exact opposite agreement bounds each assignment below 1/2. All supplied pair class counts differ, but terminal connectivity has not been established. With several components, only Σ_c(nₐ,c−nᵦ,c)v_c=0 is required, so component contributions can cancel. These describe the feasible zero modes, not usefulness at positive energy.

Apply orientation **after label-free Kron reduction**. Harmonic interiors then carry real centered canonical fields, not interior probability assignments. Unknown public-node target signs or labels are unnecessary.

## 4. Attribution and interpretation limits

| Closest inspected source | Credit and scope | Boundary relative to this hypothesis |
|---|---|---|
| Zhu, Ghahramani & Lafferty, ICML 2003, Gaussian fields/harmonic functions | §2 Eqs1–5; §3.1–3.2 Eqs6–7. Primary Eq5 visually verified. | Supplies harmonic Dirichlet extension; its label prediction/class-mass construction is not the member-column constraint above. |
| Dörfler & Bullo, arXiv:1102.2950, Kron reduction | §2.1 Eq2.1/Lemma2.1 and Theorem3.4. | Supplies Schur-complement closure and exact edge/path support; disconnected and singleton conventions above are explicitly derived extensions. |
| Chen et al., ReNode, arXiv:2110.04099 / NeurIPS 2021 | §§2.2–2.4 Eqs1–6: full-PPR influence, labeled-seed conflict, ranked cosine node weights, weighted CE. | Close topology-aware node-reweighting prior; it is not meta-learning and does not optimize balanced node×member Q. |
| Chen et al., L2R-GNN, arXiv:2312.12475 | Method Eqs1–12: graph-sample weights, decorrelation and validation meta-update through a learner step. | Close graph meta-reweighting prior; graph-level OOD weights/validation decorrelation differ from private S correction, fixed public-path energy and shared R mean/pool credit. |
| Singer & Wu, arXiv:1102.0075v1 | §3 Eqs3.1–3.8: weighted orthogonal edge maps and transported vector averaging. | Primary transport ancestry; our endpoint-sign congruence is elementary own algebra, not an imported manifold convergence guarantee. |
| Kunegis et al., SDM 2010, DOI10.1137/1.9781611972801.49 | Bibliographic identity only; publisher PDF returned 403. | Signed-graph attribution is limited to the identified family. Its body/implementation was not inspected. |

L2R's accepted AAAI 2024 locator failed; the saved preprint title is “Learning to Reweight for Graph Neural Network,” while metadata identifies the accepted “Learning to Reweight for Generalizable Graph Neural Network.” Equivalence is unverified. Its printed cluster-indicator definition also conflicts with the stated distinct-cluster decorrelation intent; no resolved implementation or guarantee is asserted. Ren/Meta-Weight-Net and the related differentiable teaching priors remain credited through the already sealed training-map decision and its primary scopes, without new full-paper reads here.

The current assignment map uses cost+τ(log(MQ)+1)+γLQ and step denominator 1/rate+2·ratio_bound+2τ·reciprocal_bound+2γ. Graph-free sets γ=0 in both places. Even when L=0 or LQ=0, the live step retains 2γ. Consequently live versus graph-free is not exclusively a graph-energy contrast. This source interpretation caveat does not authorize a new control or alter the pilot.

Topology attribution must be distinguished conceptually from class-only orientation and class-preserving permuted topology. A class-preserving permutation P commutes with J, retaining orientation and mass budgets while breaking cost/topology alignment. Graph-free alone cannot attribute topology. No control grid is instantiated.

Earlier resolvent/filter predictor-capacity closure does not decide whether this change to responsibility optimization/support geometry helps the served training map. Conversely, these algebraic possibilities do not reopen a closed capacity claim. The full graph is already public to the classifier; labels added to the prior are only permitted terminal role information. Literature search absence is not novelty evidence. This packet adopts no paper quality claims or empirical results and launches nothing.

## Audit

[READ_SCOPES.json](READ_SCOPES.json), [RETRIEVAL_LEDGER.json](RETRIEVAL_LEDGER.json) and [SOURCE_BINDINGS.json](SOURCE_BINDINGS.json) delimit the evidence. Five primary PDFs received bounded method reads; no full paper or author code was read. Thirteen HTTP requests produced five PDFs, four metadata responses and four failures. All retained packet files are byte-pinned by MANIFEST.json and SEAL.json.
