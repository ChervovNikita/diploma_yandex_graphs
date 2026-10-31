# Round21: projected private-factor curvature

2 October 2026. Mathematical/literature assessment only. All work is confined to this new repository folder and read-only sources under `postsubmission_research_20260930`. No scientific data, current outcome, label pack, checkpoint, SSH/GPU operation, study edit or execution occurred. Saved literature was consulted first; no primary was refetched. Exact reused primary scopes/passages are in `READ_SCOPES.json`.

## Decision

**Do not promote a graph-specific extension or build a pilot on this evidence.** Projecting residual-logit curvature into the existing graph-band directions defines a small, falsifiable diagnostic. It does not establish that graph bands add value beyond generic curvature-selected directions. Negative-curvature copy expansion is close SSD ancestry; the proposed graph restriction is an unvalidated choice of search space. A local TRAIN improvement would establish an optimization fact, not held-out ensemble utility or novelty.

## Exact closest priors

| Version-bound primary evidence already saved | Closest established operation | Remaining operational difference |
|---|---|---|
| **Splitting Steepest Descent**, arXiv:1910.02366v2, Sections 2.1–2.2, Eqs. 2–4, Theorem 2.3 / Algorithm 1 | Copies a neuron, separates common motion from centered copy motion, and chooses negative eigenvectors of residual-weighted neuron curvature. Loss follows averaging neuron outputs. | Restricting the curvature search to masked graph-residual VJP directions of private BatchEnsemble factors; four fixed routes rather than selected neuron growth. This restriction is not itself evidence of useful graph specialization. |
| **LoRA-GA**, arXiv:2407.05000v2, Sections 3.2/3.4 | Gradient-SVD initialization chooses compact factor coordinates to approximate the first full-weight update; base compensation preserves the initial effective function. | Graph cotangent filtering and diagonal private-factor coordinates rather than a low-rank gradient-update approximation. Supervised factor initialization is prior. |
| **BGNN**, arXiv:2101.08543v1, Section 3 / Algorithm 1 | Fits additive trees to graph-GNN input-feature updates; one inner step produces a negative graph-dependent loss gradient. | One-time private-predictor expansion rather than repeated feature-gradient/tree co-training. Graph-dependent residual/gradient ensemble learning is prior. |
| **Functional repulsive ensembles**, arXiv:2106.11642v1, Section 2.2 / Eq. 5 | Pulls output-space cotangents back through each member Jacobian transpose. | Curvature-based selection inside a fixed graph-filtered subspace rather than continuous functional repulsion. VJP projection is prior. |
| **BernNet / C&S / TabM**, saved conclusions | Bernstein graph bands, masked TRAIN residual diffusion, efficient private-factor members and supervised collective training. | These supply the graph bank, residual source and model boundary; recombining them with SSD supplies no originality evidence. |

SSD's retrieved v1/v2 HTML sign discrepancy remains preserved: its displayed `-epsilon² lambda_min/2 < 0` under a negative eigenvalue conflicts with its own Eq. 4. Algebraic substitution gives `+epsilon² lambda_min/2 < 0`. Later SSD v3, later BGNN v2, the SSD PDF and native implementations remain uninspected; no wider source-absence claim is made. MORGAN/FAGEL and B³F-GNN retain their prior access/source ambiguities. Manifold-curvature graph experts are not evidence about a loss Hessian.

## The compressed operator and its target

Freeze the shared model, biases and all parameters outside the admitted private slice \(\phi\) at the identical warm function. Stack target-node logits \(z(\phi)\), let \(L(\phi)=\ell(z(\phi))\) be mean TRAIN CE, and define at \(\phi_0\)

\[
s=\nabla_z\ell,\quad g=J^\top s,\quad
F=J^\top WJ\succeq0,\quad
C=\sum_j s_j\nabla_\phi^2z_j,\quad H_L=F+C.
\]

The residual \(s\) and logit Hessian \(W\) are zero outside TRAIN. The residual in \(C\) is **detached at the warm state**: \(C=\nabla^2_\phi\langle s_0,z(\phi)\rangle\), with \(s_0\) fixed. Differentiating the residual as well gives \(H_L\), a different operator. No label outside TRAIN is needed.

With the existing graph/mask convention, write \(h_b=J^\top M_T H_bs\), \(t_b=h_b-g(g^\top h_b)/\|g\|^2\), and \(T=[t_0,t_1,t_2,t_3]\). Since \(\sum_b t_b=0\),

\[
V_G=\operatorname{range}(T)\subseteq g^\perp,\qquad r=\operatorname{rank}(T)\le3.
\]

The graph bank therefore supplies at most three independent search directions, not four independent frequency-specialist mechanisms. For an orthonormal basis \(Q\) of \(V_G\), the relevant small matrices are \(C_G=Q^\top CQ\), \(F_G=Q^\top FQ\), and \(H_G=C_G+F_G\). Using raw band coefficients without their Gram metric mis-scores rescaled/redundant columns: the equivalent unwhitened problem is \(T^\top H_LT\beta=\nu T^\top T\beta\), restricted to nonzero Gram eigenvalues. Whitening the subspace does not make the underlying private-parameter metric gauge invariant.

Selecting the most negative eigenvector of \(C_G\) targets pooled-logit CE. Selecting one of \(H_G\) targets the actual mean-member-CE objective locally. They cannot be interchanged.

## Concrete disproofs of the stronger premise

1. **Negative residual curvature can be counterproductive for training.** A unit direction with \(v^\top Cv=-1\) and \(v^\top Fv=2\) has \(v^\top H_Lv=+1\). Balanced copies improve pooled TRAIN CE at second order but increase mean member CE relative to the same common step. A negative \(C_G\) eigenvalue alone is not a continuation-consistent diversification certificate.

2. **Graph bands can miss an available descent direction.** Choose a unit \(u\in g^\perp\cap V_G^\perp\) with nonzero class-centered \(Ju\), and set a local residual-curvature matrix \(C=-c uu^\top\), where \(c>u^\top Fu\). Then \(Q^\top H_LQ=Q^\top FQ\succeq0\), while \(u^\top H_Lu<0\). The graph search abstains although a functionally active generic negative-curvature direction exists. Smooth logit maps can realize this modification without changing warm logits, \(J\), residuals or graph-band VJPs. This refutes a guarantee based only on the filter/Jacobian identities; realization inside a particular admitted backbone would require a separate construction.

   A concrete factor-model case uses Round19's two-observation logits \(z_1=rs\), \(z_2=\epsilon(r-s)\), labels \((0,1)\), and warm \(r=s=1\). It has a nonnull gradient-orthogonal direction \(t\) with \(t^\top H_Lt=\tfrac34p\epsilon^2-p^3<0\) when \(p=\sigma(1)\) and \(0<\epsilon^2<4p^2/3\). On an all-isolate initializer graph, every cubic band is \(\binom3b I/8\), every \(h_b\) is proportional to \(g\), and \(T=0\). Graph residual directions therefore miss the available factor curvature entirely. This is a constructed degenerate graph case, not a claim about either released study graph.

3. **A restriction cannot beat an exact generic optimum at the same norm.** Rayleigh–Ritz gives

\[
\min_{v\in g^\perp,\|v\|=1}v^\top H_Lv
\ \le\ \lambda_{\min}(H_G),
\]

and the same inequality holds for \(C\). Shared functional-admission constraints preserve the set-inclusion comparison, using infima over the corresponding admissible sets when necessary. Any graph advantage must concern a finite computation budget or an inductive bias for later/generalization behavior; it cannot be a superior exact local TRAIN curvature minimum by construction. A full-space eigensolver is more expensive, so this inequality does not prove that a budget-matched generic search wins.

4. **Graph frequencies need not survive the private Jacobian.** The first-order output motion is transformed by \(JJ^\top\); its commutation with the graph bank is unestablished. Spectral bands of residuals do not establish spectral bands of predictors or matching of functional spread. Neither first-order graph filtering nor an accepted Armijo step demonstrates negative curvature.

## A falsifiable TRAIN-only diagnostic criterion

The following is a mathematical specification, **not an adopted initializer or launch protocol**. Assume smoothness in the inspected neighborhood; keep the nonzero-gradient, empty-subspace and nonduplicate-function abstentions explicit. At ReLU/other activation boundaries, a classical Hessian prediction requires qualification rather than silently modifying the architecture as SSD did with Softplus.

For a unit \(v=Qw\), require a prospectively fixed margin \(v^\top H_Lv<-\kappa\), with \(\kappa>0\) larger than the qualified operator error, and a nonnull TRAIN class-centered JVP. This implies \(v^\top Cv<-\kappa\) because \(F\succeq0\). A nonnull-JVP requirement retains the existing first-order activity contract; it can exclude some genuinely second-order improvements and is not mathematically necessary for every curvature method. No numeric margin is chosen here.

An explicit four-copy diagnostic uses distinct coefficients \(c=(-3,-1,1,3)/\sqrt5\), whose mean is zero and mean square is one. Let \(a_m=\rho c_mv\), use the **same** alpha and common descent endpoint \(\phi_C=\phi_0-\alpha g\), and set \(\phi_m=\phi_C-\alpha a_m\). The spread is one selected contrast direction, not four independent directions. For fixed \(\rho\),

\[
\begin{aligned}
D_{\rm member}&=\tfrac14\sum_mL(\phi_m)-L(\phi_C)
=\tfrac12\alpha^2\rho^2 v^\top H_Lv+o(\alpha^2),\\
D_{\rm pool}&=\ell(\tfrac14\sum_mz(\phi_m))-L(\phi_C)
=\tfrac12\alpha^2\rho^2 v^\top Cv+o(\alpha^2).
\end{aligned}
\]

The finite candidate must actually have both contrasts negative, all members pass the fixed TRAIN safeguard against warm, and actual centered predictions remain nonduplicate. The common endpoint and alpha must agree across the compared candidates; separately accepted alphas revive a first-order common-step explanation. A null/zero-gradient case is an abstention in this diagnostic definition; changing the zero-gradient rule to search for saddle escape would be another operation.

**What would falsify a graph advantage:** a nonnegative \(\lambda_{\min}(H_G)\), failure of the finite contrast, or no advantage over a generic curvature-selected subspace with the same dimension, private metric, starting point, direction-evaluation budget, dose and finite checks. A topology-permuted subspace distinguishes initializer node alignment. Random directions must themselves receive the same curvature selection; comparing curvature-selected graph directions only against unselected random motion confounds graph value with curvature selection. Seeds, rank handling, margins and search budget would need to be fixed before such a separately authorized diagnostic. TRAIN search/evaluation on the same objective is an optimization screen, not independent predictive evidence.

## Implementation limit and recommendation

The projected operator is at most \(3\times3\). It can in principle be formed from second-order products with fixed-residual logits and TRAIN JVPs, without forming the full Hessian. Its costs include the graph bank/VJPs that construct the subspace, curvature products, qualification and every finite check. A first-order AD certificate does not certify those second derivatives, and no runtime or cost advantage has been shown. TRAIN local curvature also does not certify persistence under shared-weight/private-factor Adam and member dropout.

This supports a **small future diagnostic question**, not a defensible new graph-specific method today. The current controls should finish first. If they provide no evidence that correct graph alignment helps, restricted curvature language supplies no reason to prefer graph bands. If they do provide evidence, curvature and a budget-matched generic search can be separately investigated without claiming novelty from their combination. No runner, new threshold, pilot, final-label measurement or study amendment was created.
