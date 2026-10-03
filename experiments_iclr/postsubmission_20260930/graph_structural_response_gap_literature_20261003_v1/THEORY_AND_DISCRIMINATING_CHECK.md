# Exact assumptions and a structural-response discriminator

3 October 2026. Analytic derivation. Standard binomial identities and polynomial interpolation are used without novelty claims. A separately specified and root-authorized local stdlib/Fraction check verifies only the two-node equations at fixed rational probe points; no model code or predictive experiment is involved.

## 1. Scalar feature gates versus order gates

Let a fixed shared scalar polynomial be `g(P)=Σ_{k=0}^K c_k P^k`. Scalar member input and output gates multiply to `a_m`. A linear member then has coefficients

\[
\gamma_{mk}=a_m c_k.
\]

The member-by-order coefficient matrix has rank at most one. Private order gates instead give `γ_mk=a_mk c_k`; if the shared coefficients at the relevant orders are nonzero, rows need not be proportional. They require O(M(K+1)) private scalars. Multiplicative gates cannot activate a zero shared coefficient, and rescaling shared/private factors changes parameter gauges and decay effects.

This is restricted to scalar linear gates, a scalar common filter, fixed feature input and readout. Channel-specific shared filters, nonlinear feature maps, LayerNorm and nonlinear BE paths can already have nonproportional functional responses. Parameter-row rank is not response rank on the actual bank, error complementarity, or a novelty result. A full bank costs O(N(K+1)H) storage and K graph applications; dense/private member work persists.

## 2. Normalized lazy-walk family and graph realization

For a nonnegative weighted adjacency A with positive row-degree matrix D, let `P=D^{-1}A`. Define

\[
P_t=(1-t)P+tI,\qquad 0\leq t\leq1.
\]

It is row stochastic and is the normalized adjacency of

\[
A_t=(1-t)A+tD,
\]

whose degree matrix remains D. For undirected A, the symmetric normalized operator has the same identity: `D^{-1/2}A_tD^{-1/2}=(1−t)D^{-1/2}AD^{-1/2}+tI`. This construction reallocates edge weight to self-loops; it is not a generic ordinary deletion mask followed by renormalization. Holding the original loop weights fixed and uniformly attenuating off-diagonal edges generally does not yield one common t across irregular nodes.

In the specific two-node witness, an equivalent fixed-loop realization exists. Set both loop weights to one and the symmetric nonloop edge weight to w. Then

\[
P(w)=\frac{1}{1+w}\begin{pmatrix}1&w\\w&1\end{pmatrix}.
\]

The native w=1 operator is `P_0=½[[1,1],[1,1]]`. Equating `P(w)` with `(1−t)P_0+tI` gives

\[
w(t)=\frac{1-t}{1+t},\qquad
t=\frac{1-w}{1+w}.
\]

If ordinary attenuation uses `w=1−ρ`, then `t=ρ/(2−ρ)`. Complete deletion agrees at the endpoint, but linear interpolation in normalized P and linear attenuation of the edge have different interior parameterizations. For two nodes row and symmetric normalization coincide because the degrees are equal.

## 3. Witness: native feature gradients and endpoint deletion are insufficient

Use scalar node features `X=(1,−1)^T`, a fixed shared scalar s, shared basis `{I,P,P²}`, and member scalar `b_m`:

\[
z_m(P,X)=sX+b_m(P^2-P)X.
\]

Fix binary class logits as `(z_m,0)` at each node, so the first-class probability is `σ(z_m)`. Here P is held fixed when differentiating X; topology and features are independent inputs.

Because `P_0²=P_0`, native logits and their **complete feature Jacobians** are

\[
z_m(P_0,X)=sX,\qquad
\frac{\partial z_m}{\partial X}(P_0,X)=sI.
\]

Every member thus has the same native probabilities and probability feature Jacobian. The selected true-class raw-logit full-X gradients used by the source FoRDE comparator also coincide: either a row of sI for class one, or zero for the fixed second-class logit. Normalizing equal gradients cannot distinguish members. This is not a statement about parameter gradients or all possible DICE hidden representations.

Complete edge deletion with the fixed loops gives `P_1=I`. Again `P_1²−P_1=0`, so every member's native-minus-complete-deletion probability response is zero. Nevertheless `P_t X=tX`, hence

\[
z_m(P_t,X)=\{s+b_m(t^2-t)\}X,
\]

\[
\left.\frac{\partial z_m(P_t,X)}{\partial t}\right|_{t=0}
=-b_mX,\qquad
\left.\frac{\partial\sigma(z_m(P_t,X))}{\partial t}\right|_{t=0}
=-b_m\,\sigma'(sX)\odot X.
\]

For finite s and different b values, both are different predictor-level responses. At the internal point t=1/2, `z_m=(s−b_m/4)X`, so even a finite probability probe distinguishes the members. The analytic three-point check `{0,1/2,1}` requires no labels or model fitting.

By contrast, adding a private hidden code in a final linear readout's nullspace changes no class probability for **any** t. Hidden separation and the witness's observable structural response are therefore distinguishable properties.

This does not guarantee utility. With mean-logit serving and fixed mean coefficient,

\[
\frac1M\sum_m z_m(P,X)=sX+\bar b(P^2-P)X
\]

for every P. Enlarging member contrasts while preserving `bar b` leaves the entire served predictor unchanged. Mean-probability serving is nonlinear and does not satisfy this collapse identity; neither endpoint has a quality guarantee here. No graph perturbation was asserted to preserve the true label.

## 4. K+1 probes: exact linear identification and its limits

Let `p_m(x)=Σ_{k=0}^K γ_mk x^k` and `z_m(P)=p_m(P)Xb`, with the same fixed feature vector/matrix X and readout b across probes. For a symmetric P, let u be a unit eigenvector with eigenvalue λ. The projected response is

\[
u^Tz_m(P_t)=a\,p_m\big(\lambda+(1-\lambda)t\big),
\qquad a=u^TXb.
\]

If `a≠0` and `λ≠1`, K+1 distinct t values give K+1 distinct scalar arguments. Their Vandermonde matrix is nonsingular, so they identify the degree-at-most-K polynomial p_m exactly. Consequently different scalar filters are distinguishable under these visibility assumptions. This is ordinary polynomial interpolation. The assumption can use an accessible contrast of class logits to remove softmax's additive-logit gauge; absolute multiclass logits cannot be identified from probabilities alone.

More generally, without diagonalizing P, the complete fixed-readout linear response `z_m(P_t)` is a vector polynomial in t of degree at most K. K+1 distinct exact output samples determine this **observable curve**. Equality of two such responses at all K+1 points implies equality everywhere on this family, even when their parameter coefficients are not identifiable.

Important limits:

- The λ=1 eigenspace is unchanged: only `p_m(1)` is visible. For an undirected row-normalized operator this is the componentwise constant mode; for symmetric normalization it is the degree-weighted stationary mode, not generally the ordinary constant vector.
- Modes absent from X, annihilated by the readout, or removed by the chosen output projection are unobservable. Hidden/parameter gauges remain possible.
- Finite precision, noise and clustering of sample arguments can make interpolation ill conditioned. The statement is exact algebra, not a numerically qualified recovery procedure.
- A nonlinear head on polynomial tokens generally has a nonpolynomial output response. K+1 samples do not identify all such predictors. A finite set of probes can miss differences between its points.
- This one-dimensional family preserves P's invariant eigenspaces and cannot identify sensitivity to arbitrary edge perturbations, new eigenvectors, direction changes, or every possible graph operator.

## 5. Exact cached-token intervention

Let `H_0` be fixed independently of P for the intervention, and cache `S_j=P^jH_0`, j=0,…,K. Since P and I commute, the binomial theorem gives

\[
P_t^kH_0=\sum_{j=0}^k {k\choose j}(1-t)^j t^{k-j}S_j.
\]

No new sparse graph propagation is required to obtain these tokens. The transform is triangular in order; t=1 collapses every token to H_0. For J probes the direct triangular dense mixing costs O(J N H K²), or O(J B H K²) for B selected rows when the downstream predictor is row local. Dense model forwards and member processing at all probe points still cost work and memory; the bank itself is not free.

For a predictor whose entire graph dependence is this fixed token bank followed by per-node token processing, replacing its tokens by the formula is an **exact mathematical forward intervention**, including nonlinear MLPs, per-row normalization and attention across orders. It does not imply a polynomial output, exact floating-point equivalence, or already-qualified implementation. If H_0 changes during training the bank must be rebuilt; for a current frozen forward it remains a valid intervention.

The identity cannot recover arbitrary normalized edge-deletion operators not polynomial in P, evolving private nonlinear message trajectories, graph-dependent edge scores, omitted graph positional/context inputs, or new propagation on hidden states. Cross-node normalization or processing may also defeat selected-row evaluation. A claimed cache implementation must bind all graph-dependent inputs and verify its source/forward contract before a numerical study.

For general differential graph probes,

\[
\delta(P^kH_0)=\sum_{r=0}^{k-1}P^r\,\delta P\,P^{k-1-r}H_0.
\]

If P is normalized from A, its degree-normalization derivative is part of `δP`. Differentiating raw A while silently fixing degrees is a different intervention. The lazy-walk family avoids that ambiguity by declaring P_t and its graph realization explicitly.

## 6. Disposition

Use the witness and the hidden-nullspace control as a source-level discrimination requirement; qualify any eventual implementation separately. Compare structural response to source FoRDE feature response without treating either as a universal diversity measure. Spectral augmentation, filter banks, functional disagreement and cached graph evidence all have established prior. This exact measurement boundary is useful independently of a new learner, but no successful learner, novel regularizer, or ensemble quality improvement follows from it.
