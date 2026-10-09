# Conditional Q/K kernel expressiveness

9 October 2026. Local source/literature/theory assessment. Saved kernel proofs and prior conclusions were reused first. No label, feature array, checkpoint, outcome, server, model, training or inference was opened/executed. Canonical research state, running sources, readout seals and protocols remain unchanged.

## Assessment

**The exact split admits a useful narrow claim: it enlarges the family of intrinsic global value-mixing kernels beyond reversible, nonnegative-spectrum Gram walks.** This concerns one member, head and global layer, conditioned on its node representations. It is not a theorem that tied Polynormer cannot solve directed tasks or realize the complete split predictor. The proof applies standard Gram-matrix and Markov-chain algebra to a particular compact parameterization. It does not establish a new attention primitive, theorem priority, useful learned circulation, causal mediation, ensemble advantage or generalization.

A defensible statement is:

> At a fixed global-layer feature state, tied sigmoid Q/K and the active positive channel-reweighting control generate reversible positive Gram mixing cores with spectrum in [0,1]. The exact pre-sigmoid role-scale split contains the tied case and can generate positive cores outside that class, using one projected feature matrix. This makes nonreversible direct global mixing available; whether the trained predictor uses it beneficially remains an empirical question under the frozen pilot.

The strict containment is relative to the tied parameterization. The active control has the same reversibility/PSD restriction; no containment between its complete fixed-input parameter family and the split family is asserted.

Calling this **conditional kernel expressiveness** is more accurate than calling it whole-network relational expressiveness. Even output separation for one native layer does not follow merely from unequal kernels: the actual value matrix is constrained by the value projection, and different kernels can agree on that value subspace.

## Exact operator and proposition

Suppress layer/member/head indices. The actual adapter uses

\[
a_i=W(x_i\odot r),\quad
q_i=\sigma(a_i\odot(s+\delta)+b),\quad
k_i=\sigma(a_i\odot(s-\delta)+b),\quad
C_{ij}=q_i^Tk_j,\quad P_{ij}=C_{ij}/d_i,\ d_i=\sum_j C_{ij}.
\]

Bias is outside the scales. Delta=0 exactly recovers the tied factorized map for any old r/s/W/b state. The source does not implement `sigmoid((a+b)*(s±delta))`. Output scales are node independent within a member; queries and keys retain one common projection and input factor. The split does not supply arbitrary independently rotated Q/K projections, a support mask or a free N×N matrix. Rank remains at most the head width; here512. These statements concern exact real arithmetic and finite preactivations.

**Proposition 1 — tied/active core restriction.** Let U contain finite sigmoid features, so every entry is strictly positive. In the tied case C=UUᵀ. In the active reversible control, C=U diag(exp γ) Uᵀ. Both are positive symmetric PSD matrices. With D=diag(C1),

\[
\pi_i=d_i/\sum_jd_j,\quad \pi_iP_{ij}=\pi_jP_{ji},\qquad
D^{1/2}PD^{-1/2}=D^{-1/2}CD^{-1/2}\succeq0.
\]

Thus P is a positive row-stochastic reversible kernel, with unique stationary mass π and real eigenvalues in [0,1]. P generally **is not symmetric**; unequal d_i make P_ij unequal to P_ji. It is still query dependent. Raw normalized edge asymmetry is therefore not evidence of a new directional capacity.

**Proposition 2 — strict available extension.** The split contains every tied parameter setting at delta=0. The saved finite three-node construction gives q_i=(u_i,1/2), k_i=(1−u_i,1/2), u=(1/4,1/2,3/4). It is realized by first projection coordinates (−log3,0,log3), first s=0, delta=1 and bias0. Its affinity matrix is

\[
C=\begin{pmatrix}7/16&3/8&5/16\\5/8&1/2&3/8\\13/16&5/8&7/16\end{pmatrix},\qquad
\frac{C_{12}C_{23}C_{31}}{C_{21}C_{32}C_{13}}=\frac{117}{125}\ne1.
\]

Row denominators cancel around the cycle, so its normalized P is nonreversible. The saved negative-mode calculation gives spectrum {1,0,−2/45}. To embed this construction in width512, replace the single constant coordinate by511 constants 1/(2√511), supplied by zero projection rows and finite shared biases. Their squared contributions still sum to1/4. This is an admissible operator construction, not an assertion about any graph's actual hidden states or selected weights.

There is also a local possibility at the tied start. For q=k=u at delta=0 and perturbation delta=εh,

\[
e_i=a_i\odot\sigma'(a_i\odot s+b)\odot h,\qquad
C_{ij}(\epsilon)=u_i^Tu_j+\epsilon A_{ij}+O(\epsilon^2),\quad
A_{ij}=e_i^Tu_j-u_i^Te_j=-A_{ji}.
\]

The ordinary s tangent is symmetric; the delta tangent is antisymmetric. For the same projected three-node features at s=1, the saved proof gives a triangle-log-ratio derivative 2log3/35≠0. Hence arbitrarily small splits can leave reversibility while both s±delta remain positive. This local statement does not say that every split breaks reversibility, that every delta gradient is nonzero, or that a negative eigenvalue appears immediately. The negative-mode witness is a separate finite-parameter result.

Important degeneracies are retained. One feature channel gives C_ij=q_i k_j and identical normalized rows, hence a reversible rank-one core. Every positive two-state Markov kernel is reversible. Nonzero delta, Q≠K or raw C asymmetry need not give circulation: a positive affinity can remain diagonally symmetrizable. The strict separation is an existence statement for admissible feature states, not a guarantee for every fixed graph/input.

The full-Q/K reference already contains this compact split as a parameter subfamily: choose its affine maps to match diag(s±delta)Wdiag(r), with the same unchanged bias b. Separate arbitrary affine Q/K maps, positive nonlinear feature maps and normalized linear attention are established methods. Training (s,delta) and training the effective independent scales (s+delta,s−delta) can have different Adam trajectories; function-class algebra alone does not isolate a capacity effect from optimization.

## Scope within the complete Polynormer

The global source performs V=V_lin(X), H=H_lin(X), then

\[
Y=\operatorname{ReLU}\!\left(W_{out}\left[\operatorname{LN}(P(X)V(X))\odot(H(X)+\beta)\right]+b_{out}\right),
\]

followed by dropout, which is disabled for eval. The core P is only the value aggregation before LayerNorm/gating/output maps. The source has no global `(1−beta)P+beta I` restart-walk formula.

| Exact native feature | Consequence for the theory boundary |
|---|---|
| Local GATConv on the observed graph, plus learned local linear residuals | Local coefficients are not a positive Gram construction and have no corresponding detailed-balance guarantee. Directed responses are available before the global stage, including on an undirected support. GATv1's separate static-ranking limitation does not turn it into a reversible Gram core. |
| Beta parameters have shape layers×(heads·channels), then `unsqueeze(0)` | **Beta itself is channelwise and broadcast across nodes**, not an independently learned per-node parameter. With the current beta=−1 recipe it is sigmoid transformed. The effective global `H_i(X)+beta` multiplier is node dependent and can be signed because H_lin has no ReLU. |
| Local `x=(1−beta)*LN(h*x)+beta*x` with h=ReLU(H_lin(x)) | There are nodewise multiplicative local responses, a residual term and signed LayerNorm outputs. This is a feature transformation, not a single stochastic node-mixing matrix. |
| Learned value/output maps and affine LayerNorm | Value/channel transformations can be signed and mix channels. The full feature Jacobian need not inherit the kernel's positivity, PSD or reversibility. |
| ReLU, local/global state evolution and two different global layers | Each layer computes a different conditional P at its actual input. Products of reversible kernels need not be reversible; under a common stationary measure their adjoint reverses the order, so noncommuting factors already remove self-adjointness. No PSD constraint on the whole nonlinear predictor follows. |
| Private member hidden states, heads and served mean of class probabilities | Members can differ with tied Q/K. Their served probability pool is not an average transition kernel and has no single kernel stationary measure supplied by this proposition. |
| Selected local mode | A member can serve its local head without executing either global layer. Its intrinsic global-cycle diagnostic is **not applicable**, not zero; forcing a global probe would alter the served-state question. |

Even a hypothetical product of fixed cores is a different object from the real network: X changes, values change, post-aggregation gates/nonlinearities act, and differentiating P(X)V(X) includes derivatives of P. The witness separates intrinsic mixing kernels, not classifier functions or complete input/output maps. It supports no claim that tied Polynormer cannot express orientation, heterophily, signed effects or cyclic label patterns.

## A cheap intrinsic diagnostic for a separately reviewed postclosure readout

Use a **sampled triangle cycle affinity** from the actual Q/K features. For distinct sampled nodes i,j,k compute only six inner products and

\[
r_{ij}=\log C_{ij}-\log C_{ji},\qquad
R_{ijk}=r_{ij}+r_{jk}+r_{ki}
=\log\frac{P_{ij}P_{jk}P_{ki}}{P_{ji}P_{kj}P_{ik}}.
\]

The full row normalizers cancel. There is no stationary-distribution estimation, degree approximation, dense N×N kernel, labels or classifier logits. In a strictly positive complete core, all triangle residuals vanish iff the kernel is reversible: zero triangle circulation makes the antisymmetric log-ratio field a node potential, which yields a positive diagonal symmetrization. A sampled nonzero residual is a witness on that sampled cycle; sampled zeros do not certify all cycles or reversibility.

**Prospective inactive budget:** sample 128 distinct row indices uniformly from all 11701 observed graph nodes, then 1024 distinct unordered triples and 1024 distinct unordered pairs from those indices; fixed seed 20261009 and the same samples across every operator/family/seed/member/layer/head. Store sorted row indices, orient each triple as i<j<k for the cycle i→j→k→i, and each pair as i<j. This orientation is arbitrary and fixed; signed residuals do not define a task-relevant direction. Generate and seal the exact index list before comparative outcome access. Sampling uses neither labels, valid masks, correctness, feature similarity, learned scales nor outcomes. This plan is separate from the frozen quality gate and is not adopted here.

Capture only those128 Q/K rows **after the exact feature map used for attention**, before the KV contractions, during the already required selected-state eval/no_grad member call. The actual global core uses all nodes; sampled pair scores are its entries, not a separately normalized sampled graph. A future private observer may copy/detach these rows without changing Q/K/X, consuming model RNG, restoring a different stage or mutating the serving predictor. Native/full modes need the actual sigmoid outputs; split/active modes need the actual pair outputs. A generic linear-module hook alone is insufficient for the split, whose implementation uses functional linear projection and manual scales.

The diagnostic adds **zero training, backward, optimizer or model-member inference calls**. It may accompany the108-call readout only through a separately reviewed observer that uses those existing calls. There is no observer installation, readout-source edit or extra forward in this packet. Preserve each member's actual selected mode; record local-only members/layers as not executed. The frozen full36 closure/trusted-state admission remains mandatory before observing any selected features.

Evaluate score dots/log differences in float64 from captured float32 features, in chunks of64 triangles/pairs. This approximates the real-arithmetic implicit kernel of the captured features; it does not establish bitwise equivalence to an explicitly materialized float32 attention matrix. A nonzero cycle is a mathematical witness in exact arithmetic. Floating-point residuals are observations, and a claimed resolved witness additionally requires a prospectively reviewed arithmetic-error policy; this packet supplies neither a certification threshold nor a new predictive gate. Retain all fixed sample slots and nonfinite/zero-score counts. Do not add epsilons, floors, clips, replace bad samples or rerun until residuals look favorable. If any required score is nonpositive/nonfinite, the aggregate cycle statistic is unavailable, with its failure/counts preserved.

Report RMS R, mean |R|, max |R|, signed min/max, sample counts and each actual member/layer/head. They are intrinsic magnitudes conditional on the sampled graph/state, not node-iid tests or evidence of predictive competence. Native tied and active reversible serve as mathematical/null conformance references. A material residual there calls for source/capture/precision review; it never authorizes a tiny-float chase or recipe change. No new success cutoff is added to PILOT_PROTOCOL.

An optional second obstruction uses sampled pairs:

\[
L_{ij}=\log C_{ii}+\log C_{jj}-\log C_{ij}-\log C_{ji}.
\]

For a tied/positive-weighted Gram kernel, Cauchy–Schwarz gives L_ij≥0. A genuinely negative value witnesses departure from that Gram constraint. It does **not** by itself prove that the full large kernel has a negative eigenvalue, and it is not a test of reversibility. Cycle and minor diagnostics should therefore be reported separately.

Structural cost is O((6T+4B)·512) dot-product work per active layer/head for T=B=1024, with only sampled rows and bounded chunk buffers. Captured Q/K rows require at most2×128×512×4=524288bytes per active layer/head; at most216 layer/head slots occur under108 member calls and two global layers, giving at most108MiB of row-transfer volume. These are prospective bounds, not measured runtime/memory claims. Charge actual capture, transfer, arithmetic, storage and skipped/failure slots. No optimizer-seed/member independence is inferred from the many cycles.

## Interpretation alongside the frozen comparison

| Predictive readout / intrinsic observation | Permitted conclusion |
|---|---|
| Split passes both fixed co-primary gates and has a resolved nonzero cycle | The trained split uses an available nonreversible intrinsic core and the operator package improves this selected-development comparison. It does not prove circulation caused the gain. |
| Cycle is nonzero but predictive gates fail | The capacity was used without the required utility. It cannot rescue a failed pilot. |
| Predictive gates pass but sampled cycles are zero/undefined or global mode is unserved | Operator-package utility can remain a result; learned nonreversible transport on these probes is unsupported. There may be gauge-equivalent changes, unprobed cycles, optimization effects or a local-selected predictor. |
| Native/active core has a material cycle residual | Review capture/source/arithmetic provenance; their exact intrinsic cores remain Gram/reversible for arbitrary hidden representations. Local directionality is not an explanation for a residual inside their global core. |

Different trained arms also have different earlier representations, values, gates and optimization histories. One observational cycle measure cannot establish mediation. A causal attribution to nonreversibility would need a separately frozen intervention/comparison; this assessment proposes no extra fit, post hoc ablation, credit-policy change or added gate.

## Published ancestry and reading accounting

The saved prior assessment already identifies the closest published collision: **Polynormer**, ICLR2024, §3.2 Eq.6 and its printed global-attention algorithm use separate Q/K maps and positive normalized linear attention. The tied setting is a native implementation choice. **Katharopoulos et al., Transformers are RNNs**, ICML2020, establish separate Q/K positive feature maps and associative linear-time attention. **BatchEnsemble**, ICLR2020, supplies the private input/output feature-scale primitive. **GAT**, ICLR2018, is the existing native local-attention ancestry. The saved (IA)³/feature-scale and QKNorm findings provide adjacent modulation/normalization ancestry, not an equivalence between their placements and this sigmoid split.

Detailed balance, diagonal symmetrization, the Kolmogorov cycle criterion and Gram PSD/similarity are classical algebra. Kelly's *Reversibility and Stochastic Networks* (1979) is a standard background reference. Log cycle affinities also have established network/Markov ancestry (Schnakenberg, *Network theory of microscopic and macroscopic behavior of master equation systems*, Rev. Mod. Phys.48,571,1976); the node-potential versus cycle-flow distinction is related to combinatorial Hodge formulations (Jiang, Lim, Yao and Ye, *Statistical Ranking and Combinatorial Hodge Theory*,2011). These background identities are identified from established knowledge; no new primary passage/publication audit for them was performed here. The saved published method scopes, not an invented literature-absence claim, ground the architectural ancestry.

Saved method links: [Polynormer §3.2](https://arxiv.org/html/2403.01232v2#S3.SS2), [Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html), and [BatchEnsemble](https://arxiv.org/abs/2002.06715v2). The source bindings preserve their saved read scopes and distinguish them from the background references above.

The site-specific bias-correct tangent/witness is useful source qualification, not a newly cleared theory primitive. Reuse credit: saved reversibility note, saved negative-mode addendum and saved closest-prior conclusions; targeted native local/global/beta and exact adapter source inspection. New full or scoped primary papers read:0. Remote/literature retrievals:0. Numerical fixtures:0. Labels/checkpoints/arrays/outcomes:0. No global theorem-priority or novelty clearance follows.
