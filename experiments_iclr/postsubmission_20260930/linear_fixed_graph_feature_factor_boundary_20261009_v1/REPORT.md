# The narrow boundary between feature factors and graph operators

9 October 2026. Retain a **linear common-filter diagnostic null**, with established ancestry. Close any broader assertion that feature-only BatchEnsemble cannot change a nonlinear GNN's graph response. This note supplies no new method, quality advantage, acceptance verdict, implementation or experiment.

## Exact statement and proof

Let X be an n×d0 feature matrix, A a fixed n×n node propagation matrix shared by every member, and L a fixed depth. Assume a bias-free strictly linear stack, with no residual/self branch, activation, feature-dependent normalization, attention, dropout or member-dependent node operation. In row-feature notation let

`H_(m,0)=X`, `H_(m,l+1)=A H_(m,l) B_(m,l)`.

A feature-only BE layer has `B_(m,l)=diag(r_(m,l)) W_l diag(s_(m,l))`; the statement also holds for arbitrary independent feature matrices B. Then

`H_(m,L)=A^L X P_m`, where `P_m=B_(m,0) ... B_(m,L−1)`.

**Proof.** The left-node and right-feature actions commute as operations: `(A H)B=A(HB)` by associativity, including rectangular feature dimensions. Inductively, `A[A^l X(B0...B_(l−1))]B_l=A^(l+1)X(B0...B_l)`. This does not assert `AB=BA`, or commutation of W with feature diagonals. The product P_m need not be a single rank-one BE modulation of `W0...W_(L−1)`; intermediate diagonals remain a constrained, potentially rich feature-map family.

Equivalently, the member input-output operator is the separable Kronecker product `P_m^T⊗A^L`. Its source-node u to target-node v row-feature map is `(A^L)_(v,u) P_m` (the standard column-vector Jacobian block is its transpose): every node pair uses the same member feature map multiplied by the fixed path coefficient. Feature factors cannot change the support, relative path coefficients, edge weights or depth of this node filter. They can change output feature directions and classifier quality. The same collapse holds for a prescribed sequence of member-independent A_l, replacing A^L by `A_(L−1)...A_0`; a common Fourier eigenbasis then needs additional assumptions.

For real symmetric A, write `A=U diag(lambda_i) U^T`. At frequency i,

`[U^T H_(m,L)]_i=lambda_i^L [U^T X]_i P_m`.

The transfer matrix is `lambda_i^L P_m`, so the member cannot choose an additional independent graph-frequency response f_m(lambda) for all inputs. A zero-eigenvalue component is lost for every member at positive depth. **This does not imply identical observed spectra:** different P_m can suppress or mix feature channels whose energy is concentrated at different frequencies on a particular X. It also supplies no universal oversmoothing, expressivity, optimization or accuracy theorem. Such claims need separate assumptions on A, depth, weights and readout.

A member-specific fixed node operator can escape this family. With one scalar feature, the null permits only scalar multiples of A^L; a different node matrix S not proportional to A^L cannot equal `A^L X p` for every X. This is an all-input operator distinction. Fitting a feature transform to one finite observed X, or observing a raw operator difference annihilated by X, does not establish predictor separation.

## Why the broad version fails

| Operation | Symbolic counterexample or exact boundary |
|---|---|
| Nonlinearity | Take `A=[[3/4,1/4],[1/4,3/4]]`, `X=[1,−1]^T` and scalar feature weights +1/−1. `AX=[1/2,−1/2]^T`; ReLU gives `[1/2,0]^T` and `[0,1/2]^T`. Neither is `AX p` for a scalar p. Within an activation region, the effective Jacobian includes a member/input-specific node mask. The raw A remains fixed; effective graph sensitivity need not remain the common linear filter. Positive diagonal/ReLU homogeneity preserves some special compensations, not this broad collapse. |
| Residual or independent self/neighbour branches | `H'=H B_self,m+A H B_neighbour,m` gives spectral response `B_self,m+lambda B_neighbour,m`. Even scalar `H'=H+b_m A H` has `1+b_m lambda`, changing relative frequency response when b_m changes. Deeper linear branches yield matrix-valued graph polynomials. If every branch is tied to the same B and fixed node coefficients, they can collapse to a new common node filter; branches alone are not an unconditional separation theorem. |
| Normalization | Fixed graph degree normalization merely defines the common A and does **not** invalidate the proof. Fixed bias-free channel scaling can be absorbed into B. Data-dependent LayerNorm/GraphNorm/training BatchNorm generally cannot: for nonconstant x and epsilon=0, `LN(2x)=LN(x)`, contradicting linear homogeneity. Row means/variances depend on member states and can create node-dependent effects. Shared normalization parameters do not imply shared normalization actions. Shared affine biases already violate the strictly linear premise and introduce separately propagated constant terms. |
| Feature-dependent attention | With scalar projections and shared scorer, let `x=[1,0]^T`, `q=k=t_m x`, and row-softmax scores `q k^T`. The first attention row is `[exp(t_m^2)/(exp(t_m^2)+1),1/(exp(t_m^2)+1)]`; it varies with the feature-only scale. No private attention vector is needed. For fixed realized A_m(X), right-feature multiplication still commutes locally; the all-input A is now member/state-dependent. Some attention changes are only temperature changes, and output-scale compensation or row-score offsets can preserve the final function. |

These are analytic witnesses, with no numerical fixture run. They refute an unconditional common-filter statement, not establish that any trained candidate exploits the extra possibilities usefully.

## Implications for the present hypotheses

**Q/K.** The saved native-attention assessment already rejects “shared scorer means shared graph operator”: BE-modulated hidden states enter native local/global attention. The current pre-sigmoid source uses `q=sigmoid(a*(s+delta)+b)`, `k=sigmoid(a*(s−delta)+b)`, with bias outside the role scales. Tied positive features yield a symmetric raw Gram kernel and reversible row-normalized operator; feature changes can still change that operator. Splitting Q/K can leave that restricted kernel family, but asymmetric raw scores do not by themselves prove nonreversibility, usefulness or a whole-model capacity separation. Distinct Q/K and cheap role scales have Polynormer/linear-attention/BE/adapter ancestry. Direct ownership may change optimization even where feature compensation represents the same function.

**Source supply.** The sealed IMDB pilot measures source-removal/restoration BCE, all signed U/D cells, label repairs/harms and actual member/pool competence. Those are conditional evidence-use diagnostics; they do not prove that a member changed edge weights or a graph-frequency filter. A feature map can change reliance on source-specific propagated channels under the same graph operator. The common-filter null therefore motivates careful language, not rejection of useful source specialization. Positive U or distinct source responses do not establish a sharing advantage.

**Geometry.** Factors in an incidence-map generator are outside the feature-only-value-path premise: they can change the actual normalized block graph operator. The saved NSD witness already proves both possibilities: row-sign changes leave every Gram/degree block and deployed operator unchanged, while maps aI versus bI with native +I augmentation can change normalized spectra and message action. Raw private maps are insufficient. At common incoming state H and transformed values V, the exact action difference is `(L_m−L_n)V`; a nonzero operator difference can vanish on V or in the classifier's nullspace. State feedback, residuals, normalization and ELU exclude the linear null from the complete native NSD predictor. Fixed right channel transforms commute with a fixed sheaf operator on the node×stalk axis, but stalk transforms and live member operators require their own algebra.

## Prior collision and retained use

The collapse is already the SGC simplification, explicitly reproduced in saved **SIGN §2.3 Eq.3**. Saved **MIMO-GC §3 Eq.10** gives a fixed adjacency's common feature-channel component amplification; **§4 Eq.11** supplies `sum_k A_k(X) X W_k` and endpoint-dependent matrix alternatives. BE §3.1 supplies the shared slow/private rank-one feature parameterization. NSD supplies matrix transport; BSNN already jointly learns shared network weights with sampled private sheaves and predictive ensembling; BuNN supplies several learned bundles in one joint model; HetSheaf supplies multiple type-conditioned geometry generators over common features. The local commutation identity and private-geometry ancestry cannot support methodological novelty.

Retain only this diagnostic implication: **under the declared strict linear null, feature factors cannot create a member-specific node kernel; in the actual nonlinear model, compare the deployed operator at a common state and its action on common values before attributing any prediction change to operator ownership.** A first-layer common-state comparison can separate private operator parameters from later state feedback. Gauge-equivalent/raw-map-only differences and operator differences invisible to real values remain failures of attribution. Existing source/attention/geometry notes already cover these observations; reuse them and add no observer, theory framework or experiment here. Quality still requires competent members, useful pool repairs versus harms, capable same-operator single/independent/joint controls and prospective confirmation.

Known-prior links: [SGC](https://arxiv.org/abs/1902.07153); [SIGN v3](https://arxiv.org/html/2004.11198v3); [BatchEnsemble](https://arxiv.org/abs/2002.06715); [MIMO-GC v1](https://arxiv.org/html/2505.11346v1); [Polynormer v2](https://arxiv.org/html/2403.01232v2); [NSD v4](https://arxiv.org/html/2202.04579v4); [BSNN v1](https://arxiv.org/html/2410.09590v1); [BuNN v2](https://arxiv.org/html/2405.15540v2); [HetSheaf v3](https://arxiv.org/html/2409.08036v3). Scope/identity bindings are in REUSED_BINDINGS.json and CITATIONS.json. SGC's original body was not newly read; its algebra is supported here by saved SIGN. No full-paper, priority, publication-body-equivalence or empirical-result credit is claimed.

No datasets/models/checkpoints/logits/current outcomes/server/live probes/compute or implementations were accessed or created. Only saved notes, bounded literature excerpts, source protocol/interface text and source identities were read. Existing sources, scores, gates and jobs remain untouched.
