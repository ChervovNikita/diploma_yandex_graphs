# Frozen Context9: local geometry at an ideal within-class prototype

At an ideal state where both normalized views of every labelled node equal its class's unit prototype, changing common targets to factual route targets has **zero first derivative through normalized representations**. Its second-order loss difference compares selected same-class distances. This explains a local geometric possibility; it does not identify an observed training state, establish a stationary point or stability, or show that hidden diversity repairs mistakes. The calculation is ordinary contrastive and graph-Laplacian local geometry, not a new theorem or method.

Use the actual frozen shared objective and targets in [the immutable first note](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/context_target_shared_gradient_identity_theory_20261008_v1.md>): four members, \(B=512\) panel rows, \(\lambda=.05\), \(\tau=.2\), and the mean over members. Each \(Q_m\) is row-normalized after panel restriction and supported only on pairs with the same TRAIN class. The common target is their arithmetic mean; write \(\Delta_m\) for \(Q_m-\bar Q\). Its rows sum to zero, and its support stays within class. All statements below use exact mathematical targets and smooth normalization in the unit-sphere regime; the first note's finite-precision qualification still applies.

Keep the model state, targets, panel identities and stochastic paths fixed while comparing objectives. Let \(a_{mi}(\varepsilon)\) and \(b_{mi}(\varepsilon)\) be smooth normalized representation curves starting at the unit prototype \(z_{y_i}\). Their first derivatives \(u_{mi}\) and \(v_{mi}\) are tangent to that prototype: their inner products with \(z_{y_i}\) vanish. For every pair in the support of \(\Delta_m\), the class is the same, so normalization gives

\[
a_{mi}(\varepsilon)^T b_{mj}(\varepsilon)
 =1-\frac{\varepsilon^2}{2}\|u_{mi}-v_{mj}\|^2+o(\varepsilon^2).
\tag{1}
\]

This follows by differentiating the unit-norm constraint: the prototype component of each second derivative is minus the squared norm of its first derivative. Any second-order tangent acceleration has zero inner product with the common prototype.

In the route-minus-common **loss difference**, both log-softmax normalizer terms cancel exactly because each row of \(\Delta_m\) sums to zero. What remains is minus the signed target-weighted sum of the forward and reverse dot products, with coefficient \(\lambda/(2BM\tau)\). The constant terms cancel, and the linear dot-product terms vanish by tangency. Ordinary own CE is the same function in both assignments and cancels from their difference at every compared state. Thus, writing \(D(\varepsilon)\) for total shared route loss minus total shared common loss,

\[
D(\varepsilon)=
\frac{\lambda\varepsilon^2}{4BM\tau}
\sum_{m,i,j}\Delta_{m,ij}
\left(\|u_{mi}-v_{mj}\|^2+\|v_{mi}-u_{mj}\|^2\right)
 +o(\varepsilon^2).
\tag{2}
\]

The proposed coefficient is therefore correct: the source supplies one half for the two-view average, one \(1/B\) for anchors, one \(1/M\) for members, and the normalized-dot-product expansion supplies the other half. Equation (2) is a **loss-difference coefficient**; the second directional derivative is twice its quadratic coefficient. It is not the full objective Hessian.

For identical perturbations in the two views of a route, put \(u_{mi}=v_{mi}=w_{mi}\). Let \(S_m\) be the symmetrized signed matrix \((\Delta_m+\Delta_m^T)/2\), and let \(E_{S_m}(w_m)\) denote its Dirichlet energy: half the sum of \(S_{m,ij}\) times squared pairwise distances. Equivalently this is the quadratic form of its signed degree-minus-weight Laplacian. Symmetry of the squared distances gives

\[
D(\varepsilon)=
\frac{\lambda\varepsilon^2}{2BM\tau}
\sum_{m,i,j}\Delta_{m,ij}\|w_{mi}-w_{mj}\|^2
 +o(\varepsilon^2)
=\frac{\lambda\varepsilon^2}{BM\tau}
\sum_m E_{S_m}(w_m)+o(\varepsilon^2).
\tag{3}
\]

This graph has **signed** weights: factual targets place more weight than common targets on some edges and less on others. Its energy need not be positive. Sparse class-compatible neighborhoods therefore change the relative local cost of within-class variation along a specified direction. Unselected same-class pairs remain denominator distractors in the actual loss; their normalizer contributions cancel only in this target-assignment difference. If perturbations are also identical across all routes, the entrywise sum of \(\Delta_m\) is zero, so the displayed quadratic difference cancels too; with fully identical paths the exact loss difference cancels.

Row normalization does not fix column masses. Expanding the squared distances in (2), the outgoing squared-norm terms vanish with the zero row sums, while incoming column-mass differences and cross-view inner products remain. At the prototype, any corresponding unconstrained ambient representation gradient is radial; normalization projects it away. Thus the zero first derivative refers to the normalized tangent geometry and, by the chain rule, smooth raw-representation/model paths through it. It is not a claim that the gradient treating normalized vectors as unconstrained Euclidean coordinates is zero.

Self positives also matter. Their normalized weights depend on each restricted positive count. When the two views have different tangents, the diagonal target difference contributes a signed penalty on that node's cross-view mismatch. When the views share the same tangent, self distances vanish and self-loop weights drop out of (3). Column-mass and self-view effects therefore cannot be discarded by treating every directed neighborhood as a regular undirected graph.

The native private factors cannot choose arbitrary nodewise tangent fields. A parameter direction produces only the fields reachable through its normalized representation Jacobians in the two fixed views; (2) must be restricted to that image. After accounting for sphere-normalization curvature in (1), extra second-order tangent acceleration does not change (2). This still proves neither a useful private direction nor specialization.

A classifier-null direction is a reachable perturbation of the raw pre-head representation that the fixed effective classifier map annihilates. It leaves logits unchanged to first order while its normalized image may change the distances in (2). The native representation is captured before the final linear prediction head, so this is a possible distinction between local representation geometry and immediate logits. The head and factor parameterization can restrict which such directions are reachable; first-order logit neutrality alone does not guarantee second-order CE neutrality or preserved classifications. The prototype assumption itself says nothing about whether the classifier is correct.

Source grounding is the first note's exact hash table and cited frozen dispatcher/replay. Static lines rechecked for this companion were portable_context_steering_public_interface_20261008_v1/context_alignment_objective.py:20–36 (normalization, symmetric views, scaling), context_positive_masks.py:69–85,142–147 (same-class/self support and normalized target mean); these are byte-identical to the frozen successor's bound helpers. The first note also binds core/models.py:72–83 and core/factors.py:18–26 for the pre-head capture and factorized linear maps. Its SHA-256 remains b86578d22ec1a91e6775010a80de46219518f851fe55d520183d9e2096dbb53e. Only static sources and notes were read. No implementation or arm was changed, and no array, checkpoint, numeric fixture, server, or comparative outcome was accessed.
