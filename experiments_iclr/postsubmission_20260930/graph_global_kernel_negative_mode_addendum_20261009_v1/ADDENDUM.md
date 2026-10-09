# A negative mode of the positive query/key-split core

9 October 2026. Separate analytic addendum; the original note and seals are
unchanged. No numerical test, dataset, outcome, model weight, server contact,
source change or launch is involved.

The proposed split can remove a second restriction of the native fixed core:
its nonnegative spectrum. Positive attention weights can support a negative
Markov eigenvalue. This is a structural possibility, with no whole-model
expressivity, task-quality or theorem-priority claim.

For the native tied core, let `U_ih=sigmoid(z_ih)`, `C=UU^T` and
`D=diag(C 1)`. With finite real logits, C is entrywise positive and D is
strictly positive. Its row-normalized core satisfies

\[
P=D^{-1}C,\qquad
D^{1/2}PD^{-1/2}=D^{-1/2}UU^TD^{-1/2}\succeq0.
\]

Thus every eigenvalue of P is real and nonnegative. Row stochasticity also
bounds them above by one. This property supplements the native core's detailed
balance; it does not follow from reversibility alone.

For the finite three-node witness in the preceding note, take
`q_i=(u_i,c)`, `k_i=(1-u_i,c)`, `u=(1/4,1/2,3/4)` and `c=1/2`.
Then `C_ij=u_i(1-u_j)+c²>0`. Direct expansion gives, for each i≠j,

\[
C_{ii}C_{jj}-C_{ij}C_{ji}=-c^2(u_i-u_j)^2,
\qquad
P_{ii}P_{jj}-P_{ij}P_{ji}
=-\frac{c^2(u_i-u_j)^2}{d_id_j}<0.
\]

The kernel is a sum of two rank-one matrices and has a nonzero second-order
minor, so C and P have rank two. P has eigenvalue one because it is stochastic
and zero because its determinant vanishes. The sum of its three principal
second-order minors is the remaining eigenvalue: the second characteristic
coefficient is the sum of pairwise eigenvalue products.

Here the row sums are exactly `(9/8,3/2,15/8)`. The minors for pairs
`(1,2),(1,3),(2,3)` are respectively
`-1/108,-4/135,-1/180`, yielding

\[
\operatorname{spec}(P)=\left\{1,0,-\frac2{45}\right\}.
\]

All entries remain positive. Repeated application of this *fixed* stochastic
core is bounded; its negative eigenmode alternates sign and decays in magnitude.
The original cycle-product witness separately proves that this particular core
is nonreversible. Negative eigenvalues alone do not prove nonreversibility:
positive reversible matrices can also have negative modes when they lack the
Gram/PSD restriction.

The native complete GNN already contains local attention, residual/gating paths,
learned value/output maps, nonlinearities, changing states and layer/member
combinations. Those operations can produce signed or directed effects despite
each tied global core's nonnegative spectrum. Consequently this calculation
identifies an available fixed-core mode, rather than a function the whole native
model cannot represent. It establishes neither useful learned circulation nor
accuracy, competence, stability of training or generalization.

There is a relevant control qualification. After-feature-map scaling
`Q=U D_q`, `K=U D_k` gives the symmetric kernel
`C=U diag(d_q*d_k) U^T`. When all diagonal products are nonnegative, this is
Gram/PSD; if a positive weighted channel remains, positive U also gives strictly
positive entries. The active reversible control `q=k=u*exp(c/2)` in the original
recommendation retains both restrictions. Signed **products** can destroy PSD
while symmetry remains. Their entrywise kernel positivity and valid positive
row denominators would require a separate check; signed individual scales with
nonnegative products still retain PSD. A negative mode is therefore not an
exclusive signature of pre-sigmoid splitting.

The source grounding is the pinned `GlobalAttn.forward` and its retained tied
factor semantics, plus the analytically verified witness in
`graph_global_kernel_query_key_split_design_20261009_v1/NOTE.md`. Exact input
bytes are recorded in `SOURCE_BINDINGS.json`. No original artifact is edited,
and no new experiment or resources request follows from this addendum.
