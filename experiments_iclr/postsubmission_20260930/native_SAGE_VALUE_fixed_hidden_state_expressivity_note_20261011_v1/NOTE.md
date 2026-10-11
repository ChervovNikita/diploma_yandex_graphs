# Fixed hidden state distinction for endpoint VALUE transport

At the supplied SAGE hidden states, the implemented endpoint operation E can distinguish two equal-degree neighborhoods with identical raw means that every implemented receiver-only operation N maps to the same aggregate. The example below uses a diagonal special case of the actual paired 4×4 maps. This is a local statement about the message boundary and available parameter settings.

## Exact implemented operation

Fix one layer and member, and write its live source and receiver states as $h_j,h_i\in\mathbb R^{128}$. Let $R(h)$ reshape $h$ into a $4\times32$ matrix. The helper computes

$$F_s=I_4+\tfrac12\tanh(A_s(c)),\qquad F_t=I_4+\tfrac12\tanh(A_t(c)),\qquad T=F_t^\top F_s,$$

with elementwise tanh after each affine builder is reshaped to 4×4. E uses $c=[h_j;h_i]$. N uses $c=[h_i;\operatorname{silu}(h_i)]$ and builds its map once per receiver. A nonself message is $R^{-1}(T R(h_j))$; an actual self edge bypasses transport and returns $h_i$. Native mean aggregation precedes the retained `lin_l` map and `lin_r` root addition.

This reshape applies the same 4×4 action to each column $q$: coordinates $(h_q,h_{32+q},h_{64+q},h_{96+q})$, for $q=0,\ldots,31$. It permits off-diagonal mixing. The operation is more general than coordinate scaling and is constrained relative to a full generated 128×128 matrix. The Physics source port substitutes graph dimensions while retaining this equation, width128 and d=4.

## The N aggregate factors through the raw neighbor sum

Let $J_i$ contain the nonself incoming sources, $n=|J_i|$, and $s$ be the number of propagated self entries. The degree used by native mean is $D=n+s>0$. For N, the same $T_i$ acts on every nonself source, so

$$R(\bar m_i^N)=\frac{T_i\sum_{j\in J_i}R(h_j)+sR(h_i)}{D}.$$

Consequently, for fixed receiver state, degree and self count, equal nonself sums give equal N aggregates for **every** N parameter setting. The nonlinear receiver context and paired-map product can make $T_i$ complicated, but they do not make it vary with $j$. At this boundary there is no source-wise nonlinearity after applying $T_i$ and before mean. The self bypass is included in the formula.

## Explicit attainable E counterexample

Take two receivers with the same state $h_i=0$, two nonself neighbors each, and one self entry each. Keep all hidden vectors supplied at this boundary. Let

$$v=e_0-e_1\in\mathbb R^{128},\qquad H_A=\{v,-v\},\qquad H_B=\{0,0\}.$$

The two zero vectors in $H_B$ belong to different neighbor nodes. Both neighborhoods have nonself sum zero and $D=3$. Thus $\bar m_A^N=\bar m_B^N=0$ for all N weights. The chosen $v$ has zero coordinate sum, so the example does not require an all-positive state or violate the sign structure possible after LayerNorm.

Choose any finite $a>0$. Set all target-builder weights and bias to zero, and set all source-builder weights and bias to zero except the weight from context coordinate $h_j[0]$ to raw matrix entry $(0,0)$, which is $a$. Set the builder member factors r and s to one. These are allowed settings of the actual 256→16 `FactorLinear` builders. With $E_{00}$ the 4×4 matrix unit,

$$F_t=I_4,\qquad F_s(h_j)=I_4+\tfrac12\tanh(a h_j[0])E_{00}.$$

Both nonzero coordinates of $v$ occupy the first row of $R(v)$. Therefore

$$m_E(v,0)=\bigl(1+\tfrac12\tanh(a)\bigr)v,\qquad
m_E(-v,0)=-v+\tfrac12\tanh(a)v.$$

Their sum is $\tanh(a)v$. Zero sources and the zero self state contribute zero, giving

$$\boxed{\bar m_A^E=\tfrac13\tanh(a)v\ne0=\bar m_B^E.}$$

With no self edge the coefficient is $\tfrac12$; with extra zero neighbors it is $1/D$. Hence the implementation's self bypass preserves the example. It is algebra over supplied states, with no toy accuracy experiment or altered scientific graph.

## Assumptions and limits

The claim quantifies over permitted builder parameters, with one fixed member and supplied hidden states, deterministic message evaluation and matched support counts. The actual zero builder initialization gives identity transport for both E and N; separation requires a learned nonzero setting and is no learning guarantee. The example applies to M1 as well as an M4 member and supplies no ensemble-specific result.

The aggregate difference survives the native convolution output if `lin_l` does not annihilate $v$. Equal receiver states make the root terms equal. Later FFNs, normalizations and readouts can preserve or erase the difference. The claim therefore concerns the pre-`lin_l` aggregate; it does not guarantee distinct final predictions for every downstream weight setting.

A preceding nonlinear embedding can give these neighborhoods different means before N receives them. The actual native model includes a learned stem, GELU, LayerNorm and earlier residual/message computations. The example does not hold those entire learned networks to a common attainable representation on the dataset, prove whole-GNN family containment, establish a WL hierarchy, prove universal approximation, or imply generalization or predictive utility. N also uses a different receiver feature basis, including SiLU; this one separation does not show that every N function is representable by E.

## Established principles supporting the explanation

**ECC.** Simonovsky and Komodakis, CVPR2017, section3.1/Eq1 generates an edge-dependent matrix before neighborhood averaging. Section3.2 explicitly factors a common filter through the sum when edge labels are uninformative. The distinction here uses that established conditional-filter principle with live endpoint states and a constrained paired-map parameterization. It is not a new theorem about conditional graph filters.

**GNN-FiLM.** Brockschmidt, arXiv1906.12192v5, section2.1/Eqs7–8 treats receiver-generated modulation and distinguishes nonlinearity placement. Its Eq8 applies a source-wise nonlinearity before summation, which can already break equal-raw-mean collapse even with receiver-generated coefficients. The N factorization above applies to this exact N implementation; it must not be generalized to all receiver-conditioned methods or used to claim superiority over GNN-FiLM.

**Deep Sets.** Zaheer et al., arXiv1703.06114, sections2.2 and3.1 describes learning a per-instance representation before summing, optionally conditioned on context. E's nonlinear source-dependent message is one constrained instance of this established transform-before-pooling principle. Only the architecture principle is used here; the countable-universe theorem and its appendix qualifications are not transferred to this fixed-width mean aggregator.

**Suggested method wording.** “For fixed incoming hidden states, receiver-only N applies one common linear block map to the nonself neighbor sum. Endpoint E lets that map vary nonlinearly with the sender before averaging. It can therefore retain some sender-dependent statistics that have already cancelled in the raw mean, while preceding embeddings and downstream maps determine whether this local capacity matters for the complete network.”
