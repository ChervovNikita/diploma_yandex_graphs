# Interpreting source context reassignment in conditional SAGE

The proposed intervention tests whether selected predictors use the association between a neighbor's live state as map context and that same neighbor's actual VALUE. A beneficial factual-versus-reassigned difference can support useful association reliance for those checkpoints, layers and the frozen assignment. It does not establish necessity after retraining, recovered edge geometry, or an ensemble-specific mechanism.

## Exact prospective test

The factual E, equipped M1 and equipped I4 selected states remain fixed in evaluation mode. The four conditions are factual, first layer only, last layer only and both layers. The design restores 18 selected states and serves 36 banks, comprising 108 native trajectories, across the three original fit seeds. No fitting or dropout occurs.

PCG64 seed 2026101101 determines one mapping before array access. Receivers ascend by ID; nonself incoming records are sorted by source ID and original record index. Groups of degree at least two receive one uniformly drawn nonzero cyclic offset. Self records and degree zero or one retain identity. The same mapping is used across layers, members, methods and fit seeds. This is one restricted cyclic reassignment, not a uniform sample over all permutations or an estimate over an assignment distribution.

At each selected layer, **both** endpoint builders receive the reassigned source half of their context. The receiver half, actual native $h_j$ used as VALUE, graph, weights, degree normalization, self bypass and remaining computation stay unchanged at that hook. Raw VALUE and source-context multisets are preserved locally. Transported messages are expected to change.

## What is broken while the local marginals stay fixed

At one layer, member and receiver, write $H_j=R(h_j)$ for the actual 4×32 VALUE and $T_j=F_t([h_j;h_i])^\top F_s([h_j;h_i])$ for its paired 4×4 transport. Because the receiver state is common within its incoming group and both builders use the same reassignment, the local intervention applies $T_{\pi(j)}$ to $H_j$:

$$\bar M-\bar M_\pi=\frac1D\sum_{j\ne i}(T_j-T_{\pi(j)})H_j.$$

Self contributions cancel. The local multiset of paired maps is preserved, as is the multiset of actual VALUES. Their association changes. Constant maps, repeated source states, degree-one groups, or actions in task-irrelevant directions can yield no effect despite a nonidentity record mapping.

The preservation statement is relative to the live incoming states of that forward. A first-layer intervention changes states supplied to the later layer, including later VALUES and receiver contexts. The last-layer-only condition enters that layer with factual preceding states; first-layer and both-layer conditions include downstream propagation of changed representations. Layer effects need not add, and the both-layer condition is not the sum of two importances.

## Nonzero cyclic shifts do not remove covariance cleanly

The actual reshape gives an effective 128×128 action $A_j=T_j\otimes I_{32}$. For fixed supplied states and $n\ge2$ nonself incoming records, define

$$\bar h=\frac1n\sum_jh_j,\qquad \bar A=\frac1n\sum_jA_j,\qquad
C=\frac1n\sum_j(A_j-\bar A)(h_j-\bar h).$$

Here $C$ is a vector from the contraction of the paired map/VALUE deviations. The factual nonself mean is

$$M=\frac1n\sum_j A_jh_j=\bar A\bar h+C.$$

For $\pi_\delta(j)=j+\delta\pmod n$, each $A_k$ is paired with every $h_j$ except its original partner exactly once when $\delta$ ranges uniformly over $1,\ldots,n-1$. Thus

$$\begin{aligned}
\mathbb E_\delta[M_\delta]
&=\frac1n\sum_j\frac{n\bar A-A_j}{n-1}h_j\\
&=\bar A\bar h-\frac{C}{n-1},\\
M-\mathbb E_\delta[M_\delta]&=\frac{n}{n-1}C.
\end{aligned}$$

Fixed-point exclusion leaves an opposite covariance term. This design therefore cannot be described as pure covariance removal or independent rematching. For $n=2$, the sole shifted assignment reverses $C$; its factual-minus-shifted contrast is $2C$.

With $s$ unchanged self entries, the total native mean is $\frac{n}{n+s}M+\frac{s}{n+s}h_i$. Its factual-minus-expected-shift difference is consequently $\frac{n}{n+s}\frac{n}{n-1}C$. Degree-zero or degree-one groups retain identity and are outside this formula.

These are elementary finite-sum identities for hypothetical averaging at fixed supplied states. The one frozen realized shift per receiver estimates neither this expectation nor assignment-distribution uncertainty. In the both-layer condition, later states depend on the same reused mapping, so the fixed-state expectation cannot be applied conditionally to those varying later states. Nonlinear propagation and readout also prevent replacing an expected final score by the score at an expected aggregate. No new theory or causal guarantee follows from the algebra.

## Reading the complete outcomes

| Observation | Supported interpretation |
| --- | --- |
| Logits or decisions change, with no material quality benefit for factual assignment | The fitted computation is sensitive to the reassignment. Numerical sensitivity alone does not show that the association helps prediction. |
| Factual predictions have materially better paired accuracy and compatible stable NLL than the reassigned condition | The selected predictor benefits from its native context/VALUE association under this particular intervention. |
| Reassignment preserves or improves quality despite an effective change to transported actions | This challenges useful association reliance for the tested checkpoints and assignment. A nonsignificant difference with wide uncertainty is not evidence of equivalence. |
| E, M1 and I4 show similar dependence | The evidence concerns a conditional-message mechanism available to competent single and independent predictors. It does not identify a benefit caused by shared/private committee ownership. |
| E is more sensitive than M1 or I4 | Larger sensitivity may reflect differing fitted functions, confidence, redundancy or fragility. It does not by itself explain the original E-minus-N training contrast or establish an ensemble interaction. |

Retain every condition, seed and capable reference, with raw/stable NLL, member competence, pooling losses, coverage, repairs and harms. Do not select the layer condition or metric with the largest deterioration after seeing outcomes. The number of nodes or edges does not create independent graph replications. One common mapping supplies no Monte Carlo uncertainty over mappings.

The original factual archives retain their scientific authority. Disclose factual replay discrepancies; do not silently replace original study scores. Perturbation contrasts should identify the exact factual replay used as their evaluation carrier.

## Strongest falsifier and required references

The strongest falsifier of the narrow **useful frozen-checkpoint association** claim is a complete, effective both-layer reassignment that retains materially equivalent or better predictions, with the same conclusion supported by stable NLL and the retained member and repair/harm readouts. Formal material equivalence requires a predeclared margin and adequate precision; this guide adds no numerical gate. Ineffective reassignment, a wide null interval, or surviving argmax accuracy despite worsened NLL cannot supply that conclusion alone.

Equally perturbed equipped M1 and I4 are the essential capable predictive references in this design. They distinguish association reliance shared by ordinary conditional predictors from a proposed explanation specific to E's committee. The exact receiver-only N operation is analytically invariant to source-half association when receiver states are fixed; this invariance is a source property, not a new numerical control result.

A stronger claim that persistent source assignment remains useful **after adaptation** needs a matched retrained assignment-breaking or capable receiver-only replacement, with the same data, fitting, selection and capacity accounting. Such a new study is outside this diagnostic's authorization. A causal interpretation of real source features would additionally need a justified intervention or conditional sampling model. Within-receiver blocking preserves observed marginals but does not establish exchangeability, a valid conditional sampler, or a conditional randomization test. The context/VALUE mismatch can also be outside the ordinary training distribution.

## Primary attribution and access scope

The direct saved primary reference is [Learned, Relied Upon, or Necessary? Separating Checkpoint Dependence from Task-Level Value in Sheaf GNNs, arXiv2607.25387v2](https://arxiv.org/html/2607.25387v2), section3.1–3.3, retained blocks17–40. It defines checkpoint reliance with all other weights fixed, distinguishes protocol-relative replacement after retraining, and warns that different incidence factors can induce the same paired transport. The current intervention is narrower than its general edge-map shuffle: it reassigns live sender context only within each receiver and uses a restricted cyclic mapping. Its logic is applicable; its empirical outcomes and theoretical guarantees are not transferred.

Its related-work block15 cites Hooker et al.2019 for retraining in interpretability evaluation and Fisher et al.2019 for model-class reliance. Those are citation leads from the inspected primary reference; their original papers were not newly audited here. ECC section3.1/Eq1 and GNN-FiLM section2.1/Eqs7–8 remain the already scoped ancestry for conditional message transformations and the importance of operation placement. The permutation sensitivity diagnostic is an attributed mechanism test, with no claim that permutation importance is new.

READ_SCOPES.json records the exact retained carriers and semantic access. No new primary identity, full-paper credit, download, current Physics/Penn outcome access, numerical run or reviewer is added.
