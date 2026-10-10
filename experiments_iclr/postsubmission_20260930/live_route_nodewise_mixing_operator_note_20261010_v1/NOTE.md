# Node specific route mixing before graph propagation

Author theory audit, 10 October 2026. Node specific route communication can change what a later neighborhood aggregation receives. The exact distinction below concerns a common fixed linear propagation operator. It supplies a placement hypothesis with established feature sharing and attention ancestry; accuracy improvement remains an empirical question.

## Exact fixed operator identity

Let node states be \(H_i\in\mathbb R^{M\times d}\), a common fixed scalar propagation matrix be \(A\in\mathbb R^{N\times N}\), and fixed route mixers be \(R_i\in\mathbb R^{M\times M}\). Define

\[
(P_AH)_i=\sum_j A_{ij}H_j,
\qquad (M_RH)_i=R_iH_i.
\]

Propagation after mixing minus mixing after propagation is exactly

\[
\boxed{[(P_AM_R-M_RP_A)H]_i
=\sum_j A_{ij}(R_j-R_i)H_j.}
\]

The proof is expansion: the two orders give \(\sum_j A_{ij}R_jH_j\) and \(R_i\sum_j A_{ij}H_j\). Neither positivity nor stochasticity of the matrices is required. A common feature map applied on the right multiplies the displayed difference by that map; it can annihilate a nonzero difference. Using \(I+R_i\) in place of \(R_i\) leaves the commutator unchanged.

The difference vanishes **for every state bank** if and only if \(A_{ij}(R_j-R_i)=0\) for every node pair. Necessity follows by varying one \(H_j\) at a time. Thus mixers must agree across every nonzero propagation edge, equivalently be constant on each weak support component. A diagonal propagation matrix permits arbitrary node mixers. For a **particular bank**, varying mixers can still give zero through cancellation or nullspaces. For example, if each \(H_j\) has identical route rows and every \(R_i\) is row stochastic, each difference annihilates \(H_j\).

The identity holds with gates frozen during the comparison. For a recomputed input dependent rule \(F(H)_i=R_i(H)H_i\), the second order instead uses \(R_i(P_AH)\); gate recomputation creates an additional term. Route specific propagation also breaks the assumption of a common \(A\): even constant route mixing can fail to commute with different route operators.

## What the v3 source implements

For four width 512 route states, [blocks.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/live_route_wikics_native_init_source_20261010_v3/blocks.py:50>) computes

\[
Z_i=\operatorname{LayerNorm}(H_i),\quad
R_i=\operatorname{softmax}_{\rm row}
\!\left((Z_iQ)(Z_iK)^\top/4\right),\quad
H'_i=H_i+R_i(Z_iV)U,
\]

with the diagonal attention scores masked, rank 16 projections, and \(U=0\) initially. The gate uses graph informed prefix states; it has no additional explicit neighbor summary. The transported object is a normalized, projected value, and the full residual update is not a convex mixture of raw route states. Gate variation alone has no state effect when \(U=0\).

[The coupled forward](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/live_route_wikics_native_init_source_20261010_v3/lockstep_forward.py:117>) exchanges recurrent states after local block 0 and before blocks 1–6. It retains each route's original block 0 contribution in the native accumulator. Later graph blocks, root branches and heads therefore receive continuing route states affected by the exchange. [The adapter](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/live_route_wikics_native_init_source_20261010_v3/session_adapter.py) preserves the original [wrapper's factor contexts and separate outputs](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/portable_internal_be_public_interface_20261007_v2/core/models.py:200>) and selects private native attention scorers. Sequential two-view gradient accumulation changes neither this placement nor the within-view coupling.

LayerNorm, gate recomputation, native input dependent GAT attention, private factors/scorers, ReLU, dropout, the \(h\odot x\) product and optional global attention prevent identifying the actual tail with one common fixed \(A\). The boxed identity describes an isolated linear subcase. It does not quantify the complete native order effect.

## Closest established operation and scope of distinction

The [saved primary-method conclusions](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/member_hidden_graph_conditioned_mixing_independent_assessment_20261007_v1/REPORT.md>) identify [Cross-stitch v1, §3.3](https://arxiv.org/html/1604.03539v1#S3.SS3) as the closest continuing-stream collision: learned, spatially constant per-channel stream combinations feed subsequent layers. Such a constant mixer commutes with common fixed linear propagation. Its order can still matter with native nonlinear or route specific blocks. The [saved architecture scout](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/live_route_hidden_interaction_architecture_scout_20261010_v1/CONCLUSIONS.json>) identifies Set Transformer self-attention as the nonlinear primitive and Cross-stitch/sluice as communication ancestry.

Node specific weights permit different neighbors' donor values to enter the subsequent propagation. Final-logit averaging leaves those intermediate states untouched. This is a conditional-operation and placement distinction. Concatenating the routes produces a width 2048 state with grouped native updates, tied projections and four heads; a structured wider GNN allowed the same operations reproduces the coupled forward. No additional observed graph, features or labels, expressivity separation from that comparator, novelty, or accuracy guarantee follows.

## One diagnostic after complete family closure

After the entire declared family closes, use every exchange bank already chosen by its frozen selection rule. With weights fixed and dropout off, capture only the exchange-site \(H_i,R_i,T_i=(Z_iV)U\). Predeclare one label-free \(A\): row-normalized binary incoming graph support including one self-edge per node. Freeze gates and values, then compute the realized value commutator

\[
D_i=\sum_j A_{ij}(R_j-R_i)T_j.
\]

Report \(\|D\|_F\), its ratio to \(\|H\|_F\) (zero for a zero denominator), and the \(A_{ij}\)-weighted RMS edge variation of \(\|R_j-R_i\|_F\). Replacing every mixer by their common node mean gives an exact zero reference. This is one zero-training, label-free diagnostic; it requires no terminal predictions, refitting, coefficient tuning or new checkpoint selection.

Nonzero gate variation with negligible \(D\) rejects the isolated fixed-\(A\) explanation on those captured values: projection, route redundancy or cancellation makes the variation ineffective. Nonzero \(D\) establishes realized order sensitivity in that subcase. It establishes neither useful repairs nor accuracy causality, and a small value leaves effects from native attention and nonlinear processing unresolved. Numerical near-zero criteria must be fixed before inspecting the banks and supply no utility gate.
