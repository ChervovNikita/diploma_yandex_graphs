# Endpoint conditioned VALUE transport in native graph messages

The small VALUE operation is source feasible for the existing shared SAGE, GCN and GAT routes. A pinned native message port remains necessary before integration. This packet admits no scientific launch. The existing retrieval pilot closes independently.

The broad shared weight and private endpoint map conjunction has already been tested. The complete Tolokers NSD pilot shared every native affine W/b, including the ordered endpoint incidence learner, while private BE factors selected each member. Both frozen recipes failed: centered sharing minus genuine independent pooling had mean VALID AUROC −0.009020 and NLL +0.006568; vanilla had −0.018316 and +0.018506. Centering repaired learning at two seeds but supplied few uniquely correct alternatives. These results narrow the inactive scout's earlier “untested conjunction” to **direct VALUE transport in the existing native SAGE/GCN/GAT scalar message operator**. They rule out a new broad nomination under the same description; they do not prove every native operator transfer must fail.

## Exact message operation

For each actual propagated nonself edge u→v, flatten the live native gathered message features x_j and x_i to h_u,h_v∈R^c. Two caller-owned builders receive z_e=[h_u∥h_v] and output d² entries. Each builder uses the established `FactorLinear` convention:

`a_(m,e)=s_m ⊙ W_shared(r_m ⊙ z_e)+b_shared`.

The shared bias remains outside s. Define

`F_src=I+tanh(a_src)/sqrt(d)`, `F_tgt=I+tanh(a_tgt)/sqrt(d)`, `T_(m,e)=F_tgtᵀ F_src`.

Reshape x_j to `[heads,d,channels_per_head/d]`, apply the same T within each head, then restore its native shape. No head is mixed with another. Self-edge VALUEs remain x_j. The caller then applies the original edge scalar or per-head attention coefficient and the original aggregation, bias, root path and residual stack. GAT attention is computed from the original projected x and is not recomputed from transported VALUEs.

The helper accepts caller-owned Torch, builders, live endpoint tensors and a self-edge mask. It allocates no learned parameters or persistent state and uses ordinary autograd through both endpoints and builders. It receives no label, frozen representation, teacher or extra edge data. These maps are an attributed paired block VALUE adaptation; retaining scalar native normalization does not implement NSD's normalized block Laplacian or faithful DNSD diffusion.

## Native integration and ownership

| Native message site | VALUE hook and retained native operations |
|---|---|
| GCN `message(x_j, edge_weight)` | Gather projected `x_i` from the same `x=self.lin(x)` passed to propagation. Transport x_j before multiplying by the native degree weight. GCN normalization and final bias stay native. |
| GAT `message(x_j, alpha)` | Gather projected `[E,H,C]` x_i. Transport x_j before `alpha[...,None]*x_j`. Original attention softmax/dropout, head concatenation or mean, residual and bias stay native. |
| SAGE `message(x_j)` | Gather x_i from the same propagated x pair. Transport before native mean. Keep `lin_l(aggregate)`, `lin_r(root)` and optional output normalization in their original order, then the native module's `[x∥message]` FFN input. The original constructor uses `project=False`. |

The pinned original `models.py` explicitly uses `add_self_loops=False` for GCN and GAT; the saved native `run_base.py` requests loops from its data provider. Preserve that factual support and do not insert a second set of loops. The helper's explicit mask covers any self edges actually propagated.

Use the existing `NativeModelAdapter`, shared body, `FactorLinear`, member context, RNG streams, optimizer and selector. Register two new `FactorLinear` builders under the relevant conv after `wrap_shared` factorizes the native body and before device movement and optimizer creation. The recursive member context then selects their private factors. W/b of each builder is stored once; r/s are private. Every route retains its own live features and graph forward. The existing optimizer ownership check can verify each trainable parameter once.

One bookkeeping detail needs a narrow integration amendment: `SharedRoutes.parameter_roles()` snapshots original native parameter names when wrapping. Builders registered afterward are classified as shared-block W/b and private-fast r/s. The saved `factorized_maps` count also describes only the original replacement pass. Neither field should be reported as the final native-only map count or builder ownership without updating its accounting. No second wrapper is needed.

Saved PyG sources establish the integration boundary. `MessagePassing` caches message signatures, argument lists and generated propagation templates at construction. GCN and SAGE sparse/fused `message_and_aggregate` bypass `message`; a message hook also lacks x_i and actual edge indices in the current native signatures. A blind monkeypatch is inadequate. A future narrow port must collect x_i plus actual post-loop edge indices, retain source→target flow and node ordering, and bind a nonfused factual edge-index path with `decomposed_layers=1`. The self mask must be built from that exact propagated support. This packet supplies no installer, fused rewrite or runtime qualification.

## Decisive controls and interpretation

The indispensable paired comparison is endpoint E4 against node-only N4. N4 uses the same two builders, dimensions, private factors, nonlinearity and matrix action, with `z_v=[h_v∥silu(h_v)]`. Both input halves are functional and every learned parameter is active. For a fixed recipient, its T_v is the same on all incoming nonself edges. Because it acts separately within each GAT head, it commutes with native per-head scalar attention; it also commutes with SAGE mean and GCN scalar normalization. Thus it is a capable receiver-node transform of the nonself aggregate, with the original self contribution retained, at the exact same parameter count.

Include the existing native shared baseline as context, a capable same-operator M1, and genuine independent I4. M1 must receive the same endpoint operator and M=1 factor coordinates with its own complete native body. I4 consists of four separately initialized and fitted complete native bodies and builders, with no shared fitted representation and the same factor-coordinate opportunity, raw graph information, fitting and selection rules. An ensemble-specific benefit beyond four live paths additionally requires the scout's capable joint single with those same paths and a nonlinear joint readout; its extra readout cost must be charged.

E4−N4 isolates ordered endpoint dependence under the chosen shared/private implementation. It does not isolate the causal effect of private ownership. A claim about that effect needs a prospectively matched private-versus-tied builder comparison; unused dummy parameters cannot make a weaker tied control capable. No extra factorial or scientific grid is admitted here.

The hypothesis is that changing each neighbor's VALUE before the sum improves competent correct alternatives that the ordinary probability pool retains. Current evidence does not establish that native aggregation causes the observed common errors. Report final pooled accuracy/NLL, mean and worst member quality, complete repairs and harms, acquisition and losses during pooling, with common seeds and fixed rosters. A gain explained by N4, capable M1/joint single or genuine I4 closes the corresponding endpoint or ensemble-specific claim. Map distance, loop differences or one damaged checkpoint cannot establish useful geometry. Edge-assignment necessity needs a matched retrained assignment-breaking control, not an evaluation-only shuffle.

## Closest tested mechanisms and ancestry

| Saved mechanism | Overlap and remaining placement difference |
|---|---|
| Completed original NSD Tolokers pilot | Already learned private-factor ordered endpoint incidence maps with all large affine weights shared. It used paired restriction blocks, native degree/SVD normalization and diffusion. The prospective operation keeps native scalar SAGE/GCN/GAT aggregation. |
| Closed QK36 | Changed global Polynormer Q/K operators, including private pre-sigmoid splitting; VALUE maps stayed native. Its rejected sharing/operator utility constrains broad relational claims. |
| Closed HGT private modulation | Native typed V and relation matrices with static member/relation input/output scales. FiLM supplies recipient-conditioned incoming transforms. These lack this continuously endpoint-conditioned block T per incidence. |
| Closed native privatepair and rank1 LoRA | Applied one route/layer-wide input transform before selected projections. The transform did not vary with both endpoints before each sum. Its failed transfer remains closed. |
| Frozen query VALUE gate | Applied a detached target-conditioned gate after label-message aggregation. This live graph-message helper acts before aggregation and uses no labels. The independent class-state decoder work remains separate. |

ECC and NNConv already supply learned edge-conditioned VALUE matrices. GNN-FiLM supplies recipient-conditioned message modulation; HGT supplies relation-conditioned VALUE maps; NSD/DNSD supply learned endpoint incidence maps and paired transport; BatchEnsemble supplies shared W/private r/s. This small composition carries that ancestry and makes no novelty claim. The saved scout's versioned primary method passages are retained as source context, rather than re-nominating a sheaf principle.

## Bounded size and disposition

The illustrative interface uses c=128,d=4,M=4; these are not an admitted recipe or search range. Per hooked layer, shared builders add `2(2c+1)d²=8,224` W/b parameters and private factors add `2M(2c+d²)=2,176`; N4 matches both exactly. Setting builder W/b to zero gives identity transport initially, but private factor gradients are initially zero while W is zero and become active only after W changes. This initialization is no learning guarantee; the caller owns initialization and existing member RNG.

Sharing does not remove M builder or transport evaluations. Prediction costs O(ME c d²), paired products O(ME d³), and application O(ME heads d² f) for f=channels_per_head/d. Endpoint concatenations E×2c, paired maps E×2d², activations and backward may dominate memory. Native normalization and graph costs remain. No speed, memory or quality advantage is inferred from the parameter count.

**Retain as inactive native placement feasibility only.** The helper is concise and reusable; exact pinned PyG integration, scalar/loop preservation, ownership, live gradients and resource qualification belong to a later root admission. Complete prediction evidence and capable references would be required for a scientific decision. This packet reads no partial outcomes, runs no tensor/data code and opens no TEST evidence.
