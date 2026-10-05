# Disabled native Amazon member callback and sparse responsibility port

5 October 2026. Read frozen V6 model/adapter, architecture construction and public-data accessor **source only**. No model/Torch import, execution, dataset/checkpoint/result payload, SSH, 77 or Desktop access. Prior source remains unchanged. This separate port binds repaired G0V2 source `fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3`; root reports its small CPU tensor qualification passed. That report is not a native or scientific qualification here.

## Native partition and one-member reconstruction

The frozen architecture is Polynormer 300→512→5, ten local layers, one global layer, two heads of 256, learned sigmoid gates, shared query/key and no pre-LayerNorm. The continuation uses fixed global stage and complete eval mode.

**phi_m is row m of exactly nine registered banks:** `stem.R/S/B`, `local_head.R/S/B`, `global_head.R/S/B`. Shapes per row are respectively 300/512/512, 512/5/5 and 512/5/5: 2368 scalars/member. In fixed global stage 522 local-head factors are inactive; 1846 stem+global factors are active. Keep inactive coordinates in the state and verify zero gradients/no movement. **theta is every other native named parameter**, including all three boundary weights and every interior linear/GAT, local/global gate, LayerNorm and global-attention parameter. Local-head weight is also inactive in global stage. GAT subparameter names follow the exact pinned installed implementation and must be inventoried before execution; the exclusion rule retains all of them.

The callback uses `torch.func.functional_call` on a tiny wrapper whose forward invokes **only** the existing `family.forward_member(features, edge_index, 0)`. For each call, replace the nine banks with the supplied private vector unsqueezed as row0; supply the complete theta dictionary. Therefore callback(theta,phi_m) reconstructs native member m without evaluating the other three branches. No module is constructed/reset from native code, no donor copy is trained, no bank pooling is hidden in the callback.

All 24492 nodes and native undirected+self-loop edges remain in that route: native GAT scores/skips/gates and LayerNorm, the summed local trajectory, global node-axis Q/K/V reductions, global normalization and private global head. The callback does not substitute a gathered S/R node subgraph or a cached boundary embedding. The optional `expected_nodes` override is solely for an explicitly recorded synthetic engineering fixture; the actual Amazon binding remains 24492.

Binding requires the existing complete family already in eval mode, the fixed stage, and no registered buffers. It never calls `.train()`, `.eval()`, reset, RNG or a stateful cache helper. Functional parameter substitution is temporary and strict. Its restoration, unregistered PyG attributes/caches, module modes and RNG immutability still need native checks; source inspection alone cannot certify external GAT behavior.

## Sparse map is the same map

The input public model edges must be the frozen unique reciprocal support after `to_undirected/remove_self_loops/add_self_loops`. For responsibility affinity only, remove loops and restrict endpoints to S. For each unordered pair {a,b}, include **every** S node labeled a or b, with the other class as competitor, retaining all induced edges across and within labels. Raw K is unweighted binary adjacency. Compute the same max row degree and W=K/(1+dmax) as dense G0V2.

Store fixed oriented edge rows/columns/weights and degree=W1. Implement

    LQ = degree[:,None]*Q - index_add(row, weight[:,None]*Q[column]).

`SparseLaplacian.__matmul__` supplies this operation to the **unchanged repairedV2 `_assignment_map`** through its Pair object. The eight steps, cost/RMS, entropy/graph coefficients, smooth step bounds, private independent-Q partial and total outer derivative are reused, not rewritten. `native_sparse_episode` copies only the G0V2 state orchestration to accept prepared sparse pairs, including full post-core recomputation from original phi. No solver, marginal rounding or convergence criterion changes. Dense/sparse summation order can differ in floating arithmetic; algebraic equality is the claim pending numerical comparison.

An optional supplied fixed within-S-class permutation p implements precisely K'[i,j]=K[p[i],p[j]] by remapping edges with p's inverse. It generates no permutation or RNG and changes affinity only. No query/A labels enter graph preparation.

Sparse pair storage is O(sum_p n_p + E_p), assignment columns O(|S|(C-1)M). No dense S×S or pair affinity is materialized. Pair edges can repeat across blocks; within-class edges occur in all pairs containing that class. This reduces affinity storage; full native member activation, higher-order and response work remain. The unchanged source still requires 9M member forwards, 4M private first derivatives and two assignment maps per committed episode, plus the shared-core outer backward. No production memory/time claim is supplied.

## Safe input boundary

The port provides **no data accessor**. The stock V6 `common.load_data` decodes all TRAIN (including A under the new roles), VAL panels and old fit/control labels; it is not the accessor for this gate. A separately reviewed caller may supply label-free public features/edges and only B/S/R labels, with the protocol's A exclusion. Public preprocessing remains native; self-loop removal for affinity must not alter callback edges. Acquisition, fixed role identities, heldA evaluation, scientific admission and cost authorization remain with the separate bounded protocol.

## Required numerical qualification before any scientific fit

`QUALIFICATION_PLAN.json` specifies the short sequence: one-member full-logit/gradient reconstruction; graph-product/assignment value and derivative equality; independent full shared chain with fixed-Q private partial; complete sparse directional checks; and state/RNG/commit checks. Each failure is retained, with no alternate graph, approximation, changed tolerance or fit workaround. Numerical/source execution is disabled here. Full native sparse GAT higher-order differentiation, actual memory/time and the safe caller remain pending.
