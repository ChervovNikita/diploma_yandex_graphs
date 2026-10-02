# Vectorized evaluation of the selected original GNNM/SAGE — source only

Date:2026-10-02. Supporting evaluation-efficiency lane. No model/source import, training, checkpoint/data loading, graph execution, SSH, installation or benchmark occurred. Canonical sources remain unchanged.

## Concrete opportunity

The selected original wrapper performs four complete member forwards in a Python loop. Each member has private input/output BE R/S/B and its own complete nonlinear hidden trajectory. The residual SAGE/FFN matrices and affine LayerNorm parameters are common. The same selected trained parameters can evaluate four `[member,node,feature]` states together. This replaces repeated module/kernel dispatch with leading-dimension batching; it does not create one trajectory, merge members or eliminate dense/message arithmetic.

`vectorized_eval.py` supplies an unexecuted `OriginalSAGEEval` callable. It reuses the actual selected modules and live parameters. It performs private input scaling→shared W→private output scaling/bias in the native order, then per-member feature LayerNorm, native mean SAGE, last-axis `[normalized,message]` concatenation, native FFN/dropout/GELU order and identity residual addition. Final affine norm and private output BE preserve logits `[4,N,C]`, including C=1. Evaluation dropout is identity; training/stochastic-dropout execution is rejected.

The frozen `SAGEModule` uses `torch.cat(...,dim=1)` on2D states. Calling that module directly on3D would concatenate nodes. The adapter therefore spells out the unchanged residual branch and concatenates on `dim=-1`. PyG2.7.0 MessagePassing defaults `node_dim=-2`; this is the node axis in `[member,node,H]`. COO aggregation therefore keeps member slices separate and uses unchanged directed edges, duplicate multiplicities, self loops and zero-degree handling. LayerNorm(H) normalizes only H, never members or nodes.

The source path permits complete-trajectory chunks1/2/4. Input stems have a separate chunk1/2/4<=trajectory chunk. Default hidden chunk4/stem1 avoids a fourfold private `[N,D]` temporary when input features are large, while retaining four complete hidden trajectories. Leading-dimension PyG message lifting may still require `[chunk,E,H]` storage; chunk4 can be slower or exceed memory. No speed/memory result is known.

## Closest implementation ancestry

BatchEnsemble2002.06715v2 explicitly vectorizes `(X*R)W*S`, and at testing repeats each input across all members before averaging predictions. That is the boundary batching ancestry. Its hardware-utilization claim does not remove M dense input transforms when every input needs every member. This adapter is a cited implementation extension to the selected GNNM evaluation, not a new rank-one or batching primitive.

Primary PyG2.7.0 source documents mean SAGE: aggregate neighbor features, apply biased lin_l, add bias-free lin_r(root), optionally normalize. The selected contract fixes mean/root, project=False, normalize=False, source_to_target, node_dim=-2 and no feature decomposition. MessagePassing and MeanAggregation/scatter propagate that node axis and broadcast destination degree counts. PyG Linear and PyTorch2.1.2 Linear transform the last feature axis, supporting leading dimensions. PyTorch LayerNorm(H) preserves row-local affine normalization. Exact source slices are in PRIMARY_PASSAGES.md/SOURCE_MAP.json.

Actual runtime from the parent is Torch2.1.2+cu118/PyG2.7.0. An initial prospective2.5.3 read was replaced by2.7.0 before packet freeze. Installed-source custody currently covers the Coauthor provider only; no claim is made that the remote installed SAGE/scatter bytes were retrieved by this agent. Prospective runtime qualification must bind the installed source/backend to these operations.

## Fair controls

`PackedControlEval` also drafts batched untied and head paths. Untied keeps every private selected W/bias/norm and full hidden trajectory, applies batched private dense maps and the same native parameter-free incoming mean aggregation. Heads retains its legitimate single common trunk and packs all four private two-layer heads. Selected weights are converted once; no retraining or warm-request repacking.

Cold packing time and its temporary original+packed peak count. After equivalence, `release_source()` permits packed-only private serving storage; the caller must also release its original model reference. Heads retains its shared trunk once. Warm timing charges the actual deployed representation, without requiring permanent duplicate private weights. The packed affine LayerNorm decomposition is algebraically identical but may round differently from the native fused implementation, so controls need their own numerical gates.

All controls retain the same trained cell/checkpoint, full graph, selected dimensions, four trajectories/heads, precision and runtime resources. Native serial and optimized versions are compared for every arm; the original is compared against the best qualified competent control. No toy timing or inference-only number establishes training/whole-study efficiency. No new backbone, initialization/diversity mechanism or validation tuning is introduced in this supporting lane.

## Disposition

Mathematical memberwise equivalence is supported by operator separability and source inspection, but numerical equivalence, conformal-output stability and real full-checkpoint GPU utility remain unqualified. QUALIFICATION.md defines those prospective gates. Adapter syntax was parsed only. A qualified faster path may support the new conformal utility experiment; failure or no actual advantage means keeping the canonical serial evaluator.


## Reusable literature record

PRIMARY_CONCLUSIONS.json stores canonical IDs/versions, exact read scopes, implementation takeaways, GNNM implications, limitations/next decisions and evidence paths for BatchEnsemble, PyG2.7.0 and Torch2.1.2. Consult those records and the frozen EnKF notes before reading again. The BatchEnsemble material was reused with a targeted revisit of the outstanding test-batching/work question; the EnKF REPORT/SOURCE_MAP were not edited. Root may link these entries into its literature index.

Static receipt:23 owned payloads before the exclusive manifest;14 primary/canonical custody rows,18 exact source/passage slices,two unchanged canonical source hashes,three reusable primary conclusion records andone AST-only adapter. No numerical/model/graph/checkpoint or performance execution. Corrections after freezing require a versioned supplement. Root owns the requested source-only commit/push.
