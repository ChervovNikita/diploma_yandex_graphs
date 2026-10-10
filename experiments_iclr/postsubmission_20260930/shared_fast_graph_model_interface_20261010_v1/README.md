# Small common wrapper for the original GCN, GAT and SAGE

`common_routes.py` is a callable model library. Importing it loads only stdlib. It creates no data reader, training/selection loop, optimizer, launcher, owner or provenance system. No numerical import or execution occurred during preparation. The current nine WikiCS fits and every old score remain unchanged.

The wrapper stores one original `models.Model` body, retains its native Parameter objects, and uses the existing `core/factors.py` registration to add four private rank-one input/output factor rows around each materialized dense map. GCN/GAT/SAGE remain the original residual/FFN architectures. Native residual operations, input linear → dropout → GELU, output normalization and classifier remain in their original order. The raw-logit average is the default output baseline, matching `run_tabm.evaluate_tabm`; native probability averaging is an explicit alternative that must be frozen with compatible metrics. Binary outputs gain a singleton class axis without changing values.

| Original backbone | Dense maps wrapped | Native operation retained |
|---|---|---|
| GCN | Input/head, FFN linears, each `conv.lin` | Original `GCNConv`, normalization and `add_self_loops=False`; use the original dataset's prepared graph. |
| GAT | Input/head, FFN linears, each homogeneous `conv.lin` | Original `GATConv` formula, shared `att_src/att_dst`, heads, concatenation, attention softmax/dropout and self-loop convention. |
| SAGE | Input/head, FFN linears, each `conv.lin_l/lin_r` | Original `SAGEConv` neighborhood/root operations and SAGEModule's `[x,message]` concatenation into its FFN. |

Native attention vectors, biases and normalization affines are shared slow parameters. Only r/s factor banks are private. This differs from the running WikiCS V3 source, whose private native scorer banks/start convention stay separate and unchanged. This common wrapper does not pretend to reproduce that exact experimental family.

Construct a fresh original native float32 CPU body with the already fixed model options, then wrap before moving the whole model or creating its optimizer:

```python
from common_routes import NativeModelAdapter, wrap_shared, existing_member_rng_scope

body = models.Model(**native_options)  # Original native constructor/reset only.
adapter = NativeModelAdapter(torch, body)
model = wrap_shared(torch, factors, body, adapter, members=4,
                    kind="baseline", rank=16, block_seed=seed + 870000000)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=fixed_lr, weight_decay=0)
model.check_optimizer_ownership(optimizer)
scope = lambda m: existing_member_rng_scope(torch, session.streams, m, session.cuda_index)
batch = {"graph": graph, "x": graph.x, "edge_index": graph.edge_index}
logits, representations = model(batch, route_scope=scope)  # [member,node,class/width]
raw_average = model.serve(logits, mode="logit_mean")
```

`factors` is the existing imported `portable_internal_be_public_interface_20261007_v2/core/factors.py` module. Streams remain owned/saved by the existing Session. Its scope resumes each member's same stream around native passes or prefix/tail segments, so interleaving live prefixes does not borrow another member's dropout state. The wrapper requires this scope in training; eval can omit it. The existing runner needs one explicit joint-forward call in place of its independent member-forward loop for coupled arms. Loss, graph preparation, masks, metrics, stopping and checkpoints stay in the existing runner. The model reads no labels or role masks. Optional `batch["ids"]` selects output nodes only after the full factual graph path.

Unit r/s factors preserve the native parameters and mathematical initial function. Floating-point operation order can differ, including native fused bias operations; bitwise parity and accuracy parity have not been measured. The existing first-factor Rademacher initializer can be applied explicitly before Adam if a future protocol freezes it for all matched factorized controls, but it changes the initial function and is not the default. Do not reset the native body after installing factors: original convolution resets expect native Linear objects. Reconstruct with the same factory and strictly load a matching selected state instead.

Save/load `state_dict`, optimizer state and the Session's stream state through existing checkpoint helpers; do not pickle the whole local factory class. Parameter roles use stable registered names across `.to(device)`, and optimizer coverage is checked using current parameter identities after that move.

For `kind="separable"` or `"exchange"`, all four live prefixes end after **complete** `residual_modules[0]`. They must align identical label-free node IDs; the original graph/input is supplied to every path. The block runs over `[node,member,width]`, then all original remaining residuals, normalization and head continue. Separable uses a shared route-wise residual MLP; exchange uses off-diagonal receiver query/key/value attention. Both add exactly4×width×rank weights, start with a zero final projection, and use the same placement/normalization. Source/receiver gradients stay live: no detach, cached teacher or oracle labels. This site requires at least two residual modules; baseline wrapping also supports one.

The wrapper rejects running/statistical buffers, BatchNorm, lazy maps, cached adjacency and tied module/parameter aliases. LayerNorm/Identity are stateless and remain shared; native GraphNorm is allowed only with the original one-graph invocation semantics, where each route computes its own statistics. It does not pack members into one normalization batch. Original `GAT-sep`/GATv2, GT, TAG, weighted/edge-feature models, heterogeneous/bipartite graphs, sampled node subsets and custom stateful layers need explicit adapters. No arbitrary model rewriting is claimed.

`PolynormerAdapter` maps the saved native body at the equivalent site between local blocks0/1: preserve its first x_local term, multiplicative local updates, betas, all subsequent local blocks and saved optional global path/head. It requires a fresh unmodified body. Existing private-scorer controllers and current sourceV3 Session guards require their dedicated adapter; this port does not install or alter those controllers.

Rank-one BatchEnsemble factors, Cross-stitch/sluice stream mixing and set attention are known ingredients. The proposed contribution to evaluate is whether the same controlled shared-body interface acquires and serves useful competent alternatives across native backbones, with lower stored-weight cost than genuine I4. Code establishes no gain, novelty, efficiency or acceptance. The accompanying small evaluation plan starts with capable native/factorized M1 and genuine I4 controls; failed references or weak members cannot manufacture an ensemble improvement.
