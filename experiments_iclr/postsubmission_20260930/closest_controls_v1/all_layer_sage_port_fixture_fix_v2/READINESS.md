# All-layer SAGE BatchEnsemble adaptation: source readiness

Status on 2026-09-30: **source prototype prepared; new runtime CPU gate not run**. The current PPI/MNIST studies and frozen runners were not edited. No dataset was loaded, accelerator/server job launched, checkpoint opened, or historical score recalculated. This packet is an engineering readiness audit, not a study freeze or performance result.

## Source choice and reuse

Use the existing `validation_tuning_prepared/all_layer_factor_study.py` constructor pattern (lines 47–92) around the phase `ProjectorEnsemble`, rather than importing its study runner. Its base is exactly the same `TABMModel` that the phase adapter uses. The validation, Roman, and phase `models.py` files are byte-identical: SHA-256 `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303`.

The Roman port `gnnm_external_transfer/experiments_iclr/all_layer_be_sage.py` (lines 126–198) implements the same hidden-map operation. Its `MemberFactorLinear` constructor, forward, and parameter-count property are reused in `all_layer_sage_adapter.py`; its fixed Roman dimensions, four-member restriction, `TABMAblationModel` dependency, launcher locks, and historical result handling are not suitable as a new task adapter. The validation port additionally supplies useful identity/shared-gradient audit patterns (lines 151–247), which inform the pending synthetic gate. Neither original study module is imported or executed.

`SOURCE_PROVENANCE.json` records ten preserved source copies. `artifact_manifest.json` hashes all audit inputs, specifications, and new source. The prototype imports preserved copies of the phase adapter, model, and existing MNIST graph readout. The two all-layer ports are already implemented; this packet adapts their interface and pooling without inventing another hidden-factor mechanism.

## Relation to Kim (2023)

The official thesis method excerpts are preserved in `sources/prior/kim_inspected_method_excerpts.txt`; the source-retrieval sibling contains the catalog/viewer provenance and fuller report. Printed pp. 6 and 8 establish probability averaging and shared layer matrices multiplied by member rank-one factors. Equations on pp. 10, 13, and 16–17 place those factors inside GCN, GIN, and GAT graph layers. Hidden states retain the member index. The thesis does not establish SAGE as a tested backbone.

Declare the arm **“all-layer BatchEnsemble, SAGE adaptation”** (`all_layer_be_sage_adaptation_prob`). It is a relevant prior-method adaptation, not a thesis reproduction or new novelty. The residual SAGE/MLP architecture, LayerNorm, input sign initialization, hidden identity initialization, exact bias rule, mean-member loss, PPI multilabel sigmoid pooling, and MNIST mean-node-logit graph readout are adaptation choices. The inspected thesis does not verify its factor initialization distribution or a mean-member CE training objective. Its GIN discussion uses BatchNorm and member epsilon/bias; normalization-statistic sharing is unspecified. These facts cannot be used to attribute the port's choices to the author. No exact author code was recovered.

## Exact operation and interface

`AllLayerSAGEAdapter(options, members=4)` accepts the phase `model_options` dictionary with `model_name="SAGE"`, `normalization="LayerNorm"`, `hidden_dim_multiplier=1`, and positive depth. It returns member node logits `[M,N,C]` from `forward(graph, features)`. `graph.edge_index` is the existing directed edge list; `features` is `[N,input_dim]`. It constructs on CPU and can later use normal module `.to(device)` behavior. It retains the original boundary projectors and factorizes all four hidden maps in each residual block:

| Hidden map | Input → output | Factors per member |
| --- | --- | --- |
| SAGE `lin_l` (incoming-neighbor mean) | D → D | R, S, B |
| SAGE `lin_r` (root) | D → D | R, S, B |
| concatenated SAGE FFN `linear_1` | 2D → D | R, S, B |
| FFN `linear_2` | D → D | R, S, B |

For an existing affine map with PyTorch weight `W[O,I]` and optional shared bias `b[O]`, the port computes `shared(x * R[m]) * S[m] + B[m]`. Thus `W_m = W * S[m,:,None] * R[m,None,:]`; effective bias is `b * S[m] + B[m]`, or just `B[m]` if the original has no bias. R is an input vector and S an output vector. This is the thesis's rank-one weight modulation up to vector naming/orientation; bias semantics are explicitly those of the existing port. Shared biases are trainable and scaled by S. Root `lin_r` originally has no bias, but its new B creates a member root offset. Neighbor/root offsets add and can be redundant; that is preserved from the port.

At each block, shared LayerNorm is applied separately to the current member's hidden state. Native SAGE computes the sum of the factorized incoming-neighbor mean transform and factorized root transform. The normalized state and this message are concatenated, passed through factorized FFN1 → dropout → GELU → factorized FFN2 → dropout, then added to the member's residual state. Final shared LayerNorm precedes the member output projector. The input path is input projector → dropout → GELU. LayerNorm gamma/beta are shared across members; LayerNorm computes statistics per node feature vector and has no running batch statistics. No hidden state is averaged across members.

Native `SAGEConv` remains responsible for aggregation: default incoming-neighbor `mean`, `root_weight=True`, `normalize=False`, `project=False`. Existing directed entries, duplicate multiplicity, and self edges are retained. No self loops, reverse edges, coalescing, mean operator replacement, or adjacency packing are introduced. Only its two callable linear children are wrapped. This preserves its ordinary forward/scatter path subject to the pending ABI and synthetic-scatter check.

The members execute sequentially with a complete hidden state per member, then stack outputs. Dropout uses separate draws in this fixed member order. This is a source implementation of the member ensemble, not evidence for parallel or faster execution. A selected R/S/B row is bound into the autograd graph during forward, so later member selection does not reroute an earlier backward pass. Shared parameter gradients accumulate from all members; member factors receive their own loss contributions. The pending gate checks this. The mutable `active_member` interface is not reentrant or safe for concurrent forwards. No resetting wrapped SAGE convolutions, scripting, compilation, or packed-copy ABI is declared: the wrapper intentionally does not implement `reset_parameters`, and packed conversion code assumes direct `lin_l.weight`/`lin_r.weight` access. Construct anew to reinitialize.

Initialization is exactly the existing phase initialization: both shared boundary W matrices use Xavier uniform; input R uses independent ±1 Bernoulli signs; output R and all boundary S are one; boundary B is zero. Existing hidden shared weights/biases and normalization tensors retain their original draws. Added hidden R/S are one and B is zero, with no random draw. Therefore a same-seed construction should have identical source state, post-construction CPU RNG, and initial member logits to `ProjectorEnsemble`. The gate checks this before any updates. Hidden factors become trainable immediately.

With M members, width D, depth L, and multiplier one, the added hidden parameters are `13*M*D*L`. At M=4, D=128, L=3 this is 19,968. Static source accounting gives PPI 292,496 total parameters (272,528 original projector ensemble), and MNIST 271,196 total (251,228 original); these counts remain runtime-gate targets, not measurements. The adapter supports other positive locked depths. Its source execution still performs M graph propagation calls per block; parameter sharing alone makes no latency or memory claim.

## Task readout, training and pooling

For PPI, `make_ppi_model(...)` returns `[M,N,121]` logits from 50 input features. The adaptation's training objective is mean member BCE with logits over the existing fixed observed nodes and classes; every member sees the same graph/mask. This uses the current task objective and must be declared as an adaptation. Evaluate `pooled_probabilities(logits,"ppi") = mean_m sigmoid(logits_m)`, threshold at 0.5, and use BCE on these pooled probabilities for the validation tie-break. A probability clamp, if needed for log loss, must be fixed prospectively. Do not pass the pooled probabilities to BCE-with-logits.

For MNIST, `make_mnist_model(...)` returns the existing `GraphClassifier` readout with a new node adapter. `forward(graph, features)` gives `[M,G,10]`: mean each member's node logits within its graph after node LayerNorm and output projection. The inherited source readout keeps nodes and graphs separate and rejects empty graphs. Three input channels remain released intensity plus released coordinates. Training is mean member graph CE. Ensemble evaluation applies softmax to each member's **graph** logits and averages probabilities; softmax is not applied before the node-to-graph mean. `pooled_forward` returns these class probabilities. Use their argmax and pooled negative log likelihood for the validation tie-break.

The thesis verifies probability averaging for its classification setting. The multilabel sigmoid form and graph readout here are explicit task adaptations. The current frozen PPI and MNIST evaluators pool raw logits. Adding this probability comparator requires an amended new-study arm definition and evaluator/replay evidence before any fits. The member-logit ABI remains compatible, but its pooled evaluation is not a silent drop-in to the existing raw-logit scorer. If a raw-logit placement ablation is desired, declare it separately as an adaptation with pooling matched to current GNNM; it is not the thesis pooling rule. This packet selects the probability arm and does not change either runner.

## Only remaining engineering CPU gate

Run `verify_source_cpu.py` once in the repository environment with hidden CUDA devices, no datasets, and layer counts equal to the prospectively amended task definitions. The existing repository runtime metadata records Python 3.11.14, Torch 2.1.2+cu118, and PyG 2.7.0. `EXPECTED_RUNTIME.json` binds the two relevant preflight environment manifests and exact SAGEConv/LayerNorm source hashes; a different runtime fails rather than claiming compatibility. The gate also verifies this packet's hashes.

```sh
: "${GNNM_AUDIT_PYTHON:?Set this to the approved repository interpreter}"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "$GNNM_AUDIT_PYTHON" \
  closest_controls_v1/all_layer_sage_port/verify_source_cpu.py \
  --ppi-layers 3 --mnist-layers 3 \
  --report closest_controls_v1/all_layer_sage_port/CPU_GATE_run01.json
```

Run from the phase directory using its existing confinement wrapper, after resolving the approved interpreter and repository location. No command was dispatched during this audit. Depth three is the current source default; use the depth actually declared for each new arm. The script's report cannot overwrite an existing report and must remain under this sibling.

The one synthetic gate covers nonidentity effective-weight/bias forward and derivative equivalence; all hidden-map coverage; parameter uniqueness/counts; exact initial shared/boundary state, RNG and member logits; matched train-mode dropout/shared gradients; finite hidden-factor gradients and unused-member gradient isolation; member hidden-state separation; native mean/scatter fidelity including duplicate/self edges and isolates; MNIST two-graph readout; explicit probability pooling (including a PPI fixture where it differs from raw-logit pooling); and no CUDA context. It performs zero optimizer steps and provides no research performance evidence. No broad rerun of already-passing existing protocols is needed for this source audit. Runner integration and fresh source/runtime/arm locks belong to a later prospective amendment before fits.

The source syntax and preservation checks in `SOURCE_AUDIT.json` pass without importing Torch or running the gate. Runtime readiness remains pending until that gate passes in the declared environment.
