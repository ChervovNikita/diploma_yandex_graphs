# Private sheaf transport: inactive native source packet

This packet prepares one sharing/placement experiment around the original **licensed NSD general-map model**. It contains source and adapters, not a dataset loader or training runner. No model imports, tensor operations, datasets, labels, checkpoints or training were used to prepare it. Competence and numerical equivalence remain unestablished.

## Files

- `adapter.py`: native factory, shared-weight/private-factor bank, independent native bank, joint multi-transport single, paired assignment and layer-sharing/capacity controls.
- `PROTOCOL.json`: four prospective full-task competence configurations, selection rules, controls, falsification criteria and execution gates.
- `PINNED_CORE_SHA256.json`, `SOURCE_RETRIEVAL.json`: exact retained author bytes and retrieval custody.
- `SOURCE_BINDINGS.json`, `READ_SCOPES.json`: bounded semantic/source bindings and inspected scopes.
- `static_verify.py`, `VERIFICATION.json`: source hash, license and syntax checks only.

## Author source and boundaries

The retained source is [`twitter-research/neural-sheaf-diffusion`](https://github.com/twitter-research/neural-sheaf-diffusion) at commit `11e21b561d884713ab1a18a521a7dc2fb26b9361`, tree `a0655fc770ecf026da23ade54d6818ed779f92da`. Its Apache-2.0 license and author headers are preserved. All 17 retained files match their pinned Git tree blobs. The adapter loads `models.disc_models.DiscreteGeneralSheafDiffusion` after checking core hashes and native namespace origins.

The inspected DNSD repository had no license grant in its tree, README or inspected source. Its code also uses different degree normalization and normalization/flow choices. All downloaded DNSD payloads were removed; only metadata, hashes and bounded findings remain. Nothing here vendors or adapts DNSD, and this fallback is called **original NSD**.

The native environment declares Python 3.9.9, Torch 1.11.0, PyG 2.0.4 and torch-householder 1.0.1. Importing the general-map class also requires numpy, torch_sparse and torch_scatter through its dependencies. A modern Tolokers loader is outside that original environment. The future runner must supply the official graph or qualify a compatible environment/loader while retaining the original NSD operator. No installation was attempted.

The author `exp/run.py` is retained for objective/configuration inspection. It evaluates TEST every epoch, permits test-dependent continuation and records selected scores without a serving-state restoration contract. It is **not the future runner**. A train/validation-only runner and checkpoint restoration must be qualified separately.

## Native operator and member ownership

Let `d` be stalk size, `f` hidden channels and `c=d*f` the node width; the proposed presets keep `c=64`. A native node state `[N,c]` is reshaped to `[N*d,f]`. At every layer the original nonsparse learner concatenates ordered endpoint states `[x[row],x[col]]` and applies one bias-free `Linear(2c,d²)`, tanh, then reshape to `[E_directed,d,d]`. There are not separate DNSD source and target builders.

For a replaced affine map, member `m` computes

`dense_m(x) = W (r_m * x)`, then `s_m * dense_m(x) + b`.

`W` and `b` are the actual native parameter objects shared across members; `r_m` and `s_m` are private. Bias stays outside output scaling. All native Linear maps are treated this way: input lift, optional second input Linear, left/right feature maps, incidence learner and native classifier. Native epsilon vectors are shared. No new FFN is claimed as part of the original body. Factors begin at one; this preserves the native affine algebra, while numerical parity is a future gate. In a fully deterministic synchronized training setup, identical factors could stay identical; this protocol retains native dropout and jitter with distinct member draws and checks whether useful private geometry actually develops.

Each member has a distinct complete model module and Laplacian builder, private factors, recomputed states and map diagnostics. Immutable topology/index tensors are shared by deepcopy memo; they are not mutated by the pinned forward. This avoids copying large graph indices in the shared bank. It does not eliminate member-specific map, state, gradient or sparse multiplication work. Independent native members are constructed separately and share no learned parameter identities.

The adapter delegates reverse-edge pairing, diagonal sums, normalization, sparse assembly and propagation to the author code. In particular, the general builder forms negative off-diagonal `F_left.T @ F_right` and block degree sums `F.T @ F`; computes the native symmetric matrix power/SVD on `D + diag(1+eps)`; retains training `eps ~ U(-.001,.001)` per stalk axis and evaluation zero jitter; and retains native clamping, augmentation, ELU, sparse `spmm` and epsilon/residual updates. There is no scalar-degree substitute or LayerNorm rewrite. The packet requires `normalised=True`, `deg_normalised=False`, nonlinear/nonsparse/tanh learners, both left/right maps and no fixed LP/HP dimensions.

Native graph attributes are plain tensors, not device-managed buffers. Construct edges on the final device before the model; do not move a bank to another device afterward. Support must be nonempty, int64 `[2,E_directed]`, without self-loops/duplicates, and contain exactly one reverse of each directed entry. The validator checks support without changing its order. Shared/assignment adapters reject graph mutation or `update_edge_index`; this is a static-graph packet. Native singles/independent models must follow the same caller-side contract.

The native class has no model-level reset method. Create fresh constructors, banks and optimizers for every seed/family/fold. `reset_factors()` resets only factors. Original right feature weights retain orthogonal initialization, left stalk weights identity initialization and epsilon vectors zero initialization.

## Model and control families

| Family | Objective and serving rule |
|---|---|
| Native single | Fresh original class; unscaled TRAIN NLL; native head |
| Independent native M4 | Four full fresh encoders/heads, separate learned parameters, RNG streams, optimizers and unscaled own NLL; mean probabilities |
| Shared private native M4 | Four complete native paths, shared native W/b/epsilon, private factors; `F=mean(member TRAIN NLL)`; mean probabilities |
| Joint multi-transport single | Same four transport states; concatenate live pre-head states into a capable `256->64->2` ELU readout; single NLL |
| Fixed paired assignment | Train with one saved, label-blind permutation of paired reverse-edge maps per member/layer |
| Resampled paired assignment | Fresh paired permutation each forward; one draw per member at validation/serving |
| Layer-shared incidence | One trainable full tanh incidence matrix per member/layer, repeated on all ordered incidences |
| Layer-shared capacity | Same layer-shared matrix plus private node residual capacity replacing the removed map budget |

Pair reassignment preserves the current multiset of paired incidence maps before the original builder recalculates block degrees and normalization. Its training controls allow adaptation; they are not fixed-checkpoint reliance tests. Permutation buffers are saved, and graph identity/order is bound. Resampling uses the global RNG; the future runner must record its evaluation draw schedule and isolate evaluation RNG consumption from training. No unreported eight-draw inference is included.

For M4 with no LP/HP dimensions, the removed map path has `2c*d² + M*(2c+d²)` active parameters per layer. Constants cost `M*d²`; private bias-free `c->w->c` node adapters cost `M*2c*w`. Setting `w=1+d²/M` gives an exact match: 1,040 parameters/layer at `d=2,w=2`, or 2,624 at `d=4,w=5`. The node adapter is inserted after the native left map and before the native right map. This is our original-NSD control placement; it is not the two-builder DNSD audit implementation.

The joint readout has 16,578 parameters. Native terminal heads are frozen/excluded from its optimizer but still execute and must be charged. This control receives all four live transport states and learns one nonlinear decision rule. It is not equal-capacity by default. Use its `parameter_groups()` so the readout is included. `IndependentNativeBank.member_parameter_groups()` supports separate native optimizers; each independent constructor must be called under its prospective member seed by the future runner. A pooled one-optimizer mean-NLL shortcut is not the prescribed independent baseline.

## Prospective full-task competence screen

The proposed task is **the full Tolokers graph**, official split 0, full validation AUROC. Only metadata was read here: 11,758 nodes, 10 features, binary labels, 1,038,000 directed entries and ten official splits. Remaining splits are reserved for later frozen confirmation. The future runner must confirm that this task/split is unused in local research history before accessing data.

The four presets are `d2/f32/L2`, `d2/f32/L4`, `d4/f16/L2` and `d4/f16/L4`, each at width 64. Common settings are original `.01` Adam, `.0005` regular/sheaf decay, ELU/tanh, input/layer dropout `.4/.2`, optional second input Linear enabled, 500 epochs and patience 200. They use published discrete ranges and explicit parser defaults, but they are **ours**, not an extracted winning author general-map configuration. The paper's unusual printed weight-decay bounds remain literal; no interpretation or repair is asserted.

Three fixed seeds are selected by full validation AUROC, with validation NLL and earlier epoch for ties. Select the architecture using native singles only, then freeze architecture, optimizer budget, checkpoint rules and all controls. A feature-only MLP supplies a competence diagnostic. `PROTOCOL.json` fixes provisional fit/validation criteria and allows a prospective amendment of at most four additional sensible full-task configurations if the baseline is weak. There is no local subset/subgroup selection. TEST labels, metrics and plots remain unopened through this process.

After competence and numerical qualification, the frozen pilot reports candidate own-member quality, pooled quality, best-member quality, genuine independent ensembles, assignment/capacity replacements, joint readout and measured cost. Disagreement, moved map parameters or matrix capacity alone cannot establish useful private geometry. If controls recover the candidate, report the corresponding replacement result. One split and three seeds is a representative pilot, not broad benchmark evidence.

## Cost and unperformed qualification

Every bank performs M complete native forwards, including M incidence evaluations, block-degree/SVD normalizations and sparse multiplications per layer. Joint serving adds its readout and still computes native terminal heads. Independent members also have M native constructor/preprocessing costs. Training retains the member computation graphs until backward; inference and training wall time, peak memory, persistent diagnostics and optimizer state must be measured rather than inferred from parameter counts.

The native learner `.L` stores a detached, **unnormalized negative lower-triangle transport**, not the final normalized/clamped block. These diagnostics persist per member/layer. At the proposed Tolokers size, their float32 payload alone is about 66 MB (`d2,L2,M4`) through 531 MB (`d4,L4,M4`), excluding activations, gradients and temporary maps. Any effective-operator or loop/gauge diagnostic must capture the actual normalized operator separately without replacing propagation.

Static source hashes/license and Python 3.9 syntax pass. The adapter has not been imported. Native numerical parity, synchronized-RNG gradient parity, ownership/deduplication, graph/device qualification, shuffle save/load/multiset checks, exact active capacity counts, joint-readout gradients, SVD stability, environment compatibility and full-task competence remain required gates. `PROTOCOL.json` lists them explicitly. This packet supports no novelty, performance, uncertainty, memory-saving or speedup claim and launches nothing.
