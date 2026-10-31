# Efficient graph ensemble ports: source preparation

**Unexecuted source packet.** No scientific imports, tests, data, labels, checkpoints, SSH or GPU were used in preparation. Syntax and source hashes alone were checked. This packet does not amend the running cohort or launch a comparison.

## What is prepared

- `ports.py`: four smaller untied native members with an explicit sequential reference and a prospective `torch.func.stack_module_state` / `vmap(functional_call)` path; a secondary PolyFormer MIMO model; compact-label losses and explicit reducers.
- `qualifications.py`: complete module-count checks, member isolation, output/gradient/Adam equivalence and MIMO tuple-label correspondence hooks.
- `test_ports.py`: real synthetic qualification tests for a future authorized CPU runtime. They have not run. Unsupported vmap execution produces visible skips, which **do not qualify that backend**.
- Three native source files are copied without any byte changes from the retained v3 architecture bodies. `PROVENANCE.json` and `MANIFEST.json` bind those copies, upstream pins, source scopes and new files. The pinned PyG GAT source is evidence only; runtime imports the installed PyG implementation.

## Fixed widths before any validation

The resource target is the retained M4 **boundary-factor family parameter count**, including its private factors and copied private biases. This is a parameter cap, not an equality of runtime, graph work, activation memory or optimizer bytes.

| Port/context | Fixed body width | Other width constraints | Complete parameters | Boundary-family cap | Next legal width: parameters |
|---|---:|---|---:|---:|---:|
| M4 gamma1 PolyFormer-Mono/SquirrelFiltered | 116 | 4 heads; FFN58; 13 tokens; 2 blocks | 4,177,828 | 4,430,644 | 120: 4,435,156 |
| M4 gamma1 Polynormer-r/AmazonPhoto | 248 total | 8 heads ×31; 7 local +2 global layers | 7,707,904 | 7,773,732 | 256: 8,185,920 |
| M4 PolyFormer cached-token MIMO | 208 | 4 heads; FFN104; body unchanged relative to width208 single | 4,308,428 | 4,430,644 | 212: 4,441,312 |

Each is the largest legal width below its cap. The counts were derived from the actual retained module declarations, including PyG2.7 GAT's homogeneous projection, two attention vectors, absent bias/residual/edge projection, and all inactive Photo global parameters. The future constructor count must agree before validation; `check_parameter_budget` also checks the next legal width exceeds the cap.

The native source equations are `P_poly(w)=59w²+2159w+109` and `P_photo(w)=28w²+826w+16`. Boundary overheads are `4F+11H+7C` and `4F+15H+14C`, respectively. MIMO enlarges its stem and head, giving `P_mimo(w)=59w²+8441w+124`. No outcome-based width search or tuning grid is prepared. Attention scores, edge work, caches and optimizer states still require charged measurement later.

## Interface and ownership

Callers supply the **existing prepared public graph** from the current adapter: `teacher_backbone`, `teacher_input`, `teacher_edge_index`. PolyFormer input is `[node,13,2089]`; Photo input is the existing normalized `[node,745]` feature matrix with coherent graph edges. Preprocessing is neither duplicated nor modified here.

Role arguments contain only parallel compact `nodes` and corresponding `labels`. No full label vector or data loader is accepted. Both family inference methods return `[4,node,class]`, matching the existing public output shape. These are constructors/losses/hooks, not a replacement fitter or study driver.

All M4 member maps, attention coefficients/projections, LayerNorm actions/affines, beta vectors, biases and states are private. Vectorization maps **parameters**, while coherent graph inputs remain unmapped; there is no independently shuffled node slot against unchanged topology. Photo's local/global state is set explicitly, as in the retained fitter. Stacked Adam retains private member slices, but its scalar step metadata assumes synchronized member updates and stages; selective member schedules are unsupported.

The meta template is not a registered trainable owner. No dense diagonal matrix or shared learned attention is introduced. `vmap(randomness="different")` requests distinct dropout streams. Deterministic gradient/Adam equivalence is tested with dropout disabled; identical stochastic RNG replay is not claimed. A separate training-mode test checks different streams for identical member values.

Native PyG GAT gather/scatter/softmax or backward operations may be unsupported by this Torch/PyG vmap combination. That is **unresolved**, because no runtime was executed. The vectorized path raises an explicit error rather than silently retrying. `backend="sequential"` remains an honest untied reference; four sequential calls are not labeled a measured efficient packed backend. Both packing success and any speed advantage require later qualification/measurement.

## MIMO and reducers

MIMO pairs **complete cached polynomial-token rows** after preprocessing, concatenates four feature slots per token, and uses four matching output heads. `positions` indexes the compact TRAIN pack; labels are gathered from that same compact position, not from a global-label array. The provided sampler uses independent permutations, rho0 and one exposure per TRAIN example per slot; there are no batch repetitions. Inference repeats the entire target row in all slots. Only PolyFormer is prepared for MIMO.

MIMO loss is mean over tuples and **sum over heads**, faithfully retaining the published reduction; packed members use the retained **mean member CE**. This scale difference and native Adam/regularization settings are explicit. The optimizer helper preserves PolyFormer's attention-specific LR/decay groups and Photo's native Adam settings across private parameters.

`reducer="native_probability"` computes the native PE/MIMO mean of class probabilities, stably in log space. `reducer="common_logit"` computes softmax of mean raw logits and is an explicit graph-protocol adaptation. Callers must choose the reducer before selection; the current study's reducer is unchanged.

## Qualification after root authorization

From this directory, an authorized runtime can invoke `python -m unittest -v test_ports`. The tests synthesize six-node inputs and compact labels; they do not load project datasets. The retained expected runtime is Torch2.1.2 and PyG2.7.0. Supply the packet directory first on `PYTHONPATH`, and bind the imported native/PyG source before drawing conclusions.

The test set covers actual parameter caps, both Photo stages, deterministic output/gradient and two-step Adam parameter/moment/step equivalence to untied copies, selected-member derivative isolation, private perturbations, dropout separation, complete MIMO slot/token/label correspondence (including deliberate corruption), repeated-input inference correspondence, head-summed loss gradients, and both explicit reducers. The same hooks can consume root-provided prepared graphs/compact TRAIN packs for a later full-context qualification.

No test result, support result, performance measurement, trained prediction, experimental freeze or acceptance judgment is asserted here.
