# Source reuse plan for the integrated ordered walk residual

This plan reuses existing numerical components in a future separately admitted consumer. It is source/text reasoning only; it creates no model module or job.

## Exact project components

| Existing file | Prospective reuse |
| --- | --- |
| `shared_fast_graph_model_interface_20261010_v1/common_routes.py`, SHA256 `84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773` | `NativeModelAdapter` exposes original stem/residual/output layout; `wrap_shared(kind=baseline)` provides four native routes; `existing_member_rng_scope` resumes member RNG; optimizer ownership checks retain each parameter once. Create a new adapter in a future new source folder, not a mutation of this file or active E/Penn sources. |
| `portable_internal_be_public_interface_20261007_v2/core/factors.py`, SHA256 `9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3` | Existing `FactorLinear` preserves source W/b and implements shared bias outside private s. Existing dynamic `member_context(body,m)` discovers registered path maps. Reuse the caller's actual bound factor provider; these local bytes are the inspected convention, not authority to replace a different runtime provider. |
| Root-bound original `models.Model('SAGE',2,...,'LayerNorm',...)` provider | Retain its exact input projection, dropout/activation, residual graph modules, norm and classifier. Bind the actual provider file receipt before source construction; no scientific-host fetch is performed here. Node-wise LayerNorm is essential to the stated radius-two fixed-feature dependency argument. |
| Original own-CE native acquisition/selection procedure | Mean of four full TRAIN own losses, one coherent native selector, existing role policy, RNG and optimizer. Reuse a separately bound source copy; the current Penn protocol and E family remain fixed. |

## Integration boundary

The new adapter's `native(batch)` computes the exact original stem e once, keeps it for the path branch, executes the same native residual modules, computes the path residual and adds it immediately before original output normalization. It then applies that normalization/head once. A new adapter is preferable to an unbound hook with persistent cross-route capture state. Every original SAGE convolution and its factual loops/scalars stays in its original order.

Register two caller-owned nn.Linear source maps and wrap them with the existing FactorLinear before the optimizer exists. If registration follows `wrap_shared`, dynamic member context can still cover them, but `factorized_maps` and the native-name snapshot need an explicit ledger amendment. Bind the original native IDs separately from newly added W/b and private r/s; do not report added path weights as original native inventory. The whole optimizer must own every trainable parameter exactly once.

Pass path indices/validity through the label-free batch. Existing interfaces already reject targets/masks in a model call. No query labels, learned node IDs, path-weight parameters, adjacency-row N×H weights or persistent fitted representations enter the path module. Native X/E and path metadata are immutable per call. Factor-dependent stem values and path activations cannot be cached across routes or optimizer steps.

Before scientific admission, check zero-output equality of original route logits and RNG trajectory, caller/provider identity, correct incoming orientation, mask semantics, live stem/path/factor gradients, all-parameter ownership, correct full TRAIN exposure and coherent control selection. These are source qualification obligations, not work performed or permission requested here.

## Author implementation reference

The abstract's Graph4Sec-Team URL redirects to `GISec-Team/PathMLP`; pin commit `4525359bc683d89b4b805251ace6f20c74d0b8ee`.

- `model/PathMLP.py`, SHA256 `3dce2deeee49a72a72cb6c8d5dc0927df01d58e13a2334fd1537ae1f20ebf2a7`: ordered reshape/path map and mean aggregation are the reference operations. Reimplement the small credited equations with project maps; native nonlinear stem reuse, fixed mean, residual insertion and private factors are explicit adaptations.
- `sampler.py`, SHA256 `dd64e4a235f650d901ead8ff3cbff66fdb8cb8b5fa950aa52dcf40ab2c957f71`: reference depth=4/four-position candidate construction, common subset and duplicate-neighbor behavior. A future incoming-CSR builder computes feature inner products only on actual neighbor records. The author's per-node full-edge scans, outgoing convention, reversed position order and self fallback are not transplanted silently. Preserve every deliberate difference in the protocol.
- `main.py`, SHA256 `ae6f3711096c5c8f9b952f7bc8a118b76313ff3d7d4d2c1d799668a58c46b905`: source evidence for ordinary own train-mask loss. Do not invoke its dataset loader, split creation, saved NPZ path loader or evaluation entry point. Use project role and acquisition custody.

This is a formula/source-reference reuse plan, not a faithful PathMLP reproduction. No author data arrays, saved paths or pretrained models were retrieved. Any future complete PathMLP baseline must declare paper-versus-code choices and include its actual topology/path-weight capacity rather than attributing those effects to ordering.
