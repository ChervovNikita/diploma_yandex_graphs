# Explicit final inference ownership and native Collab conversion

This immutable source successor combines two concrete compatibility fixes:
explicit BUDDY private-BN VALID/TEST ownership and OGB's native raw split
conversion. It preserves scorer v3, training producers and portablev3 sources.
**Source and scalar checks pass. Native selected-state, numerical, role and
resource qualification remain pending.** No models, tensors, datasets,
checkpoints, GPU, remote command or TEST payload were executed here.

## Genuine producer and separate inference identity

The scorer first authenticates the genuine selected descriptor/runtime and
loads its exact selected adapter state. It then binds
`buddy_private_bn_production_v3.bind_inference_owner` as a separately identified
inference API. For genuine v2 producers, the API verifies the original source,
class, callback, job and producer metadata. It replaces only the owned context
guards. The saved method metadata, checkpoint schema/layout, private buffers,
optimizer/model/manager objects and producer runtime binding remain unchanged.
No checkpoint relabeling, state conversion or retraining is needed merely to
add the inference API. The RESULT records `inference_API_binding` separately.

`inference_operation(role)` owns the actual `VALID` or `TEST` role. It enters
eval/no-grad mode and the original sequential native route/private manager
path, protects all native and active/inactive private bank objects, pointers,
shape, dtype, device and values, and restores original model modes on exit.
Inference diagnostics are returned locally and do not add TEST entries to the
saved TRAIN/VALID ledger. State access, sampling/hydration and nested owners
remain guarded. Scorer checks the literal role and independently protects all
banks. It never calls TEST through a VALID operation.

## Original Collab split conversion

Retained raw split PT files can contain NumPy int64 arrays. OGB's
`PygLinkPropPredDataset.get_edge_split` calls
`replace_numpy_with_torchtensor` after reading each raw split. Scorer v3 loaded
raw VALID/TEST and demanded tensors without that conversion. V4's `load_split`
calls the exact OGB helper, preserving nested containers, dtype and order.
No `.long()`, dtype coercion, raw cache rewrite or alternative population is
introduced. BUDDY's narrowly versioned role-context helper applies this same
loader to raw TRAIN/VALID, retaining the original graph, weighted topology,
year filter and HashDataset AST. Other source inference helpers are unchanged.

The frozen training port already converted top-level TRAIN/VALID fields with
dtype-preserving `torch.as_tensor`; that path did not have the raw-array
failure for the actual retained schema. It is preserved. The new final role
helper uses OGB's exact recursive conversion API for all raw roles. Upstream
metadata `COLLAB_NATIVE_CONVERSION_14.json` discloses root's TEST metadata
inspection without scoring; this source patch performs no data access.

## Coherent root review and rebind

`REQUIRED_REBIND.json` records the exact source/API identities and prospective
registry delta. The compatibility path keeps the current v2 producer registry
entry unchanged. Scorer v4's source closure separately binds the inference API.
Future training may explicitly choose the new v3 callback, but its optional
registry entry must not replace old producer provenance. No producer/portable
registry, descriptor, checkpoint or existing integration package is edited.

The final source authority schema is `final-selected-scoring-source-v4`.
Existing campaign GO/roster/seal/outcome API schemas remain v2. Root must review
this coherent patch and deliberately bind the new scoring source before
execution. This package creates no campaign authority or giant training seal.
`score_fit` remains campaign-bound and proceeds from whole VALID to TEST; a
bounded engineering qualifier must call the exact components and stop after
VALID without fabricating those authorities.

Metadata and scalar fixtures establish control flow and source preservation,
not prediction parity or whole native restore. Every actual final fit still
needs its own complete selected VALID reconstruction with exact IDs/labels/
endpoints/state and fixed `rtol=1e-4`, `atol=1e-5`. No tolerance tuning or tiny
difference retry is introduced. Native TEST topology/cache/resources and the
other existing final-readiness gates remain unqualified.
