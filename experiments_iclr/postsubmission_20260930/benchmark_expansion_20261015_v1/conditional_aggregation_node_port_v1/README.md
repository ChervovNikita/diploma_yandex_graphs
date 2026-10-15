# C6 native node bridge v1

This bridge ports the already frozen C6 mean/global/conditional methods to the
exact native `NodeSession` PolyFormer boundary source. It imports the unchanged
`conditional_aggregation_v1/core.py` under the full original package manifest
`edef4b1f552a495e09e64855afbd96c7eb14ce5ef02d26cc13ec530d80e9805f`.
The fixed recipe, seeds, positive floor.05, detached entropy/JS features and
0/4/20 parameter counts are unchanged. No new recipe variant, grid, copied
backbone, teacher, or novelty/quality claim is added.

## Native contract

TRAIN calls the unmodified source `_routes(True)` using nonreentrant activation
checkpointing and source RNG preservation. Each route's original CE is reduced
over the complete source TRAIN nodes. F is their mean. L is full-TRAIN CE of the
actual served raw logits. Every shared/native, boundary, and gate parameter
receives J=.5F+.5L through one backward and one original Adam transition.

All original per-parameter Adam groups and options are retained. Gate parameters
join the existing `native.lin3.weight` group and inherit its native learning
rate and weight decay. The source has no separate explicit loss regularizer;
native optimizer weight decay is not reconstructed as a loss penalty. This
also means W can move through inherited Adam weight decay on the first update
despite its initial zero task-loss gradient.

VALID follows `_routes(False)`, evaluates actual served logits on the complete
source VALID role, and returns those outputs/accuracy for selection. Same-bank
mean outputs/loss/accuracy use the same four forwards as diagnostics. No gate
parameter or feature is fit/selected on VALID. Native transductive graph inputs,
source feature normalization, cached basis, class order and split are retained.

The bridge authenticates frozen source files, original source methods/types,
native Parameter identity, exact boundary sites, and exhaustive optimizer
ownership. State wraps the original native state with exact method/recipe/
source/input/role/seed bindings and module training modes. Full features,
edges and source basis are hashed; only TRAIN/VALID targets are hashed.
Original Parameter objects remain in place on restore. Cross-method and
active-route save/restore are rejected. The controller owns global RNG.

## Runtime binding

Use the existing source callback, `family=node`, one of the same C6 procedure
IDs, `base_procedure=gnnm_boundary_native`, `TEST_access=false`, and
`session_transform={path: conditional_aggregation_node_port_v1/transform.py,
function: transform, sha256: <actual source hash>}`. Pin this bridge's exact
`MANIFEST.json` hash as the only `method_settings.manifest_sha256` value.
Bind `recipe._repository_root` explicitly to the actual repository root, and
preserve the source recipe, `_seed`, redesign root and native controller.

This source freeze precedes the separately requested exact CiteSeer-Cheb CPU
qualification, whose fixed plan, attempts, results and costs live under
`qualification/conditional_aggregation_node_port_v1_cpu_bridge_v1`. Other node
cells, CUDA/resource admission, links, full scientific fits and final selected
TEST reconstruction remain unqualified/pending. No Physics task is performed.
The original C6 package and molecular bridge remain byte-identical.
