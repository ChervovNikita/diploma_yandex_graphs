# Proposed NCN interaction integration

Status: source design only. The four link interactions are **not implemented**
and must remain pending in the expanded study. No native construction,
optimizer transition, dataset load, fit or pilot is authorized by this plan.

## One native source and one explicit owner

Create a new `interaction_link_ncn_v1` standalone callback and session proxy.
It constructs a fresh inner `link_native_integration_v1` session using the
unaltered `gnnm_boundary_native` base and explicit actual repository root. It
then admits only actual NCN, four original routes, zero updates, no existing
candidate/transform, original 65,536-positive + 65,536-negative source batch,
the original 64-wide encoder-input and task-final boundaries, and TEST closed.
Reject NCNC and BUDDY. No standalone residual, contrast, responsibility or
aggregation transform is called on another transform or relaxed.

The new factory verifies the exact ingredient packages and native sources,
assembles their tensor helpers directly, owns its complete method metadata,
and owns all state/load/close/update/validation dispatch. Existing frozen
callbacks, source model classes, native Parameters, sampler, adjacency mask,
source optimizers and original route/dropout RNG streams remain unchanged.

## Four recipes

| Recipe | Private components | Shared credit | Private credit | Serving |
|---|---|---|---|---|
| I1 `interaction_geometry_input` | Retained BE + portable input correction; original endpoint query contrast | F | F + 0.05 A_endpoint | Mean raw logits |
| I2 `interaction_responsibility_private_pool` | Original BE; corrected endpoint responsibility v2 | F | 0.5 W_endpoint_v2 + 0.5 L_mean | Mean raw logits |
| I3 `interaction_input_meanJ` | Retained BE + portable input correction | J = 0.5 F + 0.5 L_mean | Same scalar J | Mean raw logits |
| I4 `interaction_input_conditionalJ` | Retained BE + portable input correction; conditional20 readout | J = 0.5 F + 0.5 L_actual | Same scalar J | Actual conditional raw logits |

F is the mean of four original native positive-mean-plus-negative-mean route
losses over **all** source query rows. A uses the existing source-compatible
endpoint contrast, not node class geometry or C3 factual-context affinity.
This task-specific I1 has exactly its original shared-F/private-(F+0.05A)
credit. Do not silently replace it with an all-block contrast objective.

W is the corrected v2 semantic-query endpoint weight law with floor 0.25 and
normalization separately within each original source group and route in the
current minibatch. It is not the older `link-ext-v1` responsibility hash. Its
normalized marginals are not claimed matched or unbiased.

L_mean and L_actual use the same original native source reduction on the
served raw logits. The readout does not use labels, hidden target information
or new model forwards. Conditional descriptors are detached confidence and
JS-to-mean values; weights retain the fixed 0.05-per-route floor. The frozen
conditional helper has 20 parameters; mean has none.

## Direct component assembly

For I1/I3/I4, use only `portable_private_input_v2/core.py`'s `build_residual` at
`encoder.convs.0.lin` encoded width 64. Append its hook after the retained BE
output hook. Keep one native body and both BE boundary banks. Rank is 8;
four disjoint QR frames fit this width; up matrices start at zero. Factor seed
is fit seed + 901003, matching the node/molecular interaction convention.
The correction has 4,096 trainable parameters and joins the native encoder
anchor's existing Adam group (LR 0.0082), without changing group options.

For I4, use `conditional_aggregation_v1/core.py` directly with binary task,
conditional mode and the actual fit seed. Its parameters join the native
task head's existing Adam group (LR 0.0037). I3 uses its parameter-free mean
helper. Do not create dummy learned readouts in I1 or I2.

Extend the native core's private-parameter accessor once to include actual
residual/readout Parameters. Verify original native Parameters by object
identity and verify that shared/private blocks form a complete disjoint
partition of the original optimizer's now-extended ownership. Check original
group objects and all options before/after assembly.

## Native observation and gradient assignment

Wrap only the original `predictor.multidomainforward` call boundary and native
`_step` orchestration, following the already checked NCN observer pattern.
For all four routes authenticate actual positive and negative endpoint arrays
against the source sampler order. The predictor still executes its original
whole-minibatch positive-edge mask and original graph operators. A prepended
final-leaf prehook captures the live prehead before final BE input factors.
No extra predictor or backbone pass is added.

I1 calls the frozen `link.build_objectives` with endpoint contrast, private
auxiliary credit, coefficient 0.05, temperature 0.2, 1,024 anchor limit and 16
existing opposite-group endpoint alternatives. Preserve its zero-support
policy and actual per-group coverage. Use the four real capture signatures.

I2 forms actual four-route query batches using the corrected
`structured_responsibility_control_v2.sampler`, authenticates each route in
that namespace, and calls its objective helper in endpoint mode. Add the
existing mean-raw-logit native pool loss to obtain the declared private
criterion. Do not substitute an old responsibility law or fabricate four
signatures from one unobserved route.

I1/I2 use `link.step_blocks`: shared gradients from F and private gradients
from the declared private criterion, collected at the same old state, then
one original Adam transition. Count two reverse collections. I3/I4 use one
ordinary scalar backward on J, then one original Adam transition. Check
finite gradients before transition. Preserve None gradients of genuinely
inactive native Parameters. Log F, L, W/A where relevant, both block credits,
coverage and actual source-work counts separately.

The proxy `train_epoch` must call its own `train_update`, keeping the native
epoch accumulator and exact dropped-tail source semantics. Do not allow the
inner epoch driver to bypass the new observer.

## Serving, evidence and state

Full VALID runs the original native four routes once over all 60,084 positive
and 100,000 negative queries. I4 applies its gate to those same raw outputs;
I1–I3 serve the mean. Preserve the same-call arithmetic mean output, native
loss and Hits@50 as fixed diagnostics for the selected I4 bank. Cache those
diagnostics with the source update clock and invalidate before TRAIN or any
state restore. A zero-forward sidecar getter must require a fresh current
VALID. Generic runtime production export must bind the selected checkpoint
and persist both served and same-bank mean fields explicitly; this remains
a separate integration gate.

Wrap the inner native integration state once. Include exact method,
ingredient hashes, source/data bindings, initialization, optimizer assignment,
native model/Adam/sampler/global-and-private-RNG/module-mode/epoch state and
any fixed component definition. Rebuild fixed plans from saved native query
identities; include actual plan/signature receipts. Reuse the v2 device-safe
endpoint gather rule: each index_select uses that tensor's own device, then
authenticate endpoints on CPU. No legacy checkpoint migration.

Close only this factory's hooks/observers and restore its patched callables.
Do not alter any host setting, native source file or running worker.

## Gates before campaign inclusion

First implement and freeze the actual new package with all ingredient pins.
Then separately propose a finite real-data CPU lifecycle check for all four
recipes, including fresh state restore and whole VALID, source reduction
parity, ownership, both credit rules, correction/readout gradient support,
and original route/sampler/RNG work counts. No native execution is part of
the present source plan. CUDA lifecycle, two-worker admission, selected-bank
production export and final TEST reconstruction remain pending. A successful
smoke check is not an accuracy, convergence or novelty result.
