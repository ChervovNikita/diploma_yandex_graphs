# Portable boundary core and four training rules

These are preparation primitives. Torch is injected by the caller; these files
import no Torch, model, dataset, optimizer or native worker at module load.
No model, numerical probe or fit was run to prepare them. Source-specific
adapters and data providers are **not qualified by this core**.

## Boundary API

`attach_boundary_core(torch, native_model, bindings, factor_seed=...)` accepts
exactly two `BoundaryBinding` objects: one input and one output module path.
Only exact `nn.Linear` and `nn.Embedding` leaves are supported; the output must
currently be Linear for raw-logit/prehead capture. No model scan chooses sites,
and no module is replaced. Existing hooks and native W/b objects remain intact.

- Linear hooks compute `native_affine(x * R_m) * S_m + delta_b_m`.
- Embedding hooks compute `native_embedding(IDs) * R_m[IDs][...,None] * S_m + delta_b_m`.
  IDs, categorical encoding calls and native embedding options remain intact.
- The first input R uses an isolated Torch CPU generator and Rademacher signs.
  Other R/S start at one and bias residuals at zero. This is one fixed initializer
  shared by all four methods, not another hypothesis.
- Unit R/S and zero bias residuals give the neutral native operation by the
  hook construction law. The actual first-R sign start is not neutral. No
  numerical equivalence or tolerance result is claimed.

`core.run_member(member, *native_args, **native_kwargs)` runs the original native
forward and returns its unchanged output-interface prediction, live prehead
tensor and raw output-boundary logits. The original output activation stays in
the original forward. Both bound leaves must execute exactly once; branching,
reused leaves and keyword-only leaf inputs require explicit source ports. Outside
a route context, the hooks are inert. `core.remove_hooks()` removes only the new
handles. The wrapper core registers the original model as `native` for complete
checkpoint ownership; callers must preserve native-specific extra state/RNG too.

Prehead means the output boundary's existing input tensor before its input
factor. Capture is passive and live. A caller must establish that its selected
output leaf actually produces native pre-activation logits with aligned axes;
the core cannot establish that fact from a Linear type alone.

## Native optimizer frontend

The caller constructs its exact source optimizer over the original native
parameters. `extend_native_optimizer_groups(core, optimizer)` appends private
R/S to the native boundary weight's group and delta_b to the native bias group
(weight group where no native bias exists). Existing native parameter objects,
group dictionaries/options and optimizer state are retained. Group assignment
is validated completely before mutation; no optimizer is created. SparseAdam
and closure-based LBFGS require separate source adapters and are rejected here.

The core does not change native normalization, BN buffers, residual equations,
dropout or RNG ownership. A frontend must pin four-route order and native buffer
semantics. Four forwards can advance native BN state four times; the core does
not claim a four-route bank has M1's state trajectory. Existing clipping,
schedulers, precision/AMP, distributed synchronization and accumulation must be
ported from the native trainer without substituting generic defaults.

## Fixed loss and gradient API

`native_loss_terms` receives four native predictions and raw-logit tensors,
caller-owned native loss functions, and the unchanged native regularizer.
F is mean own task loss plus that regularizer once. L is loss on the mean raw
logits, interpreted by the original native activation/loss interface, plus the
same regularizer once. J is `0.5 F + 0.5 L`. Target masks, native reductions,
label smoothing and class weights belong to those native loss functions.
The same `mean_raw_logits` helper fixes GNNM and ordinary I4's aggregation domain;
M1 keeps its own original source prediction path.

The caller uses its native `zero_grad` before the four forwards, supplies live
scalar losses, and calls exactly one rule:

| Helper | Shared native parameters | Boundary private parameters |
| --- | --- | --- |
| `step_global_half_pool` | J | J |
| `step_private_half_pool` | F | J |
| `step_private_responsibility` | F | weighted F |
| `step_own_supcon` | F + 0.05 A | F + 0.05 A |

All reverse collections precede gradient assignment and one `optimizer.step()`.
An optional source clipping callback runs after all gradients are assigned.
None-gradient behavior is retained for unused native parameters. These helpers
do not create another optimizer, take a private-first transition, run another
forward or change the update budget. Native AMP/DDP/accumulation adaptations
require a source frontend before those recipes can be admitted.

`draw_fixed_responsibilities` draws positive Exp(1) weights once with an isolated
Torch CPU generator and normalizes within observed TRAIN class/native reduction
group. Its law is attributed to the earlier fixed-weight primitive; its Torch
stream is explicitly different from that source's NumPy PCG64 exact draws.
Persist weights and metadata once. `weighted_native_own_loss` uses the caller's
existing per-record loss and reduction, including native positive/negative
groups. It does not invent missing-target handling or class groups.

`within_route_node_supcon` accepts only `task_kind="node_categorical"`. It
requires four live native prehead tensors in exact full TRAIN anchor order,
integer categorical labels and a nonself same-class positive for every anchor.
Every nonself TRAIN row is in the denominator; temperature is 0.2, normalization
epsilon is 1e-12, coefficient is 0.05. No projector, extra stochastic view,
alignment/residual contrast, graph augmentation or inference correction is added.
The native provider must authenticate the supplied complete TRAIN IDs/labels;
the primitive cannot certify data-role custody from tensor equality alone.

## Remaining recipe and provider work

The parent proposal reports 27 adopted source presets and 2,480 planned fits.
This core neither re-audits that roster nor admits those fits. Every adopted
preset still needs its exact input/output bindings and capture semantics, native
constructor/settings/optimizer frontend, source normalization/buffer/RNG policy,
task-loss reductions/regularizers, and source-specific checkpoint state map.
PyG Linear leaves, multiple categorical embeddings, custom/fused heads, opaque
logits and reused boundary leaves are explicit port gaps; no fallback placement
is implemented. Deferred Polynormer head/horizon conflicts and excluded HGB
hidden settings are not resolved or reintroduced by this core.

Native providers must preserve complete raw/categorical inputs, graph/support
semantics, source splits/masks/targets, original samplers and preprocessing,
TRAIN/SELECT/evaluation separation, full native selection and serving interfaces.
They must supply existing native loss callbacks and full TRAIN anchor custody.
No provider, launcher, model-specific binding table or selected state is included.

`metadata_check.py` performs syntax/AST/structural checks and hashes these files
without importing them or constructing a numerical runtime. Its receipt is
static preparation evidence, not numerical or adapter qualification.
