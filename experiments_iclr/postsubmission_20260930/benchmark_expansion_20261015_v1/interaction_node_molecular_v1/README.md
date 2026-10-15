# Explicit node and molecular interaction ports v1

Four new recipe identities from expanded_study_design_v1. One shared native
body and the original native four-route BE boundaries remain in every recipe.
This factory admits a fresh native session using the frozen source validators,
then assembles the ingredients itself. It never invokes, relaxes or edits an
existing standalone transform. All component manifests and source files are
pinned in BINDINGS.json and the complete wrapped checkpoint metadata.

| Recipe | Components and credit |
| --- | --- |
| interaction_geometry_input (I1) | Portablev2 BE input residual plus frozen task alignment+variance: all-block F+.05(A+V). |
| interaction_responsibility_private_pool (I2) | C2 structured fixed TRAIN table; shared dF, private d(.5W+.5L_mean), using the native old-state block-gradient helper. No input residual. |
| interaction_input_meanJ (I3 mean control) | Portablev2 BE input residual; all-block .5F+.5L_mean. |
| interaction_input_conditionalJ (I3) | Same input residual and frozen20-parameter conditional readout; all-block .5F+.5L_actual. |

## Exact ingredient bindings

- Portablev2's unchanged pure builder, rank=min(8,width), CPU float32 QR frame
  policy, zero up matrices, seed=fitseed+901003. Attach after lin1+BE for nodes
  or the complete nine-field AtomEncoder+BE for molecules. Residual parameters
  join the original native input-anchor group; no LR/WD/group option changes.
- Node geometry uses the pinned multiclass objective and fixed512 TRAIN cohort
  selector. Molecular geometry uses the pinned original graph objective on all
  original source-minibatch graphs. Temperature.2, coefficient.05, std
  fraction.5, variance epsilon1e-8 and normalization epsilon1e-12 are unchanged.
- C2 uses the exact fixed-table helper. Native incoming edge records within
  TRAIN class define node quartiles; actual TRAIN graph atom-row counts within
  observed class define molecular quartiles. Exact tied-rank overlap, w=.5+2q,
  no permutation or minibatch renormalization. Interaction state owns this
  table; the original native session table slot stays empty.
- C6 mean/conditional pure helpers, descriptors, floor, uniform start, seed
  offset1301009 and original native head-anchor group remain unchanged. Mean
  adds zero parameters. I3 conditional adds20. No output bias or tuning.

Native operators, categorical inputs, topology, basis, losses, reductions,
source Adam, per-parameter LR/WD, four-route order, BN/dropout behavior, native
batch32/skip law and node activation checkpointing remain unchanged. All new
parameters have exactly one original optimizer owner. Native Parameter objects
are preserved. I2 is block credit with two reverse collections and one step,
not a claimed global scalar objective. Other recipes use one scalar backward
and one step. The native observed-label mask/CE population remains complete.

These are new combined states, not aliases of the running old molecular pilot
or standalone portablev2/C1/C2/C6 states. No old checkpoint is migrated or
stripped. The complete source model/Adam/loader state, component recipes,
fixed table/weights, auxiliary configuration, source/data signatures and
module modes are retained. Controller-owned global RNG remains external.

## Serving, selection and mean export

Native validation returns actual serving logits/metric and same-bank fixed
mean logits/loss/metric from the same complete VALID forwards. Only actual
serving outputs reach the own selector. There is no VALID readout fitting or
best-readout selection. `selected_mean_sidecar(session)` retrieves a detached
same-call artifact with IDs/labels and both outputs, with zero new forwards.
Its cache is invalidated on TRAIN and restore and refuses stale retrieval.

The getter does not select a checkpoint or fabricate an epoch binding. Frozen
generic runtime_v3 adapters discard native extra fields; the eventual new
production finalizer must explicitly persist this getter after validating its
own loaded selected checkpoint, and attach the recorded checkpoint/epoch/job
authority. That export hook is an explicit pipeline boundary, not silently
claimed complete here. The frozen selected-VALID replayer does not recognize
this new interaction generation.

## API and qualification

Use transform(session,job) on a freshly constructed native boundary session.
Require family=node/ogb, one of the four exact method IDs, base_procedure=
gnnm_boundary_native, TEST_access=false, a non-reference policy, explicit
absolute recipe._repository_root, and method_settings containing only this
frozen package manifest SHA256. No composable transform stack or adjustable
coefficient is accepted. Native CPU engineering qualification is separate and
must be frozen before its maximum16 optimizer updates across8 cases. No GPU,
TEST, scientific fitting, convergence, predictive advantage or novelty claim.
