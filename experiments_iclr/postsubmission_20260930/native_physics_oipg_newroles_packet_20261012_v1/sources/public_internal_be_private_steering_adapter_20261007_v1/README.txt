Prospective GNCL/private steering control — callable source adapter
=================================================================

Scientific purpose
------------------
The current hidden contrast can separate internal codes without improving the
prediction actually served. This control asks whether steering existing internal
member factors with the published own/pool risk mixture is more useful, while
the live shared core and input/score-head private factors retain exact mean-own
supervision. It adds no capacity, teacher, warm acquisition, router or new graph
view. It does not guarantee competent members, useful diversity or heldout gain.

This is an attributed GNCL/NCL and mixed block update adaptation. The saved
3October graph_mixed_block_objectives_scout already defines reverse shared-own/
private-pool supervision, and mixed_block_objective_closest_priors covers ONE/PCL
shared/member/ensemble objectives and classifier routing. The current difference
is an audited INTERNAL-only phi mixture/alignment, input/classifier psi own-only,
live shared theta own-only, and WikiCS mean-probability vs binary mean-logit pool.
No new split-loss principle, new primitive or novelty certificate is claimed.
Those saved reports and the7October NOTE were reused; no primary paper reopened.

Implementation status and limit
-------------------------------
This packet is source-ready for review as a CALLABLE adapter, not a ready full-fit
CLI or an adopted experiment. No numerical imports, model/dataset/outcome work,
GPU/server/18.77 operations or launch occurred. PublicV2 core/recipe/train.py and
all fixed current families are unchanged. Original portable three-task code
remains the separate full-training interface; its previous runtime/data receipts
do not qualify this new two-gradient-pass control.

Do not directly replace train.py's Session with this adapter. Original train.py
requires result['auxiliary'] and public-Session attribute delegation; this
adapter intentionally exposes a private_loss and has no such delegation. Direct
replacement would currently fail. Full driver integration requires a separately
reviewed successor with explicit method metadata and logging semantics; it is
not silently registered in train.py or the24-cell queue.

Exact callable interface (future authorized use, not executed here)
------------------------------------------------------------------
Place the phase directory on sys.path for this package and the unchanged publicV2
directory on sys.path for its public library. Example one-update use:

  import sys
  from pathlib import Path
  phase = Path('/your/project/phase')
  sys.path.insert(0, str(phase))
  sys.path.insert(0, str(phase/'portable_internal_be_public_interface_20261007_v2'))
  from portable import Session
  from data_interface import load_train_valid, train_batches
  from public_internal_be_private_steering_adapter_20261007_v1 import PrivateSteeringAdapter

  train, valid, data_identity = load_train_valid('molhiv', '/your/project/train.npz', '/your/project/valid.npz')
  session = Session('molhiv', 'be_init_contrastive',6101,'cpu')
  control = PrivateSteeringAdapter(session, mode='alignment_private')
  batch, labels = next(train_batches('molhiv',train,epoch=1,seed=6101,device=session.device))
  charged_update = control.train_step(batch,labels)
  method_identity = control.annotate_run({'data':data_identity})

Use a separately fresh otherwise-identical Session for each matched mode:
alignment_private, hidden_private, gncl_private. GNCL additionally REQUIRES an
explicit fixed Python lambda_value in[0,1]. No value/default/search is adopted:

  control = PrivateSteeringAdapter(fresh_session, mode='gncl_private',
                                   lambda_value=lambda_chosen_by_separate_protocol)

Reference modes reject a lambda; GNCL rejects omitted/nonfinite/learnable tensor
lambda. Jointly minimizing lambda by ordinary inner loss is excluded. Lambda
must be recorded in method_identity. The model constructor arm controls the
unchanged BE initialization; it is not the name of this new loss policy.

WikiCS Session takes polynormer=the pinned model.py path. Collab takes ncn_model
and ncn_utils at the pinned public source hashes. Molhiv needs neither native
repository. Only existing four-member shared BE banks are supported; ordinary
single/untied utility references continue to use their separate public interface.
There is no new parameter, optimizer or checkpoint acquisition in this adapter.

For validation/selection, continue to call unchanged publicV2 evaluate(),
joint_snapshot() and core/selection.local_transition with the underlying session
and control.annotate_run(run). Retain original strict joint selection, WikiCS
local/global transition, live end-local RNG and serving modes. The adapter keeps
Session.forward()/its persistent streams, makes exactly two own-view forwards,
and increments the underlying Session.steps once. Do not use an unannotated
snapshot/run to label this update as the frozen hidden-full arm.

Audited parameter permissions
-----------------------------
theta: ALL shared weights/biases, embeddings, norms, native attention/common
parameters. psi: existing input and classifier boundary private r/s. Both learn
g=gradient(mean-own), including every TRAIN label/member and original1/M.
phi: only explicitly inventoried existing INTERNAL r/s and Molhiv message factors.

Task       Own-only private boundaries                         phi/psi tensors
WikiCS     body.lin_in, body.pred_local, body.pred_global       56 /6
Collab     encoder.convs.0.lin, decoder.lin.8                  12 /4
Molhiv     input, head                                         37 /4

WikiCS global_attn.lin_out transforms hidden features and is INTERNAL despite its
name. Both phase-specific score heads remain protected own-only; inactive
phase parameters retain None gradients rather than fabricated zeros.
For the exact constructed CNLinkPredictor, xcnlin*beta+xij is a width64 hidden
sum before decoder.lin, with ONE scalar score projection at lin.8. Endpoint,
common-neighbor and score-hidden MLPs are internal; they are not separate additive
score heads. Other NCN predictor classes/additive score architectures are not
covered. Native GCN's single input projection is the own-only input boundary.
Molhiv GINE/virtual-node MLP factors and existing pre-message factors are internal;
categorical atom/bond embeddings and trainable virtual node remain shared own.

BACKBONE_AUDIT.json records exact source hashes, named maps/shapes and exhaustive
allowed shared/private names. permissions.py rejects unknown/missing/aliased maps,
extra/unclassified parameters, changed optimizer coverage or shapes; it never
defaults all private parameters to internal. Runtime module/parameter catalog
matching remains unexecuted. No third-party native source is redistributed or
new licensing grant made.

Loss and update semantics
-------------------------
For each stochastic view separately, calculate original per-member own loss and
the actual served-pool loss, then average the TWO view objectives. WikiCS pool
risk is stable -log(mean_m softmax(z_m)_y), without epsilon clipping or replacing
it with softmax(mean logits). Binary pools use BCE-with-logits(mean_m z_m).
Collab sums positive mean logistic risk PLUS negative mean logistic risk, even
when counts differ. Molhiv uses the uniform mean across every finite target/output
entry. No invalid/missing label is silently omitted and pool risk uses all rows.

phi objective by mode:
  alignment_private: L_mean +.05 A
  hidden_private:    L_mean +.05 A +.05 hidden residual contrast
  gncl_private:      per-view[(1-lambda)L_mean +lambda L_actual_pool] average +.05 A

GNCL omits hidden residual entirely. Existing A/hidden residual primitives and
the original at-most512 evenly spaced TRAIN auxiliary positions are reused.
Own/pool risks are FULL-label losses, not that bounded auxiliary subset.
All three modes have IDENTICAL phi eligibility and theta/psi permissions.
The current running hidden-full arm also steers shared W, so it cannot substitute
for the matched hidden_private reference without that permission confound.

First collect own gradients for ALL parameters with retain_graph=True. Then
collect the selected private objective's gradients ONLY for phi. Both collections
use the same old parameters/views and are checked finite. Shared/classifier
operations remain connected for private pullback; no hidden/logit peer detach
removes earlier-factor or cross-member paths. Assign theta/psi own gradients and
phi mixed gradients, preserving unused None entries. Commit ONE unchanged Adam
step; there is no theta-first step, alternating learner, clipping or multiplier.
Post-step parameter/Adam finiteness uses the unchanged public helper.

Charge two autograd.grad reverse sweeps per update, vs one reverse sweep in the
source shared update, plus retained graphs/stored gradients and objective work.
The counters/return record those sweeps,2M member forwards and one Adam step.
No timing/memory parity is inferred from equal updates. This mixed vector field
is generally not the gradient of one scalar global objective; native Adam does
not guarantee descent, competence or served prediction utility.

Only conditional next representative scope
------------------------------------------
If separately authorized, one bounded matched fixture can resolve actual module
catalog/eligibility, lambda0 equivalence to alignment-private as a mathematical
reference, finite old-state/supplied gradients, protected own boundaries, live
core/private updates, source RNG/selection behavior and charged extra backward
memory/time. Use the exact original full TRAIN batch/two views and one externally
declared lambda for GNCL; no coefficient grid or full cohort is proposed here.
WikiCS local/global inactive blocks need catalog coverage if that task is used.
Existing data/runtime evidence can supply inputs; it does not certify this new
gradient adapter. Failure/missing scope is preserved. No compute/adoption or
new heldout label/diagnostic panel follows automatically from this source packet.
