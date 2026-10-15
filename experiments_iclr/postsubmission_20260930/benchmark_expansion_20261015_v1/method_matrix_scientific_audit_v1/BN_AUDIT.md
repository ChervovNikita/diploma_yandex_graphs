# Read-only audit of private BN state

Scope: `private_bn_state_v1/banks.py`, its molecular extension, the frozen OGB
boundary route API and session checkpoint methods. Parent reports six CPU
semantic checks and one actual MolBACE GIN32-TRAIN-graph forward/gradient audit
with zero optimizer transitions. This audit did not execute or repeat them and
does not infer VALID/CUDA/resume or prediction-quality evidence.

Subsequent parent update: the two entry-point/checkpoint hazards reported here
were repaired in NEW `private_bn_state_v3`. Parent reports seven allocation CPU
tests and the32-graph native equality audit passed with zero optimizer steps;
the intervening v2 Torch2.1 hook-availability failure was preserved. This file
retains the original v1 findings and reviewed snapshot, rather than representing
them as unresolved defects in v3. The v3 implementation was not independently
executed or re-audited by this agent.

## What is sound in the inspected design

- Private running mean/variance/counters are registered child-module buffers,
  not extra learned Parameters or copies of the native backbone. Shared affine
  Parameters remain original objects and need no new optimizer ownership.
- Temporarily binding BN buffer pointers avoids repeatedly copying values into
  a buffer saved by an outstanding autograd graph. Restoring original pointers
  in `finally` is appropriate when four route forwards precede one backward.
- The wrapper's `run_member` enters private BN state before the unchanged native
  route forward and restores canonical objects afterward. Existing molecular
  TRAIN/VALID callbacks use this entry point, including internal GIN/virtual-node
  BN calls that execute inside the forward.
- Outside contexts, `model.state_dict()` contains both original resident BN
  buffers and every private bank. Default strict `load_state_dict(assign=False)`
  copies into registered objects. Reconstructing the same attached extension
  **before** loading therefore has a viable checkpoint path. A base model
  cannot silently accept the extra keys under strict load.
- For ordinary TRAIN-mode BN, current batch statistics determine output;
  ownership of running buffers changes eval behavior, not mathematical TRAIN
  normalization. No quality conclusion follows from matching TRAIN gradients.

## Substantive hazards in the inspected entry-point-only version

### 1. Direct forward bypass can silently use initial resident statistics

The extension wraps `model.run_member`, but the wrapper's `forward`,
`model.native`, and `with model.route(m): model(batch)` remain callable without
the BN manager. The original resident buffers are deliberately restored outside
contexts and remain at their initial state during private-bank training. Such a
call can therefore return plausible finite predictions using **initial** BN
statistics, not trained private statistics or the counterfactual shared EMA.

README labels direct calls unsupported, but that does not make bypass fail.
A final-serving or diagnostic path must reconstruct the extension and use its
route entry point. Prefer an actual native-forward guard that checks the BN
manager's active member agrees with the boundary wrapper's active member. It
should reject both a boundary-only route and a BN-only route. This is a direct
same-context consistency requirement, not an architecture change.

### 2. Saving/loading inside an active route needs explicit rejection

During a context, native BN buffer names point to one registered bank's tensors;
the model also registers those tensors through the bank module. Calling
`state_dict()` then can serialize active private values under native names as
well as bank names. Loading at that point can copy through aliased paths and
does not represent the canonical outside-context checkpoint contract.

Normal controller checkpoint calls after the forward are outside the context,
so this is not evidence that the reported checkpoint flow already failed.
However, the source has no explicit save/load guard for active state. Add
state/load preconditions or hooks rejecting an active manager and checking
canonical identities before serializing/restoring. Fresh-process/default strict
load, mode restoration and next-update replay remain actual qualification needs.

### 3. Bypassing does not recover a same-weight shared-state control

The canonical resident buffers are an identity anchor, not a saved shared EMA
history. A same-weight comparison of private versus shared statistics cannot
use them as the shared condition. A real shared EMA history must be recorded
prospectively, or supplied by a correctly paired shared-state run. Old shared
checkpoints cannot reconstruct private histories; private checkpoints cannot
reconstruct the absent shared history either.

The mathematical equality of TRAIN normalization permits such a future
state/readout study under matching weights/inputs/RNG, but equality must not be
assumed across controller-dependent stopping or different selected epochs.

## Boundaries already disclosed, not new defects

Post-attachment device/dtype migration can replace buffer objects retained in
identity snapshots; current identity checks should fail, and README explicitly
excludes migration. Attach after final device placement and reconstruct before
load. SyncBatchNorm, concurrent/nested routes and activation recomputation
through BN are excluded. Recomputed native forwards after route exit would
need a different context law; the present ordinary forward audit does not cover
that case. Untracked BN has no running-state object to privatize, and unused
branches must keep their counters unchanged.

The molecular extension pins banks.py and the full runner additionally binds
the job/extension source. Its native session state alone is not a universal
algorithm-version receipt. Final serving must retain those source bindings and
reconstruct the bank manager before restoring selected model state. BUDDY
requires its own route bridge; no BN should be inserted into BN-free backbones.

These findings refer to the reviewed source snapshot. A later guard repair
should retain a new source receipt and focused qualification; it does not
retroactively supply prediction-quality evidence.
