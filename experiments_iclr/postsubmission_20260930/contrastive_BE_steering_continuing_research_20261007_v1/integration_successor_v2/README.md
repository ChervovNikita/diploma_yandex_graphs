# Explicit context-target integration successor v2

Source preparation only. Scientific release is disabled. No new model fit,
heldout scoring or native/fullgraph/CUDA qualification has run here.

## The corrected independent reference

`independent4_route_context` constructs the unchanged original
`independent4_contrastive` parameter bodies, Adam optimizers and member dropout
streams. A fixed-target facade replaces the legacy contrastive functions with
route-specific alignment and differentiable zero residual repulsion. It never
normalizes or aliases `model.arm`.

The registered selector restores each member's own local model and Adam history,
preserving the live end-local RNG streams. Each member's final own-selected model
and its own local/global flag forms an evaluation-only bank. That bank is not a
mixed-epoch resumable trajectory. Shared and one-member methods delegate their
local transition to the original joint selector.

Legacy `independent4_contrastive` is a different, coupled method with cross-member
residual contrast and joint selection. It must not enter this selector through
an identity alias. The explicit source contract rejects nonzero residual weight.
The frozen Wiki24 source and all results remain untouched.

## Exact gradient scaling

The original Session uses sum-own-loss plus M times average auxiliary loss for
untied models. The fixed context auxiliary is a mean over independent member
losses. Therefore the untied objective is exactly
sum_m [own_m + .05 alignment_m], with every original model parameter receiving
its own unscaled gradient. Shared predictors retain mean-own-loss plus .05
average alignment. M1 receives its original own loss plus .05 Qbar alignment.

The predecessor CPU helper verified the objective algebra and gradients. The
successor replay uses the same explicit untied sum/shared mean expression.
No original optimizer is replaced and no optimizer steps before the complete
member/view reverse accumulation. Actual native/CUDA gradients remain to be
qualified on the intended runtime; source/fixture checks do not claim that pass.

## Materialized source and costs

- `train_context.py` is an explicit copy of the public full driver with the
  registered context constructor and own/joint selection policy. Complete
  1100 epochs, strict-first selections and native stage schedule are retained.
- `frozen_targets.py` consumes the root-preflight archive and receipt, verifies
  TRAIN/source/panel/permutation identities and hashes, and caches immutable
  logical targets. `context_dispatch.py` uses that same archive for every
  condition; it does not generate or reselect relations.
- `context_recompute.py` is a narrow successor to the qualified Wiki12 execution
  strategy. It handles M1, shared M4 and untied M4 with their correct scaling.
- Every update charges 2M shadow forwards, 2M replay forwards, 2M reverse
  collections and one step of each original optimizer. The two stochastic
  views and exact member RNG endpoints are retained. These extra forwards are
  real training cost and no bitwise author parity is claimed.
- `DRIVER_SUCCESSOR.diff` and `REPLAY_SUCCESSOR.diff` expose every change against
  the bound predecessors. No frozen source file was edited.

The command requires a separate explicit root release with exact source/input,
role-manifest, target-archive and preprocessing-receipt bindings, full staged
protocol adoption, numerical qualification and external supervision. Before
model construction the CPU and actual CUDA panels and ordered TRAIN IDs/labels
are checked against the original frozen archive; each RUN records those verified
logical array hashes. Raw targets remain server-only. The supplied release template is disabled. Stage2 also requires
whole-nine-fit Stage1 gate closure. This package does not launch a family or
automatically admit a passing pilot.

## Checks and next decision

Eight stdlib fixture checks passed: own local model/Adam histories, live stream
preservation, per-member final stage flags, joint delegation for shared/single,
and rejection of coupled objectives or hidden arm normalization. AST checks
also verify the materialized strict-first and replay scaling sites. These are
source checks without frameworks, scientific inputs or numerical training.

Root's complete TRAIN-only preflight passed with distinct target relations,
using the fixed991327 permutation and original512 panel. This is eligibility,
not semantic subclass, accuracy or novelty evidence. The v2 consume-only
contract passed11 NumPy/static checks for changed identities, labels, panels,
permutations and relation contents; the real raw archive was not opened locally. Native model, backward, selector, replay and resource qualification remain
required before scientific admission. Existing Wiki12 and Mol18 are unchanged.
