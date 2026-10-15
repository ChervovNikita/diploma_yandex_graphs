# Route-specific BatchNorm running state

This is **v2**, a guarded repair of prepared v1. Review found that calling the
native model directly could bypass v1's route wrapper and silently use canonical
initial buffers. V2 rejects that bypass, including direct tracked-BN calls. It
also rejects saving/loading model state during an active route, when native
buffers are temporarily aliased to a route bank. V1 and its scoped qualification
remain preserved. No v1 scientific fit ran and no frozen pilot was changed.

This candidate tests a specific source of disagreement between training and
evaluation in a shared ensemble. Each route may produce different activation
statistics. The current molecular wrapper applies four routes successively to
the same native running means and variances, then uses that common state for
every route during evaluation. Here each route keeps its own running state.
All learned backbone parameters, including BatchNorm affine parameters, remain
shared. No new learned parameters, backbone copies or label inputs are added.

`banks.py` binds registered route buffers to the unchanged native BatchNorm
module for each original forward. It restores the exact original buffer objects
outside that context. Buffer pointer binding avoids invalidating saved autograd
tensors when four forwards precede one backward. State is included in the normal
model checkpoint. TRAIN updates only the active route; eval never recalibrates.
Untracked normalization is unchanged, unused branches stay unused, and models
without tracked BatchNorm are inapplicable. No normalization layer is inserted.

For ordinary TRAIN-mode BatchNorm, current batch statistics determine its
normalized outputs. Under matching inputs, weights and RNG, the candidate has
the same mathematical TRAIN outputs and gradients as shared running state.
Its eval outputs and therefore checkpoint selections can differ. This is a
testable mechanism, **not** a promise of better prediction quality. It is a
normalization-state control with domain-specific normalization ancestry;
see `../catalogue/BN_STATE.md` for the scoped sources. No novelty is claimed.

The molecular `session_extension.py` is a new, separately pinned callback. It
does not modify the frozen 24-fit pilot. `test_banks.py` checks delayed backward,
momentum=None, eval ownership, state restoration, exceptions, inactive branches
and zero added optimizer parameters on small CPU modules. The first invocation
failed because the bundled Python alone did not include Torch; using the existing
Torch cache resolved that import. V2 adds a bypass/checkpoint test to those six
original semantic cases; qualification results have separate receipts.

`qualify_native_cpu.py` is a finite source-bound audit on one real TRAIN batch.
It constructs native models and compares their four-route forwards and
gradients, with zero optimizer steps. Its result is distinct from training and
from actual VALID/CUDA/BUDDY/runtime qualification. Scientific prediction-quality
fits have not been launched. A future experiment must freeze seeds, source
settings, controls and selection before execution; original paper results and
the active pilot remain untouched.

Not supported by this version: concurrent route contexts, SyncBatchNorm,
activation recomputation through BN, model migration after attachment, or direct
native forwards outside a route. The existing molecular TRAIN/VALID serving
uses `run_member`; a new backend must bind the same context explicitly.
