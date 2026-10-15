# Compact production authority for original BUDDY inputs

This new native-base callback constructs the original BUDDY source at the
actual requested seed. A compact static manifest authenticates unchanged
GRAPH/TRAIN/VALID files, native fixed settings and static graph/hash/features.
It contains no sampled negatives, initialization, TRAIN sampler or hydrated
tensors. No saved seed0 oracle tensor file is read or injected.

The original source returns every sampled ID and hydrated input object. The
provider records actual source-specific initial parameter, negative-set and RNG
fingerprints, a chain over native sampler calls, and compact first/second/tail
input observations. The audit consumes no additional RNG, source forward,
hydration or optimizer transition. Existing full native state and the audit
ledger have a new checkpoint identity.

An explicit query-only mode supports one finite nonzero-seed input parity
gate. It blocks every model forward and any attempt to save its traversed
sampler as a trained fit. Traversing all1585 source batches is not a training
epoch; only three TRAIN batches and two whole VALID roles are hydrated.

This source supports the native gnnm_boundary_native base only. New integration
versions for expansion,C3,privateBN and interaction callbacks remain pending.
Input parity cannot qualify their optimization/restore lifecycle, full epochs,
selected serving or CUDA. All frozen seed0 oracle sources/receipts remain
unchanged. This artifact adds no scientific run or TEST permission.
