# interaction_BUDDY_production_v1

Source successor of frozen `interaction_BUDDY_v1`. Only the input authority and checkpoint
nesting change. The original method objectives, coefficients, native joint
mixed-batch BCE, query targets, optimizer ownership, shared/private BN law and
served readout remain as defined by the pinned parent.

The callback creates exactly one genuine native base through
`buddy_native_provider_v1/callback.py`, then assembles this one declared method.
The production runner must call its exact `seed_all` at the requested fit seed
before construction. This callback never resets a seed or supplies stored
negative queries, sample indices or hydrated tensors. Static metadata is shared;
actual inputs and sampler/negative-set identities remain source-owned per fit.

Required settings are the exact method manifest, provider manifest, compact
static manifest, and `provider_mode=production_native`.

The provider owns input roles and audit ledger; the outer method owns its
original method state. New checkpoint schemas include both. Old checkpoint
migration is unsupported. Frozen seed0 oracle packages are unchanged.

Source-only contract checks do not qualify nonzero-seed optimization, restore,
selected serving, full native epochs or CUDA. Those gates and compiler overlay
remain pending. No TEST or scientific campaign execution is authorized here.
