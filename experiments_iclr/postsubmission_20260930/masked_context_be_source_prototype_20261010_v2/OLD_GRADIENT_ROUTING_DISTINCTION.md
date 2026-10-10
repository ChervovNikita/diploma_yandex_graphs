# Verified prior auxiliary-gradient ownership

The executed WikiCS A/R/C/P source chain is
`internal_BE_WikiCS_unit_mechanism_ablation_source_20261007_v2/run_cell.py`
→ sealed `portable_internal_be_public_interface_20261007_v2` Session
→ V2 `recompute.py`. `make_session` installs only its ObjectivesFacade and
VJP replay; it does not install the separate private-steering adapter.

The V2 replay builds mean two-view own CE +.05 alignment +.05 residual,
with inactive terms replaced by differentiable zero. It differentiates that
ordinary scalar with respect to logits and captured representations, restores
the same dropout streams, then calls unrestricted `torch.autograd.backward`
on each member/view's logits and representation. All eight VJPs accumulate
before one optimizer step. Shared matrices and upstream private factors
receive auxiliary gradients. The captured classifier map receives CE only,
because it is downstream from the auxiliary representation.

The canonical SupCon full3 frozen protocol and normal-host source bindings
point to `portable_wikics_supcon_loss_comparison_20261008_v1`. Its exact V2
replay uses the same unrestricted scalar-gradient routing, replacing the
alignment formula with SupCon Eq2 and setting residual to zero. It does not
make auxiliary gradients private-only.

A different source,
`public_internal_be_private_steering_adapter_20261007_v1/adapter.py`, collects
own-loss gradients for all parameter blocks and a separate selected private
objective gradient for audited internal factors. It supplies own CE only to
shared/boundary blocks, the private objective to internal factors, then performs
one optimizer step. That source is not in the verified A/R/C/S chain.

Therefore the earlier component/SupCon evidence must not be scoped as a
private-only-auxiliary experiment. It tested shared auxiliary gradients for
the particular label-aware representation objectives, provider, population
and initialization. It did not test masked raw-feature reconstruction,
mask-before-all-polynomial-preprocessing, masked CE or a shared reconstruction
decoder. No Q/K or scientific score payload was opened to make this distinction.
Hash bindings and actual inspected source scopes are in `SOURCE_BINDINGS.json`
and `READ_SCOPES.json`.
