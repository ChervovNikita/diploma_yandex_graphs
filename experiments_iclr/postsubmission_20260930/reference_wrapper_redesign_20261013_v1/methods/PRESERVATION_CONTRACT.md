# Native-preserving boundary-wrapper contract

This is a source specification for a required port. It is not a paper review, a numerical-equivalence report, or execution admission. The roster remains the same **four training-only hypotheses**; the primary references remain native M1, GNNM and genuine ordinary I4. No extra baselines are introduced.

## What the historical source does and what the new port must do

The historical [TABMModel / BEBlock source](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/common_wrapper_native_source_20261010_v1/models.py>) uses a generic residual scaffold, creates new bias-free W, initializes W with Xavier, and uses member B in place of a native affine bias. Its first R has Rademacher signs. Copying that constructor into a newly acquired paper model would not preserve that model's W/b, encoder, normalization or initialization.

The new **native-preserving boundary adaptation** must retain the adopted native model and all its existing parameter objects. For an explicitly declared continuous affine boundary, its operation is:

**y_m = S_m ⊙ native_affine(R_m ⊙ x) + Δb_m.**

The original native affine module still owns and computes with its W and b. Δb is an additive private residual; it does not replace b. Input/output boundary dimensions remain native. Only those two boundary sites receive diagonal private factors and bias residuals; no new dense weights, internal factors, heads or graph operations are added. This bias-preserving adaptation is explicitly distinct from the historical bias-free-W/member-B implementation.

At **R=1, S=1, Δb=0**, the wrapped operation must recover the native function for the same input, state and RNG on a corresponding forward. This is a required construction/algebraic property. No numerical model or small tolerance comparison was run here. Native initialization remains unchanged. The roster adds no initializer hypothesis. The adopted common first-input R initializer is now fixed Rademacher from an isolated Torch CPU generator, with other R/S one and Δb zero. The implemented sign start cannot be called native-equal at initialization; the neutral unit/zero construction property remains required.

## Checklist required before a recipe is executable

- **W/b and other learned objects:** retain native parameter identity, values, ownership and initialization. Do not rebuild native maps or copy weights into the TABM scaffold.
- **Forward equations:** preserve nonlinearities, residuals, attention and native module call order. The wrapper only multiplies continuous boundary coordinates and adds Δb at predeclared sites.
- **Normalization and state:** retain BN/LN/GraphNorm types, affine parameters, eps, momentum, running buffers, counters and train/eval behavior. Declare buffer ownership and route order ahead of time. Repeated route forwards can update native state repeatedly; do not claim the bank has M1's state trajectory, concatenate routes into a new BN batch, freeze BN, or substitute another normalization.
- **Preprocessing and categorical encodings:** preserve feature transforms, atom/bond/type encoding, graph/self-loop/edge ordering, masks, missing targets, samplers and existing augmentation. Categorical IDs cannot be multiplied by private factors.
- **Loss and regularizers:** retain source reductions, masks, class weights, label smoothing, native loss interface and explicit native regularizers. Preserve the regularizer's original parameter set and coefficient once; adding wrapper parameters must not silently extend a native weight penalty to them.
- **Optimizer groups:** preserve native optimizer class, all native group memberships/options, LR/decay/betas/eps, scheduler and update schedule. Predeclare the new private-factor assignment to the corresponding native boundary group; do not inherit the prototype's generic Adam/AdamW defaults.
- **Exact boundary sites:** write the input and output module paths/call sites, native types, shapes and factor dimensions into the recipe before execution. No scan that replaces every linear map and no internal placement search.
- **Raw-logit interface and aggregation:** name the passive native capture site and output-axis alignment before execution. Native output activation stays in its forward. GNNM and I4 use the same predetermined raw-logit aggregation.
- **Fixed paper settings:** retain the assigned native width/depth/attention/dropout, learning rates, duration and selector. The four method coefficients remain fixed, with no placement, capacity or coefficient tuning.

## Input embedding and private-factor limitations

The affine formula above is valid for an explicitly supported continuous input boundary. AtomEncoder/BondEncoder, typed embeddings, lookup tables, multiple input channels and opaque fused stems require a source-specific mapping. Their native categorical encoding and input schema remain intact. A post-encoding diagonal adapter must have its precise continuous tensor site, dimensions and source law pinned ahead of time; it is not automatically the same as the historical input R/S operation. If that mapping is unavailable, mark the recipe **unsupported / needs a port**. Do not substitute a generic float input stem, scale categorical IDs, collapse atom fields or choose a convenient internal projection.

Likewise, multiple heads, fused output activations, conditional decoders and an inaccessible raw-logit tensor need explicit adapters. A unique prehead hook from [the existing native hook primitives](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_fast_graph_model_interface_20261010_v1/common_routes.py>) is evidence only for its declared native layouts. It is not a generic hook for every acquired paper model. Hidden SupCon is included only under its existing categorical/prehead eligibility contract.

## Prediction and aggregation contract

For new GNNM and ordinary I4, use one predeclared domain and rule: **arithmetic mean of four native raw-logit tensors**, then the unchanged native output activation/interpretation where required. Capture those raw logits during the original forward before output activation; retain that activation and the source outputs. Do not remove the activation or recover logits by taking an inverse of predicted probabilities. Native M1 returns its original source prediction unaltered.

For training, L uses the native loss interface associated with this exact aggregate. F and L retain the unchanged native regularizer contribution, so the half mixture includes that contribution once. Where the source loss expects logits, pass the aggregate logits; where it expects the source activated representation, apply the unchanged activation. The mean-probability loss in [the existing GNCL primitive](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/native_SAGE_GNCL_SupCon_2x2_source_20261010_v1/run_family.py>) is implementation provenance, not permission to change this fixed mean-logit rule.

Historical original scores, predictions and files remain unchanged. New results would belong to the explicitly named native-preserving adaptation; this document does not retroactively reaggregate the historical source or claim its score identity.

## Per-recipe specification

[PRESERVATION_CONTRACT.json](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/methods/PRESERVATION_CONTRACT.json>) lists the required fields and an unassigned template. Each adopted recipe must pin native source hashes/paper settings; exact input/output sites and shapes; native W/b identities; encoder/normalization/buffer behavior; optimizer memberships; masks/reductions/regularizers; native output activation and raw-logit capture; fixed initializer; and aggregation. An unassigned template is not an admitted model.

The portable boundary-hook core and four training helpers are **implemented with static source checks only**. They do not qualify any adopted native source-preset adapter, exact site map, buffer/RNG/optimizer frontend or native data provider. No numerical equivalence, model runtime or fit was verified. nn.Linear boundaries preserve the native affine call; nn.Embedding input boundaries factor lookup outputs by R[IDs] and S plus Δb without changing the categorical IDs. Source-specific interfaces beyond these exact leaves remain port gaps. Only source/specification files were read or written: no models, fits, numerical probes, remote access, paper review or additional baselines.
