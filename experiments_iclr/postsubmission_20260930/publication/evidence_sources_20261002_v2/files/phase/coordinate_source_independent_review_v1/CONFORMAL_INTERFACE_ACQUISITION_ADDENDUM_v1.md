# Conformal interface and acquisition addendum

This is a source-only addendum to `CONFORMAL_DRAFT_INTERFACE_REVIEW_v1.md`. The author has sealed the pilot source package; no sealed source was edited and no independent hash signoff is asserted here.

## Selected-only serving follow-up

Static source now implements the requested economical deployment paths:

- HeAD: `head_variant_scores` evaluates only the selected literal transform. Serving supplies canonical point probabilities, APS scores, degree and canonical edges, avoiding full member features and the other family transforms. Source selection still evaluates and charges all five choices.
- APS fallback: returns canonical randomized APS before any member feature, covariance or family preparation.
- CF: uses canonical pooled point probabilities, the selected corrector and APS; no full member multiset, covariance or family transform.
- Pooled gate: `prepare_fixed_inputs(feature_mode='pooled')` retains the declared pstar, pbar and degree; zeroes the K*C member-vector slots and both edge channels; skips member sorting and covariance. pbar still requires member softmax means and must not be removed from this already defined arm.
- Marginal gate: retains its complete nodewise member multiset but skips unused centered deviations/variance and covariance work. Shared all-feature source preparation remains a separately charged operation.

No remaining concrete blocker was found in this narrow serving interface check. The author's seal and source hashes were received as author-reported metadata; this reviewer did not recompute them.

## Root acquisition bridge check

Reviewed `coordinate_conformal_execution_root_v1/acquire_role_packs_v1.py` as source, before data admission. The bridge uses the prospectively fixed Squirrel Git blob `a59f6d2e0dc1f11b8e503c426e86f53124c21eb1`, size 208215 bytes, and verifies the Git blob digest over the decoded contents. Expected Squirrel fields, dimensions, published mask orientation and canonical edge conversion match the declared interfaces. Photo uses the bound existing graph and raw archive. Graph/mask preparation occurs before raw-label dereference within each graph. Source packs contain only compact nodes/labels for train/validation/A/B/D; final pool packs are written under a separate sealed-pool root. No training or scoring call is present.

Two actionable items were sent to the coordinator before freezing this bridge:

1. **Bind extraction to the exact provider bytes used.** The initial bridge verifies Photo raw by path digest and Squirrel by Git blob, but later NPZ label extraction reopens filesystem paths. Parse the Photo raw archive from a byte buffer whose SHA256 is verified and use the Git-verified Squirrel buffer for its reads; retain that extraction provenance. Recheck the separate driver binding at completion, rather than depending on its optional duplication in `protected_files`.
2. **Describe inherited Photo feature processing accurately.** The generic metadata phrase “raw values cast toFP32; no normalization” does not describe a reused PyG public graph's upstream provider transformation. The retained provider `read_npz` reconstructs CSR features, casts FP32, binarizes positive values, removes self loops and symmetrizes adjacency. Declare reuse of those bound provider features and no additional feature normalization, separately from the new canonical edge conversion.

The retained Photo public manifest explicitly binds the supplied graph SHA256 `615509190e55162b25fa3f382f0d09c18622d11d40b0a57914a870140fae05df` and raw SHA256 `bdb1feb8e6ff42ee44024b04479145029c563fd42fc31bff80af20887ba0439a`, with matching graph dimensions and pinned loader provenance. Its provider source preserves row identities when attaching the raw label vector. This metadata supports the intended graph/raw label pairing; no raw labels or tensors were loaded during this review.

These acquisition findings concern the initial unsealed bridge. Their implementation and final root source freeze remain with the coordinator. No acquisition, prototype/model import, execution, compilation, test, tensor load, evidence rehash or SSH was performed by this reviewer.
