# Literal raw-family SeHGNN source views

This separate inactive packet implements concrete source views for the exact sealed native IMDB driver. It supplies no loader, model construction, fit, optimizer, DataLoader, launcher or supervisor. The caller supplies the root-qualified runtime and the native once-loaded StaticContext and frozen-role SeedContext. The public native reference packet and every earlier seal remain unchanged.

## Actual view construction

| Source | Raw destination/source relations removed | Native feature paths through source | Native TRAIN-label paths through source |
| --- | --- | --- | --- |
| director | MD and DM | 12 of 25 | 6 of 12 |
| actor | MA and AM | 12 of 25 | 6 of 12 |
| keyword | MK and KM | 12 of 25 | 6 of 12 |

Numeric raw relation IDs come from RoleData. Each view deletes both canonical raw edge arrays before DGL aggregation or sparse row normalization. It retains all six typed relation keys, with genuinely empty arrays for the two removed directions and exact original pairs for the other four. No self loops or replacement edges are added.

DGL receives explicit original node counts: 4,932 movies, 2,393 directors, 6,124 actors and 7,971 keywords. This keeps the isolated source population and all query nodes present after removal. All original typed attributes, including the native dense keyword identity, remain attached at graph initialization. The movie-own `M` feature channel remains exactly equal to the full native context.

The unchanged author feature helper runs the full four hops. The unchanged sparse helper rebuilds the full four-hop TRAIN-label products on the newly normalized supports; the original `remove_diag(product) @ TRAIN_label_source` follows without renormalization. All 25 feature and 12 label keys, their full-movie shapes, dtypes and separate namespace order are retained. Every dependent sparse product must be empty, and every dependent feature/label channel must be zero as a result of those native operators. If an installed native provider does not produce the required keys/shapes/zeros for empty relations, construction fails and records the failed cost scope. No synthetic zero cache is inserted as a repair.

This is literal removal of raw source connectivity. It is implemented before support normalization and aggregation. Derived-channel masking would require its own algebra and provider/state checks; it is absent here. Channels that avoid the removed type are unchanged in the typed-walk algebra, but this version recomputes them too. It claims no saved preprocessing work or speedup. Native affine biases and cross-channel normalization can still make later hidden states nonzero; those learned computations remain part of the qualified model.

## Callable seam

```python
views = build_family_views(
    rt, full_static, full_ctx, costs,
    config=ViewConfig(),  # inactive by default
)
```

Root fills and enables an external config using the reviewed source seal, successful native numerical qualification receipt, full factual context binding and frozen role binding. All four bindings are SHA256 strings. The builder checks its own source/dependency seals and exact native helper/model/state-helper files before numerical construction. It imports no numerical provider itself. The native runtime is already adopted by the caller.

The source bindings preserve the reviewed v1 native seam as provenance. A root-accepted metadata-only native successor may supply the actual qualified context while retaining these exact model/helper/state-helper bytes and native protocol source commitment. The qualification binding records the actual admitted caller. Any earlier failed qualification evidence remains preserved; no success is inferred from a metadata correction.

The result maps `director`, `actor` and `keyword` to NativeFamilyView objects. Their `.feats`, `.label_feats` and `.data_size` match the native cache interface. They reuse the factual context's `.data`, `.seed`, `.train_index`, `.valid_index`, `.targets`, `.targets_cuda`, `.train_count` and `.valid_count`. The identical target/index objects and frozen full role order are retained. Non-role targets remain NaN; label propagation reads only TRAIN. No TEST identity/truth is loaded or inferred.

Each completed view exposes `.binding`, `.metadata` and `.source_supply_descriptor(helper_module)`. The descriptor for the unchanged helper is produced after actual raw support/cache checks pass. Its declaration is not a source-only numerical certificate. Metadata binds the actual typed raw IDs, original/retained pair counts and digests, removed family, complete channel keys/shapes, frozen role IDs and source/factual/qualification provenance. The semantic binding excludes timing and receipt-location fields. `SOURCE_VIEW_<family>.json`, `COST_EVENTS.jsonl` and `SOURCE_VIEWS_COMPLETE.json` preserve construction and partial failures. The builder requires fresh receipt names and provides no cache-adoption path.

## Callback integration

The separate adapter/callback author supplies the unchanged helper's `native_forward(member, source_or_None, mode, replay_token)`:

- `source=None` selects the full factual SeedContext.
- Every peer for a recipient's source selects that same named rebuilt view, regardless of the peer's own assignment.
- Complete fixed TRAIN IDs from the full context slice either cache into isolated per-call staging. The source builder creates no iterator, seed or model forward.
- Tokens bind the full TRAIN row order and selected view binding. The callback uses native FP32 source paths and returns FP32 logits for the Bernoulli marginal score, following root's precision amendment. Native own updates retain AMP; native full selection remains FP32.
- Member buffers, modes, canonical cache identity/version and per-view RNG must be guarded by the callback transaction. Native BatchNorm scratch buffers must not leak source calls into the member trajectory. Restoring buffers must preserve the current private trial candidate and the replay autograd tape.

The view builder preserves the caller's existing Python/NumPy/Torch streams without reseeding. It does not provide model/buffer transactions or claim those callback checks passed. Real empty-relation kernel behavior, state-safe helper callbacks and the full native candidate remain separately root-qualified before any admitted fit.

## Costs and verification

All three views rebuild raw features/graphs, normalize supports, run native full feature propagation, clone complete role caches, reconstruct TRAIN-label products, scan shapes/finiteness/structural zeros and release transient graphs/products. These scopes and receipt writes are recorded with the exact native Costs interface. Scopes are nested; their times must not be summed as if disjoint. Root's inclusive caller/fit accounting must also include entry/receipt bookkeeping and retain actual process/GPU peaks. No unchanged-channel reuse or cached-free-cost assumption is made.

`verify_static.py` checks ASTs, exact source hashes, disabled config, native call arguments and the published four-type star's symbolic walks. It constructs no graph or numerical model, reads no data and imports no numerical provider. `STATIC_VERIFICATION.json` records those passed source checks. It explicitly does not qualify numerical empty-support behavior.

`SOURCE_BINDINGS.json` pins the native reference, isolated RoleData seam and unchanged source-supply helper. `LICENSE_SCOPE.json` binds the existing internal attribution/license locator record and claims no blanket redistribution grant.
