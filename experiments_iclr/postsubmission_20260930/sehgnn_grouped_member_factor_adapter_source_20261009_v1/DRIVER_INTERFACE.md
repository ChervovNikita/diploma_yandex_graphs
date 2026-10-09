# Exact native-driver seam

The coordinated driver is the separate inactive `sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v1` packet. Its public runner continues to require the exact native class and installs no adapter. This packet adds no alternate input loader, feature preprocessing, training loop, optimizer or supervisor.

The driver agent's declared seams are:

- `prepare_static(runtime, RoleData, costs)` returns raw full feature caches, row-normalized typed adjacencies and data_size.
- `prepare_seed(runtime, static, frozen_role, seed, costs)` consumes the literal native seeded NumPy shuffle, computes all TRAIN-only label channels, and returns complete TRAIN/VALID IDs/targets, loaders, eval batches, feats/label_feats and data_size. No TEST truth is introduced.
- `make_model(runtime, SeedContext, costs)` returns the exact CPU-constructed native model after ordinary final placement, before its optimizer. `runtime['model_module']` is the module loaded as `_owned_sehgnn_IMDB_model_v1`; `runtime['model_class']` is its SeHGNN.

For a separately released bank, use that already initialized/placed prototype:

```python
bank = install_member_bank(
    runtime['model_module'], placed_native_prototype,
    AdapterConfig(enabled=False, members=4),
)
```

This delivered example is inactive. Enablement belongs to root after native/source qualification. Construct the optimizer **after installation**, with shared slow Parameters deduplicated once and all private factors included. `bank.slow_parameters()` and `bank.private_parameters(member)` expose the groups. No optimizer/GradScaler is received or modified by this adapter. Do not reuse an optimizer built for the prototype, reset native parameters after installation, or move/retype the bank afterward; reconstruct a fresh natively placed prototype and bank instead.

Each `bank.members[m]` retains the native signature:

`model(batch, feature_dict, label_dict, mask=None)`.

`bank.forward_member(m,...)` delegates to it. The bank intentionally provides no stacked four-tape `forward`; a later reviewed fit must stream the complete member own losses at the same old state, then take one deduplicated optimizer step. The exact original reference loop is not edited to call a bank. Its native label/feature preparation and full population scoring are reused only through a separately qualified release.

The namespace/order is exactly `sorted(feat_keys)` followed by `sorted(label_feat_keys)`. A path string present in both namespaces remains **two distinct channel rows and two distinct native embedding namespaces**. The grouped W has shape `[C,cin,cout]`, never a guessed Conv/Linear orientation. For `fc_after_concat`, the author first transposes to `[B,H,C]` and flattens; input factor index is `h*C+c`, not `c*H+h`. Q/K/V factors apply to the native last hidden axis and are shared across meta-channel positions within a member.

The source model calls `semantic_fusion(x,mask=None)` regardless of its external mask argument. A family-removed view cannot be implemented merely by passing `mask`. The qualified source-view seam must reconstruct native source support/derived feature and TRAIN-label paths, retain all keys/order/shapes, and make every path involving the assigned family unavailable. Movie own-feature `M` remains. Label diagonal removal and no post-removal renormalization remain native. Feature/label source construction belongs to the driver/root, not this adapter. Do not collapse the 25+12 native cache channels or drop the keyword type.

For the unchanged `source_supply.py` helper, pass `bank.members`, `bank.slow_names` and `bank.private_names` to its complete ownership verifier. No other member-owned slow blocks exist in the shared bank. Its source VJPs target the entire declared private factor block, not just the assigned family's first-layer rows. Parameter names of every original slow object are preserved by the wrappers. The factor fields are `.input_factor` and `.output_factor` under each of the six explicit native sites.

All registered native buffers, including each of the three task BatchNorm running mean/variance/counters, are copied per member; **no registered buffer is shared** in this adapter. Parameter sharing does not imply a common normalization trajectory. Module modes, buffers, prediction-relevant caches and per-view RNG must be captured/restored for every source reference/replay/trial/eval guard. Additional source-view TRAIN calls must not train those statistics or leak them into evaluation. Preserve the current private trial candidate during buffer restoration. The adapter does not supply that native callback/state transaction.

Selected reconstruction must recreate the exact bank with the same native constructor and installation order. Check `bank.verify_shared_state_dict(state)` before loading: duplicated state_dict keys that refer to one shared slow object must have equal values, rather than rely on last-write-wins. Restore each member's distinct buffers/factors plus deduplicated optimizer/scaler/RNG state, verify ownership, then serve every native full-input member with the unchanged helper's explicit full-input probability mean. The original prototype module tree is not modified by installation, although its slow Parameters are the bank's actual shared objects.
