# Engine and future member adapter interface

The public reference runner forces `rt['model_class']` to the exact registered native `model_module.SeHGNN`. It does not install member factors or run a grouped bank.

Numerical providers are imported after source/release/adoption gates. The runtime map supplies `torch`, `numpy`, `dgl`, `SparseTensor`, `remove_diag`, `native`, `model_module`, `model_class`, `device`, and frozen `protocol`.

- `prepare_static(rt, RoleData, costs)` builds the once-only original full typed graph, feature caches and row-normalized sparse supports.
- `prepare_seed(rt, static, frozen_role, seed, costs)` reuses loaded input buffers, verifies native shuffled role identity, clones all features, makes masked targets and TRAIN-only label caches, and constructs the native loader/prebuilt known-role serving list.
- `make_model(rt, context, costs)` constructs the exact native model on CPU then places it ordinarily. Its signature is the author factory signature including `data_size`.
- `evaluate(rt, model, context, counters, costs, scope)` serves complete known roles while preserving current RNG position.
- `run_one(rt, static, role, seed, folder, costs, scalar, identity, qualify=False)` owns a single literal native fit or qualifier. Root passes one master scalar across the ordered author seed loop. The `costs` argument is the cohort record context; per-fit events are written in a separate local Costs object. Final `RESULT.json` and `COMPLETE.json` must be checked after the call.

The native module is registered as `_owned_sehgnn_IMDB_model_v1` and exposed through `rt['model_module']`. The native IMDB forward takes `(batch, feature_dict, label_dict, mask=None)`. Feature and label namespaces have all 25 and 12 keys respectively, including repeated path strings in separate namespaces. Native model key sorting determines embedding/channel order.

The separate inactive member adapter may consume an already-placed native prototype after root numerical qualification. Its intended sites are both grouped `LinearPerMetapath` layers, semantic query/key/value and `fc_after_concat`. Native task MLP, embedding dictionaries, normalization, activation, dropout, raw-residual flag, internal task residuals and zero initial semantic gamma remain unchanged. The reference source is isolated from that future bank.

Native forward passes `mask=None` into semantic fusion, so source-view family removals must modify the feature/TRAIN-label channel caches while preserving their complete keys/order. A mask argument alone cannot implement those views. Adapter optimizer ownership and grouped update scheduling belong to a separate root-reviewed driver and are absent here.
