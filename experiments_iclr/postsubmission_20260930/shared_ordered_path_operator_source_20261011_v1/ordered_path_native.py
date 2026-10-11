"""Factories around caller-owned native providers; stdlib-only import.

No numerical/model/data import, optimizer, reader, selector, loop or launcher.
Original models.Model and existing common_routes/factors are supplied by root.
"""


def parameter_formula(feature_dim, classes, *, width=128, members=4, depth=2):
    """Integer source derivation: multiplier1 SAGE, affine LayerNorm, all maps BE."""
    for value in (feature_dim, classes, width, members, depth):
        if type(value) is not int or value <= 0:
            raise ValueError("Positive integer source dimensions required")
    shared = (feature_dim * width + 5 * depth * width * width
              + (3 + 5 * depth) * width + width * classes + classes)
    private = members * (feature_dim + (2 + 9 * depth) * width + classes)
    path_shared = 5 * width * width + 2 * width
    path_private = 7 * members * width
    return {"native_shared": shared, "native_private": private,
            "native_total": shared + private,
            "path_shared": path_shared, "path_private": path_private,
            "path_increment": path_shared + path_private,
            "native_plus_path": shared + private + path_shared + path_private}


def _check_sage(body, depth, width):
    if body.model_name != "SAGE" or len(body.residual_modules) != depth:
        raise ValueError("Exact saved native SAGE propagation depth required")
    if tuple(body.input_linear.weight.shape)[0] != width:
        raise ValueError("Native stem width differs")
    if body.output_normalization.__class__.__name__ != "LayerNorm":
        raise ValueError("Node-wise native LayerNorm required")
    for residual in body.residual_modules:
        if residual.normalization.__class__.__name__ != "LayerNorm":
            raise ValueError("All original block normalizations must be LayerNorm")
        module = residual.module
        if module.__class__.__name__ != "SAGEModule":
            raise ValueError("Original SAGEModule layout required")
        conv = module.conv
        if conv.__class__.__name__ != "SAGEConv" or conv.aggr != "mean":
            raise ValueError("Original mean SAGE operator required")
        if conv.project or conv.normalize or not conv.root_weight:
            raise ValueError("Native project=False/normalize=False/root_weight=True")
        if conv.lin_l.bias is None or conv.lin_r.bias is not None:
            raise ValueError("Saved native SAGE left/root bias convention differs")
        feed = module.feed_forward_module
        if tuple(feed.linear_1.weight.shape) != (width, 2 * width) or tuple(feed.linear_2.weight.shape) != (width, width):
            raise ValueError("Exact native multiplier1 two-map FFN required")


def _native_body(models, *, depth, feature_dim, classes, width, dropout):
    # Caller controls original constructor seed/provider; no reseeding here.
    return models.Model("SAGE", depth, feature_dim, width, classes, 1, 8,
                        "LayerNorm", dropout)


def _build_native_shared(torch, models, factors, routes, *, depth, feature_dim,
                         classes, width, dropout, members):
    body = _native_body(models, depth=depth, feature_dim=feature_dim, classes=classes,
                        width=width, dropout=dropout)
    _check_sage(body, depth, width)
    work = {"native_route_calls": 0, "native_residual_calls": 0,
            "native_node_block_rows": 0}

    class CountedAdapter(routes.NativeModelAdapter):
        def native(self, batch):
            result = super().native(batch)
            work["native_route_calls"] += 1
            work["native_residual_calls"] += depth
            work["native_node_block_rows"] += depth * len(batch["x"])
            return result

    adapter = CountedAdapter(torch, body)
    model = routes.wrap_shared(torch, factors, body, adapter, members=members,
                               kind="baseline")
    model.path_packet_work = work
    model.path_packet_kind = "depth%d_shared_native" % depth
    return model


def build_depth2_shared(torch, models, factors, routes, *, feature_dim,
                        classes, width=128, dropout=.2, members=4):
    """Unchanged shared2 comparator, including its original native forward."""
    return _build_native_shared(torch, models, factors, routes, depth=2,
                                feature_dim=feature_dim, classes=classes,
                                width=width, dropout=dropout, members=members)


def build_depth3_shared(torch, models, factors, routes, *, feature_dim,
                        classes, width=128, dropout=.2, members=4):
    """Capable radius3 control: unchanged three-layer native source and wrapper."""
    return _build_native_shared(torch, models, factors, routes, depth=3,
                                feature_dim=feature_dim, classes=classes,
                                width=width, dropout=dropout, members=members)


def build_path_shared(torch, models, factors, routes, path_factory, *,
                      feature_dim, classes, init_seed, mode="ordered",
                      width=128, dropout=.2, members=4):
    """Depth2 native plus registered path maps; root creates optimizer afterward.

    path_factory is make_path_block from this packet, supplied without importing
    numerical modules. All modes differ only in the path input/pooling operation.
    """
    if mode not in ("ordered", "marginal", "order_breaking"):
        raise ValueError("Exact prospectively chosen path/control mode required")
    body = _native_body(models, depth=2, feature_dim=feature_dim, classes=classes,
                        width=width, dropout=dropout)
    _check_sage(body, 2, width)
    work = {"native_route_calls": 0, "native_residual_calls": 0,
            "native_node_block_rows": 0, "native_stem_calls": 0}

    class PathAdapter(routes.NativeModelAdapter):
        def check(self, batch):
            super().check(batch)
            if "walk_indices" not in batch or "walk_swap_mask" not in batch:
                raise ValueError("Root must supply the common frozen label-free table")

        def native(self, batch):
            b = self.body
            # Exact saved Model.forward order; stem generated once per route.
            stem = b.act(b.dropout(b.input_linear(batch["x"])))
            hidden = stem
            for residual in b.residual_modules:
                hidden = residual(batch["graph"], hidden)
            hidden = hidden + b.path_residual(stem, batch["walk_indices"],
                                            batch["walk_swap_mask"], mode=mode)
            representation = b.output_normalization(hidden)
            logits = b.output_linear(representation).squeeze(1)
            work["native_route_calls"] += 1
            work["native_residual_calls"] += 2
            work["native_node_block_rows"] += 2 * len(batch["x"])
            work["native_stem_calls"] += 1
            return logits.unsqueeze(-1) if logits.ndim == 1 else logits, representation

    adapter = PathAdapter(torch, body)
    model = routes.wrap_shared(torch, factors, body, adapter, members=members,
                               kind="baseline")
    # Existing native-name snapshot stays native. New W/b -> shared_block;
    # dynamic member_context/parameter_roles discover added private vectors.
    native_maps = model.factorized_maps
    body.path_residual = path_factory(torch, factors, width=width,
                                     members=members, init_seed=init_seed)
    model.native_factorized_maps = native_maps
    model.path_factorized_maps = 2
    model.factorized_maps = native_maps + 2
    model.path_packet_kind = mode
    model.path_packet_work = work
    return model
