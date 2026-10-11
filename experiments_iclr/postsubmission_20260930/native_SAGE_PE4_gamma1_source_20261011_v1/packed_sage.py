"""Small declared native SAGE Packed-Ensembles graph adaptation.

Stdlib-only module. The existing root owner supplies Torch and four freshly
constructed ordinary native bodies. Packing preserves their initialized values
without another reset or draw. This file owns no reader, optimizer or launcher.
"""
MEMBERS = 4
GAMMA = 1
WIDTH = 59
BASE_PARAMETER_CAP = 216674


def native_count(width):
    """Whole ordinary Model/SAGE: depth2, F300, C10, FFN1, LayerNorm."""
    if type(width) is not int or width < 1:
        raise ValueError("Positive integer native width required")
    return 10 * width * width + 323 * width + 10


def width_contract():
    current, following = MEMBERS * native_count(WIDTH), MEMBERS * native_count(WIDTH + 1)
    if not current <= BASE_PARAMETER_CAP < following:
        raise ValueError("Fixed largest-width contract differs")
    return {"members": MEMBERS, "gamma": GAMMA, "width": WIDTH,
            "parameters": current, "BASE_parameter_cap": BASE_PARAMETER_CAP,
            "next_width": WIDTH + 1, "next_parameters": following}


def pack_native(torch, bodies):
    """Pack private affine/norm blocks; adopt native parameter-free mean only.

    Members occupy disjoint coordinates of the leading parameter axis. The
    grouped tensors are the only fitted owners after construction. Callers must
    discard the temporary native bodies before fitting or memory measurement.
    """
    if len(bodies) != MEMBERS or len({id(b) for b in bodies}) != MEMBERS:
        raise ValueError("Four distinct fresh native bodies required")
    original_parameters = [p for b in bodies for p in b.parameters()]
    if len({id(p) for p in original_parameters}) != len(original_parameters):
        raise ValueError("Native bodies share parameters")
    if len({p.untyped_storage().data_ptr() for p in original_parameters}) != len(original_parameters):
        raise ValueError("Native constructor storages alias")
    for b in bodies:
        if b.model_name != "SAGE" or len(b.residual_modules) != 2:
            raise ValueError("Original ordinary depth2 SAGE required")
        if sum(p.numel() for p in b.parameters()) != native_count(WIDTH):
            raise ValueError("Native complete count differs from fixed width59")
        if tuple(b.input_linear.weight.shape) != (WIDTH, 300) or tuple(b.output_linear.weight.shape) != (10, WIDTH):
            raise ValueError("Native stem/head dimensions differ")
        if b.act.__class__.__name__ != "GELU" or b.act.approximate != "none" or b.dropout.p != .2:
            raise ValueError("Native stem activation/dropout differs")
        for r in b.residual_modules:
            conv, ff = r.module.conv, r.module.feed_forward_module
            if (r.module.__class__.__name__ != "SAGEModule" or conv.__class__.__name__ != "SAGEConv"
                    or conv.aggr != "mean" or conv.project or conv.normalize or not conv.root_weight
                    or conv.flow != "source_to_target" or conv.node_dim != -2 or conv.decomposed_layers != 1
                    or conv.aggr_module.__class__.__name__ != "MeanAggregation"
                    or any(True for _ in conv.aggr_module.parameters())
                    or conv.lin_l.bias is None or conv.lin_r.bias is not None):
                raise ValueError("Native scalar SAGE mean/root contract differs")
            if (tuple(ff.linear_1.weight.shape) != (WIDTH, 2 * WIDTH)
                    or tuple(ff.linear_2.weight.shape) != (WIDTH, WIDTH)
                    or ff.act.__class__.__name__ != "GELU" or ff.act.approximate != "none"
                    or ff.dropout_1.p != .2 or ff.dropout_2.p != .2):
                raise ValueError("Native FFN multiplier/order/dropout differs")
        for norm in [r.normalization for r in b.residual_modules] + [b.output_normalization]:
            if norm.__class__.__name__ != "LayerNorm" or norm.normalized_shape != (WIDTH,) or norm.eps != 1e-5:
                raise ValueError("Native feature LayerNorm differs")

    class PrivateLinear(torch.nn.Module):
        def __init__(self, maps):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.stack([p.weight.detach() for p in maps]).clone())
            present = [p.bias is not None for p in maps]
            if any(present) != all(present):
                raise ValueError("Bias contract differs among members")
            self.bias = torch.nn.Parameter(torch.stack([p.bias.detach() for p in maps]).clone()) if all(present) else None

        def forward(self, x):
            out = torch.einsum("nmi,moi->nmo", x, self.weight)
            return out if self.bias is None else out + self.bias

    class PrivateNorm(torch.nn.Module):
        def __init__(self, norms):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.stack([p.weight.detach() for p in norms]).clone())
            self.bias = torch.nn.Parameter(torch.stack([p.bias.detach() for p in norms]).clone())

        def forward(self, x):
            # Native fused feature normalization/affine for each private route.
            return torch.stack([torch.nn.functional.layer_norm(
                x[:, m], (WIDTH,), self.weight[m], self.bias[m], 1e-5)
                for m in range(MEMBERS)], dim=1)

    class PackedSAGE(torch.nn.Module):
        def __init__(self, native):
            super().__init__()
            self.members, self.kind, self.width = MEMBERS, "packed_native_SAGE_gamma1", WIDTH
            self.input = PrivateLinear([b.input_linear for b in native])
            self.norms = torch.nn.ModuleList([PrivateNorm([b.residual_modules[l].normalization for b in native]) for l in range(2)])
            self.neighbors = torch.nn.ModuleList([PrivateLinear([b.residual_modules[l].module.conv.lin_l for b in native]) for l in range(2)])
            self.roots = torch.nn.ModuleList([PrivateLinear([b.residual_modules[l].module.conv.lin_r for b in native]) for l in range(2)])
            self.first = torch.nn.ModuleList([PrivateLinear([b.residual_modules[l].module.feed_forward_module.linear_1 for b in native]) for l in range(2)])
            self.second = torch.nn.ModuleList([PrivateLinear([b.residual_modules[l].module.feed_forward_module.linear_2 for b in native]) for l in range(2)])
            self.final_norm = PrivateNorm([b.output_normalization for b in native])
            self.output = PrivateLinear([b.output_linear for b in native])
            self.mean = native[0].residual_modules[0].module.conv.aggr_module
            self.operation_counts = {"packed_mean_calls": 0, "edge_channel_values": 0,
                                     "grouped_affine_calls": 0, "member_dropout_calls": 0}

        def check_optimizer_ownership(self, optimizer):
            expected = {id(p) for p in self.parameters() if p.requires_grad}
            actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError("Optimizer must own every grouped private parameter once")

        def dropout(self, x, route_scope):
            if not self.training:
                return x
            if route_scope is None:
                raise ValueError("Existing persistent member RNG scopes required")
            rows = []
            for m in range(MEMBERS):
                with route_scope(m):
                    rows.append(torch.nn.functional.dropout(x[:, m], p=.2, training=True))
            self.operation_counts["member_dropout_calls"] += MEMBERS
            return torch.stack(rows, dim=1)

        def forward(self, batch, *, route_scope=None):
            if any(k in batch for k in ("y", "labels", "train_mask", "valid_mask", "test_mask", "edge_attr", "edge_weight")):
                raise ValueError("One label-free factual unweighted graph required")
            x, edge = batch["x"], batch["edge_index"]
            if (x.ndim != 2 or x.shape[1] != 300 or edge.ndim != 2 or edge.shape[0] != 2
                    or edge.dtype != torch.long or batch["graph"].edge_index is not edge):
                raise ValueError("Original graph/input interface differs")
            versions = x._version, edge._version
            h = self.input(x[:, None, :].expand(-1, MEMBERS, -1))
            h = torch.nn.functional.gelu(self.dropout(h, route_scope))
            for l in range(2):
                z = self.norms[l](h)
                messages = z.flatten(1).index_select(0, edge[0])
                a = self.mean(messages, index=edge[1], dim_size=len(x), dim=0).reshape_as(z)
                self.operation_counts["packed_mean_calls"] += 1
                self.operation_counts["edge_channel_values"] += edge.shape[1] * MEMBERS * WIDTH
                message = self.neighbors[l](a) + self.roots[l](z)
                branch = self.first[l](torch.cat([z, message], dim=-1))
                branch = torch.nn.functional.gelu(self.dropout(branch, route_scope))
                branch = self.dropout(self.second[l](branch), route_scope)
                h = h + branch
            hidden = self.final_norm(h)
            logits = self.output(hidden)
            self.operation_counts["grouped_affine_calls"] += 10
            if versions != (x._version, edge._version):
                raise ValueError("Factual graph/input was mutated")
            ids = batch.get("ids", torch.arange(len(x), device=x.device))
            return logits[ids].permute(1, 0, 2), hidden[ids].permute(1, 0, 2)

    model = PackedSAGE(bodies)
    if sum(p.numel() for p in model.parameters()) != width_contract()["parameters"]:
        raise ValueError("Registered packed parameter count differs")
    return model
