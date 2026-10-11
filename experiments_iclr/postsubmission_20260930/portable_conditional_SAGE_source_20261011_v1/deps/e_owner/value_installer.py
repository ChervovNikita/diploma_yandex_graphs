"""Narrow factual VALUE port; stdlib-only until the caller supplies Torch.

Reuse native SAGE mean aggregation and native already-constructed lin_l/lin_r.
No native weight reset, new graph support, label input or optimizer is owned here.
"""
from math import sqrt


def _counts_add(total, unit):
    for key, value in unit.items():
        total[key] = total.get(key, 0) + value


def install(torch, factors, body, transport_values, *, members, conditioning, edge_metadata):
    if conditioning not in ("endpoint", "node") or members not in (1, 4):
        raise ValueError("One fixed endpoint/node operation and M1/M4 required")
    if getattr(body, "model_name", None) != "SAGE" or len(body.residual_modules) != 2:
        raise ValueError("Native depth-two SAGE only")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in body.parameters()):
        raise ValueError("Install on fresh float32 CPU body before movement/optimizer")
    if (edge_metadata.get("nodes") != 11701 or type(edge_metadata.get("edges")) is not int
            or edge_metadata["edges"] <= 0 or type(edge_metadata.get("self_edges")) is not int
            or not 0 <= edge_metadata["self_edges"] <= edge_metadata["edges"]):
        raise ValueError("Exact actual full-graph edge metadata required")
    from torch_geometric.nn.conv import MessagePassing

    def builder():
        # Preserve the constructor/native/dropout RNG trajectory. Only these
        # added affine tensors start at zero; existing native tensors are reused.
        with torch.random.fork_rng(devices=[]):
            source = torch.nn.Linear(256, 16, bias=True, device="cpu", dtype=torch.float32)
            with torch.no_grad():
                source.weight.zero_(); source.bias.zero_()
        return factors.FactorLinear(source, members)

    class EndpointValueSAGEConv(MessagePassing):
        def __init__(self, native):
            self.in_channels = native.in_channels
            self.out_channels = native.out_channels
            self.normalize = native.normalize
            self.root_weight = native.root_weight
            self.project = native.project
            # Inspect the new message signature at construction. An explicit
            # generic PyG propagate delegation avoids stale/generated SAGE
            # x_j-only templates. This uses PyG's ordinary collector/mean.
            MessagePassing.__init__(self, aggr="mean", flow="source_to_target",
                                    node_dim=-2, decomposed_layers=1)
            self.fuse = False
            self.aggr_module = native.aggr_module
            self.lin_l = native.lin_l
            self.lin_r = native.lin_r
            self.value_source_builder = builder()
            self.value_target_builder = builder()
            self.value_conditioning = conditioning
            self.value_counts_train = {}
            self.value_counts_eval = {}
            self.value_last_unit = None
            self._node_maps = None

        def propagate(self, edge_index, size=None, **kwargs):
            return MessagePassing.propagate(self, edge_index, size=size, **kwargs)

        def forward(self, x, edge_index, size=None):
            if (not torch.is_tensor(x) or x.ndim != 2 or x.shape != (11701, 128)
                    or x.dtype != torch.float32):
                raise ValueError("Actual homogeneous full WikiCS native width128 states required")
            if (not torch.is_tensor(edge_index) or edge_index.layout != torch.strided
                    or edge_index.ndim != 2 or edge_index.shape[0] != 2
                    or edge_index.dtype != torch.long or edge_index.device != x.device):
                raise ValueError("Native factual COO edge_index, source-to-target, required")
            if self.project or self.normalize or not self.root_weight:
                raise ValueError("Preserve admitted native SAGE project=False/normalize=False/root=True")
            if size is not None and tuple(size) != (11701, 11701):
                raise ValueError("No induced/bipartite/subsampled qualification graph")
            if self.fuse or self.decomposed_layers != 1 or self.explain:
                raise ValueError("VALUE needs the explicit nonfused native message path")
            edges, nodes, c, d = edge_index.shape[1], x.shape[0], 128, 4
            if edges != edge_metadata["edges"]:
                raise ValueError("Factual edge support size differs from admitted full graph")
            if conditioning == "node":
                context = torch.cat((x, torch.nn.functional.silu(x)), dim=-1)
                identity = torch.eye(d, dtype=x.dtype, device=x.device)
                src = identity + self.value_source_builder(context).tanh().reshape(nodes, d, d) / sqrt(d)
                tgt = identity + self.value_target_builder(context).tanh().reshape(nodes, d, d) / sqrt(d)
                self._node_maps = tgt.transpose(-2, -1) @ src
            # This is exactly native SAGE's project=False forward order:
            # propagate -> lin_l -> lin_r(root) addition. SAGEModule retains
            # original [x, message] FFN, residual and outer LayerNorm/dropout.
            try:
                out = self.propagate(edge_index, x=(x, x), size=size)
                out = self.lin_l(out)
                out = out + self.lin_r(x)
            finally:
                self._node_maps = None
            rows = edges if conditioning == "endpoint" else nodes
            unit = {"value_conv_calls": 1, "value_builder_calls": 2,
                    "value_builder_rows": 2 * rows,
                    "value_builder_MACs": 4 * rows * c * d * d,
                    "value_pair_matrix_products": rows,
                    "value_pair_matrix_MACs": rows * d * d * d,
                    "value_message_rows": edges,
                    "value_message_application_MACs": edges * c * d,
                    "value_self_mask_rows": edges,
                    "value_self_rows_preserved": edge_metadata["self_edges"],
                    "value_gathered_source_coordinates": edges * c,
                    "value_gathered_target_coordinates": edges * c if conditioning == "endpoint" else 0,
                    "value_context_coordinates": rows * 2 * c,
                    "value_generated_incidence_coordinates": 2 * rows * d * d,
                    "value_generated_transport_coordinates": rows * d * d,
                    "value_node_transport_gather_coordinates": edges * d * d if conditioning == "node" else 0}
            self.value_last_unit = unit
            _counts_add(self.value_counts_train if self.training else self.value_counts_eval, unit)
            return out

        def message(self, x_j, x_i, edge_index_i, edge_index_j):
            return transport_values(torch, x_j, x_i,
                self.value_source_builder, self.value_target_builder,
                stalk_dim=4, self_edge=edge_index_i.eq(edge_index_j), conditioning="endpoint")

    class NodeValueSAGEConv(EndpointValueSAGEConv):
        def message(self, x_j, edge_index_i, edge_index_j):
            # Same pre-mean pointwise action, but build receiver maps once per
            # node, not redundantly E times. No sum/matrix order is changed.
            if self._node_maps is None:
                raise ValueError("Receiver map must be built from this live forward")
            moved = (self._node_maps[edge_index_i] @ x_j.reshape(-1, 4, 32)).reshape_as(x_j)
            return torch.where(edge_index_i.eq(edge_index_j)[:, None], x_j, moved)

    native_ids = {id(p) for p in body.parameters()}
    replacements = []
    for index, residual in enumerate(body.residual_modules):
        native = residual.module.conv
        if (residual.module.__class__.__name__ != "SAGEModule"
                or native.__class__.__name__ != "SAGEConv"
                or native.in_channels != 128 or native.out_channels != 128
                or native.aggr != "mean" or native.flow != "source_to_target"
                or native.node_dim != -2 or native.decomposed_layers != 1
                or native.project or native.normalize or not native.root_weight
                or native.lin_r.bias is not None
                or any(bool(values) for name, values in vars(native).items() if name.endswith("_hooks"))):
            raise ValueError("Fresh admitted native SAGE constructor differs")
        if not all(isinstance(getattr(native, key), factors.FactorLinear) for key in ("lin_l", "lin_r")):
            raise ValueError("Use existing native dense-map factor replacement before VALUE registration")
        cls = EndpointValueSAGEConv if conditioning == "endpoint" else NodeValueSAGEConv
        replacement = cls(native)
        if replacement.lin_l is not native.lin_l or replacement.lin_r is not native.lin_r:
            raise ValueError("Native map ownership changed")
        residual.module.conv = replacement
        replacements.append({"layer": index, "conditioning": conditioning,
            "stalk_dim": 4, "channels": 128, "members": members,
            "native_map_parameter_identity_preserved": True,
            "self_loop_policy": "unchanged factual support; self VALUE unchanged",
            "native_bias_policy": "lin_l original bias; lin_r no bias",
            "builder_shared_parameters": 8224, "builder_private_parameters": 544 * members,
            "builder_factor_initialization": "existing FactorLinear ones; native stem Rademacher unchanged",
            "builder_W_bias_initialization": "zero, RNG preserved, identity VALUE transport",
            "propagate": "native PyG generic nonfused collector and MeanAggregation"})
    if not native_ids.issubset({id(p) for p in body.parameters()}):
        raise ValueError("Installer lost existing native/factor parameter ownership")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in body.parameters()):
        raise ValueError("All registered native/builder parameters must remain float32 CPU until caller movement")
    return replacements


def transport_counts(body, members, completed_updates):
    """Account actual hooks; split final selected read from eval without replay."""
    total, train, evaluation, selected = {}, {}, {}, {}
    for residual in body.residual_modules:
        conv = residual.module.conv
        if not hasattr(conv, "value_counts_train"):
            continue
        _counts_add(train, conv.value_counts_train)
        _counts_add(evaluation, conv.value_counts_eval)
        if conv.value_last_unit is None:
            raise ValueError("VALUE operator never exercised")
        _counts_add(selected, {key: value * members for key, value in conv.value_last_unit.items()})
    if not train:
        return {"TRAIN": {}, "VALID_selector": {}, "selected_VALID": {}, "total": {}}
    expected = 2 * members * completed_updates
    if (train["value_conv_calls"] != expected
            or evaluation["value_conv_calls"] != expected + 2 * members):
        raise ValueError("Actual VALUE hook count differs from native fitted/selected trajectories")
    selector = {key: evaluation[key] - selected[key] for key in evaluation}
    if any(value < 0 for value in selector.values()):
        raise ValueError("Selected cost cannot exceed recorded evaluation work")
    _counts_add(total, train); _counts_add(total, evaluation)
    return {"TRAIN": train, "VALID_selector": selector, "selected_VALID": selected, "total": total}
