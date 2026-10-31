"""New-study bond-aware molecular controls; not historical benchmark models.

forward(graph, features=None) -> [members, graphs, targets].
forward_member(graph, features, member) -> [graphs, targets].
pooled_forward(graph, features=None) -> mean raw logits [graphs, targets].

`features` are original categorical OGB atom fields, not encoded floats.
"""
import copy
import importlib.util
from pathlib import Path

import torch
from torch import nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import GINEConv, global_mean_pool

_SOURCE = Path(__file__).resolve().parents[1] / "source_snapshot" / "models.py"
_SPEC = importlib.util.spec_from_file_location("gnnm_gine_boundary_source_20260930", _SOURCE)
_SOURCE_MODELS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SOURCE_MODELS)
BEBlock = _SOURCE_MODELS.BEBlock


def gine_options(output_dim=1, hidden_dim=128, num_layers=3, dropout=0.1,
                 hidden_dim_multiplier=2):
    return dict(output_dim=output_dim, hidden_dim=hidden_dim,
                num_layers=num_layers, dropout=dropout,
                hidden_dim_multiplier=hidden_dim_multiplier)


def _atoms(graph, features):
    features = graph.x if features is None else features
    if features.ndim != 2 or features.size(1) != 9 or features.dtype != torch.long:
        raise ValueError("GINE requires original OGB atom fields: long[N,9].")
    if features.size(0) != graph.num_nodes:
        raise ValueError("Atom features do not match graph nodes.")
    if graph.edge_attr is None or graph.edge_attr.ndim != 2 or graph.edge_attr.size(1) != 3:
        raise ValueError("GINE requires original OGB bond fields: long[E,3].")
    if graph.edge_attr.dtype != torch.long or graph.edge_attr.size(0) != graph.edge_index.size(1):
        raise ValueError("Bond fields do not match graph edges.")
    return features


def _pool(graph, features):
    batch = getattr(graph, "batch", None)
    if batch is None:
        batch = torch.zeros(features.size(0), device=features.device, dtype=torch.long)
    return global_mean_pool(features, batch)


class GINEBlock(nn.Module):
    """Pre-LayerNorm residual GINE, with an in-block learned bond encoder."""
    def __init__(self, dim, hidden_dim_multiplier=2, dropout=0.1):
        super().__init__()
        hidden = int(dim * hidden_dim_multiplier)
        self.normalization = nn.LayerNorm(dim)
        self.bond_encoder = BondEncoder(dim)
        self.conv = GINEConv(nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)), train_eps=True)

    def forward(self, graph, x):
        edge_features = self.bond_encoder(graph.edge_attr)
        return x + self.conv(self.normalization(x), graph.edge_index, edge_features)


class MolecularSingle(nn.Module):
    def __init__(self, options=None):
        super().__init__()
        self.options = gine_options() if options is None else dict(options)
        o = self.options
        self.members, self.output_dim = 1, o["output_dim"]
        d = o["hidden_dim"]
        self.encoder = AtomEncoder(d)
        self.input_projector = nn.Linear(d, d)
        self.dropout, self.act = nn.Dropout(o["dropout"]), nn.GELU()
        self.blocks = nn.ModuleList([GINEBlock(d, o["hidden_dim_multiplier"], o["dropout"])
                                    for _ in range(o["num_layers"])])
        self.output_normalization = nn.LayerNorm(d)
        self.output_projector = nn.Linear(d, self.output_dim)

    def encode(self, graph, features=None):
        x = self.encoder(_atoms(graph, features))
        x = self.act(self.dropout(self.input_projector(x)))
        for block in self.blocks:
            x = block(graph, x)
        return _pool(graph, self.output_normalization(x))

    def forward_member(self, graph, features=None, member=0):
        if member != 0:
            raise IndexError("A single model has only member 0.")
        return self.output_projector(self.encode(graph, features))

    def forward(self, graph, features=None):
        return self.forward_member(graph, features).unsqueeze(0)

    forward_members = forward

    def pooled_forward(self, graph, features=None):
        return self.forward(graph, features).mean(0)

    def probe_blocks(self):
        return {str(i): tuple(block.parameters()) for i, block in enumerate(self.blocks)}


class MolecularProjectorEnsemble(nn.Module):
    """Boundary-only BatchEnsemble factors, optionally private GINE blocks.

    The AtomEncoder, boundary W/factors and output LayerNorm are shared in
    both the tied and copied-private intervention. Each block includes its
    own BondEncoder. Privatize before creating an optimizer, or rebuild the
    optimizer and record a deliberate warmup-state transfer policy.
    """
    def __init__(self, options=None, members=4):
        super().__init__()
        if members < 1:
            raise ValueError("members must be positive")
        self.options = gine_options() if options is None else dict(options)
        o = self.options
        self.members, self.output_dim = members, o["output_dim"]
        d = o["hidden_dim"]
        self.encoder = AtomEncoder(d)
        self.input_projector = BEBlock(d, d, True, "cpu", members)
        self.dropout, self.act = nn.Dropout(o["dropout"]), nn.GELU()
        self.blocks = nn.ModuleList([GINEBlock(d, o["hidden_dim_multiplier"], o["dropout"])
                                    for _ in range(o["num_layers"])])
        self.private_blocks = nn.ModuleDict()
        self.output_normalization = nn.LayerNorm(d)
        self.output_projector = BEBlock(d, self.output_dim, False, "cpu", members)

    def privatize(self, indices):
        for index in sorted(set(indices)):
            if index < 0 or index >= len(self.blocks):
                raise ValueError("GINE block index out of range")
            key = str(index)
            if key in self.private_blocks:
                raise ValueError("GINE block is already private")
            self.private_blocks[key] = nn.ModuleList([
                copy.deepcopy(self.blocks[index]) for _ in range(self.members)])
            # Remove the now-unused shared parameters from optimizer/storage.
            self.blocks[index] = nn.Identity()

    def forward_member(self, graph, features=None, member=0):
        if member < 0 or member >= self.members:
            raise IndexError("Member index out of range")
        x = self.encoder(_atoms(graph, features))
        x = self.act(self.dropout(self.input_projector(x, member)))
        for index, shared in enumerate(self.blocks):
            key = str(index)
            block = self.private_blocks[key][member] if key in self.private_blocks else shared
            x = block(graph, x)
        x = _pool(graph, self.output_normalization(x))
        return self.output_projector(x, member)

    def forward(self, graph, features=None):
        return torch.stack([self.forward_member(graph, features, member)
                            for member in range(self.members)])

    forward_members = forward

    def pooled_forward(self, graph, features=None):
        return self.forward(graph, features).mean(0)

    def probe_blocks(self):
        if self.private_blocks:
            raise ValueError("Probe a fully shared molecular warmup state")
        return {str(i): tuple(block.parameters()) for i, block in enumerate(self.blocks)}


def copied_private(original, private_indices=None):
    if not isinstance(original, MolecularProjectorEnsemble):
        raise TypeError("Copy a MolecularProjectorEnsemble")
    if original.private_blocks:
        raise ValueError("Start from a fully tied model")
    result = copy.deepcopy(original)
    if private_indices is None:
        private_indices = range(len(result.blocks))
    result.privatize(private_indices)
    return result


class MolecularSharedMultihead(nn.Module):
    """One bond-aware hidden-state path and independent linear graph heads."""
    def __init__(self, options=None, members=4):
        super().__init__()
        self.options = gine_options() if options is None else dict(options)
        self.members, self.output_dim = members, self.options["output_dim"]
        self.trunk = MolecularSingle(self.options)
        self.trunk.output_projector = nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(self.options["hidden_dim"], self.output_dim)
                                    for _ in range(members)])

    def forward_member(self, graph, features=None, member=0):
        if member < 0 or member >= self.members:
            raise IndexError("Member index out of range")
        return self.heads[member](self.trunk.encode(graph, features))

    def forward(self, graph, features=None):
        hidden = self.trunk.encode(graph, features)
        return torch.stack([head(hidden) for head in self.heads])

    forward_members = forward

    def pooled_forward(self, graph, features=None):
        return self.forward(graph, features).mean(0)


class MolecularIndependentEnsemble(nn.Module):
    """Independent initialization, separate atom/bond encoders and stacks."""
    def __init__(self, options=None, members=4):
        super().__init__()
        self.options = gine_options() if options is None else dict(options)
        self.members, self.output_dim = members, self.options["output_dim"]
        self.models = nn.ModuleList([MolecularSingle(self.options) for _ in range(members)])

    def forward_member(self, graph, features=None, member=0):
        return self.models[member].forward_member(graph, features)

    def forward(self, graph, features=None):
        return torch.stack([self.forward_member(graph, features, member)
                            for member in range(self.members)])

    forward_members = forward

    def pooled_forward(self, graph, features=None):
        return self.forward(graph, features).mean(0)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())
