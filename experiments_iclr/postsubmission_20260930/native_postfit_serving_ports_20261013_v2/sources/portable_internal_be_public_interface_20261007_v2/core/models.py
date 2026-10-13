"""Three substantive task backbones. Existing author source is hash-bound.

Polynormer/NCN bytes are not recopied here; unresolved NCN redistribution
permission remains unresolved. Imports never invoke author dataset/main scripts.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from factors import FactorLinear, install_factors, initialize_first_factor, member_context


def load_source(path, digest, name):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Bound source changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_sources(phase, bindings):
    poly = bindings["polynormer"]
    p = load_source(Path(phase)/poly["path"], poly["sha256"], "be_suite_polynormer")
    ncn = bindings["ncn"]
    utils = load_source(Path(phase)/ncn["utils"]["path"], ncn["utils"]["sha256"], "be_suite_ncn_utils")
    old = sys.modules.get("utils")
    try:
        sys.modules["utils"] = utils
        n = load_source(Path(phase)/ncn["model"]["path"], ncn["model"]["sha256"], "be_suite_ncn")
    finally:
        if old is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = old
    return p, n


@contextmanager
def seeded(seed):
    # Private constructor randomness cannot alter another member's training RNG.
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


class WikiBackbone(nn.Module):
    def __init__(self, native, options):
        super().__init__()
        self.body = native.Polynormer(**options)
        self.body.reset_parameters()

    @property
    def stem(self):
        return self.body.lin_in

    def set_global(self, value):
        self.body._global = bool(value)

    def forward(self, batch):
        head = self.body.pred_global if self.body._global else self.body.pred_local
        captured = []
        hook = head.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        try:
            logits = self.body(batch["x"], batch["edge_index"])
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError("Exact native representation capture failed")
        ids = batch["ids"]
        return logits[ids], captured[0][ids]


class CollabBackbone(nn.Module):
    def __init__(self, native, options):
        super().__init__()
        d = options["hidden"]
        self.encoder = native.GCN(options["features"], d, d, 1, .1, ln=True, res=True,
            conv_fn="gcn", edrop=.25, xdropout=.25, taildropout=.05)
        self.decoder = native.CNLinkPredictor(d, d, 1, 3, .3, edrop=0., ln=True,
            cndeg=-1, use_xlin=True, tailact=True, twolayerlin=False, beta=1.)

    @property
    def stem(self):
        return self.encoder.convs[0].lin

    def forward(self, batch):
        captured = []
        hook = self.decoder.lin[-1].register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        try:
            h = self.encoder(batch["x"], batch["adj"])
            out = self.decoder(h, batch["adj"], batch["query"].t().contiguous())
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError("NCN representation capture failed")
        return out, captured[0]


class MoleculeBackbone(nn.Module):
    """Bond-aware residual GINE + trainable virtual node; original atom/bond fields.

    An explicit contemporary baseline adaptation, not a reproduced OGB table.
    Node LayerNorm avoids mixing one member's normalization state with another.
    """
    def __init__(self, options):
        super().__init__()
        d, layers, p = options["hidden"], options["layers"], options["dropout"]
        self.atom = AtomEncoder(d)
        self.input = nn.Linear(d, d)
        self.bonds = nn.ModuleList([BondEncoder(d) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.convs = nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(d, 2*d), nn.ReLU(),
            nn.Dropout(p), nn.Linear(2*d, d)), train_eps=True) for _ in range(layers)])
        self.virtual = nn.Parameter(torch.zeros(1, d))
        self.virtual_mlps = nn.ModuleList([nn.Sequential(nn.Linear(d, 2*d), nn.ReLU(),
            nn.Linear(2*d, d)) for _ in range(layers - 1)])
        self.output_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.dropout = p
        self.message_factors = None
        self.member = 0

    def enable_message_factors(self, members):
        self.message_factors = nn.Parameter(torch.ones(len(self.convs), members, self.input.weight.shape[0]))

    @property
    def stem(self):
        return self.input

    def forward(self, batch):
        graph = batch["graph"]
        if graph.x.dtype != torch.long or graph.x.shape[1] != 9 or graph.edge_attr.dtype != torch.long or graph.edge_attr.shape[1] != 3:
            raise ValueError("Complete original OGB chemical categorical fields required")
        assignment = graph.batch
        graphs = int(graph.num_graphs)
        h = F.relu(self.input(self.atom(graph.x)))
        virtual = self.virtual.expand(graphs, -1)
        for i, (conv, bond, norm) in enumerate(zip(self.convs, self.bonds, self.norms)):
            message = norm(h + virtual[assignment])
            if self.message_factors is not None:
                message = message * self.message_factors[i, self.member]
            # Private channels enter GINE's relu(x_j + bond) BEFORE sum aggregation.
            update = conv(message, graph.edge_index, bond(graph.edge_attr))
            h = h + F.dropout(F.relu(update), self.dropout, self.training)
            if i < len(self.convs) - 1:
                virtual = virtual + F.dropout(self.virtual_mlps[i](
                    global_add_pool(h, assignment, size=graphs) + virtual), self.dropout, self.training)
        representation = global_mean_pool(self.output_norm(h), assignment, size=graphs)
        return self.head(representation), representation


class Ensemble(nn.Module):
    def __init__(self, task, arm, seed, config, sources):
        super().__init__()
        self.task, self.arm = task, arm
        self.contrastive = arm.endswith("_contrastive")
        kind = arm.removesuffix("_contrastive")
        self.independent = kind in ("single", "independent4")
        self.members = 1 if kind == "single" else 4
        def construct():
            if task == "wikics":
                return WikiBackbone(sources[0], config)
            if task == "collab":
                return CollabBackbone(sources[1], config)
            if task == "molhiv":
                return MoleculeBackbone(config)
            raise ValueError(task)
        self.models = nn.ModuleList()
        for m in range(self.members if self.independent else 1):
            with seeded(seed + 1009 * m):
                self.models.append(construct())
        if not self.independent:
            install_factors(self.models[0], self.members)
            if task == "molhiv":
                self.models[0].enable_message_factors(self.members)
            if kind == "be_init":
                initialize_first_factor(self.models[0].stem, seed + 900001)
            elif kind != "be_unit":
                raise ValueError("Unregistered arm")

    def set_global(self, value):
        if self.task != "wikics":
            raise ValueError("Only native WikiCS local/global schedule")
        for model in self.models:
            model.set_global(value)

    def member_forward(self, batch, member):
        model = self.models[member if self.independent else 0]
        old = getattr(model, "member", 0)
        if hasattr(model, "member"):
            model.member = member
        try:
            with member_context(model, 0 if self.independent else member):
                return model(batch)
        finally:
            if hasattr(model, "member"):
                model.member = old

    def forward(self, batch):
        rows = [self.member_forward(batch, m) for m in range(self.members)]
        return torch.stack([x[0] for x in rows]), torch.stack([x[1] for x in rows])
