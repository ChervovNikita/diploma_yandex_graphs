"""Source-only Pubmed shared4 bridge; no runner, loader or execution release.

Reuse the qualified completion decoder with the exact native Pubmed encoder.
The caller must authenticate the new source, inputs and runtime under a new
external root release before loading numerical modules or calling this factory.
Previously saved native/engineering states are never accepted by the factory.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
MODES = ("private", "pooled_after_clamp")
NODES, FEATURES, WIDTH, MEMBERS = 19717, 500, 256, 4
SIGN_DOMAIN = "ncnc-pubmed-shared4-extension-v1|base={seed}|domain=factor-signs"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def factor_sign_seed(seed):
    require(seed in SEEDS, "Outside prespecified Pubmed seed block")
    label = SIGN_DOMAIN.format(seed=seed)
    return int.from_bytes(sha256(label.encode("ascii")).digest()[:8], "little") % (2**31 - 1)


def expected_parameter_counts():
    # Native puregcn: input Linear plus active one-layer jkparams, no encoder LN.
    encoder = FEATURES * WIDTH + WIDTH + 1
    decoder = 7 * WIDTH**2 + 9 * WIDTH + 2 + MEMBERS * (24 * WIDTH + 3)
    unused_ptlin = WIDTH**2 + 2 * WIDTH + 1 + MEMBERS * (3 * WIDTH + 1)
    return {"encoder": encoder, "decoder": decoder, "total": encoder + decoder,
            "active": encoder + decoder - unused_ptlin, "unused_fixed_ptlin": unused_ptlin}


def authenticate_modules(phase_root, bodies, prototype, graph_ops):
    """Check source identities only; this does not grant execution authority."""
    phase_root = Path(phase_root).resolve()
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    pins = {row["path"]: row for row in bindings["external_source_pins"]}
    modules = {
        "pubmed_heart_native_numerical_qualification_source_20261004_v2/native_bodies.py": bodies,
        "graph_ncNC_member_completion_qualification_preparation_20261003_v2/prototype.py": prototype,
        "graph_ncNC_member_completion_qualification_preparation_20261003_v2/graph_ops.py": graph_ops,
        "pubmed_heart_available_inspector_native_adapter_20261004_v1/public_author_code/HeaRT/benchmarking/baseline_models/NCN/model.py": sys.modules[bodies.GCN.__module__],
    }
    for relative, module in modules.items():
        expected = phase_root / relative
        actual = Path(module.__file__).resolve()
        require(actual == expected and not expected.is_symlink(), "Numerical module identity differs: " + relative)
        payload = actual.read_bytes()
        require(len(payload) == pins[relative]["bytes"] and sha256(payload).hexdigest() == pins[relative]["sha256"],
                "Numerical source bytes differ: " + relative)
    require(prototype.Graph is graph_ops.Graph, "Prototype graph module was shadowed")


def make_shared4(phase_root, bodies, prototype, graph_ops, seed, mode, x, train):
    """Fresh native-shaped unit compatible with unchanged native TRAIN/VALID.

    Returns (native_encoder, predictor_facade, Adam, x, native_Data, None).
    No checkpoint argument, model copying, reset, post-factory reseed or resume.
    The routing mode has no constructor/RNG effect and is fixed for the fit.
    """
    require(seed in SEEDS and mode in MODES, "Outside prespecified arm/seed")
    authenticate_modules(phase_root, bodies, prototype, graph_ops)
    import torch
    from torch import nn
    require(tuple(x.shape) == (NODES, FEATURES) and x.dtype == torch.float32 and x.device.type == "cpu",
            "Exact raw CPU Pubmed feature geometry required")
    require(tuple(train.shape) == (37676, 2) and train.dtype == torch.long and train.device.type == "cpu",
            "Exact released CPU TRAIN record geometry required")
    require(str(bodies.device) == "cuda:0", "Qualified native device differs")

    class PredictorFacade(nn.Module):
        def __init__(self, recipe):
            super().__init__()
            self.decoder = prototype.CompletionDecoder(recipe)
            self.mode = mode

        @staticmethod
        def graph(adj):
            require(tuple(adj.sizes()) == (NODES, NODES), "Native adjacency dimensions differ")
            rowptr, col, value = adj.csr()
            row = adj.coo()[0]
            require(value is None or bool((value == 1).all()), "Only native unweighted adjacency is supported")
            return graph_ops.Graph(NODES, row, col, rowptr)

        def multidomainforward(self, h, adj, tar_ei, filled1=False, cndropprobs=()):
            require(self.mode == mode, "Fit routing mode changed")
            require(not filled1 and len(cndropprobs) == 0, "Native unfilled/no-domain-dropout path required")
            require(tar_ei.ndim == 2 and tar_ei.shape[0] == 2 and tar_ei.dtype == torch.long,
                    "Native endpoint tensor geometry differs")
            return self.decoder(h, self.graph(adj), tar_ei.t(), self.mode)

        def forward(self, h, adj, tar_ei, filled1=False):
            # NCNC serving pools unbounded raw member logits. No sigmoid.
            return self.multidomainforward(h, adj, tar_ei, filled1, ()).mean(1, keepdim=True)

    edge = bodies.to_undirected(train.t())
    adj = bodies.SparseTensor.from_edge_index(edge, sparse_sizes=(NODES, NODES)).to_symmetric().coalesce()
    data = bodies.Data(x=x, edge_index=edge, adj_t=adj, num_nodes=NODES, max_x=-1).to(bodies.device)
    # Match the resolved Pubmed native factory's seed point and encoder exactly.
    bodies.seed_native(seed)
    encoder = bodies.GCN(FEATURES, WIDTH, WIDTH, 1, 0.1, True, False, -1,
                         "puregcn", True, 0.0, xdropout=0.3, taildropout=0.0,
                         noinputlin=False).to(bodies.device)
    recipe = prototype.Recipe(features=FEATURES, hidden=WIDTH, input_dropout=0.3,
        encoder_dropout=0.1, encoder_edge_dropout=0.0, decoder_dropout=0.1,
        scale=5.3, offset=0.5, alpha=0.3, pt=0.5,
        encoder_lr=0.001, decoder_lr=0.001, completion_depth=1,
        candidate_split_size=-1, member_count=MEMBERS, native_target_mask=True)
    recipe.validate()
    predictor = PredictorFacade(recipe).to(bodies.device)
    # Same sorted assignment/local CPU generator policy as the established
    # shared4 factory; a prospectively declared task domain supplies the seed.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(factor_sign_seed(seed))
    with torch.no_grad():
        for name, parameter in sorted(predictor.decoder.named_parameters()):
            if name.endswith(".r") or name.endswith(".s"):
                signs = 2 * torch.randint(0, 2, parameter.shape, dtype=torch.int64,
                                          generator=generator, device="cpu") - 1
                parameter.copy_(signs.to(device=parameter.device, dtype=parameter.dtype))
    optimizer = torch.optim.Adam([
        {"params": encoder.parameters(), "lr": 0.001},
        {"params": predictor.parameters(), "lr": 0.001}], weight_decay=0)
    counts = expected_parameter_counts()
    require(sum(p.numel() for p in encoder.parameters()) == counts["encoder"], "Native encoder count differs")
    require(sum(p.numel() for p in predictor.parameters()) == counts["decoder"], "Shared4 decoder count differs")
    return encoder, predictor, optimizer, x, data, None
