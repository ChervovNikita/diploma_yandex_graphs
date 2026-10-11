"""Exact pinned OGB SAGE class only; never import its stock provider/test driver."""
import ast
import hashlib
from pathlib import Path
import sys
import types

AUTHOR_COMMIT = "61e9784ca76edeaa6e259ba0f836099608ff0586"
AUTHOR_SHA256 = "050b6b7a0fc86ef99b237438f1506f2868a442833d963cfe03efa3419e60d365"
SEEDS = (8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623)
POINT = {"nodes": 169343, "input_dim": 128, "classes": 40, "layers": 3,
         "hidden": 256, "dropout": .5, "normalization": "BatchNorm1d",
         "activation": "ReLU", "optimizer": "Adam", "learning_rate": .01,
         "weight_decay": 0., "updates": 500, "patience": None, "scheduler": None}


def model_class(torch, F, SAGEConv, path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUTHOR_SHA256:
        raise ValueError("Exact pinned official author gnn.py required")
    tree = ast.parse(raw.decode())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SAGE"]
    if len(classes) != 1:
        raise ValueError("One exact original SAGE class")
    module = types.ModuleType("pinned_ogb_author_SAGE")
    module.__file__ = str(Path(path).resolve())
    module.__dict__.update(torch=torch, F=F, SAGEConv=SAGEConv)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=classes, type_ignores=[]), module.__file__, "exec"), module.__dict__)
    return module.SAGE


def inspect_model(torch, model):
    if (len(model.convs) != 3 or len(model.bns) != 2 or model.dropout != .5
            or [(c.in_channels, c.out_channels) for c in model.convs] != [(128, 256), (256, 256), (256, 40)]
            or any(c.aggr != "mean" or c.project or c.normalize or not c.root_weight or c.lin_r.bias is not None for c in model.convs)):
        raise ValueError("Fixed ordinary public SAGE structure differs")
    if any(not isinstance(b, torch.nn.BatchNorm1d) or b.num_features != 256 or b.eps != 1e-5 or b.momentum != .1 or not b.affine or not b.track_running_stats for b in model.bns):
        raise ValueError("Original BatchNorm defaults required")
    params = list(model.parameters()); buffers = dict(model.named_buffers())
    if any(p.dtype != torch.float32 for p in params) or sum(p.numel() for p in params) != 218664:
        raise ValueError("Fixed float32 constructor count differs")
    expected = {f"bns.{i}.{name}" for i in (0, 1) for name in ("running_mean", "running_var", "num_batches_tracked")}
    if set(buffers) != expected or sum(b.numel() * b.element_size() for b in buffers.values()) != 4112:
        raise ValueError("All original BatchNorm buffers must be registered")
    return {"parameters": 218664, "parameter_bytes": sum(p.numel() * p.element_size() for p in params),
            "BatchNorm_buffer_bytes": 4112, "buffer_names": sorted(buffers),
            "author_commit": AUTHOR_COMMIT, "author_source_sha256": AUTHOR_SHA256}
