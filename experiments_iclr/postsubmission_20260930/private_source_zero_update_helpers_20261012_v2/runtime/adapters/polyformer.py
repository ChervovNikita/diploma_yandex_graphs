"""Pinned PolyFormer mono source bridge; no numerical imports at module load.

PolyFormer is air029/PolyFormer, not Polynormer. One adapter is one physical
body/optimizer/selector. Genuine independent ensembles use separate paid fits.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

try:
    from .classifier import ClassifierAdapter
    from .node_losses import node_objective
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from adapters.node_losses import node_objective

AUTHOR_REPOSITORY = "https://github.com/air029/PolyFormer"
AUTHOR_COMMIT = "d390f39e88d0eaac80318fdc7704bd3bf3cf8b13"
AUTHOR_DIRECTORY = "coordinate_source_independent_review_v1/strong_backbones_v1/sources/polyformer_code/node_classification"
SOURCE_FILES = {
    "block": {"path": AUTHOR_DIRECTORY + "/layers/PolyFormerBlock.py", "bytes": 4418,
              "sha256": "12a225dcaed730681b7af4a29278d4c52ebeafdcd4536e1fee801845e1ba7f0e"},
    "model": {"path": AUTHOR_DIRECTORY + "/mymodels.py", "bytes": 2271,
              "sha256": "a873231a52b02f6637036f22c3b6e764cae58a9d02ec37a618036d18a522d90f"},
    "utils": {"path": AUTHOR_DIRECTORY + "/utils.py", "bytes": 16618,
              "sha256": "2db2f0bbf42bcdf3c49715d4bcb5a197cc592521ce957395f2e7aea75268732b"},
    "factors": {"path": "learnable_internal_be_contrastive_multitask_suite_20261007_v1/factors.py", "bytes": 3271,
                "sha256": "acd5f681f2408dc8aaee71b5953f44bb2d71f83cec5805fcb424b4833fea49f7"},
}
NATIVE_FIELDS = {"K", "hidden", "nlayer", "n_head", "d_ffn", "q", "multi", "dropout", "dprate"}
METHODS = {"polyformer_single": (1, False), "polyformer_factorized_single": (1, True),
           "polyformer_internal_be": (4, True)}


def _path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("PolyFormer input/source must remain inside the project")
    return path


def _file_receipt(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked_source(root, key):
    row = SOURCE_FILES[key]
    path = _path(root, row["path"])
    payload = path.read_bytes()
    if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError("Pinned PolyFormer dependency changed: " + key)
    return path


def _definitions(path, names, environment):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names) or len(nodes) != len(names):
        raise ValueError("Complete unique pinned native definitions required")
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), environment)
    return environment


def native_class(torch, project_root):
    """Exact author ASTs, omitting global seed42/import/data-loader side effects."""
    environment = {"__name__": "bound_polyformer_author", "torch": torch,
                   "nn": torch.nn, "F": torch.nn.functional, "Linear": torch.nn.Linear}
    _definitions(checked_source(project_root, "block"), ("PolyAttn", "FFNNetwork", "FFN", "PolyFormerBlock"), environment)
    _definitions(checked_source(project_root, "model"), ("PolyFormer",), environment)
    return environment["PolyFormer"]


def factor_definitions(torch, project_root):
    # PolyFormer has only nn.Linear affine sites. Preserve the source type
    # predicate for the general archived installer with its normal PyG class.
    from torch_geometric.nn.dense.linear import Linear as PyGLinear
    environment = {"__name__": "bound_polyformer_factors", "torch": torch, "nn": torch.nn,
                   "F": torch.nn.functional, "PyGLinear": PyGLinear, "contextmanager": contextmanager}
    return _definitions(checked_source(project_root, "factors"),
                        ("FactorLinear", "install_factors", "initialize_first_factor", "member_context"), environment)


def mono_tokens(torch, project_root, x, edge_index, *, K, preprocessing):
    """Exact author gcn_norm -> SciPy float32 COO -> successive spmm path."""
    import numpy as np
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import is_undirected, to_scipy_sparse_matrix, to_undirected
    if (x.ndim != 2 or x.dtype != torch.float32 or x.requires_grad or not torch.isfinite(x).all().item()
            or edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long
            or type(K) is not int or K < 0):
        raise ValueError("Finite target-free FP32 features and long COO graph required")
    if edge_index.numel() and (edge_index.min().item() < 0 or edge_index.max().item() >= len(x)):
        raise ValueError("Graph endpoint outside complete feature rows")
    if preprocessing == "author_undirected_then_mono":
        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)
    elif preprocessing != "pinned_mono_base":
        raise ValueError("Declare literal mono preprocessing or author undirected preprocessing")
    environment = {"np": np, "torch": torch, "gcn_norm": gcn_norm,
                   "to_scipy_sparse_matrix": to_scipy_sparse_matrix}
    _definitions(checked_source(project_root, "utils"), ("sparse_mx_to_torch_sparse_tensor", "mono_base"), environment)
    return environment["mono_base"](K, x, edge_index, None)


@contextmanager
def _constructor_seed(torch, seed, device):
    devices = [] if device.type == "cpu" else [device.index]
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        if devices:
            torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(seed).get_state(), device.index)
        yield


def _arguments(native, features, classes, dataset):
    if set(native) != NATIVE_FIELDS:
        raise ValueError("Declare every native mono constructor field")
    for key in ("hidden", "nlayer", "n_head", "d_ffn"):
        if type(native[key]) is not int or native[key] < 1:
            raise ValueError("Positive native shape required: " + key)
    if type(native["K"]) is not int or native["K"] < 0 or native["hidden"] % native["n_head"]:
        raise ValueError("Nonnegative mono order and head-divisible hidden width required")
    for key in ("q", "multi", "dropout", "dprate"):
        if type(native[key]) not in (int, float) or not math.isfinite(native[key]):
            raise ValueError("Finite native scalar required: " + key)
    if native["multi"] <= 0 or int(native["hidden"] * native["multi"]) < 1:
        raise ValueError("Positive material token network width required")
    if any(not 0 <= native[key] <= 1 for key in ("dropout", "dprate")):
        raise ValueError("Native dropout probabilities must lie in0..1")
    if type(classes) is not int or classes < 2:
        raise ValueError("This packet implements categorical node CE, not author binary-node BCE")
    return SimpleNamespace(**native, base="mono", num_features=features, num_classes=classes, dataset=dataset)


def build_bank(torch, project_root, tokens, *, native, classes, dataset, members, factorized,
               initialization_seed, dropout_seeds, factor_initialization="unit", factor_seed=None):
    """One actual native body with optional complete internal factor rows."""
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Pinned PolyFormer construction requires native float32 default dtype")
    if members not in (1, 4) or (members == 4 and not factorized) or type(initialization_seed) is not int:
        raise ValueError("One ordinary body or one equipped M1/M4 body with explicit seed required")
    if (not isinstance(dropout_seeds, list) or len(dropout_seeds) < members
            or any(type(seed) is not int for seed in dropout_seeds)):
        raise ValueError("Explicit private dropout seeds required for every used route/view slot")
    if (not tokens or len(tokens) != native["K"] + 1 or any(t.ndim != 2 or t.shape != tokens[0].shape
            or t.device != tokens[0].device or t.dtype != torch.float32 or t.requires_grad
            or not torch.isfinite(t).all().item() for t in tokens)):
        raise ValueError("Exact aligned frozen mono token bank required")
    device = tokens[0].device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This pinned bridge supports normal CPU/CUDA execution")
    args = _arguments(native, tokens[0].shape[1], classes, dataset)
    constructor = native_class(torch, project_root)
    with _constructor_seed(torch, initialization_seed, device):
        body = constructor(None, args).to(device)
    slow = dict(body.named_parameters())
    factors = factor_definitions(torch, project_root) if factorized else None
    sites = factors["install_factors"](body, members) if factors else 0
    body.to(device)
    expected_sites = 3 + native["nlayer"] * (2 * (native["K"] + 1) + 4)
    if factorized and (sites != expected_sites or any(dict(body.named_parameters())[name] is not parameter for name, parameter in slow.items())):
        raise ValueError("Every native affine site and original slow parameter object must be preserved")
    if factor_initialization not in {"unit", "first_rademacher"} or (not factorized and factor_initialization != "unit"):
        raise ValueError("Declared source factor initialization required")
    if factor_initialization == "first_rademacher":
        if type(factor_seed) is not int:
            raise ValueError("Explicit private first-stem Rademacher seed required")
        factors["initialize_first_factor"](body.lin1, factor_seed)

    class Bank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body, self.members, self.last_hidden = body, members, None
            self.factor_sites = sites
            for index, token in enumerate(tokens):
                self.register_buffer("token_" + str(index), token, persistent=False)
            self.streams = []
            for seed in dropout_seeds:
                self.streams.append({"cpu": torch.Generator(device="cpu").manual_seed(seed).get_state(),
                                     "cuda": torch.Generator(device=device).manual_seed(seed).get_state()
                                     if device.type == "cuda" else None})

        @contextmanager
        def stream(self, slot):
            state = self.streams[slot]
            devices = [] if device.type == "cpu" else [device.index]
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(state["cpu"])
                if devices:
                    torch.cuda.set_rng_state(state["cuda"], device.index)
                yield
                state["cpu"] = torch.get_rng_state()
                if devices:
                    state["cuda"] = torch.cuda.get_rng_state(device.index)

        def forward_route(self, route, *, slot=None):
            if not 0 <= route < self.members:
                raise ValueError("Unknown native factor route")
            captured = []
            handle = self.body.lin3.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
            try:
                data = SimpleNamespace(list_mat=[getattr(self, "token_" + str(i)) for i in range(len(tokens))])
                member_scope = factors["member_context"](self.body, route) if factors else _empty_context()
                with member_scope:
                    if self.training:
                        with self.stream(route if slot is None else slot):
                            logits = self.body(data)
                    else:
                        logits = self.body(data)
                if (len(captured) != 1 or logits.shape != (len(tokens[0]), classes)
                        or captured[0].shape != (len(tokens[0]), native["hidden"])):
                    raise ValueError("Complete native raw logits and live preclassifier H required")
                if not torch.isfinite(logits).all().item() or not torch.isfinite(captured[0]).all().item():
                    raise FloatingPointError("Nonfinite native PolyFormer forward")
                return logits, captured[0]
            finally:
                handle.remove()

        def forward(self):
            pairs = [self.forward_route(member) for member in range(self.members)]
            self.last_hidden = torch.stack([hidden for _, hidden in pairs])
            return torch.stack([logits for logits, _ in pairs])

        def capture_runtime_state(self):
            return {"streams": [{key: value.clone() if value is not None else None for key, value in state.items()}
                                for state in self.streams],
                    "factor_members": {name: module.member for name, module in self.body.named_modules()
                                       if factors and isinstance(module, factors["FactorLinear"])},
                    "poly_attn_bias": {name: module.bias.detach().cpu().clone() for name, module in self.body.named_modules()
                                       if type(module).__name__ == "PolyAttn"}}

        def restore_runtime_state(self, state):
            if len(state["streams"]) != len(self.streams):
                raise ValueError("Private dropout stream roster changed")
            self.streams = [{key: value.clone() if value is not None else None for key, value in stream.items()}
                            for stream in state["streams"]]
            modules = dict(self.body.named_modules())
            for name, member in state["factor_members"].items():
                modules[name].member = member
            for name, bias in state["poly_attn_bias"].items():
                modules[name].bias = bias.clone()  # source uses this unregistered CPU tensor via .to(src.device)

    return Bank().to(device)


@contextmanager
def _empty_context():
    yield


def make_adapter(*, torch, project_root, study, run, device):
    """Factory for run_study.py; full-budget scheduling remains in the engine."""
    config = run["config"]
    mode = config.get("polyformer_mode", run["method"])
    if mode not in METHODS or config.get("fusion") != "mean_probabilities":
        raise ValueError("Explicit PolyFormer method and native probability-mean serving required")
    members, factorized = METHODS[mode]
    inputs = config["node_inputs"]
    input_receipts = {key: _file_receipt(_path(project_root, inputs[key])) for key in ("graph", "train", "select")}
    graph = torch.load(_path(project_root, inputs["graph"]), map_location="cpu", weights_only=False)
    train = torch.load(_path(project_root, inputs["train"]), map_location="cpu", weights_only=False)
    select = torch.load(_path(project_root, inputs["select"]), map_location="cpu", weights_only=False)
    if set(graph) != {"x", "edge_index"} or set(train) != {"ids", "y"} or set(select) != {"ids", "y"}:
        raise ValueError("Target-free graph and separate categorical TRAIN/SELECT files required")
    rows = len(graph["x"])
    for role in (train, select):
        if (role["ids"].ndim != 1 or role["ids"].dtype != torch.long or role["y"].dtype != torch.long
                or role["y"].shape != role["ids"].shape or role["ids"].numel() == 0
                or len(torch.unique(role["ids"])) != len(role["ids"])
                or role["ids"].min().item() < 0 or role["ids"].max().item() >= rows
                or role["y"].min().item() < 0 or role["y"].max().item() >= config["classes"]):
            raise ValueError("Exact unique nonempty ordered categorical role authority required")
    if torch.isin(train["ids"], select["ids"]).any().item():
        raise ValueError("TRAIN and SELECT overlap")
    tokens = mono_tokens(torch, project_root, graph["x"], graph["edge_index"], K=config["native"]["K"],
                         preprocessing=config["preprocessing"])
    tokens = [token.to(device) for token in tokens]
    views = config["train_views"]
    loss = config.get("loss", {"variant": "own", "coefficients": {}})
    if loss["variant"] == "own" and loss["coefficients"] != {}:
        raise ValueError("Native own CE has no hidden coefficient overrides")
    if type(views) is not int or views < 1 or (loss["variant"] == "own" and members == 4 and views != 1):
        raise ValueError("Declare TRAIN views; equipped4 native own procedure uses one view per route")
    seeds = config["dropout_seeds"]
    if len(seeds) < max(members, views):
        raise ValueError("Private factual stream seeds required for every own view")
    model = build_bank(torch, project_root, tokens, native=config["native"], classes=config["classes"],
                       dataset=study["dataset"], members=members, factorized=factorized,
                       initialization_seed=config["initialization_seed"], dropout_seeds=seeds,
                       factor_initialization=config.get("factor_initialization", "unit"),
                       factor_seed=config.get("factor_seed"))
    groups = []
    for name, parameter in model.named_parameters():
        attention = "attnmodule" in name
        lr = config["attn_lr"] if attention else config["learning_rate"]
        decay = config["attn_weight_decay"] if attention else config["weight_decay"]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (lr, decay)) or lr == 0:
            raise ValueError("Declared finite positive learning rates/nonnegative Adam decay required")
        groups.append({"params": [parameter], "lr": lr, "weight_decay": decay})
    optimizer = torch.optim.Adam(groups, eps=1e-8, betas=(.9, .999))
    train_ids, train_y = train["ids"].to(device), train["y"].to(device)
    select_ids, select_y = select["ids"].to(device), select["y"].to(device)

    def own_training(adapter, step):
        adapter.optimizer.zero_grad(set_to_none=True)
        loss_values = []
        for view in range(views):
            for route in range(members):
                logits, _ = model.forward_route(route, slot=route if members > 1 else view)
                ce = torch.nn.functional.cross_entropy(logits[train_ids], train_y)
                (ce / (members * views)).backward()
                loss_values.append(adapter._scalar(ce))
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("Nonfinite native old-state CE gradient")
        adapter.optimizer.step()
        mean = sum(loss_values) / len(loss_values)
        return {"objective": mean, "ce": mean, "regularization": 0., "own_view_ce": loss_values,
                "operation_counts": {"optimizer_updates": 1, "model_forwards": members * views,
                                     "route_predictions": members * views, "backward_calls": members * views,
                                     "supervised_examples": len(train_ids) * members * views}}

    callback, custom_train = None, own_training
    if loss["variant"] != "own":
        custom_train = None
        paired = loss["variant"].startswith("two_view_")
        if views != (2 if paired else 1):
            raise ValueError("Scalar loss variants must explicitly match their one/two TRAIN views")

        def callback(adapter, step, logits, hidden):
            second_logits, second_hidden = None, None
            if paired:
                second_logits = model()
                second_hidden = model.last_hidden
            result = node_objective(torch, torch.nn.functional, logits[:, train_ids], hidden[:, train_ids], train_y,
                                    variant=loss["variant"], coefficients=loss["coefficients"],
                                    paired_logits=second_logits[:, train_ids] if paired else None,
                                    paired_hidden=second_hidden[:, train_ids] if paired else None)
            return {**result, "operation_counts": {"model_forwards": members * views,
                                                    "route_predictions": members * views,
                                                    "supervised_examples": len(train_ids) * members * views}}

    selector = {"metric": run["selection"]["metric"], "direction": run["selection"]["direction"],
                "ties": run["selection"]["ties"], "save_every": run["selection"].get("save_every", 1),
                "patience_steps": run["selection"].get("patience_steps", run["budget"]["maximum"])}
    return ClassifierAdapter(torch, torch.nn.functional, model, optimizer,
                             logits_fn=lambda model, training, step: model(), train_ids=train_ids, train_y=train_y,
                             select_ids=select_ids, select_y=select_y, regularizer=lambda model: tokens[0].new_zeros(()),
                             fusion="mean_probabilities", selector=selector,
                             provenance={"run_id": run["run_id"], "data": study["data"], "config": config,
                                         "semantic_method_id": run["method"], "polyformer_mode": mode,
                                         "actual_model_seed": config["initialization_seed"],
                                         "private_dropout_seeds": list(seeds),
                                         "factor_initialization": config.get("factor_initialization", "unit"),
                                         "factor_seed": config.get("factor_seed"),
                                         "source": study["source"], "author_repository": AUTHOR_REPOSITORY,
                                         "author_commit": AUTHOR_COMMIT, "native_source_files": SOURCE_FILES,
                                         "input_receipts": input_receipts},
                             external_capture=model.capture_runtime_state, external_restore=model.restore_runtime_state,
                             custom_train=custom_train, objective_callback=callback,
                             objective_provenance=loss if callback else None)


def assemble_independent(torch, adapters):
    """Serve separately selected M1 fits; no pooled selector, update or new model."""
    if len(adapters) != 4 or any(adapter.model.members != 1 for adapter in adapters):
        raise ValueError("Four separately selected native/equipped M1 adapters required")
    try:
        from ..engine import inspect_independence
    except ImportError:
        from engine import inspect_independence
    inspect_independence(adapters)
    if any(adapter.provenance["data"] != adapters[0].provenance["data"] for adapter in adapters[1:]):
        raise ValueError("Independent serving requires matching declared data/role custody")
    logits = []
    with torch.no_grad():
        for adapter in adapters:
            adapter.model.eval()
            logits.append(adapter.model()[0])
    raw = torch.stack(logits)
    return {"raw_logits": raw, "probabilities": raw.softmax(-1).mean(0),
            "fusion": "mean_probabilities", "additional_optimizer_updates": 0}
