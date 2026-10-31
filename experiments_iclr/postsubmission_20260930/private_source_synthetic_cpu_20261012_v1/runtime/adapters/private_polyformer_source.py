"""Owned bridge utilities for pinned private PolyFormer source sessions.

Numerics are supplied at execution. This is a source/data/state bridge, with no
learning objective, optimizer rule, model fit or remote action at import.
"""
import ast
from contextlib import contextmanager
import copy
import hashlib
import math
from pathlib import Path
import random
from types import SimpleNamespace
import time

try:
    from .polyformer import native_class, factor_definitions, mono_tokens, _constructor_seed
    from .classifier import ClassifierAdapter
    from ..spec import confined, fingerprint, normalized_id, ORIGINAL_PAPER_DATASETS
except ImportError:
    from adapters.polyformer import native_class, factor_definitions, mono_tokens, _constructor_seed
    from adapters.classifier import ClassifierAdapter
    from spec import confined, fingerprint, normalized_id, ORIGINAL_PAPER_DATASETS

NATIVE = {"K": 2, "hidden": 256, "nlayer": 2, "n_head": 8, "d_ffn": 32,
          "q": 1.6, "multi": 2., "dropout": .5, "dprate": .8}
DOMAIN = [{"learning_rate": lr, "dropout": dropout} for lr in (.0025, .005, .01) for dropout in (.3, .5, .7)]
SOURCE_ANCHOR = {"learning_rate": .005, "dropout": .5}


def receipt(root, relative):
    path = confined(root, relative)
    if path.is_symlink():
        raise ValueError("Immutable source/input may not be symlinked")
    raw = path.read_bytes()
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def bound(root, row):
    if set(row) != {"path", "bytes", "sha256"} or receipt(root, row["path"]) != row:
        raise ValueError("Pinned source/input changed")
    return confined(root, row["path"])


def source_tree(root, row):
    return ast.parse(bound(root, row).read_bytes())


def definitions(root, row, names, environment, transform=None):
    tree = source_tree(root, row)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(nodes) != len(names) or {n.name for n in nodes} != set(names):
        raise ValueError("Complete exact source API definitions required")
    tree = ast.Module(body=nodes, type_ignores=[])
    if transform is not None:
        tree = transform.visit(tree)
    exec(compile(ast.fix_missing_locations(tree), str(bound(root, row)), "exec"), environment)
    return environment


def load_roles(torch, root, bindings, classes):
    if set(bindings) != {"graph", "train", "select"}:
        raise ValueError("Fit may open only target-free GRAPH and separate TRAIN/SELECT")
    values = {k: torch.load(bound(root, row), map_location="cpu", weights_only=True) for k, row in bindings.items()}
    graph, train, select = (values[k] for k in ("graph", "train", "select"))
    if set(graph) != {"x", "edge_index"} or any(set(role) != {"ids", "y"} for role in (train, select)):
        raise ValueError("Separate exact feature/edge and role authorities required")
    x, edges = graph["x"], graph["edge_index"]
    if x.ndim != 2 or x.dtype != torch.float32 or x.requires_grad or not torch.isfinite(x).all():
        raise ValueError("Complete finite target-free float32 public features required")
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2 or not edges.numel() or edges.min() < 0 or edges.max() >= len(x):
        raise ValueError("Complete valid homogeneous int64 COO graph required")
    for role in (train, select):
        ids, y = role["ids"], role["y"]
        if ids.ndim != 1 or ids.dtype != torch.long or y.dtype != torch.long or y.shape != ids.shape or not len(ids) or len(ids.unique()) != len(ids) or ids.min() < 0 or ids.max() >= len(x) or y.min() < 0 or y.max() >= classes:
            raise ValueError("Complete unique ordered categorical role IDs/targets required")
    if torch.isin(train["ids"], select["ids"]).any():
        raise ValueError("TRAIN/SELECT overlap")
    return graph, train, select


@contextmanager
def seeded(torch, np, seed, device):
    py, nr, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with _constructor_seed(torch, seed, device):
        try:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            yield
        finally:
            random.setstate(py)
            np.random.set_state(nr)
            torch.set_rng_state(cpu)


def constructor(torch, root, recipe, seed, device):
    import numpy as np
    with seeded(torch, np, seed, device):
        return native_class(torch, root)(None, SimpleNamespace(**recipe)).to(device)


def recipe(config, x, dataset):
    value = {**NATIVE, "dropout": config["dropout"], "base": "mono", "dataset": dataset,
             "num_features": x.shape[1], "num_classes": config["classes"],
             "lr": config["learning_rate"], "attn_lr": config["learning_rate"] / 10,
             "weight_decay": .001, "attn_wd": 1e-8}
    return value


def optimizer_inventory(model, optimizers):
    expected = {id(p) for p in model.parameters()}
    owned = [p for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]]
    if len(owned) != len({id(p) for p in owned}) or {id(p) for p in owned} != expected:
        raise ValueError("Every source encoder/decoder parameter must have exactly one optimizer owner")
    return [{"parameters": len([p for g in o.param_groups for p in g["params"]]),
             "coordinates": sum(p.numel() for g in o.param_groups for p in g["params"])} for o in optimizers]


class OptimizerOwners:
    """Checkpoint container; source sessions retain every actual Adam step owner."""
    def __init__(self, optimizers):
        self.optimizers = tuple(optimizers)

    @property
    def param_groups(self):
        return [g for optimizer in self.optimizers for g in optimizer.param_groups]

    @property
    def state(self):
        return {p: state for optimizer in self.optimizers for p, state in optimizer.state.items()}

    def state_dict(self):
        return {"schema": "source-optimizer-owner-roster-v1", "optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state):
        if state["schema"] != "source-optimizer-owner-roster-v1" or len(state["optimizers"]) != len(self.optimizers):
            raise ValueError("Exact source optimizer owner roster required")
        for optimizer, saved in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(saved)

    def step(self):
        raise RuntimeError("Only the exact source session may commit its optimizer owners")


class SourceCosts:
    def __init__(self):
        self.records = []

    @contextmanager
    def measure(self, scope):
        started = time.perf_counter()
        record = {"scope": scope}
        try:
            yield record
        finally:
            record["wall_seconds"] = time.perf_counter() - started
            self.records.append(record)


def runtime_capture(torch, bodies):
    return {"factor_members": [{name: int(module.member) for name, module in body.named_modules() if hasattr(module, "member")}
        for body in bodies], "poly_attn_bias": [{name: module.bias.detach().cpu().clone() for name, module in body.named_modules()
        if type(module).__name__ == "PolyAttn"} for body in bodies]}


def runtime_restore(torch, bodies, state):
    if len(bodies) != len(state["factor_members"]) or len(bodies) != len(state["poly_attn_bias"]):
        raise ValueError("Selected full source body roster changed")
    for body, members, biases in zip(bodies, state["factor_members"], state["poly_attn_bias"]):
        modules = dict(body.named_modules())
        for name, member in members.items():
            modules[name].member = member
        for name, bias in biases.items():
            modules[name].bias = bias.clone()


class SourceClassifierAdapter(ClassifierAdapter):
    """Use exact counts and stable probability NLL at the own selected state."""
    def validate(self, step):
        self.model.eval()
        with self.t.no_grad():
            logits = self.logits_fn(self.model, False, step)
            selected = logits[:, self.select_ids]
            lp = selected.double().log_softmax(-1)
            pool = self.t.logsumexp(lp, 0) - math.log(len(lp))
            correct = int(pool.argmax(-1).eq(self.select_y).sum())
            ce = self.t.stack([self.F.cross_entropy(z, self.select_y) for z in selected]).mean()
            return {"objective": self._scalar(ce), "ce": self._scalar(ce), "regularization": 0.,
                "accuracy": correct / len(self.select_y), "correct": correct, "examples": len(self.select_y),
                "pooled_nll": self._scalar(-pool.gather(1, self.select_y[:, None]).mean()),
                "selection_metric": correct / len(self.select_y), "fusion": "mean_probabilities",
                "operation_counts": {"optimizer_updates": 0, "backward_calls": 0, "model_forwards": len(logits),
                    "route_predictions": len(logits), "validation_metric_calls": 1, "evaluated_examples": len(self.select_y)}}


def common_config_check(study, run, *, unit):
    c = run["config"]
    if c["classes"] < 3 or c.get("native") != {**NATIVE, "dropout": c["dropout"]}:
        raise ValueError("Pinned K2/256-wide two-layer PolyFormer recipe required")
    if {"learning_rate": c["learning_rate"], "dropout": c["dropout"]} not in DOMAIN:
        raise ValueError("Only the declared useful source-shaped nine-point domain is admitted")
    if run["budget"] != {"unit": unit, "policy": "full_budget", "maximum": 5000} or run["selection"] != {
            "metric": "accuracy", "direction": "max", "ties": "first", "evaluation_interval": 1, "checkpoint0": False}:
        raise ValueError("Bind full5000 updates/macro cycles and source strict earliest SELECT")
    if normalized_id(study["dataset"]) in ORIGINAL_PAPER_DATASETS or study["data"]["task"] != "node_classification":
        raise ValueError("Only compatible new categorical node providers; original paper reruns excluded")
    return c


def require(value, message):
    if not value:
        raise ValueError(message)


class ProviderConstants(ast.NodeTransformer):
    """Explicit provider/LR substitutions; mathematical routines stay exact."""
    def __init__(self, replacements):
        self.replacements = replacements
        self.changed = {}

    def visit_Constant(self, node):
        key = (type(node.value), node.value)
        if key in self.replacements:
            name = self.replacements[key]
            self.changed[name] = self.changed.get(name, 0) + 1
            return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
        return node


class UpdateAuthority:
    """No-GO exports can restore; only the claimed engine fit may update."""
    def __init__(self, execution_context=None, *, engineering=False):
        self.allowed = engineering or (isinstance(execution_context, dict)
            and set(execution_context) == {"registry", "output_root", "paid_parent_checkpoints"}
            and execution_context["registry"] is not None)

    def require(self, *names):
        if "scientific" in names and not self.allowed:
            raise RuntimeError("Restoration-only factory: fit execution requires claimed engine context after explicit GO")


def coordinate_counts(features, classes):
    h, tokens, layers, ff, heads, wide = 256, 3, 2, 32, 8, 512
    native = features*h+h+h*h+h+h*classes+classes + layers*(tokens*(h*wide+wide+wide*h+h)
        +2*h*h+heads*tokens+2*h+(h*ff+ff+ff*h+h)+2*h)
    private = 4*((features+h)+(h+h)+(h+classes)+layers*(tokens*2*(h+wide)+2*(h+h)+2*(h+ff)))
    return native, private


def source_namespace(torch, root, row):
    environment = {"__name__": "pinned_private_state_helpers", "random": random}
    definitions(root, row, ("capture_rng", "restore_rng", "synchronize", "parameter_counts", "cpu_tree", "exact"), environment)
    return SimpleNamespace(**environment)


def source_adapter(torch, study, run, session, model, bodies, optimizers, logits, training, capture, restore):
    c = run["config"]
    return SourceClassifierAdapter(torch, torch.nn.functional, model, OptimizerOwners(optimizers),
        logits_fn=logits, train_ids=session.train_ids if hasattr(session, "train_ids") else session.inputs.train.ids,
        train_y=session.train_y if hasattr(session, "train_y") else session.inputs.train.targets,
        select_ids=session.select_ids, select_y=session.select_y, regularizer=lambda model: next(model.parameters()).new_zeros(()),
        fusion="mean_probabilities", selector={"metric": "accuracy", "direction": "max", "ties": "first",
            "save_every": 1, "patience_steps": 5001}, provenance={"run_id": run["run_id"], "data": study["data"],
            "source": study["source"], "config": c, "source_shape_adaptation": True,
            "source_training_math_retained": True, "actual_optimizer_inventory": optimizer_inventory(model, optimizers)},
        custom_train=training, external_capture=capture, external_restore=restore)
