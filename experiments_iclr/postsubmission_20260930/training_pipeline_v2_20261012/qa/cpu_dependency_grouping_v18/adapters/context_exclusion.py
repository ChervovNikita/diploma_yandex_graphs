"""Pinned SAGE context-positive exclusion mechanism on explicit v2 roles.

Import is stdlib-only. The sole source-function AST change replaces the fixed
580-row guard with >=2 rows; targets, masks, reductions and gradients are
unchanged. Full5000, arbitrary admitted TRAIN/SELECT counts and standardized
provider roles are declared controller/data adaptations, not source replication.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

try:
    from .classifier import ClassifierAdapter
    from ..engine import AdapterStepError
    from ..spec import confined, fingerprint, normalized_id, ORIGINAL_PAPER_DATASETS
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from engine import AdapterStepError
    from spec import confined, fingerprint, normalized_id, ORIGINAL_PAPER_DATASETS


SOURCE_FILES = {
    "family": ("native_SAGE_context_positive_exclusion_source_20261010_v1/run_family.py", "739ebf06e2a9c47de383a8e4c6ce9536af965c35430a8c5b227140b521fe1c32"),
    "objective": ("native_SAGE_context_positive_exclusion_source_20261010_v1/context_objective.py", "7197b90a9e2bf58894d86e88c17edfae768df191e01f6c3148b5473f6758379c"),
    "native": ("common_wrapper_native_source_20261010_v1/models.py", "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303"),
    "routes": ("shared_fast_graph_model_interface_20261010_v1/common_routes.py", "84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773"),
    "factors": ("portable_internal_be_public_interface_20261007_v2/core/factors.py", "9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3"),
    "seedlaw": ("common_wrapper_native_source_20261010_v1/run_common.py", "e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc"),
    "base_trainer": ("common_wrapper_native_source_20261010_v1/run_base.py", "ca78469a7c1dcc9e98fba054ae1df714f6b19c3e91690ee7b48538005b4e52b1"),
}
SOURCE_ARMS = {
    "ordinary_m1_context_mixture": ("single", 1, "mixture"),
    "factorized_m1_context_mixture": ("factorized_single", 1, "mixture"),
    "shared4_context_private_masked": ("internal_be", 4, "private"),
    "shared4_context_pair_mixture_masked": ("internal_be", 4, "mixture"),
    "shared4_context_private_full": ("internal_be", 4, "private_full"),
    "shared4_context_shuffled_masked": ("internal_be", 4, "shuffled_private"),
    "ordinary_m1": ("single", 1, "none"),
    "factorized_m1": ("factorized_single", 1, "none"),
    "original_boundary": ("original_gnnm", 4, "none"),
}
SOURCE_CONTRACT = {
    "mechanism": "own_CE_plus_.05_fixed_context_exclusion",
    "contexts": ["X", "X-PX", "PX", "P^2X"], "neighbors": 16,
    "temperature": .2, "coefficient": .05, "shuffle_seed": 9100019,
    "target_policy": "TRAIN_sameclass_top16_nonself;float64_signatures;stable_ties",
    "masked_denominator": "positive_union_differentclass;exclude_unselected_classmates",
    "mixture": "mean_four_complete_Q_E_losses;never_union_denominator",
    "shuffle": "joint_Q_support_axes_within_class_and_positive_degree;no_retry_if_trivial",
    "gradients": "one_live_H_Gram_per_route;both_sides_live;all_slow_and_fast_receive_same_scalar",
    "initialization": "source_Model;unit_factors;only_input_r_row_Rademacher_seed+2000003",
    "dropout": "persistent_source_fork_rng;seed+3000007+1000003*body_or_route",
    "TRAIN_order": "shared_bank:provider_train_ids;M1/I4:stable_sorted_ids_boolean_mask_order",
    "source_horizon": "max1000/patience300;strict_first_accuracy_every_update",
    "v2_horizon": "full5000;strict_first_accuracy_every_update;checkpoint0_false",
    "source_function_delta": "only_hidden.shape[1]!=580_to_hidden.shape[1]<2",
    "source_exact_training_recipe": False,
}


def receipt(root, path):
    path = confined(root, path)
    raw = path.read_bytes()
    return {"path": str(path.relative_to(Path(root).resolve())), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def read_bound(root, binding):
    if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
        raise ValueError("Exact path/bytes/SHA256 receipt required")
    if receipt(root, binding["path"]) != binding:
        raise ValueError("Source/input receipt changed")
    return confined(root, binding["path"])


def source_paths(root, bindings):
    if set(bindings) != set(SOURCE_FILES):
        raise ValueError("Complete exact family/operator/helper dependency roster required")
    result = {}
    for key, (_, digest) in SOURCE_FILES.items():
        if bindings[key].get("sha256") != digest:
            raise ValueError("Pinned context source changed: " + key)
        result[key] = read_bound(root, bindings[key])
    return result


def objective_module(path):
    """Compile pinned pure helpers; perform exactly one declared guard edit."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_FILES["objective"][1]:
        raise ValueError("Pinned context objective required")
    tree = ast.parse(raw)
    changed = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1
                and isinstance(node.ops[0], ast.NotEq) and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == 580
                and ast.unparse(node.left) == "hidden.shape[1]"):
            node.ops[0] = ast.Lt();node.comparators[0] = ast.Constant(value=2);changed += 1
    if changed != 1:
        raise ValueError("The sole fixed-row source guard was not found exactly once")
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return SimpleNamespace(**{k: v for k, v in namespace.items() if not k.startswith("__")})


def import_source(path, key):
    name = "v2_context_source_" + key + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec);sys.modules[name] = module
        try:spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None);raise
    return sys.modules[name]


def seed_function(path, torch, np):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "set_seed"]
    if len(nodes) != 1:raise ValueError("Exact native set_seed function required")
    env = {"torch": torch, "np": np, "random": random}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env["set_seed"]


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes(order="C")).hexdigest()


def target_payload(torch, np, helper, graph, train, ancestry):
    """Target custody only: public graph/features and ordered TRAIN, no model."""
    arrays, metadata = helper.prepare_targets(np, graph["x"].numpy(), graph["edge_index"].numpy(),
                                               train["ids"].numpy(), train["y"].numpy())
    order = np.argsort(train["ids"].numpy(), kind="stable")
    banks = {"bank": helper.tensor_bank(torch, arrays, np.arange(len(order), dtype=np.int64), "cpu"),
             "single": helper.tensor_bank(torch, arrays, order, "cpu")}
    hashes = {name: {key: tensor_hash(value) for key, value in bank.items()} for name, bank in banks.items()}
    return {"schema": "context-exclusion-targets-v1", "ordered_train_ids": train["ids"].clone(),
            "ordered_train_y": train["y"].clone(), "single_order": torch.from_numpy(order.copy()),
            "banks": banks, "tensor_hashes": hashes, "ancestry": copy.deepcopy(ancestry),
            "preparation": metadata, "source_contract": SOURCE_CONTRACT}


def validate_config(study, run):
    c = run["config"]
    if c.get("context_arm") not in SOURCE_ARMS:
        raise ValueError("A declared exact source arm/control is required")
    architecture, members, mode = SOURCE_ARMS[c["context_arm"]]
    if (c.get("architecture"), c.get("members"), c.get("context_mode")) != (architecture, members, mode):
        raise ValueError("Arm architecture/routes/context allocation differ")
    if (c.get("backbone"), c.get("width"), c.get("depth"), c.get("weight_decay")) != ("SAGE", 128, 2, 0):
        raise ValueError("This scoped port fixes source SAGE128/depth2/AdamW decay0")
    if c.get("learning_rate") not in (.0005, .001, .002) or c.get("dropout") not in (.1, .2, .4):
        raise ValueError("Only the frozen nine LR/dropout points are supported")
    if (run["budget"] != {"unit": "optimizer_updates", "policy": "full_budget", "maximum": 5000}
            or run["selection"] != {"metric": "accuracy", "direction": "max", "ties": "first",
                                    "evaluation_interval": 1, "checkpoint0": False}):
        raise ValueError("Declare standardized full5000 and source strict-first accuracy selection")
    if study["data"]["task"] != "node_classification" or c.get("classes", 0) < 3:
        raise ValueError("Categorical full-graph node classification only")
    if normalized_id(study["dataset"]) in ORIGINAL_PAPER_DATASETS:
        raise ValueError("Original GNNM paper datasets are excluded")
    if c.get("fusion") != ("mean_logits" if architecture == "original_gnnm" else "mean_probabilities"):
        raise ValueError("Retain native boundary logit mean and source context probability mean")
    if set(c["node_inputs"]) != {"graph", "train", "select"}:
        raise ValueError("Fit opens only GRAPH/TRAIN/SELECT authorities")
    if mode != "none" and not isinstance(c.get("context_targets"), dict):
        raise ValueError("An immutable precomputed TRAIN-only target pack is required")


class ContextExclusionAdapter(ClassifierAdapter):
    def validate(self, step):
        try:
            result = super().validate(step)
            counts = dict(self.model.last_counts)
            counts.update(validation_metric_calls=1, evaluated_examples=int(self.select_y.numel()))
            result["operation_counts"] = counts
            return result
        except BaseException as error:
            raise AdapterStepError(str(error), operation_counts=dict(self.model.last_counts)) from error


def make_adapter(*, torch, project_root, study, run, device):
    validate_config(study, run)
    import numpy as np
    from torch.nn import functional as F
    c = run["config"];paths = source_paths(project_root, c["context_source"])
    native = import_source(paths["native"], "native")
    routes = import_source(paths["routes"], "routes")
    factors = import_source(paths["factors"], "factors")
    set_seed = seed_function(paths["seedlaw"], torch, np)
    data = {k: torch.load(read_bound(project_root, row), map_location="cpu", weights_only=True)
            for k, row in c["node_inputs"].items()}
    graph, train, select = (data[k] for k in ("graph", "train", "select"))
    if set(graph) != {"x", "edge_index"} or any(set(r) != {"ids", "y"} for r in (train, select)):
        raise ValueError("Exact target-free graph and separate TRAIN/SELECT dictionaries required")
    n, features = graph["x"].shape
    if graph["x"].dtype != torch.float32 or not torch.isfinite(graph["x"]).all():raise ValueError("Finite float32 X required")
    edges = graph["edge_index"]
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2 or edges.min() < 0 or edges.max() >= n:
        raise ValueError("Valid homogeneous int64 COO required")
    for role in (train, select):
        ids, y = role["ids"], role["y"]
        if (ids.dtype != torch.long or y.dtype != torch.long or ids.ndim != 1 or y.shape != ids.shape
                or not len(ids) or len(torch.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= n
                or y.min() < 0 or y.max() >= c["classes"]):raise ValueError("Exact unique categorical role IDs/labels required")
    if torch.isin(train["ids"], select["ids"]).any():raise ValueError("TRAIN/SELECT overlap")
    architecture, members, mode = SOURCE_ARMS[c["context_arm"]]
    order = torch.arange(len(train["ids"])) if members == 4 else torch.argsort(train["ids"], stable=True)
    train_ids, train_y = train["ids"][order].to(device), train["y"][order].to(device)
    helper, targets, hashes = None, None, None
    if mode != "none":
        helper = objective_module(paths["objective"])
        payload = torch.load(read_bound(project_root, c["context_targets"]), map_location="cpu", weights_only=True)
        if (set(payload) != {"schema", "ordered_train_ids", "ordered_train_y", "single_order", "banks",
                            "tensor_hashes", "ancestry", "preparation", "source_contract"}
                or payload["schema"] != "context-exclusion-targets-v1" or payload["source_contract"] != SOURCE_CONTRACT
                or payload["ancestry"] != {"graph": c["node_inputs"]["graph"], "train": c["node_inputs"]["train"],
                                          "objective_sha256": SOURCE_FILES["objective"][1]}
                or not torch.equal(payload["ordered_train_ids"], train["ids"])
                or not torch.equal(payload["ordered_train_y"], train["y"])
                or not torch.equal(payload["single_order"], torch.argsort(train["ids"], stable=True))):
            raise ValueError("Target graph/TRAIN/source/order ancestry differs")
        name = "bank" if members == 4 else "single"
        targets = {k: v.to(device) for k, v in payload["banks"][name].items()}
        hashes = {k: tensor_hash(v) for k, v in targets.items()}
        if hashes != payload["tensor_hashes"][name]:raise ValueError("Target tensor bytes changed")
        b = len(train_ids)
        if (set(targets) != {"weights", "shuffled_weights", "retained", "shuffled_retained", "full_allowed"}
                or targets["full_allowed"].shape != (b, b)):
            raise ValueError("Complete aligned target/support bank required")
    initialization = run["seed"] + 1000003 * run["body"]
    set_seed(initialization)
    args = ("SAGE", 2, features, 128, c["classes"], 1, 8, "LayerNorm", c["dropout"])
    body = native.TABMModel(*args, tabm_inits=4, device=device) if architecture == "original_gnnm" else native.Model(*args)
    source_adapter = None if architecture == "original_gnnm" else routes.NativeModelAdapter(torch, body)
    wrapped = None
    if architecture in {"factorized_single", "internal_be"}:
        wrapped = routes.wrap_shared(torch, factors, body, source_adapter, members=members, kind="baseline", rank=16,
                                     block_seed=run["seed"]+4000037)
        with torch.no_grad():
            for row in range(members):
                seed = run["seed"] + 1000003*(run["body"]+row)+2000003
                generator = torch.Generator(device="cpu").manual_seed(seed)
                body.input_linear.r[row].copy_((torch.randint(0, 2, (features,), generator=generator)*2-1).to(body.input_linear.r))
    streams = []
    for row in range(members):
        set_seed(run["seed"]+1000003*(run["body"]+row)+3000007)
        stream = {"cpu": torch.get_rng_state().clone()}
        if str(device).startswith("cuda"):stream["cuda"] = torch.cuda.get_rng_state(device).clone()
        streams.append(stream)
    cuda_index = torch.device(device).index if str(device).startswith("cuda") else None
    x = graph["x"].to(device);edge_index = edges.to(device);g = SimpleNamespace(edge_index=edge_index)
    batch = {"x": x, "edge_index": edge_index, "graph": g}

    class Bank(torch.nn.Module):
        def __init__(self):
            super().__init__();self.source_model = wrapped if members == 4 and wrapped is not None else body
            self.last_hidden = None;self.last_counts = {}
        def scope(self, row):return routes.existing_member_rng_scope(torch, streams, row, cuda_index)
        def forward(self):
            self.last_counts = {"model_forward_attempts": members, "model_forwards": 0, "route_predictions": 0}
            if architecture == "original_gnnm":
                rows = []
                for row in range(members):
                    with self.scope(row):rows.append(self.source_model(g, x, tabm_seed=row))
                    self.last_counts["model_forwards"] += 1;self.last_counts["route_predictions"] += 1
                self.last_hidden = None;return torch.stack(rows)
            if members == 4:
                logits, hidden = self.source_model(batch, route_scope=lambda row:self.scope(row))
            else:
                with self.scope(0):logits, hidden = source_adapter.native(batch)
                logits, hidden = logits.unsqueeze(0), hidden.unsqueeze(0)
            self.last_hidden = hidden
            self.last_counts.update(model_forwards=members, route_predictions=members)
            return logits

    model = Bank().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=0)
    if wrapped is not None:wrapped.check_optimizer_ownership(optimizer)
    versions = {k: v._version for k, v in targets.items()} if targets else {}
    def guard_targets(*, check_bytes=False):
        if targets and (any(v._version != versions[k] for k, v in targets.items())
                        or (check_bytes and {k: tensor_hash(v) for k, v in targets.items()} != hashes)):
            raise ValueError("Immutable context targets changed")
    def capture():
        guard_targets(check_bytes=True);state = np.random.get_state()
        return {"streams": copy.deepcopy(streams), "numpy_rng": [state[0], state[1].tolist(), *state[2:]],
                "target_hashes": hashes, "target_versions": versions,
                "factor_members": {name: module.member for name, module in model.named_modules()
                                   if isinstance(module, factors.FactorLinear)}}
    def restore(state):
        guard_targets(check_bytes=True)
        if state["target_hashes"] != hashes or state["target_versions"] != versions or len(state["streams"]) != members:
            raise ValueError("Checkpoint context/RNG/route ancestry differs")
        streams[:] = copy.deepcopy(state["streams"])
        nr = state["numpy_rng"];np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), *nr[2:]))
        for name, member in state["factor_members"].items():dict(model.named_modules())[name].member = member
    def train_step(adapter, step):
        counts = {"backward_attempts": 0, "backward_calls": 0, "optimizer_update_attempts": 0,
                  "optimizer_updates": 0, "supervised_examples": len(train_ids),
                  "training_label_opportunities": members*len(train_ids)}
        try:
            guard_targets();optimizer.zero_grad()
            logits = model();counts.update(model.last_counts)
            own = F.cross_entropy(logits[:, train_ids].flatten(0, 1), train_y.repeat(members))
            auxiliary = own*0;route_losses = pair_losses = None
            if mode != "none":
                counts["context_loss_attempts"] = 1
                auxiliary, route_losses, pair_losses, paid = helper.context_auxiliary(
                    torch, F, model.last_hidden[:, train_ids], targets, mode)
                counts["context_loss_calls"] = 1
                counts.update(paid)
            objective = own + .05*auxiliary
            if not torch.isfinite(objective):raise FloatingPointError("Nonfinite source context objective")
            counts["backward_attempts"] = 1;objective.backward();counts["backward_calls"] = 1
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite source context gradient")
            counts["optimizer_update_attempts"] = 1;optimizer.step();counts["optimizer_updates"] = 1
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError("Nonfinite parameter")
            if any(torch.is_tensor(v) and not torch.isfinite(v).all()
                   for state in optimizer.state.values() for v in state.values()):
                raise FloatingPointError("Nonfinite AdamW state")
            components = {"context_mode": mode, "auxiliary": float(auxiliary.detach())}
            if route_losses is not None:
                components.update(route_losses=route_losses.detach().cpu().tolist(), pair_losses=pair_losses.detach().cpu().tolist())
            return {"objective": float(objective.detach()), "ce": float(own.detach()),
                    "regularization": float((.05*auxiliary).detach()), "operation_counts": counts,
                    "objective_components": components}
        except BaseException as error:
            counts.update(model.last_counts)
            raise AdapterStepError(str(error), operation_counts=counts) from error
    adapter = ContextExclusionAdapter(torch, F, model, optimizer, logits_fn=lambda model, training, step:model(),
        train_ids=train_ids, train_y=train_y, select_ids=select["ids"].to(device), select_y=select["y"].to(device),
        regularizer=lambda model:next(model.parameters()).sum()*0, fusion=c["fusion"],
        selector={"metric":"accuracy", "direction":"max", "ties":"first", "patience_steps":5000, "save_every":1},
        provenance={"actual_model_seed": initialization, "source_contract": SOURCE_CONTRACT,
                    "config_hash": fingerprint(c), "TRAIN_H_order": "provider_ids" if members == 4 else "stable_sorted_ids",
                    "source_bindings": c["context_source"], "inputs": c["node_inputs"], "context_targets":c.get("context_targets")},
        external_capture=capture, external_restore=restore, custom_train=train_step)
    adapter.context_targets = targets;adapter.source_streams = streams
    return adapter
