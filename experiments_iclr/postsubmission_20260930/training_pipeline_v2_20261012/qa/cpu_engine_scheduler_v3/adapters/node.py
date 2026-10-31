"""Executable node-classification bridge for the original GNNM model family.

It supports the native single, original boundary projectors, internal diagonal
factors and coupled untied controls. Other mechanisms need their own adapters;
they are not silently mapped to one of these implementations.
"""
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace

try:
    from .classifier import ClassifierAdapter
except ImportError:
    from adapters.classifier import ClassifierAdapter


def source_module(project_root, relative):
    path = (Path(project_root) / relative).resolve()
    if not path.is_relative_to(Path(project_root).resolve()):
        raise ValueError("Model source must stay inside the project")
    name = "gnnm_native_models_" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def make_adapter(*, torch, project_root, study, run, device):
    from torch.nn import functional as F
    config = run["config"]
    method = run["method"]
    supported = {"single", "original_gnnm", "internal_be", "factorized_single", "joint_untied"}
    if method not in supported:
        raise ValueError("This adapter does not implement: " + method)
    inputs = config["node_inputs"]
    def load(relative):
        path = (Path(project_root) / relative).resolve()
        if not path.is_relative_to(Path(project_root).resolve()):
            raise ValueError("Input path escapes the repository")
        return torch.load(path, map_location="cpu", weights_only=False)
    graph_inputs = load(inputs["graph"])
    train = load(inputs["train"])
    select = load(inputs["select"])
    if set(graph_inputs) != {"x", "edge_index"} or set(train) != {"ids", "y"} or set(select) != {"ids", "y"}:
        raise ValueError("Training adapter requires graph plus separate TRAIN/SELECT targets")
    train_ids, select_ids = train["ids"], select["ids"]
    if torch.isin(train_ids, select_ids).any().item():
        raise ValueError("TRAIN and SELECT overlap")
    graph = SimpleNamespace(edge_index=graph_inputs["edge_index"].to(device))
    x = graph_inputs["x"].to(device)
    train_ids, select_ids = train_ids.to(device), select_ids.to(device)
    train_y, select_y = train["y"].to(device), select["y"].to(device)
    members = 1 if method in {"single", "factorized_single"} else config["members"]
    model_seed = config.get("initialization_seed", run["seed"] + 1000003 * run["body"])
    torch.manual_seed(model_seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(model_seed)
    random.seed(model_seed)
    native = source_module(project_root, config["native_model_source"])
    arguments = (config["backbone"], config["depth"], x.shape[1], config["width"],
                 config["classes"], config["ffn_multiplier"], config["heads"],
                 config["normalization"], config["dropout"])

    class NativeBank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if method == "original_gnnm":
                self.body = native.TABMModel(*arguments, tabm_inits=members, device=device)
            elif method == "joint_untied":
                self.bodies = torch.nn.ModuleList(native.Model(*arguments) for _ in range(members))
            else:
                self.body = native.Model(*arguments)
            if method in {"internal_be", "factorized_single"}:
                self.fast_r, self.fast_s = torch.nn.ParameterList(), torch.nn.ParameterList()
                self.sites = []
                for name, parameter in self.body.named_parameters():
                    if parameter.ndim == 2 and name.endswith("weight"):
                        self.sites.append(name)
                        r = torch.empty(members, parameter.shape[1], device=parameter.device)
                        s = torch.ones(members, parameter.shape[0], device=parameter.device)
                        r.bernoulli_(.5).mul_(2).sub_(1)
                        self.fast_r.append(torch.nn.Parameter(r))
                        self.fast_s.append(torch.nn.Parameter(s))
                if not self.sites:
                    raise ValueError("Backbone exposes no declared two-dimensional weight maps")

        def forward(self):
            if method == "original_gnnm":
                return torch.stack([self.body(graph, x, tabm_seed=m) for m in range(members)])
            if method == "joint_untied":
                return torch.stack([body(graph, x) for body in self.bodies])
            if method == "single":
                return self.body(graph, x).unsqueeze(0)
            parameters = dict(self.body.named_parameters())
            outputs = []
            for m in range(members):
                effective = dict(parameters)
                for i, name in enumerate(self.sites):
                    effective[name] = parameters[name] * self.fast_s[i][m, :, None] * self.fast_r[i][m, None, :]
                outputs.append(torch.func.functional_call(self.body, effective, (graph, x)))
            return torch.stack(outputs)

    model = NativeBank().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    selector = {"metric": run["selection"]["metric"], "direction": run["selection"]["direction"],
                "ties": run["selection"]["ties"],
                "patience_steps": run["selection"].get("patience_steps", run["budget"]["maximum"]),
                "save_every": run["selection"].get("save_every", 1)}
    return ClassifierAdapter(torch, F, model, optimizer, logits_fn=lambda model, training, step: model(),
                             train_ids=train_ids, train_y=train_y, select_ids=select_ids, select_y=select_y,
                             regularizer=lambda model: x.new_zeros(()), fusion=config["fusion"],
                             selector=selector, provenance={"run_id": run["run_id"],
                                 "data": study["data"], "source": study["source"], "config": config,
                                 "actual_model_seed": model_seed})
