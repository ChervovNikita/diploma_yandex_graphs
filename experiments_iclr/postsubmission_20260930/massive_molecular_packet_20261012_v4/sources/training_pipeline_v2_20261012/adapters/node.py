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
    method = config.get("architecture", run["method"])
    supported = {"single", "original_gnnm", "internal_be", "factorized_single", "joint_untied"}
    if method not in supported:
        raise ValueError("This adapter does not implement: " + method)
    if config["classes"] < 2 or config.get("target_law", "multiclass_ce") != "multiclass_ce":
        raise ValueError("This packet supports multiclass CE; native binary BCE needs its own adapter")
    objective_variant = config.get("objective_variant", "own")
    needs_hidden = objective_variant not in {"own", "gncl"}
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
                        initialization = config.get("fast_input_initialization", "rademacher")
                        if initialization == "rademacher":
                            r.bernoulli_(.5).mul_(2).sub_(1)
                        elif initialization == "ones":
                            r.fill_(1)
                        elif initialization == "gaussian":
                            deviation = config["fast_input_std"]
                            if not isinstance(deviation, (int, float)) or deviation <= 0:
                                raise ValueError("Positive fast-factor initialization deviation required")
                            r.normal_(1, deviation)
                        else:
                            raise ValueError("Unknown input fast-factor initialization")
                        self.fast_r.append(torch.nn.Parameter(r))
                        self.fast_s.append(torch.nn.Parameter(s))
                if not self.sites:
                    raise ValueError("Backbone exposes no declared two-dimensional weight maps")
            self._hidden_parts = []
            self.last_hidden = None
            if needs_hidden:
                bodies = list(self.bodies) if method == "joint_untied" else [self.body]
                for body in bodies:
                    head = body.output_be_block if method == "original_gnnm" else body.output_linear
                    head.register_forward_pre_hook(self._capture_hidden)

        def _capture_hidden(self, module, arguments):
            # No detach: auxiliary losses must train the actual upstream states.
            self._hidden_parts.append(arguments[0])

        def forward(self):
            self._hidden_parts = []
            if method == "original_gnnm":
                result = torch.stack([self.body(graph, x, tabm_seed=m) for m in range(members)])
            elif method == "joint_untied":
                result = torch.stack([body(graph, x) for body in self.bodies])
            elif method == "single":
                result = self.body(graph, x).unsqueeze(0)
            else:
                parameters = dict(self.body.named_parameters())
                outputs = []
                for m in range(members):
                    effective = dict(parameters)
                    for i, name in enumerate(self.sites):
                        effective[name] = parameters[name] * self.fast_s[i][m, :, None] * self.fast_r[i][m, None, :]
                    outputs.append(torch.func.functional_call(self.body, effective, (graph, x)))
                result = torch.stack(outputs)
            self.last_hidden = torch.stack(self._hidden_parts) if needs_hidden else None
            return result

    model = NativeBank().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    selector = {"metric": run["selection"]["metric"], "direction": run["selection"]["direction"],
                "ties": run["selection"]["ties"],
                "patience_steps": run["selection"].get("patience_steps", run["budget"]["maximum"]),
                "save_every": run["selection"].get("save_every", 1)}
    def objective_callback(adapter, step, logits, hidden):
        try:
            from .node_losses import node_objective
        except ImportError:
            from adapters.node_losses import node_objective
        first_logits = logits[:, train_ids]
        first_hidden = hidden[:, train_ids] if hidden is not None else None
        second_logits, second_hidden = None, None
        paired = objective_variant.startswith("two_view_")
        if paired:
            second_logits = adapter.logits_fn(adapter.model, True, step)[:, train_ids]
            second_hidden = adapter.model.last_hidden[:, train_ids]
        # A fixed TRAIN-only panel is shared between paired views; it never
        # changes the complete supervised TRAIN CE population.
        panel = config.get("auxiliary_max_rows")
        auxiliary_labels = train_y
        if panel and first_hidden is not None and first_hidden.shape[1] > panel:
            first_hidden = first_hidden[:, :panel]
            second_hidden = second_hidden[:, :panel] if second_hidden is not None else None
            auxiliary_labels = train_y[:panel]
        result = node_objective(torch, F, first_logits, first_hidden, train_y,
                                variant=objective_variant,
                                coefficients=config.get("objective_coefficients", {}),
                                paired_logits=second_logits, paired_hidden=second_hidden,
                                auxiliary_labels=auxiliary_labels, independent=False)
        result["operation_counts"] = {"model_forwards": members * (2 if paired else 1),
                                      "route_predictions": members * (2 if paired else 1),
                                      "supervised_examples": int(train_ids.numel()) * (2 if paired else 1)}
        return result

    return ClassifierAdapter(torch, F, model, optimizer, logits_fn=lambda model, training, step: model(),
                             train_ids=train_ids, train_y=train_y, select_ids=select_ids, select_y=select_y,
                             regularizer=lambda model: x.new_zeros(()), fusion=config["fusion"],
                             selector=selector, provenance={"run_id": run["run_id"],
                                 "data": study["data"], "source": study["source"], "config": config,
                                 "actual_model_seed": model_seed},
                             objective_callback=objective_callback,
                             objective_provenance={"variant": objective_variant,
                                 "coefficients": config.get("objective_coefficients", {}),
                                 "port_kind": "standardized_native_node_port",
                                 "exact_archival_recipe_replication": False})
