"""Source boundary GNNM adapted to the pinned PolyFormer mono backbone.

Only lin1/lin3 are replaced. Plain attention/FFN/lin2 remain shared. Private
constructor/dropout streams, grouped LRs, roles/horizon/selector are declared
standardized adaptations rather than unchanged published-task replication.
"""
from contextlib import contextmanager
import copy
import hashlib
import math
from types import SimpleNamespace

from . import polyformer as p

BOUNDARY_SOURCE = {"path": "common_wrapper_native_source_20261010_v1/models.py", "bytes": 10977,
                   "sha256": "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303"}
TRAIN_SOURCE = {"path": "source_snapshot/run_tabm.py", "bytes": 11240,
                "sha256": "b1d748e50c854802b93f4d4fa7485f096daa1cd36d3dd3bd1c89fbfa63cbd124"}
M4 = "polyformer_original_boundary_own"
M1 = "polyformer_boundary_equipped_M1_own"
METHODS = {M4: 4, M1: 1}


def source_class(torch, project_root):
    path = p._path(project_root, BOUNDARY_SOURCE["path"])
    if p._file_receipt(path) != {k: BOUNDARY_SOURCE[k] for k in ("bytes", "sha256")}:
        raise ValueError("Pinned original BEBlock source changed")
    return p._definitions(path, ("BEBlock",), {"torch": torch, "nn": torch.nn})["BEBlock"]


def validate_config(config, run):
    mode = config.get("polyformer_mode")
    if mode not in METHODS or run["method"] != mode or config.get("members") != METHODS[mode]:
        raise ValueError("Declare the distinct original boundary M4 or matched boundary M1")
    if (config.get("boundary_source") != BOUNDARY_SOURCE or config.get("fusion") != "mean_logits"
            or config.get("train_views") != 1 or config.get("loss") != {"variant": "own", "coefficients": {}}
            or config.get("factor_initialization") != "source_rsb_xavier" or config.get("factor_seed") is not None
            or config.get("is_original_gnnm") != (mode == M4)):
        raise ValueError("Bind exact source boundary law, one CE/member and mean raw logits")
    if (config.get("optimizer") != {"name": "AdamW", "betas": [.9, .999], "eps": 1e-8,
                                     "grouping": "attnmodule_vs_other", "weight_decay": 0}
            or config.get("weight_decay") != 0 or config.get("attn_weight_decay") != 0):
        raise ValueError("One grouped AdamW with every decay locked zero required")
    for key in ("learning_rate", "attn_lr"):
        if type(config.get(key)) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Positive explicit joint learning-rate pair required")
    seeds = config.get("dropout_seeds")
    if (type(config.get("initialization_seed")) is not int or not isinstance(seeds, list)
            or len(seeds) != config["members"] or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Private constructor and one distinct dropout stream per route required")
    if (run["budget"].get("unit") != "optimizer_updates" or run["budget"].get("policy") != "full_budget"
            or type(run["budget"].get("maximum")) is not int or run["budget"]["maximum"] < 1):
        raise ValueError("Explicit standardized full-update horizon required")
    if (run["selection"].get("metric") != "accuracy" or run["selection"].get("direction") != "max"
            or run["selection"].get("ties") != "first" or run["selection"].get("save_every", 1) != 1):
        raise ValueError("Own SELECT first accuracy maximum required")
    return copy.deepcopy(config)


def build_bank(torch, project_root, tokens, *, native, classes, dataset, members,
               initialization_seed, dropout_seeds):
    if torch.get_default_dtype() != torch.float32 or members not in (1, 4):
        raise ValueError("Native float32 source boundary M1/M4 required")
    if (not tokens or len(tokens) != native["K"] + 1 or any(t.ndim != 2 or t.shape != tokens[0].shape
            or t.device != tokens[0].device or t.dtype != torch.float32 or t.requires_grad
            or not bool(torch.isfinite(t).all()) for t in tokens)):
        raise ValueError("Exact aligned finite frozen mono tokens required")
    if type(initialization_seed) is not int or len(dropout_seeds) != members:
        raise ValueError("Explicit boundary constructor/dropout roster required")
    device = tokens[0].device
    args = p._arguments(native, tokens[0].shape[1], classes, dataset)
    constructor, BEBlock = p.native_class(torch, project_root), source_class(torch, project_root)

    class Boundary(torch.nn.Module):
        def __init__(self, inputs, outputs, first):
            super().__init__()
            self.block = BEBlock(inputs, outputs, first, "cpu", members)
            self.member = 0
        def forward(self, x):
            return self.block(x, self.member)

    with p._constructor_seed(torch, initialization_seed, device):
        body = constructor(None, args)
        prior = dict(body.named_parameters())
        body.lin1 = Boundary(args.num_features, args.hidden, True)
        body.lin3 = Boundary(args.hidden, args.num_classes, False)
    retained = {name: parameter for name, parameter in prior.items() if not name.startswith(("lin1.", "lin3."))}
    current = dict(body.named_parameters())
    if any(current[name] is not parameter for name, parameter in retained.items()):
        raise ValueError("Every ordinary interior parameter object must be retained")
    removed = [parameter for name, parameter in prior.items() if name.startswith(("lin1.", "lin3."))]
    if any(parameter is old for parameter in current.values() for old in removed):
        raise ValueError("Discarded native boundary tensors must leave the learned body")
    body.to(device)

    class Bank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body, self.members, self.last_hidden = body, members, None
            self.boundary_sites = ("lin1", "lin3")
            self.factor_sites = 0
            self.interior_parameter_names = tuple(retained)
            for index, token in enumerate(tokens):
                self.register_buffer("token_" + str(index), token, persistent=False)
            self.streams = [{"cpu": torch.Generator(device="cpu").manual_seed(seed).get_state(),
                             "cuda": torch.Generator(device=device).manual_seed(seed).get_state()
                             if device.type == "cuda" else None} for seed in dropout_seeds]

        @contextmanager
        def stream(self, slot):
            state = self.streams[slot]
            devices = [] if device.type == "cpu" else [device.index]
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(state["cpu"])
                if devices:
                    torch.cuda.set_rng_state(state["cuda"], device.index)
                try:
                    yield
                finally:
                    state["cpu"] = torch.get_rng_state().clone()
                    if devices:
                        state["cuda"] = torch.cuda.get_rng_state(device.index).clone()

        def forward_route(self, route, *, slot=None):
            if not 0 <= route < members or slot not in (None, route):
                raise ValueError("One private boundary route stream per own view required")
            captured = []
            hook = self.body.lin3.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
            old = (self.body.lin1.member, self.body.lin3.member)
            self.body.lin1.member = self.body.lin3.member = route
            data = SimpleNamespace(list_mat=[getattr(self, "token_" + str(i)) for i in range(len(tokens))])
            try:
                if self.training:
                    with self.stream(route):
                        logits = self.body(data)
                else:
                    logits = self.body(data)
                if (len(captured) != 1 or logits.shape != (len(tokens[0]), classes)
                        or captured[0].shape != (len(tokens[0]), native["hidden"])):
                    raise ValueError("Native raw logits and live preclassifier representation required")
                if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(captured[0]).all()):
                    raise FloatingPointError("Nonfinite source boundary PolyFormer forward")
                return logits, captured[0]
            finally:
                self.body.lin1.member, self.body.lin3.member = old
                hook.remove()

        def forward(self):
            pairs = [self.forward_route(member) for member in range(members)]
            self.last_hidden = torch.stack([hidden for _, hidden in pairs])
            return torch.stack([logits for logits, _ in pairs])

        def capture_runtime_state(self):
            return {"streams": [{k: v.clone() if v is not None else None for k, v in s.items()} for s in self.streams],
                    "boundary_members": {name: getattr(self.body, name).member for name in self.boundary_sites},
                    "poly_attn_bias": {name: module.bias.detach().cpu().clone() for name, module in self.body.named_modules()
                                       if type(module).__name__ == "PolyAttn"}}

        def restore_runtime_state(self, state):
            if len(state["streams"]) != members or set(state["boundary_members"]) != set(self.boundary_sites):
                raise ValueError("Boundary runtime roster changed")
            self.streams = [{k: v.clone() if v is not None else None for k, v in s.items()} for s in state["streams"]]
            for name, member in state["boundary_members"].items():
                getattr(self.body, name).member = member
            modules = dict(self.body.named_modules())
            for name, bias in state["poly_attn_bias"].items():
                modules[name].bias = bias.clone()

    return Bank().to(device)


def grouped_optimizer(torch, model, config):
    groups = [{"params": [parameter], "lr": config["attn_lr"] if "attnmodule" in name else config["learning_rate"],
               "weight_decay": 0} for name, parameter in model.named_parameters()]
    return torch.optim.AdamW(groups, eps=1e-8, betas=(.9, .999), weight_decay=0)


def own_training(adapter, step):
    model, t = adapter.model, adapter.t
    adapter.optimizer.zero_grad(set_to_none=True)
    values = []
    for route in range(model.members):
        logits, _ = model.forward_route(route)
        ce = adapter.F.cross_entropy(logits[adapter.train_ids], adapter.train_y)
        (ce / model.members).backward()
        values.append(adapter._scalar(ce))
    if any(parameter.grad is not None and not bool(t.isfinite(parameter.grad).all()) for parameter in model.parameters()):
        raise FloatingPointError("Nonfinite source boundary old-state CE gradient")
    adapter.optimizer.step()
    mean = sum(values) / len(values)
    return {"objective": mean, "ce": mean, "regularization": 0., "own_view_ce": values,
            "operation_counts": {"optimizer_updates": 1, "model_forwards": model.members,
                                 "route_predictions": model.members, "backward_calls": model.members,
                                 "supervised_examples": len(adapter.train_ids) * model.members}}


def make_adapter(*, torch, project_root, study, run, device):
    config = validate_config(run["config"], run)
    inputs = config["node_inputs"]
    if set(inputs) != {"graph", "train", "select"}:
        raise ValueError("Boundary learner opens only target-free graph, TRAIN and SELECT")
    packs = {key: torch.load(p._path(project_root, path), map_location="cpu", weights_only=True) for key, path in inputs.items()}
    graph, train, select = packs["graph"], packs["train"], packs["select"]
    if set(graph) != {"x", "edge_index"} or any(set(role) != {"ids", "y"} for role in (train, select)):
        raise ValueError("Separate categorical TRAIN/SELECT authority required")
    for role in (train, select):
        if (role["ids"].ndim != 1 or role["ids"].dtype != torch.long or role["y"].dtype != torch.long
                or role["y"].shape != role["ids"].shape or not len(role["ids"])
                or len(torch.unique(role["ids"])) != len(role["ids"]) or int(role["ids"].min()) < 0
                or int(role["ids"].max()) >= len(graph["x"]) or int(role["y"].min()) < 0
                or int(role["y"].max()) >= config["classes"]):
            raise ValueError("Exact nonempty ordered categorical role authority required")
    if bool(torch.isin(train["ids"], select["ids"]).any()):
        raise ValueError("TRAIN and SELECT overlap")
    tokens = [token.to(device) for token in p.mono_tokens(torch, project_root, graph["x"], graph["edge_index"],
              K=config["native"]["K"], preprocessing=config["preprocessing"])]
    model = build_bank(torch, project_root, tokens, native=config["native"], classes=config["classes"],
                       dataset=study["dataset"], members=config["members"], initialization_seed=config["initialization_seed"],
                       dropout_seeds=config["dropout_seeds"])
    optimizer = grouped_optimizer(torch, model, config)
    selector = {"metric": "accuracy", "direction": "max", "ties": "first", "save_every": 1,
                "patience_steps": run["selection"].get("patience_steps", run["budget"]["maximum"])}
    return p.ClassifierAdapter(torch, torch.nn.functional, model, optimizer, logits_fn=lambda model, training, step: model(),
        train_ids=train["ids"].to(device), train_y=train["y"].to(device), select_ids=select["ids"].to(device),
        select_y=select["y"].to(device), regularizer=lambda model: tokens[0].new_zeros(()), fusion="mean_logits", selector=selector,
        provenance={"run_id": run["run_id"], "data": study["data"], "config": config, "source": study["source"],
            "semantic_method_id": run["method"], "polyformer_mode": config["polyformer_mode"],
            "actual_model_seed": config["initialization_seed"], "private_dropout_seeds": config["dropout_seeds"],
            "factor_initialization": "source_rsb_xavier", "factor_seed": None, "boundary_source": BOUNDARY_SOURCE,
            "native_source_files": p.SOURCE_FILES, "boundary_sites": ["lin1", "lin3"],
            "constructor_order": "native_PolyFormer_then_lin1_BEBlock_then_lin3_BEBlock_inside_one_private_CPU_constructor_stream",
            "optimizer_law": "one_AdamW_zero_decay_per_named_parameter_attention_or_ordinary_LR_group",
            "input_receipts": {key: p._file_receipt(p._path(project_root, path)) for key, path in inputs.items()},
            "original_member_law": "CE/M_separate_backwards_old_state_one_update_mean_raw_logits",
            "published_author_protocol_replication": False, "same_backbone_source_based_adaptation": True,
            "declared_changes": ["PolyFormer_mono_backbone", "joint_LR_groups", "private_constructor_and_dropout_streams",
                                 "four_roles_full2000_horizon_first_SELECT_accuracy_selector"]},
        external_capture=model.capture_runtime_state, external_restore=model.restore_runtime_state, custom_train=own_training)
