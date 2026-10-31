"""Headless GINE/internal-BE graph-classification packet.

Only stdlib loads at import. The approved worker injects Torch; numerical
providers, archived source, and split-specific TRAIN/SELECT packs load only in
make_adapter. This packet never loads an OGB dataset or CAL/DEV/TEST targets.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import random
import re
import sys

try:
    from ..engine import AdapterStepError
except ImportError:
    from engine import AdapterStepError


PACK_KEYS = {"x", "edge_index", "edge_attr", "node_ptr", "edge_ptr", "ids", "y"}
ARMS = {"single", "single_contrastive", "factorized_single", "be_unit", "be_init",
        "be_unit_contrastive", "be_init_contrastive", "independent4_body",
        "independent4_body_contrastive", "factorized_independent4_body",
        "factorized_independent4_body_contrastive"}
SOURCE_NAMES = {"models", "factors", "objectives"}


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")


def _finite_number(value, name, *, minimum=0., maximum=None, strict=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    if (value <= minimum if strict else value < minimum) or (maximum is not None and value >= maximum):
        raise ValueError("Invalid " + name)


def _receipt(value, name):
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(name + " needs an exact path/bytes/SHA256 receipt")
    path = Path(value["path"])
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(name + " must use a project-relative path")
    if type(value["bytes"]) is not int or value["bytes"] < 0:
        raise ValueError("Invalid " + name + " byte count")
    if not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
        raise ValueError("Invalid " + name + " SHA256")


def validate_config(config, run):
    """Pure schema/selector admission; no module, data or numerical imports."""
    required = {"molecular_inputs", "molecular_source", "model", "optimizer", "training",
                "arm", "fusion", "initialization_seed", "dropout_seed", "shuffle_seed",
                "head_policy", "contrastive", "target_policy"}
    optional = {"maximum_cuda_bytes", "independence_group"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - optional:
        raise ValueError("Incomplete molecular config or unsupported/heldout config fields")
    if set(config["molecular_inputs"]) != {"train", "select"}:
        raise ValueError("Fit inputs must contain only separate TRAIN and SELECT packs")
    for role, receipt in config["molecular_inputs"].items():
        _receipt(receipt, role)
    if config["molecular_inputs"]["train"]["path"] == config["molecular_inputs"]["select"]["path"]:
        raise ValueError("TRAIN and SELECT must be separate authorities")
    if set(config["molecular_source"]) != SOURCE_NAMES:
        raise ValueError("Bind exact archived models, factors and objectives source")
    for name, receipt in config["molecular_source"].items():
        _receipt(receipt, "source " + name)
    model = config["model"]
    if set(model) != {"hidden", "layers", "dropout", "outputs"}:
        raise ValueError("Declare GINE width/layers/dropout/output tasks")
    for key in ("hidden", "layers", "outputs"):
        _positive_integer(model[key], key)
    _finite_number(model["dropout"], "dropout", maximum=1.)
    if config["head_policy"] not in {"archived_scalar", "declared_multitask_linear"}:
        raise ValueError("Declare the archived scalar or adapted multitask head")
    if (config["head_policy"] == "archived_scalar") != (model["outputs"] == 1):
        raise ValueError("Multitask outputs require the explicitly adapted head")
    if config["arm"] not in ARMS or config["fusion"] != "mean_logits":
        raise ValueError("Unsupported molecular arm or serving law")
    if config["target_policy"] not in {"complete_binary", "observed_binary_mask"}:
        raise ValueError("Declare complete binary or explicitly masked binary targets")
    training = config["training"]
    if set(training) != {"batch_size", "eval_batch_size", "views_per_update"}:
        raise ValueError("Declare all minibatch/view decisions")
    for key in ("batch_size", "eval_batch_size"):
        _positive_integer(training[key], key)
    if training["views_per_update"] != 2:
        raise ValueError("Archived molecular objectives require two own-loss TRAIN views")
    optimizer = config["optimizer"]
    if set(optimizer) != {"name", "learning_rate", "weight_decay", "eps", "betas"} or optimizer["name"] != "Adam":
        raise ValueError("Declare the source Adam recipe explicitly")
    _finite_number(optimizer["learning_rate"], "learning_rate", strict=True)
    _finite_number(optimizer["weight_decay"], "weight_decay")
    _finite_number(optimizer["eps"], "Adam epsilon", strict=True)
    if not isinstance(optimizer["betas"], (list, tuple)) or len(optimizer["betas"]) != 2:
        raise ValueError("Declare both Adam betas")
    for beta in optimizer["betas"]:
        _finite_number(beta, "Adam beta", maximum=1.)
    for key in ("initialization_seed", "dropout_seed", "shuffle_seed"):
        if type(config[key]) is not int or not 0 <= config[key] < 2**63:
            raise ValueError("Explicit nonnegative " + key + " required")
    if config["initialization_seed"] >= 2**32:
        raise ValueError("Archived NumPy constructor seeding requires a uint32 initialization seed")
    contrastive = config["contrastive"]
    if set(contrastive) != {"max_objects", "alignment_weight", "residual_weight", "temperature"}:
        raise ValueError("Declare the full contrastive recipe")
    _positive_integer(contrastive["max_objects"], "max_objects")
    for key in ("alignment_weight", "residual_weight"):
        _finite_number(contrastive[key], key)
    _finite_number(contrastive["temperature"], "temperature", strict=True)
    if config["arm"].endswith("_contrastive") and (model["outputs"] != 1 or config["target_policy"] != "complete_binary"):
        raise ValueError("Source contrastive labels require one complete binary target")
    if "independent4_body" in config["arm"]:
        if not 0 <= run["body"] < 4 or not isinstance(config.get("independence_group"), str) or not config["independence_group"]:
            raise ValueError("Declare one of four separately acquired bodies and its independence_group")
    elif "independence_group" in config:
        raise ValueError("An independence group belongs only to separately acquired I4 body runs")
    if run["budget"]["unit"] != "epochs" or run["budget"]["policy"] != "full_budget":
        raise ValueError("Molecular fitting requires a declared full-epoch budget")
    _positive_integer(run["budget"]["maximum"], "epoch budget")
    selection = run["selection"]
    if (selection["metric"] not in {"rocauc", "ce", "brier"}
            or selection["direction"] != ("max" if selection["metric"] == "rocauc" else "min")
            or selection["ties"] not in {"first", "last"}
            or selection.get("save_every", 1) != 1):
        raise ValueError("Declare a compatible molecular selector")
    return copy.deepcopy(config)


def validate_independent_roster(study, run):
    """I4 is four paid trajectories and four own selectors, never one bank fit."""
    config = run["config"]
    if "independent4_body" not in config["arm"]:
        return
    group = config["independence_group"]
    members = [r for r in study["runs"] if r["kind"] == "model_fit"
               and r["config"].get("independence_group") == group]
    if len(members) != 4 or {r["body"] for r in members} != {0, 1, 2, 3}:
        raise ValueError("An I4 independence group requires exactly four distinct paid body runs")
    for key in ("initialization_seed", "dropout_seed", "shuffle_seed"):
        if len({r["config"].get(key) for r in members}) != 4:
            raise ValueError("Independent bodies need four distinct " + key + " bindings")
    baseline = {k: v for k, v in config.items() if k not in {"initialization_seed", "dropout_seed", "shuffle_seed"}}
    for member in members:
        validate_config(member["config"], member)
        other = {k: v for k, v in member["config"].items() if k not in {"initialization_seed", "dropout_seed", "shuffle_seed"}}
        if (other != baseline or member["budget"] != run["budget"] or member["selection"] != run["selection"]
                or any(member.get(key) != run.get(key) for key in ("adapter", "method", "split"))):
            raise ValueError("Independent molecular cohort recipe, roles, budget or selector differs")


def validate_role_authorities(study, config):
    """The fit's pack receipts must be its prospectively declared role authority."""
    if study["data"]["task"] not in {"graph_binary_classification", "graph_multilabel_classification"}:
        raise ValueError("Molecular packet requires an explicitly binary graph task")
    for role in ("train", "select"):
        authority = study["data"]["roles"][role]
        if authority != config["molecular_inputs"][role]:
            raise ValueError("Molecular " + role + " pack differs from its declared role authority")


def bound_file(project_root, receipt):
    _receipt(receipt, "file")
    root = Path(project_root).resolve()
    path = root / receipt["path"]
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError("Missing, symlinked or unconfined molecular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    if path.stat().st_size != receipt["bytes"] or digest.hexdigest() != receipt["sha256"]:
        raise ValueError("Molecular source/input receipt changed: " + receipt["path"])
    return path


def _module(path, namespace):
    name = "gnnm_molecular_" + namespace + "_" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def load_source(project_root, receipts):
    paths = {key: bound_file(project_root, value) for key, value in receipts.items()}
    factors = _module(paths["factors"], "factors")
    previous = sys.modules.get("factors")
    try:
        sys.modules["factors"] = factors
        models = _module(paths["models"], "models_" + receipts["factors"]["sha256"][:16])
    finally:
        if previous is None:
            sys.modules.pop("factors", None)
        else:
            sys.modules["factors"] = previous
    return models, factors, _module(paths["objectives"], "objectives")


class MolecularEpochMeans:
    """Whole-epoch means with explicit graph and observed-target denominators."""
    def __init__(self):
        self.graphs = self.labels = self.updates = 0
        self.sums = {key: 0. for key in ("objective", "ce_graph_weighted", "regularization", "alignment", "residual")}
        self.ce_sum = 0.

    def add(self, metrics, *, graphs, observed_targets):
        _positive_integer(graphs, "batch graph count")
        _positive_integer(observed_targets, "observed target count")
        for key in ("objective", "ce", "regularization", "alignment", "residual"):
            _finite_number(metrics[key], key, minimum=-1e300)
        for key in self.sums:
            self.sums[key] += graphs * metrics["ce" if key == "ce_graph_weighted" else key]
        self.ce_sum += observed_targets * metrics["ce"]
        self.graphs += graphs
        self.labels += observed_targets
        self.updates += 1

    def finish(self):
        if not self.graphs:
            raise ValueError("An empty molecular epoch cannot complete")
        return {**{key: value / self.graphs for key, value in self.sums.items()},
                "ce": self.ce_sum / self.labels, "sample_count": self.graphs,
                "observed_target_count": self.labels, "ce_numerator": self.ce_sum,
                "ce_denominator": self.labels, "objective_denominator": self.graphs,
                "aggregation": "objective_and_aux_graph_weighted; CE_observed_target_weighted",
                "operation_counts": {"optimizer_updates": self.updates, "supervised_examples": self.graphs,
                                     "observed_targets": self.labels}}


def validate_pack(torch, pack, outputs, target_policy):
    if not isinstance(pack, dict) or set(pack) != PACK_KEYS or any(not torch.is_tensor(v) for v in pack.values()):
        raise ValueError("Use an exact plain-tensor graph pack with split-specific labels")
    if any(v.device.type != "cpu" or v.layout != torch.strided for v in pack.values()):
        raise ValueError("Graph packs must be ordinary CPU tensors")
    if any(v.requires_grad for v in pack.values()):
        raise ValueError("Molecular input and target packs must be immutable non-gradient data")
    x, edges, attributes = pack["x"], pack["edge_index"], pack["edge_attr"]
    if x.dtype != torch.long or x.ndim != 2 or x.shape[1] != 9:
        raise ValueError("Preserve all nine OGB atom categorical fields")
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("Use graph-local int64 edge indices")
    if attributes.dtype != torch.long or attributes.shape != (edges.shape[1], 3):
        raise ValueError("Preserve all three bond categorical fields")
    ids = pack["ids"]
    if ids.dtype != torch.long or ids.ndim != 1 or not ids.numel() or len(torch.unique(ids)) != len(ids) or bool((ids < 0).any()):
        raise ValueError("Unique nonnegative graph identities required")
    for name, end, strict in (("node_ptr", len(x), True), ("edge_ptr", edges.shape[1], False)):
        ptr = pack[name]
        if (ptr.dtype != torch.long or ptr.shape != (len(ids)+1,) or int(ptr[0]) != 0 or int(ptr[-1]) != end
                or bool(((ptr[1:] - ptr[:-1]) <= 0 if strict else (ptr[1:] - ptr[:-1]) < 0).any())):
            raise ValueError("Invalid " + name + " graph partition")
    y = pack["y"]
    if y.ndim == 1 and outputs == 1:
        y = y[:, None]
    if y.shape != (len(ids), outputs) or not y.is_floating_point() or bool(torch.isinf(y).any()):
        raise ValueError("Use float graph-by-task binary targets; missing labels are NaN only")
    observed = torch.isfinite(y)
    if target_policy == "complete_binary" and not bool(observed.all()):
        raise ValueError("This source recipe requires complete binary targets")
    if not bool(observed.any(1).all()) or bool(((y[observed] != 0) & (y[observed] != 1)).any()):
        raise ValueError("Each graph needs an observed binary target")
    for graph in range(len(ids)):
        ns, ne = map(int, pack["node_ptr"][graph:graph+2])
        es, ee = map(int, pack["edge_ptr"][graph:graph+2])
        local = edges[:, es:ee]
        if local.numel() and (int(local.min()) < 0 or int(local.max()) >= ne-ns):
            raise ValueError("Edges must stay within their original graph-local node universe")
    result = dict(pack)
    result["y"] = y.to(dtype=torch.float32)
    return result


class MolecularAdapter:
    def __init__(self, torch, model, optimizer, *, factors, losses, train, select, config,
                 selector, provenance, data_class, batch_class, auc_function, numpy):
        self.t, self.model, self.optimizer = torch, model, optimizer
        self.F = torch.nn.functional
        self.factors, self.losses = factors, losses
        self.train, self.select, self.config = train, select, config
        self.selector, self.provenance = dict(selector), copy.deepcopy(provenance)
        self.Data, self.Batch, self.auc_function, self.np = data_class, batch_class, auc_function, numpy
        self.device = next(model.parameters()).device
        self.members = 4 if config["arm"].startswith("be_") else 1
        self.cuda_devices = [] if self.device.type != "cuda" else [self.device.index]
        self.streams = []
        for member in range(self.members):
            seed = config["dropout_seed"] + 1009 * member
            stream = {"cpu": torch.Generator(device="cpu").manual_seed(seed).get_state()}
            if self.cuda_devices:
                stream["cuda"] = torch.Generator(device=self.device).manual_seed(seed).get_state()
            self.streams.append(stream)
        self._best = self._best_step = None
        self._epoch = 0

    def _batch(self, pack, positions):
        rows = []
        for index in positions.tolist():
            ns, ne = map(int, pack["node_ptr"][index:index+2])
            es, ee = map(int, pack["edge_ptr"][index:index+2])
            rows.append(self.Data(x=pack["x"][ns:ne], edge_index=pack["edge_index"][:, es:ee],
                                  edge_attr=pack["edge_attr"][es:ee]))
        return {"graph": self.Batch.from_data_list(rows).to(self.device)}, pack["y"][positions].to(self.device)

    def _forward(self, batch, counts=None):
        rows = []
        for member, stream in enumerate(self.streams):
            with self.t.random.fork_rng(devices=self.cuda_devices):
                self.t.set_rng_state(stream["cpu"])
                if self.cuda_devices:
                    self.t.cuda.set_rng_state(stream["cuda"], self.device)
                old = self.model.member
                self.model.member = member
                try:
                    with self.factors.member_context(self.model, member):
                        if counts is not None:
                            counts["forward_attempts"] += 1
                        rows.append(self.model(batch))
                        if counts is not None:
                            counts["model_forwards"] += 1
                            counts["route_predictions"] += int(rows[-1][0].shape[0])
                finally:
                    self.model.member = old
                stream["cpu"] = self.t.get_rng_state().clone()
                if self.cuda_devices:
                    stream["cuda"] = self.t.cuda.get_rng_state(self.device).clone()
        return self.t.stack([row[0] for row in rows]), self.t.stack([row[1] for row in rows])

    def _number(self, value):
        result = float(value.detach().item())
        if not math.isfinite(result):
            raise FloatingPointError("Nonfinite molecular loss")
        return result

    def _own(self, logits, labels):
        if self.config["model"]["outputs"] == 1 and self.config["target_policy"] == "complete_binary":
            return self.losses.own_supervision(logits, labels[:, 0], "molhiv")
        mask = self.t.isfinite(labels)
        safe = labels.masked_fill(~mask, 0)
        values = self.F.binary_cross_entropy_with_logits(logits, safe[None].expand_as(logits), reduction="none")
        return values[:, mask].mean(1)

    def _finite_state(self):
        def check(value):
            if self.t.is_tensor(value):
                if (value.is_floating_point() or value.is_complex()) and not bool(self.t.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite molecular parameter/optimizer state")
            elif isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check(item)
        check(self.model.state_dict())
        check(self.optimizer.state_dict())

    def train_step(self, epoch):
        self.model.train()
        accumulator = MolecularEpochMeans()
        counts = {"optimizer_updates": 0, "optimizer_update_attempts": 0, "model_forwards": 0,
                  "forward_attempts": 0, "backward_calls": 0, "backward_attempts": 0,
                  "supervised_examples": 0, "observed_targets": 0, "route_predictions": 0,
                  "training_label_opportunities": 0}
        try:
            generator = self.t.Generator(device="cpu").manual_seed(self.config["shuffle_seed"] + 19709 + 1000*epoch)
            order = self.t.randperm(len(self.train["ids"]), generator=generator)
            for positions in order.split(self.config["training"]["batch_size"]):
                batch, labels = self._batch(self.train, positions)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = []
                for _ in range(2):
                    outputs.append(self._forward(batch, counts))
                (la, ha), (lb, hb) = outputs
                if any(not bool(self.t.isfinite(value).all()) for value in (la, lb, ha, hb)):
                    raise FloatingPointError("Nonfinite TRAIN prediction/representation")
                own = .5 * (self._own(la, labels) + self._own(lb, labels))
                graphs, observed = len(positions), int(self.t.isfinite(labels).sum().item())
                counts["supervised_examples"] += graphs
                counts["observed_targets"] += observed
                counts["training_label_opportunities"] += 2 * self.members * observed
                alignment = residual = own.sum() * 0
                if self.config["arm"].endswith("_contrastive"):
                    c = self.config["contrastive"]
                    index = self.t.linspace(0, len(labels)-1, steps=min(len(labels), c["max_objects"]), device=self.device).long()
                    target = labels[index, 0]
                    alignment = self.losses.alignment_loss(ha[:, index], hb[:, index], target, "molhiv", temperature=c["temperature"])
                    residual = self.losses.residual_member_contrast(ha[:, index], hb[:, index], target, temperature=c["temperature"])
                c = self.config["contrastive"]
                auxiliary = c["alignment_weight"] * alignment + c["residual_weight"] * residual
                objective = own.mean() + auxiliary
                metrics = {"objective": self._number(objective), "ce": self._number(own.mean()),
                           "regularization": self._number(auxiliary), "alignment": self._number(alignment),
                           "residual": self._number(residual)}
                counts["backward_attempts"] += 1
                objective.backward()
                counts["backward_calls"] += 1
                active = [p for p in self.model.parameters() if p.grad is not None]
                if not active or any(not bool(self.t.isfinite(p.grad).all()) for p in active):
                    raise FloatingPointError("Nonfinite or empty molecular gradient")
                counts["optimizer_update_attempts"] += 1
                self.optimizer.step()
                counts["optimizer_updates"] += 1
                self._finite_state()
                accumulator.add(metrics, graphs=graphs, observed_targets=observed)
            result = accumulator.finish()
            if result["sample_count"] != len(self.train["ids"]):
                raise ValueError("Incomplete molecular TRAIN epoch")
            result["operation_counts"] = counts
            result["loss_timing"] = "preupdate; dropout_on; two native own-loss views"
            self._epoch = epoch
            return result
        except BaseException as error:
            raise AdapterStepError(str(error), operation_counts=counts) from error

    def predict_pack(self, pack):
        """Predict an already admitted pack; this method opens no role files."""
        self.model.eval()
        rows = []
        with self.t.no_grad():
            for positions in self.t.arange(len(pack["ids"])).split(self.config["training"]["eval_batch_size"]):
                batch, _ = self._batch(pack, positions)
                logits, _ = self._forward(batch)
                if not bool(self.t.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite molecular serving logits")
                rows.append(logits.detach().cpu())
        return self.t.cat(rows, dim=1)

    def validate(self, epoch):
        logits = self.predict_pack(self.select)
        y = self.select["y"]
        pooled = logits.mean(0)
        mask = self.t.isfinite(y)
        ce = self._number(self._own(pooled[None], y).mean())
        probabilities = pooled.sigmoid()
        brier = self._number((probabilities[mask]-y[mask]).square().mean())
        aucs, eligible = [], []
        for task in range(y.shape[1]):
            valid = mask[:, task]
            values = y[valid, task]
            # This is the OGB binary macro ROC-AUC task eligibility rule.
            if bool((values == 0).any()) and bool((values == 1).any()):
                aucs.append(float(self.auc_function(values.numpy(), pooled[valid, task].numpy())))
                eligible.append(task)
        if not aucs and self.selector["metric"] == "rocauc":
            raise ValueError("No SELECT task has both binary classes for ROC-AUC")
        result = {"objective": ce, "ce": ce, "regularization": 0., "brier": brier,
                  "sample_count": len(y), "observed_target_count": int(mask.sum().item()),
                  "rocauc_eligible_tasks": eligible, "fusion": "mean_raw_logits_then_sigmoid",
                  "operation_counts": {"optimizer_updates": 0,
                      "validation_graphs": len(y), "validation_route_predictions": self.members*len(y)}}
        if aucs:
            result["rocauc"] = sum(aucs)/len(aucs)
        result["selection_metric"] = result[self.selector["metric"]]
        return result

    def observe_validation(self, epoch, metrics):
        value = float(metrics["selection_metric"])
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite molecular selector")
        if self._best is None:
            improve = True
        elif self.selector["direction"] == "max":
            improve = value > self._best if self.selector["ties"] == "first" else value >= self._best
        else:
            improve = value < self._best if self.selector["ties"] == "first" else value <= self._best
        if improve:
            self._best, self._best_step = value, epoch
        return {"save_selected": improve, "stop_requested": False,
                "best_saved_metric": self._best, "best_saved_step": self._best_step,
                "full_budget": True, "selection_scope": "this paid molecular trajectory only"}

    def capture_state(self):
        name, keys, position, has_gauss, cached = self.np.random.get_state()
        return {"provenance": copy.deepcopy(self.provenance), "epoch": self._epoch,
                "selector": {"best": self._best, "best_step": self._best_step},
                "streams": [{k: v.clone() for k, v in stream.items()} for stream in self.streams],
                "torch_cpu": self.t.get_rng_state().clone(), "python": random.getstate(),
                "torch_cuda": [s.clone() for s in self.t.cuda.get_rng_state_all()] if self.t.cuda.is_initialized() else [],
                "numpy": (name, keys.tolist(), position, has_gauss, cached)}

    def restore_state(self, state):
        if state["provenance"] != self.provenance or len(state["streams"]) != self.members:
            raise ValueError("Molecular checkpoint source/input/recipe ancestry differs")
        self._epoch = state["epoch"]
        self._best, self._best_step = state["selector"]["best"], state["selector"]["best_step"]
        self.streams = [{k: v.clone() for k, v in stream.items()} for stream in state["streams"]]
        self.t.set_rng_state(state["torch_cpu"])
        random.setstate(state["python"])
        if state["torch_cuda"]:
            self.t.cuda.set_rng_state_all(state["torch_cuda"])
        name, keys, position, has_gauss, cached = state["numpy"]
        self.np.random.set_state((name, self.np.asarray(keys, dtype="uint32"), position, has_gauss, cached))


def make_adapter(*, torch, project_root, study, run, device):
    config = validate_config(run["config"], run)
    validate_independent_roster(study, run)
    validate_role_authorities(study, config)
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from torch_geometric.data import Batch, Data
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Archived molecular source requires explicit float32 arithmetic")
    packs = {}
    for role, receipt in config["molecular_inputs"].items():
        path = bound_file(project_root, receipt)
        pack = torch.load(path, map_location="cpu", weights_only=True)
        packs[role] = validate_pack(torch, pack, config["model"]["outputs"], config["target_policy"])
        for key, limits in (("x", get_atom_feature_dims()), ("edge_attr", get_bond_feature_dims())):
            for column, limit in enumerate(limits):
                values = pack[key][:, column]
                if bool(((values < 0) | (values >= limit)).any()):
                    raise ValueError("Categorical molecular field outside its OGB vocabulary")
    if bool(torch.isin(packs["train"]["ids"], packs["select"]["ids"]).any()):
        raise ValueError("TRAIN and SELECT graph identities overlap")
    models, factors, losses = load_source(project_root, config["molecular_source"])
    options = {key: config["model"][key] for key in ("hidden", "layers", "dropout")}
    with models.seeded(config["initialization_seed"]):
        model = models.MoleculeBackbone(options)
        if config["head_policy"] == "declared_multitask_linear":
            # A declared output-width adaptation of the exact source backbone.
            model.head = torch.nn.Linear(options["hidden"], config["model"]["outputs"])
        if config["arm"].startswith("be_") or config["arm"].startswith("factorized_"):
            members = 4 if config["arm"].startswith("be_") else 1
            factors.install_factors(model, members)
            model.enable_message_factors(members)
            if config["arm"].startswith("be_init"):
                factors.initialize_first_factor(model.stem, config["initialization_seed"] + 900001)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"], eps=config["optimizer"]["eps"],
        betas=tuple(config["optimizer"]["betas"]))
    provenance = {"run_id": run["run_id"], "body": run["body"], "dataset": study["dataset"],
                  "data": study["data"], "source": study["source"], "config": config,
                  "budget": copy.deepcopy(run["budget"]), "selection": copy.deepcopy(run["selection"]),
                  "implementation": "archived MoleculeBackbone + archived factors/objectives; v2 full-epoch controller",
                  "author_protocol_replication": False,
                  "head_adaptation": config["head_policy"], "independent_selector": "one paid body per run"}
    return MolecularAdapter(torch, model, optimizer, factors=factors, losses=losses,
        train=packs["train"], select=packs["select"], config=config, selector=run["selection"],
        provenance=provenance, data_class=Data, batch_class=Batch, auc_function=roc_auc_score, numpy=np)
