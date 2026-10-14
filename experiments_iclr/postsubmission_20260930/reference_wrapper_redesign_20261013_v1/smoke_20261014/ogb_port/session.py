"""Controller-facing real TRAIN/VALID OGB smoke callback.

Importing this file loads only stdlib. create_session performs genuine native
construction/data loading when invoked by the authorized smoke controller.
"""
import copy
import csv
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys


PROCEDURES = ("native_single", "gnnm_boundary_native", "boundary_global_half_pool",
              "boundary_private_half_pool", "boundary_private_fixed_responsibility")
ARCHITECTURES = ("gcn", "gcn-virtual", "gin", "gin-virtual")
DATASETS = ("ogbg-molhiv", "ogbg-molbace")
_DATASET_CACHE = {}


def _sibling(name):
    path = Path(__file__).resolve().parent / (name + ".py")
    key = "_ogb_smoke_port_" + name + "_" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _configuration(recipe):
    config = recipe.get("native_fixed_config", {})
    dataset = recipe.get("dataset", config.get("dataset"))
    architecture = config.get("gnn", recipe.get("architecture", config.get("architecture")))
    architecture = str(architecture).lower()
    drop = float(config.get("drop_ratio", config.get("dropout", 0.5)))
    if dataset not in DATASETS or architecture not in ARCHITECTURES or drop not in (0.0, 0.5):
        raise ValueError("Only the fixed MolHIV/MolBACE four native architectures/dropout0/.5 presets are supported")
    for keys, default in ((("num_layer", "num_layers"), 5), (("emb_dim",), 300), (("batch_size",), 32)):
        if any(int(config[key]) != default for key in keys if key in config):
            raise ValueError("Native OGB setting must remain " + keys[0] + "=" + str(default))
    if config.get("feature", "full") != "full" or any(
            float(config[key]) != 0.001 for key in ("lr", "learning_rate") if key in config):
        raise ValueError("Full categorical features and native Adam LR .001 are required")
    if config.get("residual", False) or config.get("JK", "last") != "last" or config.get("graph_pooling", "mean") != "mean":
        raise ValueError("Native residual=False, JK=last, graph_pooling=mean must remain unchanged")
    return {"dataset": dataset, "gnn": architecture, "drop_ratio": drop, "num_layer": 5,
            "emb_dim": 300, "batch_size": 32, "feature": "full", "lr": 0.001,
            "residual": False, "JK": "last", "graph_pooling": "mean"}


def _dataset_parent(data_root, dataset_name):
    root, dirname = Path(data_root).resolve(), dataset_name.replace("-", "_")
    if (root / dirname / "processed/geometric_data_processed.pt").is_file():
        return root
    if root.name == dirname and (root / "processed/geometric_data_processed.pt").is_file():
        return root.parent
    raise FileNotFoundError("Real processed OGB store required; automatic download is disabled. Expected "
                            + str(root / dirname / "processed/geometric_data_processed.pt"))


def _load_dataset(recipe, data_root, dataset_name):
    if "_dataset" in recipe:
        dataset = recipe["_dataset"]
    else:
        from ogb.graphproppred import PygGraphPropPredDataset
        root = _dataset_parent(data_root, dataset_name)
        key = (str(root), dataset_name)
        if key not in _DATASET_CACHE:
            _DATASET_CACHE[key] = PygGraphPropPredDataset(name=dataset_name, root=str(root))
        dataset = _DATASET_CACHE[key]
    if dataset.name != dataset_name or dataset.num_tasks != 1 or "classification" not in dataset.task_type:
        raise ValueError("Exact native single-task molecular classification provider required")
    if getattr(dataset, "transform", None) is not None:
        raise ValueError("The native full-feature preset has no additional runtime transform")
    return dataset


def _train_valid_indices(dataset, torch):
    """Read only original scaffold TRAIN/VALID role files; no TEST dependency."""
    folder = Path(dataset.root) / "split/scaffold"
    roles, pins = {}, {}
    for role in ("train", "valid"):
        pt = folder / (role + ".pt")
        if pt.is_file():
            value = torch.load(pt, map_location="cpu", weights_only=True)
            if isinstance(value, dict) and role in value:
                value = value[role]
            if not isinstance(value, torch.Tensor):
                raise TypeError("Source scaffold role file must contain its native index tensor")
            path, indices = pt, value.detach().cpu().reshape(-1)
        else:
            path = folder / (role + ".csv.gz")
            if not path.is_file():
                raise FileNotFoundError("Native scaffold " + role + " indices required: " + str(pt))
            with gzip.open(path, "rt") as handle:
                indices = torch.tensor([int(row[0]) for row in csv.reader(handle)], dtype=torch.long)
        if indices.dtype != torch.long or not len(indices) or len(torch.unique(indices)) != len(indices) \
                or int(indices.min()) < 0 or int(indices.max()) >= len(dataset):
            raise ValueError("Complete original scaffold indices required for " + role)
        roles[role] = indices.clone()
        pins[role] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "count": len(indices)}
    if torch.isin(roles["train"], roles["valid"]).any().item():
        raise ValueError("TRAIN and VALID scaffold roles overlap")
    return roles, pins


class _RoleView:
    """Select only authenticated TRAIN or VALID indices, never TEST."""
    def __init__(self, dataset, ids, torch, role):
        self.dataset, self.ids, self.torch, self.role = dataset, ids.detach().cpu().clone(), torch, role
        if role not in ("TRAIN", "VALID"):
            raise ValueError("No TEST view exists in this smoke port")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, position):
        graph_id = int(self.ids[int(position)])
        # Extra ID metadata is unused by the native forward; inputs/labels remain native.
        data = self.dataset[graph_id].clone()
        data.smoke_graph_id = self.torch.tensor([graph_id], dtype=self.torch.long)
        return data


class _TrainBatcher:
    """Native RandomSampler law and PyG Batch collation, with replayable cursor."""
    def __init__(self, torch, view, *, seed):
        from torch_geometric.data import Batch
        self.torch, self.view, self.Batch = torch, view, Batch
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.order, self.cursor, self.epoch = None, 0, 0

    def next(self):
        if self.order is None or self.cursor == len(self.order):
            sampler = self.torch.utils.data.RandomSampler(self.view, replacement=False, generator=self.generator)
            self.order = self.torch.tensor(list(iter(sampler)), dtype=self.torch.long)
            self.cursor, self.epoch = 0, self.epoch + 1
        positions = self.order[self.cursor:self.cursor + 32]
        self.cursor += len(positions)
        return self.Batch.from_data_list([self.view[int(i)] for i in positions]), positions.clone()

    def state(self):
        return {"order": None if self.order is None else self.order.clone(), "cursor": self.cursor,
                "epoch": self.epoch, "generator": self.generator.get_state().clone()}

    def load_state(self, state):
        self.order = None if state["order"] is None else state["order"].detach().cpu().clone()
        self.cursor, self.epoch = int(state["cursor"]), int(state["epoch"])
        self.generator.set_state(state["generator"].detach().cpu())
        if self.order is not None and (len(self.order) != len(self.view) or not 0 <= self.cursor <= len(self.order)):
            raise ValueError("Checkpoint sampler does not match the native TRAIN view")


class OGBSmokeSession:
    def __init__(self, torch, recipe, procedure, data_root, device):
        if procedure not in PROCEDURES:
            raise ValueError("Molecular SupCon and unlisted procedures are not supported")
        self.torch, self.recipe, self.procedure = torch, recipe, procedure
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("The original OGB construction requires the native FP32 default")
        self.config, self.device = _configuration(recipe), torch.device(device)
        self.redesign_root = Path(recipe.get("_redesign_root", recipe.get("redesign_root", Path(__file__).resolve().parents[2]))).resolve()
        loader = _sibling("native_loader")
        GNN, self.source_pins = loader.load_native_class(self.redesign_root)
        self.rules, self.rules_pin = loader.load_training_rules(self.redesign_root)
        dataset = _load_dataset(recipe, data_root, self.config["dataset"])
        split, self.split_pins = _train_valid_indices(dataset, torch)
        self.train_ids, self.valid_ids = split["train"].detach().cpu().clone(), split["valid"].detach().cpu().clone()
        # The TEST key and TEST records are never indexed.
        self.train_view = _RoleView(dataset, self.train_ids, torch, "TRAIN")
        self.valid_view = _RoleView(dataset, self.valid_ids, torch, "VALID")
        seed = int(recipe.get("_seed", recipe.get("seed", recipe.get("comparison_seed", 0))))
        self.factor_seed = int(recipe.get("factor_seed", seed + 2000003))
        self.sampler = _TrainBatcher(torch, self.train_view, seed=int(recipe.get("sampler_seed", seed + 70000001)))
        self.valid_generator = torch.Generator(device="cpu").manual_seed(seed + 70000003)
        architecture = self.config["gnn"]
        native = GNN(gnn_type=architecture.split("-")[0], virtual_node=architecture.endswith("-virtual"),
                     num_tasks=dataset.num_tasks, num_layer=5, emb_dim=300,
                     drop_ratio=self.config["drop_ratio"]).to(self.device)
        self.encoding_dependencies = {}
        for label, module in (("AtomEncoder", native.gnn_node.atom_encoder),
                              ("BondEncoder", native.gnn_node.convs[0].bond_encoder)):
            path = Path(inspect.getsourcefile(type(module))).resolve()
            self.encoding_dependencies[label] = {"module": type(module).__module__, "path": str(path),
                                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        # Keep native constructor defaults residual=False, JK=last and mean pooling.
        original = tuple(native.named_parameters())
        optimizer = torch.optim.Adam(native.parameters(), lr=0.001)
        self.optimizer_assignment = []
        self.model = native
        if procedure != "native_single":
            boundary = _sibling("categorical_boundary")
            self.model = boundary.attach_ogb_boundary(torch, native, factor_seed=self.factor_seed)
            self.optimizer_assignment = boundary.attach_native_optimizer_groups(self.model, optimizer)
        self.optimizer, self.optimizers = optimizer, [optimizer]
        prefix = "" if procedure == "native_single" else "native."
        self.source_parameter_references = {prefix + name: parameter for name, parameter in original}
        self.allow_missing_grad_names = ()
        self.full_responsibilities = None
        self.responsibility_metadata = None
        if procedure == "boundary_private_fixed_responsibility":
            store = getattr(dataset, "_data", None)
            if store is None:
                store = dataset.data
            train_y = store.y.index_select(0, self.train_ids).reshape(len(self.train_ids), 1)
            observed = ~torch.isnan(train_y[:, 0])
            values = train_y[observed, 0]
            if not ((values == 0) | (values == 1)).all().item():
                raise ValueError("Native observed MolHIV/MolBACE TRAIN binary labels required")
            observed_weights, self.responsibility_metadata = self.rules.draw_fixed_responsibilities(
                torch, values.to(torch.long), seed=seed)
            self.full_responsibilities = torch.ones(4, len(self.train_ids), dtype=torch.float64, device="cpu")
            self.full_responsibilities[:, observed] = observed_weights.cpu()
        from ogb.graphproppred import Evaluator
        from torch_geometric.loader import DataLoader
        self.evaluator = Evaluator(self.config["dataset"])
        self.valid_loader = DataLoader(self.valid_view, batch_size=32, shuffle=False, num_workers=0,
                                       generator=self.valid_generator)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.updates = 0
        self.last_train_positions = None

    def _native_loss(self, prediction, batch):
        mask = batch.y == batch.y
        if not mask.any().item():
            raise ValueError("Native batch has no observed targets; do not replace its loss")
        return self.criterion(prediction.to(self.torch.float32)[mask], batch.y.to(self.torch.float32)[mask])

    def _next_train_batch(self):
        skipped = 0
        while True:
            batch, positions = self.sampler.next()
            # Retain the exact main_pyg.train skipped-batch conditions.
            if batch.x.shape[0] == 1 or int(batch.batch[-1]) == 0:
                skipped += 1
                if skipped > (len(self.train_view) + 31) // 32:
                    raise ValueError("No source-eligible TRAIN batch in a native epoch")
                continue
            return batch.to(self.device), positions, skipped

    def train_update(self):
        torch = self.torch
        self.model.train()
        batch, positions, skipped = self._next_train_batch()
        self.last_train_positions = positions.clone()
        if self.procedure == "native_single":
            prediction = self.model(batch)
            self.optimizer.zero_grad()
            loss = self._native_loss(prediction, batch)
            loss.backward()
            self.optimizer.step()
            work = {"optimizer_transitions": 1, "reverse_collections": 1}
        else:
            routes = [self.model.run_member(member, batch) for member in range(4)]
            predictions, logits = [r[0] for r in routes], [r[1] for r in routes]
            self.optimizer.zero_grad()
            terms = self.rules.native_loss_terms(torch, predictions, logits,
                native_loss=lambda p: self._native_loss(p, batch),
                pooled_native_loss=lambda p: self._native_loss(p, batch),
                native_regularizer=torch.zeros((), device=self.device, dtype=torch.float32))
            F, L = terms["F"], terms["L"]
            if self.procedure == "gnnm_boundary_native":
                loss = F
                loss.backward()
                self.optimizer.step()
                work = {"optimizer_transitions": 1, "reverse_collections": 1}
            elif self.procedure == "boundary_global_half_pool":
                loss = terms["J"]
                work = self.rules.step_global_half_pool(torch, self.model, self.optimizer, F=F, L=L)
            elif self.procedure == "boundary_private_half_pool":
                loss = terms["J"]
                work = self.rules.step_private_half_pool(torch, self.model, self.optimizer, F=F, L=L)
            else:
                mask = (batch.y == batch.y).reshape(-1)
                target = batch.y.to(torch.float32)
                records = torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(
                    p.to(torch.float32), target, reduction="none").reshape(-1)[mask] for p in predictions])
                weights = self.full_responsibilities[:, positions][:, mask.detach().cpu()].to(self.device)
                weighted = self.rules.weighted_native_own_loss(torch, records, weights,
                    native_reduce=lambda row: row.mean(),
                    native_regularizer=torch.zeros((), device=self.device, dtype=torch.float32))
                loss = weighted
                work = self.rules.step_private_responsibility(torch, self.model, self.optimizer, F=F, F_weighted=weighted)
            self.model.assert_source_identity()
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise ValueError("Nonfinite real TRAIN smoke objective")
        self.updates += 1
        return {"loss": value, "optimizer_updates": 1, "update": self.updates,
                "TRAIN_graph_ids": self.train_ids[positions].tolist(), "native_skipped_batches": skipped,
                "native_batch_graphs": len(positions), "native_route_forwards": 1 if self.procedure == "native_single" else 4,
                "source_bn_policy": "unchanged shared native buffers, sequential native route forwards",
                "work": work}

    def validate(self):
        torch = self.torch
        self.model.eval()
        predictions, labels, graph_ids = [], [], []
        loss_sum, observed_count, skipped = 0.0, 0, 0
        if self.recipe.get("smoke_valid_batch_limit") is not None:
            raise ValueError("This campaign smoke callback requires complete official VALID, not a prefix")
        with torch.no_grad():
            for batch_index, batch in enumerate(self.valid_loader):
                batch = batch.to(self.device)
                if batch.x.shape[0] == 1:  # original main_pyg.eval condition
                    skipped += 1
                    continue
                if self.procedure == "native_single":
                    pred = self.model(batch)
                else:
                    pred = self.rules.mean_raw_logits(torch, [self.model.run_member(m, batch)[1] for m in range(4)])
                if not torch.isfinite(pred).all().item():
                    raise ValueError("Nonfinite real VALID outputs")
                y = batch.y.view(pred.shape)
                mask = y == y
                if mask.any().item():
                    loss_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(
                        pred.to(torch.float32)[mask], y.to(torch.float32)[mask], reduction="sum").cpu())
                    observed_count += int(mask.sum())
                predictions.append(pred.detach().cpu())
                labels.append(y.detach().cpu())
                graph_ids.append(batch.smoke_graph_id.detach().cpu())
        if not predictions or not observed_count:
            raise ValueError("Real VALID source output/observed-label support required")
        outputs, truth = torch.cat(predictions), torch.cat(labels)
        eligible = truth[:, 0] == truth[:, 0]
        have_both_classes = len(torch.unique(truth[eligible, 0])) == 2
        if not have_both_classes:
            raise ValueError("Complete official VALID lacks the source ROC-AUC support")
        metrics = self.evaluator.eval({"y_true": truth.numpy(), "y_pred": outputs.numpy()})
        return {"outputs": outputs, "loss": loss_sum / observed_count, "metrics": metrics,
                "VALID_graph_ids": torch.cat(graph_ids).tolist(), "VALID_graphs": len(outputs),
                "VALID_scope": "complete_source_VALID",
                "native_skipped_batches": skipped, "TEST_accessed": False}

    def state(self):
        return {"schema": "ogb-native-boundary-smoke-state-v1", "procedure": self.procedure,
                "config": dict(self.config), "model": copy.deepcopy(self.model.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()), "sampler": self.sampler.state(),
                "valid_generator": self.valid_generator.get_state().clone(), "updates": self.updates,
                "module_training": {name: module.training for name, module in self.model.named_modules()},
                "train_ids": self.train_ids.clone(), "valid_ids": self.valid_ids.clone(),
                "responsibilities": None if self.full_responsibilities is None else self.full_responsibilities.clone(),
                "responsibility_metadata": copy.deepcopy(self.responsibility_metadata),
                "last_train_positions": None if self.last_train_positions is None else self.last_train_positions.clone(),
                "global_RNG_owner": "controller captures Python/NumPy/Torch CPU/CUDA state separately"}

    def load_state(self, state):
        if state["procedure"] != self.procedure or state["config"] != self.config \
                or not self.torch.equal(state["train_ids"].cpu(), self.train_ids) \
                or not self.torch.equal(state["valid_ids"].cpu(), self.valid_ids):
            raise ValueError("Checkpoint recipe/data-role custody differs")
        self.model.load_state_dict(state["model"], strict=True)
        # Optimizer load_state_dict may retain same-device tensor objects (in
        # particular CPU Adam step scalars). Own a private copy so a resumed
        # update cannot mutate the caller's reusable checkpoint payload.
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.sampler.load_state(state["sampler"])
        self.valid_generator.set_state(state["valid_generator"].cpu())
        self.updates = int(state["updates"])
        self.full_responsibilities = None if state["responsibilities"] is None else state["responsibilities"].cpu().clone()
        self.responsibility_metadata = copy.deepcopy(state["responsibility_metadata"])
        self.last_train_positions = None if state["last_train_positions"] is None else state["last_train_positions"].cpu().clone()
        for name, module in self.model.named_modules():
            module.training = bool(state["module_training"][name])
        if self.procedure != "native_single":
            self.model.assert_source_identity()
        current = dict(self.model.named_parameters())
        if any(current[name] is not parameter for name, parameter in self.source_parameter_references.items()):
            raise ValueError("Checkpoint restoration replaced original native parameters")

    def metadata(self):
        return {"schema": "ogb-real-data-smoke-callback-v1", "procedure": self.procedure,
                "config": dict(self.config), "source_pins": self.source_pins, "training_rules_pin": self.rules_pin,
                "encoding_dependencies": self.encoding_dependencies,
                "scaffold_role_pins": self.split_pins,
                "input_encoding": "all native categorical atom fields; no IDs changed",
                "boundary": None if self.procedure == "native_single" else self.model.metadata(),
                "optimizer": "native torch.optim.Adam lr=.001, original defaults, no explicit loss regularizer",
                "optimizer_private_assignment": self.optimizer_assignment,
                "sampler": "native RandomSampler without replacement, private persisted generator/order/cursor",
                "batch_collation": "native PyG Batch.from_data_list, batch32, full features",
                "global_RNG": "controller-owned; factory does not reseed it",
                "TRAIN_graphs": len(self.train_view), "VALID_graphs": len(self.valid_view), "TEST_accessed": False,
                "full_training_started": False, "checkpoint_state_complete_for_smoke": True}

    def close(self):
        if self.procedure != "native_single":
            self.model.close()


def create_session(recipe: dict, procedure: str, data_root: Path, device: str):
    import torch
    return OGBSmokeSession(torch, recipe, procedure, data_root, device)
