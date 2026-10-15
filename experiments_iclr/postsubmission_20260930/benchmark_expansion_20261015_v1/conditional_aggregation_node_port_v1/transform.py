"""Native NodeSession PolyFormer bridge to the unchanged frozen C6 core.

No numerical import, model constructor, data provider, fit loop or teacher.
TRAIN uses the exact source _routes(True) activation-checkpointed graph path.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PHASE = EXPANSION.parent
REDESIGN = PHASE / "reference_wrapper_redesign_20261013_v1"
C6 = EXPANSION / "conditional_aggregation_v1"
C6_MANIFEST = "edef4b1f552a495e09e64855afbd96c7eb14ce5ef02d26cc13ec530d80e9805f"
METHODS = {"aggregation_mean_globalJ_v1": "mean", "aggregation_global_globalJ_v1": "global",
           "aggregation_conditional_globalJ_v1": "conditional"}
STATE_SCHEMA = "conditional-aggregation-node-wrapped-state-v1"


def receipt(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _verify_manifest(folder, expected):
    folder = Path(folder).resolve()
    manifest = receipt(folder / "MANIFEST.json")
    if manifest["sha256"] != expected:
        raise ValueError("Declare the exact frozen source manifest: " + str(folder))
    result = [manifest]
    for row in json.loads((folder / "MANIFEST.json").read_text())["files"]:
        path = (folder / row["path"]).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Manifest path escapes source package")
        actual = receipt(path)
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen source changed: " + str(path))
        result.append(actual)
    return result


def source_bindings(expected, root):
    root = Path(root).resolve()
    if not HERE.is_relative_to(root) or not C6.is_relative_to(root):
        raise ValueError("Repository root must contain actual bridge and frozen core")
    result = _verify_manifest(HERE, expected) + _verify_manifest(C6, C6_MANIFEST)
    for row in json.loads((HERE / "SOURCES.json").read_text())["source_files"]:
        path = (PHASE / row["path_relative_to_phase"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Native dependency escapes declared repository")
        actual = receipt(path)
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen native node dependency changed: " + row["path_relative_to_phase"])
        result.append(actual)
    return result


def load_core():
    path = C6 / "core.py"
    name = "_conditional_node_core_" + receipt(path)["sha256"]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def tensor_binding(value):
    array = value.detach().contiguous().cpu().numpy()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}


def data_binding(session):
    """Bind full transductive inputs and TRAIN/VALID targets, never TEST targets."""
    torch, data = session.torch, session.data
    valid_ids = torch.nonzero(data.val_mask, as_tuple=False).flatten()
    return {"features": tensor_binding(data.x), "edge_index": tensor_binding(data.edge_index),
            "source_basis": [tensor_binding(matrix) for matrix in data.list_mat],
            "TRAIN_ids": tensor_binding(session.train_ids), "TRAIN_targets": tensor_binding(session.targets),
            "VALID_ids": tensor_binding(valid_ids), "VALID_targets": tensor_binding(data.y[valid_ids]),
            "TEST_targets_read": False}


def ownership(session, references):
    current = dict(session.model.named_parameters())
    if any(current.get(name) is not parameter for name, parameter in references.items()):
        raise ValueError("Native Parameter identity or registration changed")
    owned = [p for group in session.optimizer.param_groups for p in group["params"]]
    if len(owned) != len({id(p) for p in owned}) or {id(p) for p in owned} != {id(p) for p in current.values()}:
        raise ValueError("Every native/boundary/gate parameter requires one existing optimizer owner")
    session.core.assert_native_parameter_identity()


def require_source(session, job):
    expected = REDESIGN / "smoke_20261014/node_port.py"
    if Path(inspect.getsourcefile(type(session))).resolve() != expected or session.model is not session.core \
            or Path(inspect.getsourcefile(type(session.core))).resolve() != REDESIGN / "runtime/boundary_core.py":
        raise TypeError("Exact native NodeSession and original PolyFormer boundary core required")
    if session.procedure != "gnnm_boundary_native" or session.updates != 0 or session.responsibility is not None:
        raise ValueError("Attach once to a fresh standalone native mean-own boundary session")
    if set(session.core._modules) != {"native", "banks"} or session.core._active_member is not None:
        raise ValueError("No model-extension combination or active route is allowed")
    for name in ("_routes", "train_update", "validate", "state", "load_state"):
        if getattr(getattr(session, name), "__func__", None) is not getattr(type(session), name):
            raise ValueError("Another session extension changed a native source method")
    declared = copy.deepcopy(job["recipe"])
    for key in ("_seed", "seed"):
        if key in declared and int(declared[key]) != int(job["seed"]):
            raise ValueError("Declared source/gate seeds differ")
        if key in session.recipe:
            declared[key] = job["seed"]  # runtime_v2 injects these two seed aliases
    if Path(session.recipe["_redesign_root"]).resolve() != REDESIGN or session.recipe != declared:
        raise ValueError("Source recipe/root differs from the declared method job")
    bindings = [(bank.binding.module_path, bank.binding.kind, bank.binding.role) for bank in session.core.banks]
    if bindings != [("lin1", "linear", "input"), ("lin3", "linear", "output")]:
        raise ValueError("Preserve the exact original node logical boundaries")
    native, torch = session.core.native, session.torch
    if type(native.lin1) is not torch.nn.Linear or type(native.lin3) is not torch.nn.Linear \
            or native.lin1.in_features != session.data.x.shape[-1] \
            or native.lin3.out_features != session.source.num_classes:
        raise TypeError("Native PolyFormer input/class dimensions changed")
    if type(session.optimizer) is not torch.optim.Adam or len(session.optimizers) != 1 \
            or session.optimizers[0] is not session.optimizer:
        raise TypeError("Preserve the one original Adam and per-source-parameter groups")
    if int(job["seed"]) != int(session.recipe["_seed"]):
        raise ValueError("Paired source and gate job seeds must match")
    return native.lin3


def transform(session, job):
    method = job.get("procedure")
    if job.get("family") != "node" or method not in METHODS or job.get("TEST_access") is not False \
            or job.get("base_procedure") != "gnnm_boundary_native":
        raise ValueError("Declare exact node C6 method/base and TEST_access=false")
    if job.get("policy", {}).get("track") == "source_reference":
        raise ValueError("C6 is a scientific extension, not a native source-reference baseline")
    settings = job.get("method_settings", {})
    if set(settings) != {"manifest_sha256"} or not isinstance(settings["manifest_sha256"], str) \
            or len(settings["manifest_sha256"]) != 64:
        raise ValueError("Only an explicit exact bridge manifest SHA256 is accepted")
    repository = job.get("recipe", {}).get("_repository_root")
    if not isinstance(repository, str) or not Path(repository).is_absolute():
        raise ValueError("Bind recipe._repository_root explicitly to the actual absolute checkout root")
    receipts = source_bindings(settings["manifest_sha256"], repository)
    head = require_source(session, job)
    torch = session.torch
    library = load_core()
    references = dict(session.source_parameter_references)
    ownership(session, references)
    native_groups = tuple(session.optimizer.param_groups)
    options = [{k: copy.deepcopy(v) for k, v in group.items() if k != "params"} for group in native_groups]
    owners = [index for index, group in enumerate(native_groups) if any(p is head.weight for p in group["params"])]
    if len(owners) != 1:
        raise ValueError("Native lin3.weight must retain exactly one source owner group")
    gate = library.make_aggregator(torch, mode=METHODS[method], task_kind="multiclass", seed=job["seed"], reference=head.weight)
    session.core.add_module("conditional_aggregation", gate)
    native_groups[owners[0]]["params"].extend(gate.parameters())
    if len(session.optimizer.param_groups) != len(native_groups) or any(group is not old or
            {k: v for k, v in group.items() if k != "params"} != old_options
            for group, old, old_options in zip(session.optimizer.param_groups, native_groups, options)):
        raise ValueError("Original native Adam groups or options changed")
    ownership(session, references)
    metadata = {"schema": "conditional-aggregation-native-node-v1", "method_id": method,
        "aggregation": gate.metadata(), "source_receipts": receipts,
        "declared_repository_root": str(Path(repository).resolve()), "source_recipe": copy.deepcopy(session.recipe),
        "source_data": data_binding(session), "source_metadata": copy.deepcopy(session.source_metadata),
        "objective": "all-block global J=.5F+.5L_actual_served; F=mean full-TRAIN native CE",
        "explicit_loss_regularizer": "none in source; native Adam weight decay retained per group",
        "optimizer_assignment": {"anchor": "native.lin3.weight", "original_group": owners[0],
                                 "native_head_options": options[owners[0]], "new_groups": 0, "options_changed": False},
        "training_routes": "unchanged source _routes(True), nonreentrant activation checkpointing, preserve_rng_state=True",
        "extra_backbone_copies": 0, "native_parameters_replaced": 0,
        "selection": "actual declared served VALID outputs/accuracy only; same-bank mean is diagnostic",
        "native_runtime_qualification": "pending; later exact-case engineering evidence stays separate",
        "final_selected_state_TEST_reconstruction": "unsupported/pending; preserve wrapped state",
        "global_RNG_owner": "controller captures Python/NumPy/Torch CPU/CUDA separately",
        "novelty_or_quality_claim": False, "TEST_accessed": False}
    session.conditional_aggregation_node_metadata = metadata
    session.source_metadata = {**copy.deepcopy(session.source_metadata), "conditional_aggregation_node": copy.deepcopy(metadata)}
    source_state, source_load = session.state, session.load_state

    def train_update(self):
        self.model.train()
        self.optimizer.zero_grad()
        routes = self._routes(True)  # EXACT frozen source checkpoint path
        terms = library.native_objective(torch,
            native_member_losses=[torch.nn.functional.cross_entropy(route.prediction[self.train_ids], self.targets) for route in routes],
            raw_logits=[route.raw_logits[self.train_ids] for route in routes], aggregator=gate,
            served_native_loss=lambda logits: torch.nn.functional.cross_entropy(logits, self.targets),
            native_regularizer=head.weight.new_zeros(()))
        terms["J"].backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("Nonfinite native/boundary/gate global-J gradient")
        self.optimizer.step()
        ownership(self, references)
        self.updates += 1
        return {"loss": float(terms["J"].detach().cpu()), "native_own_loss": float(terms["F"].detach().cpu()),
            "actual_served_loss": float(terms["L"].detach().cpu()),
            "native_member_losses": terms["native_member_losses"].detach().cpu().tolist(),
            "mean_route_weights": terms["weights"].detach().mean(1).cpu().tolist(),
            "optimizer_updates": 1, "logical_update": self.updates, "train_records": len(self.train_ids),
            "graph_nodes": int(self.data.num_nodes), "whole_native_train": True,
            "native_route_forward_views": 4, "activation_checkpointing": True,
            "checkpoint_recomputation_work_not_counted": True,
            "work": {"reverse_collections": 1, "optimizer_transitions": 1, "extra_backbone_forward_views": 0}}

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            ids = torch.nonzero(self.data.val_mask, as_tuple=False).flatten()
            raw = [route.raw_logits[ids] for route in self._routes(False)]
            outputs = gate(raw)["served_raw_logits"]
            mean = torch.stack(raw).mean(0)
            targets = self.data.y[ids]
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            mean_loss = torch.nn.functional.cross_entropy(mean, targets)
            accuracy = float(outputs.argmax(-1).eq(targets).float().mean().cpu())
            mean_accuracy = float(mean.argmax(-1).eq(targets).float().mean().cpu())
        ownership(self, references)
        return {"outputs": outputs, "loss": float(loss.cpu()), "metrics": {"accuracy": accuracy},
            "role": "VALID", "VALID_scope": "complete_source_VALID", "VALID_nodes": len(ids),
            "VALID_node_ids": ids.detach().cpu(), "test_scoring": False, "TEST_accessed": False,
            "same_bank_mean_outputs": mean, "same_bank_mean_loss": float(mean_loss.cpu()),
            "same_bank_mean_metrics": {"accuracy": mean_accuracy},
            "selection_uses": "actual declared served outputs/accuracy only"}

    def state(self):
        if self.core._active_member is not None:
            raise RuntimeError("Checkpoint only outside active source/recomputed routes")
        ownership(self, references)
        return {"schema": STATE_SCHEMA, "method": copy.deepcopy(metadata),
                "module_training": {name: module.training for name, module in self.model.named_modules()},
                "source_state": copy.deepcopy(source_state())}

    def load_state(self, value):
        if self.core._active_member is not None:
            raise RuntimeError("Restore only outside active source/recomputed routes")
        if value.get("schema") != STATE_SCHEMA or value.get("method") != metadata:
            raise ValueError("Node checkpoint method/source/recipe/data/seed binding differs")
        if set(value["module_training"]) != {name for name, _ in self.model.named_modules()}:
            raise ValueError("Node checkpoint module-mode identity differs")
        source_load(copy.deepcopy(value["source_state"]))
        for name, module in self.model.named_modules():
            module.training = bool(value["module_training"][name])
        ownership(self, references)

    session.train_update, session.validate = types.MethodType(train_update, session), types.MethodType(validate, session)
    session.state, session.load_state = types.MethodType(state, session), types.MethodType(load_state, session)
    return session
