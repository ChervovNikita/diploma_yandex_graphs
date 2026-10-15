"""Pinned molecular TRAIN/VALID transform; native execution remains pending.

Attach only to a freshly constructed, otherwise unextended OGB boundary source
session. This file imports only stdlib; torch comes from the source session.
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
PHASE = HERE.parent.parent
REDESIGN = PHASE / "reference_wrapper_redesign_20261013_v1"
METHODS = {"aggregation_mean_globalJ_v1": "mean",
           "aggregation_global_globalJ_v1": "global",
           "aggregation_conditional_globalJ_v1": "conditional"}
STATE_SCHEMA = "conditional-aggregation-molecular-wrapped-state-v1"


def _receipt(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _bindings(expected_manifest):
    manifest_pin = _receipt(HERE / "MANIFEST.json")
    if manifest_pin["sha256"] != expected_manifest:
        raise ValueError("Declare the exact frozen conditional aggregation manifest")
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    receipts = [manifest_pin]
    for row in manifest["files"]:
        path = (HERE / row["path"]).resolve()
        if not path.is_relative_to(HERE):
            raise ValueError("Conditional method file escapes its package")
        actual = _receipt(path)
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen conditional method file changed: " + row["path"])
        receipts.append(actual)
    for row in json.loads((HERE / "SOURCES.json").read_text())["source_files"]:
        if row["role"] != "molecular_runtime_dependency":
            continue
        path = (PHASE / row["path_relative_to_phase"]).resolve()
        if not path.is_relative_to(PHASE):
            raise ValueError("Native source escapes the phase checkout")
        actual = _receipt(path)
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Pinned molecular source changed: " + row["path_relative_to_phase"])
        receipts.append(actual)
    return receipts


def _core():
    path = HERE / "core.py"
    name = "_conditional_aggregation_v1_" + _receipt(path)["sha256"]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _ownership(session, original_references):
    current = dict(session.model.named_parameters())
    if any(current.get(name) is not parameter for name, parameter in original_references.items()):
        raise ValueError("An original native Parameter was replaced or renamed")
    owned = [p for group in session.optimizer.param_groups for p in group["params"]]
    if len(owned) != len({id(p) for p in owned}) or {id(p) for p in owned} != {id(p) for p in current.values()}:
        raise ValueError("Every native/boundary/aggregation parameter requires exactly one original optimizer owner")
    session.model.assert_source_identity()


def _require_source(session):
    expected = REDESIGN / "smoke_20261014/ogb_port"
    if Path(inspect.getsourcefile(type(session))).resolve() != expected / "session.py" \
            or Path(inspect.getsourcefile(type(session.model))).resolve() != expected / "categorical_boundary.py":
        raise TypeError("Only the pinned original OGBSmokeSession and categorical boundary are supported")
    if session.procedure != "gnnm_boundary_native" or session.updates != 0:
        raise ValueError("Use a fresh untrained mean-own native boundary source session")
    if set(session.model._modules) != {"native", "input_bank", "output_bank"}:
        raise ValueError("Standalone aggregation cannot attach over another model extension")
    for name in ("train_update", "validate", "state", "load_state", "metadata"):
        if getattr(getattr(session, name), "__func__", None) is not getattr(type(session), name):
            raise ValueError("Standalone aggregation cannot attach over another session transform")
    if Path(session.redesign_root).resolve() != REDESIGN:
        raise ValueError("The source session is bound to another redesign checkout")
    tables = tuple(session.model.native.gnn_node.atom_encoder.atom_embedding_list)
    head = session.model.native.graph_pred_linear
    if len(tables) != 9 or any(table.embedding_dim != 300 for table in tables) \
            or head.in_features != 300 or head.out_features != 1 or session.config.get("feature") != "full":
        raise TypeError("Require complete native nine-field300-wide single-task molecular source shapes")
    if type(session.optimizer) is not session.torch.optim.Adam \
            or len(session.optimizers) != 1 or session.optimizers[0] is not session.optimizer:
        raise TypeError("One original source Adam optimizer required")
    if session.full_responsibilities is not None:
        raise ValueError("No risk-assignment combination is defined for this standalone method")
    return head


def transform(session, job):
    method = job.get("procedure")
    if job.get("family") != "ogb" or method not in METHODS or job.get("TEST_access") is not False:
        raise ValueError("Explicit OGB aggregation method and TEST_access=false required; node/link ports are pending")
    if job.get("base_procedure") != "gnnm_boundary_native" or job.get("policy", {}).get("track") == "source_reference":
        raise ValueError("Declare the unchanged GNNM source base and a scientific extension track")
    settings = job.get("method_settings", {})
    if set(settings) != {"manifest_sha256"} or not isinstance(settings["manifest_sha256"], str) \
            or len(settings["manifest_sha256"]) != 64:
        raise ValueError("The only setting is the required exact package manifest SHA256; no tuning")
    repository = job.get("recipe", {}).get("_repository_root")
    if not isinstance(repository, str) or not Path(repository).is_absolute():
        raise ValueError("Bind recipe._repository_root explicitly to an absolute repository root")
    root = Path(repository).resolve()
    if not PHASE.is_relative_to(root) or not HERE.is_relative_to(root):
        raise ValueError("Declared repository root must contain the actual phase and method source")
    receipts = _bindings(settings["manifest_sha256"])
    head = _require_source(session)
    torch = session.torch
    core = _core()
    original_references = dict(session.source_parameter_references)
    _ownership(session, original_references)
    group_objects = tuple(session.optimizer.param_groups)
    group_options = [{k: copy.deepcopy(v) for k, v in group.items() if k != "params"} for group in group_objects]
    owners = [i for i, group in enumerate(group_objects) if any(p is head.weight for p in group["params"])]
    if len(owners) != 1:
        raise ValueError("Original native head weight must have one original optimizer group")
    gate = core.make_aggregator(torch, mode=METHODS[method], task_kind="binary", seed=job["seed"], reference=head.weight)
    session.model.add_module("conditional_aggregation", gate)
    session.optimizer.param_groups[owners[0]]["params"].extend(gate.parameters())
    if len(session.optimizer.param_groups) != len(group_objects) or any(
            group is not old or {k: v for k, v in group.items() if k != "params"} != options
            for group, old, options in zip(session.optimizer.param_groups, group_objects, group_options)):
        raise ValueError("Original native optimizer groups/options changed")
    _ownership(session, original_references)
    metadata = {"schema": "conditional-aggregation-molecular-session-v1", "method_id": method,
                "declared_repository_root": str(Path(repository).resolve()),
                "aggregation": gate.metadata(), "source_receipts": receipts,
                "native_source_pins": copy.deepcopy(session.source_pins),
                "training_rules_pin": copy.deepcopy(session.rules_pin),
                "encoding_dependencies": copy.deepcopy(session.encoding_dependencies),
                "source_scaffold_roles": copy.deepcopy(session.split_pins),
                "training_objective": "global J=.5*(mean native own loss)+.5*(native loss of actual served raw logits)",
                "gradient_scope": "same J to every native/boundary/aggregation parameter",
                "native_regularizer": "original source has no explicit loss regularizer",
                "optimizer_assignment": {"original_group": owners[0], "anchor": "native.graph_pred_linear.weight",
                                         "options_changed": False, "new_groups": 0},
                "native_backbone_copies": 1, "native_parameters_replaced": 0,
                "BN_dropout_and_source_route_order": "unchanged shared native state; routes0,1,2,3 sequentially",
                "selection_output": "actual served output, no gate/weight fit on VALID",
                "native_runtime_qualification": "pending; source and synthetic interface checks only",
                "final_selected_state_reconstruction": "unsupported/pending; do not strip wrapped state",
                "novelty_or_quality_claim": False, "TEST_accessed": False}
    session.conditional_aggregation_metadata = metadata
    source_state, source_load, source_metadata = session.state, session.load_state, session.metadata

    def train_update(self):
        self.model.train()
        batch, positions, skipped = self._next_train_batch()
        self.last_train_positions = positions.clone()
        self.optimizer.zero_grad()
        routes = [self.model.run_member(member, batch) for member in range(4)]
        terms = core.native_objective(torch,
            native_member_losses=[self._native_loss(route[0], batch) for route in routes],
            raw_logits=[route[1] for route in routes], aggregator=gate,
            served_native_loss=lambda logits: self._native_loss(logits, batch),
            native_regularizer=head.weight.new_zeros(()))
        terms["J"].backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("Nonfinite native/boundary/aggregation J gradient")
        self.optimizer.step()
        _ownership(self, original_references)
        self.updates += 1
        return {"loss": float(terms["J"].detach().cpu()), "native_own_loss": float(terms["F"].detach().cpu()),
                "actual_served_loss": float(terms["L"].detach().cpu()),
                "native_member_losses": terms["native_member_losses"].detach().cpu().tolist(),
                "mean_route_weights": terms["weights"].detach().mean(1).cpu().tolist(),
                "optimizer_updates": 1, "update": self.updates, "TRAIN_graph_ids": self.train_ids[positions].tolist(),
                "native_skipped_batches": skipped, "native_batch_graphs": len(positions), "native_route_forwards": 4,
                "aggregation_method": method,
                "work": {"optimizer_transitions": 1, "reverse_collections": 1, "extra_backbone_forwards": 0}}

    def validate(self):
        self.model.eval()
        predictions, means, labels, graph_ids = [], [], [], []
        loss_sum, mean_loss_sum, observed_count, skipped = 0.0, 0.0, 0, 0
        if self.recipe.get("smoke_valid_batch_limit") is not None:
            raise ValueError("Complete official source VALID required")
        with torch.no_grad():
            for batch in self.valid_loader:
                batch = batch.to(self.device)
                if batch.x.shape[0] == 1:  # exact original main_pyg.eval condition
                    skipped += 1
                    continue
                logits = [self.model.run_member(member, batch)[1] for member in range(4)]
                pred = gate(logits)["served_raw_logits"]
                mean = torch.stack(logits).mean(0)
                y = batch.y.view(pred.shape)
                mask = y == y
                if mask.any().item():
                    loss_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(
                        pred.to(torch.float32)[mask], y.to(torch.float32)[mask], reduction="sum").cpu())
                    mean_loss_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(
                        mean.to(torch.float32)[mask], y.to(torch.float32)[mask], reduction="sum").cpu())
                    observed_count += int(mask.sum())
                predictions.append(pred.detach().cpu())
                means.append(mean.detach().cpu())
                labels.append(y.detach().cpu())
                graph_ids.append(batch.smoke_graph_id.detach().cpu())
        if not predictions or not observed_count:
            raise ValueError("Complete source VALID output/observed-label support required")
        outputs, mean_outputs, truth = torch.cat(predictions), torch.cat(means), torch.cat(labels)
        eligible = truth[:, 0] == truth[:, 0]
        if len(torch.unique(truth[eligible, 0])) != 2:
            raise ValueError("Complete official VALID lacks source ROC-AUC support")
        metrics = self.evaluator.eval({"y_true": truth.numpy(), "y_pred": outputs.numpy()})
        mean_metrics = self.evaluator.eval({"y_true": truth.numpy(), "y_pred": mean_outputs.numpy()})
        _ownership(self, original_references)
        return {"outputs": outputs, "loss": loss_sum / observed_count, "metrics": metrics,
                "VALID_graph_ids": torch.cat(graph_ids).tolist(), "VALID_graphs": len(outputs),
                "VALID_scope": "complete_source_VALID", "native_skipped_batches": skipped,
                "aggregation_method": method, "same_bank_mean_outputs": mean_outputs,
                "same_bank_mean_loss": mean_loss_sum / observed_count, "same_bank_mean_metrics": mean_metrics,
                "selection_uses": "actual served outputs/metrics only", "TEST_accessed": False}

    def state(self):
        if self.model._member is not None:
            raise RuntimeError("Checkpoint only outside an active native route")
        _ownership(self, original_references)
        return {"schema": STATE_SCHEMA, "method": copy.deepcopy(metadata),
                "source_state": copy.deepcopy(source_state())}

    def load_state(self, value):
        if self.model._member is not None:
            raise RuntimeError("Restore only outside an active native route")
        if value.get("schema") != STATE_SCHEMA or value.get("method") != metadata:
            raise ValueError("Checkpoint conditional method/source/seed differs")
        source_load(copy.deepcopy(value["source_state"]))
        _ownership(self, original_references)

    def combined_metadata(self):
        value = source_metadata()
        value["conditional_aggregation"] = copy.deepcopy(metadata)
        return value

    session.train_update = types.MethodType(train_update, session)
    session.validate = types.MethodType(validate, session)
    session.state = types.MethodType(state, session)
    session.load_state = types.MethodType(load_state, session)
    session.metadata = types.MethodType(combined_metadata, session)
    return session
