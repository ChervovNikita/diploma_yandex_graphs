"""Explicit new combined recipes on one untouched native node/molecular body.

No existing standalone transform is called or relaxed. This factory admits a
fresh native session, assembles declared components, then owns one full state.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
METHODS = ("interaction_geometry_input", "interaction_responsibility_private_pool",
           "interaction_input_meanJ", "interaction_input_conditionalJ")
STATE_SCHEMA = "explicit-node-molecular-interaction-state-v1"


def sibling(name):
    path = HERE / (name + ".py")
    key = "_interaction_" + hashlib.sha256(path.read_bytes()).hexdigest()
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def selected_mean_sidecar(session):
    """Zero-forward getter; caller must bind its loaded-selected epoch/checkpoint.

    Native validation cache is invalidated before TRAIN and on state restore.
    Frozen generic runtime adapters discard extra fields; they must explicitly
    persist this sidecar in a future generation. This getter selects nothing.
    """
    value = getattr(session, "_interaction_last_VALID", None)
    if value is None or value["source_updates"] != session.updates:
        raise ValueError("A fresh complete VALID of the currently loaded state is required")
    return copy.deepcopy(value)


def transform(session, job):
    binding = sibling("bindings")
    family, method = job.get("family"), job.get("procedure")
    if family not in ("node", "ogb") or method not in METHODS or job.get("base_procedure") != "gnnm_boundary_native" \
            or job.get("TEST_access") is not False or job.get("policy", {}).get("track") == "source_reference":
        raise ValueError("Explicit new node/molecular interaction recipe with native boundary base and TEST closed required")
    root = job.get("recipe", {}).get("_repository_root")
    settings = job.get("method_settings", {})
    if not isinstance(root, str) or not Path(root).is_absolute() or set(settings) != {"manifest_sha256"}:
        raise ValueError("Actual absolute recipe repository root and only the exact interaction manifest required")
    receipts, ingredients = binding.source_bindings(root, settings["manifest_sha256"], family)
    # These immutable validators admit only the fresh native source. Assembly
    # begins after they return; no parent transform is called on another port.
    if family == "node":
        validator = binding.load("conditional_aggregation_node_port_v1/transform.py")
        head = validator.require_source(session, job)
        source_data = validator.data_binding(session)
        if not session.torch.equal(session.targets, session.data.y[session.train_ids]):
            raise ValueError("Exact source TRAIN targets required")
    else:
        validator = binding.load("conditional_aggregation_v1/molecular_transform.py")
        head = validator._require_source(session)
        if int(session.recipe.get("_seed", -1)) != int(job["seed"]):
            raise ValueError("Actual source and component fit seed must match")
        source_data = {"scaffold_roles": copy.deepcopy(session.split_pins),
            "TRAIN_ids": session.train_ids.cpu().tolist(), "VALID_ids": session.valid_ids.cpu().tolist(),
            "native_fixed_config": copy.deepcopy(session.config), "native_sources": copy.deepcopy(session.source_pins),
            "encoding_dependencies": copy.deepcopy(session.encoding_dependencies), "TEST_targets_read": False}
    torch, core = session.torch, session.model
    references = dict(session.source_parameter_references)
    binding.ownership(session, references)
    groups = tuple(session.optimizer.param_groups)
    options = [{k: copy.deepcopy(v) for k, v in g.items() if k != "params"} for g in groups]
    original_private = core.private_parameters
    residuals, input_sites = None, []
    if method != "interaction_responsibility_private_pool":
        residuals, input_sites = binding.attach_residual(session, family, int(job["seed"]))
    gate_library = binding.load("conditional_aggregation_v1/core.py")
    mode = "conditional" if method == "interaction_input_conditionalJ" else "mean"
    gate = gate_library.make_aggregator(torch, mode=mode, task_kind="multiclass" if family == "node" else "binary",
        seed=job["seed"], reference=head.weight)
    core.add_module("interaction_readout", gate)
    group = binding.load("portable_private_input_v2/core.py").assign_private_parameters(
        session.optimizer, head.weight, gate.parameters())
    extra_parameters = (() if residuals is None else tuple(residuals.parameters())) + tuple(gate.parameters())

    def private_parameters(self):
        return tuple(original_private()) + extra_parameters

    core.private_parameters = types.MethodType(private_parameters, core)
    if len(session.optimizer.param_groups) != len(groups) or any(g is not old or {
            k: v for k, v in g.items() if k != "params"} != before for g, old, before in zip(session.optimizer.param_groups, groups, options)):
        raise ValueError("Original source optimizer groups/options changed")
    binding.ownership(session, references)
    auxiliary = positions = table = weights = None
    geometry_record = responsibility_record = None
    if method == "interaction_geometry_input":
        auxiliary, positions, geometry_record = binding.geometry(session, family, job["seed"])
    if method == "interaction_responsibility_private_pool":
        table, weights, responsibility_record = binding.structural_table(session, family)
    metadata = {"schema": "explicit-node-molecular-interaction-v1", "method_id": method, "family": family,
        "manifest_sha256": settings["manifest_sha256"], "ingredient_bindings": ingredients,
        "source_receipts": receipts, "actual_repository_root": str(Path(root).resolve()),
        "source_recipe": copy.deepcopy(session.recipe), "source_data": source_data, "fit_seed": job["seed"],
        "input_residual": input_sites, "aggregation": gate.metadata(), "geometry": geometry_record,
        "responsibility": responsibility_record, "readout_optimizer_group": group,
        "native_optimizer_group_options": options, "new_optimizer_groups": 0, "source_options_changed": False,
        "objective": "all-block F+.05*(alignment+variance)" if auxiliary is not None else
            "shared F; private .5W_structured+.5L_mean; not a global scalar objective" if table is not None else
            "all-block .5F+.5L_actual_served",
        "native_task_reduction": "full native TRAIN CE mean per route" if family == "node" else
            "native original mixed source-batch FP32 observed-label BCE mean per route",
        "member_reduction": "mean of four native per-route losses; no class reweighting except private W",
        "native_regularizer": "no extra scalar loss regularizer; original source Adam WD retained",
        "TRAIN_routes": "unchanged _routes(True) activation checkpointing" if family == "node" else
            "four sequential native source routes, original BN/dropout/batch32/skip law",
        "native_backbone_copies": 1, "native_parameter_replacements": 0, "extra_backbone_forwards": 0,
        "selection": "own selector sees actual serving outputs only; same-bank mean is a fixed diagnostic",
        "mean_export_boundary": "native validate returns/cache same-call mean; generic frozen runtime drops fields until explicit production finalizer hook",
        "legacy_molecular_pilot_alias": False, "TEST_accessed": False, "novelty_or_accuracy_claim": False}
    session.interaction_metadata = copy.deepcopy(metadata)
    session._interaction_last_VALID = None
    source_state, source_load = session.state, session.load_state
    source_metadata = getattr(session, "metadata", None)
    if family == "node":
        session.source_metadata = {**copy.deepcopy(session.source_metadata), "interaction": copy.deepcopy(metadata)}
    active = [False]

    def gradients():
        result = {}
        components = {"input_residual": () if residuals is None else residuals.parameters(), "readout": gate.parameters()}
        for name, parameters in components.items():
            result[name] = [{"shape": list(p.shape), "grad_present": p.grad is not None,
                "gradient_L2": None if p.grad is None else float(p.grad.detach().norm())} for p in parameters]
        return result

    def finite_before_step(loss):
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                for p in session.model.parameters()):
            raise FloatingPointError("Nonfinite actual native interaction loss/gradient before transition")

    def train_update(self):
        if active[0]:
            raise ValueError("No nested interaction update")
        active[0] = True
        self._interaction_last_VALID = None
        try:
            self.model.train(); self.optimizer.zero_grad()
            if family == "node":
                routes = self._routes(True)
                prediction = [r.prediction[self.train_ids] for r in routes]
                raw = [r.raw_logits[self.train_ids] for r in routes]
                hidden = torch.stack([r.prehead[self.train_ids] for r in routes])
                native_loss = lambda z: torch.nn.functional.cross_entropy(z, self.targets)
                identity = {"train_records": len(self.train_ids), "whole_native_train": True,
                    "graph_nodes": int(self.data.num_nodes), "activation_checkpointing": True,
                    "checkpoint_recomputation_work_not_counted": True}
            else:
                batch, source_positions, skipped = self._next_train_batch()
                self.last_train_positions = source_positions.clone()
                routes = [self.model.run_member(m, batch) for m in range(4)]
                prediction, raw = [r[0] for r in routes], [r[1] for r in routes]
                hidden = torch.stack([r[2] for r in routes])
                native_loss = lambda z: self._native_loss(z, batch)
                observed_y = batch.y.reshape(-1)[batch.y.reshape(-1) == batch.y.reshape(-1)]
                identity = {"TRAIN_graph_ids": self.train_ids[source_positions].tolist(),
                    "native_batch_graphs": len(source_positions), "native_skipped_batches": skipped,
                    "source_batch_observed_class_counts": {str(c): int((observed_y == c).sum()) for c in (0, 1)}}
            zero = head.weight.new_zeros(())
            terms = self.rules.native_loss_terms(torch, prediction, raw, native_loss=native_loss,
                pooled_native_loss=native_loss, native_regularizer=zero)
            record = {"F": float(terms["F"].detach()), "L_mean": float(terms["L"].detach())}
            if auxiliary is not None:
                if family == "node":
                    extra = auxiliary(prehead=hidden[:, positions], labels=self.targets[positions],
                        node_ids=self.train_ids[positions], role="TRAIN")
                    extra_loss, diagnostics = extra["loss"], extra["diagnostics"]
                else:
                    extra = auxiliary(prehead=hidden, labels=batch.y,
                        graph_ids=batch.smoke_graph_id.reshape(-1), role="TRAIN")
                    extra_loss, diagnostics = extra.loss, extra.diagnostics
                loss = terms["F"] + extra_loss
                record.update(auxiliary_loss=float(extra_loss.detach()), auxiliary_diagnostics=diagnostics)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite native+geometry scalar")
                loss.backward(); finite_before_step(loss)
                work = {"reverse_collections": 1, "optimizer_transitions": 1}
                self.optimizer.step()
            elif table is not None:
                if family == "node":
                    records = torch.stack([torch.nn.functional.cross_entropy(z, self.targets, reduction="none") for z in raw])
                    batch_weights = weights
                else:
                    mask = (batch.y == batch.y).reshape(-1)
                    records = torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(
                        z.float(), batch.y.float(), reduction="none").reshape(-1)[mask] for z in prediction])
                    batch_weights = weights[:, source_positions][:, mask.cpu()].to(self.device)
                weighted = self.rules.weighted_native_own_loss(torch, records, batch_weights,
                    native_reduce=lambda row: row.mean(), native_regularizer=zero)
                private = .5 * weighted + .5 * terms["L"]
                work = self.rules.step_private_responsibility(torch, core, self.optimizer,
                    F=terms["F"], F_weighted=private, before_step=lambda: finite_before_step(private))
                loss = private
                record.update(W_structured=float(weighted.detach()), private_objective=float(private.detach()),
                    loss_semantics="logged private credit; shared receives F, no global scalar objective")
            else:
                joint = gate_library.native_objective(torch, native_member_losses=[native_loss(z) for z in prediction],
                    raw_logits=raw, aggregator=gate, served_native_loss=native_loss, native_regularizer=zero)
                loss = joint["J"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite joint native risk")
                loss.backward(); finite_before_step(loss)
                work = {"reverse_collections": 1, "optimizer_transitions": 1}
                self.optimizer.step()
                record.update(L_actual=float(joint["L"].detach()), mean_route_weights=joint["weights"].detach().mean(1).cpu().tolist())
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                raise FloatingPointError("Nonfinite actual native interaction loss/gradient")
            binding.ownership(self, references)
            self.updates += 1
            return {"loss": float(loss.detach()), **record, **identity, "component_gradients": gradients(),
                "optimizer_updates": 1, "update": self.updates, "native_route_forward_views": 4,
                "work": {**work, "extra_forward_views": 0}}
        finally:
            active[0] = False

    def validate(self):
        if active[0]:
            raise ValueError("VALID cannot occur inside an update")
        value = sibling("validation").validate(self, family, gate)
        binding.ownership(self, references)
        self._interaction_last_VALID = {"schema": "same-call-interaction-VALID-sidecar-v1", "role": "VALID",
            "method_id": method, "method_manifest_sha256": settings["manifest_sha256"], "source_updates": self.updates,
            "ids": value["VALID_ids"].cpu(), "labels": value["VALID_labels"].cpu(),
            "actual_served_raw_logits": value["outputs"].detach().cpu(), "actual_loss": value["loss"],
            "actual_metrics": copy.deepcopy(value["metrics"]),
            "same_bank_mean_raw_logits": value["same_bank_mean_outputs"].detach().cpu(),
            "same_bank_mean_loss": value["same_bank_mean_loss"],
            "same_bank_mean_metrics": copy.deepcopy(value["same_bank_mean_metrics"]),
            "selected_checkpoint_bound": False, "caller_must_bind_loaded_selected_epoch_checkpoint": True,
            "extra_forward_calls": 0, "TEST_accessed": False}
        return value

    def state(self):
        if active[0] or (core._active_member if family == "node" else core._member) is not None:
            raise ValueError("Checkpoint only outside complete updates/routes")
        binding.ownership(self, references)
        return {"schema": STATE_SCHEMA, "method": copy.deepcopy(metadata),
            "auxiliary_state": None if auxiliary is None else auxiliary.state_dict(),
            "table_record": None if table is None else table.record(),
            "fixed_weights": None if weights is None else weights.detach().cpu().clone(),
            "module_training": {n: m.training for n, m in self.model.named_modules()},
            "source_state": copy.deepcopy(source_state())}

    def load_state(self, value):
        if active[0] or (core._active_member if family == "node" else core._member) is not None \
                or value.get("schema") != STATE_SCHEMA or value.get("method") != metadata:
            raise ValueError("Combined source/recipe/site/seed/component/role binding changed")
        if value["table_record"] != (None if table is None else table.record()) or set(value["module_training"]) != {
                n for n, _ in self.model.named_modules()}:
            raise ValueError("Combined table or module identity changed")
        if weights is None:
            if value["fixed_weights"] is not None:
                raise ValueError("Unexpected responsibility weights")
        elif not torch.equal(value["fixed_weights"], weights.cpu()):
            raise ValueError("Fixed responsibility weights changed")
        if auxiliary is not None:
            auxiliary.load_state_dict(value["auxiliary_state"])
        elif value["auxiliary_state"] is not None:
            raise ValueError("Unexpected geometry state")
        source_weight_key = "responsibility" if family == "node" else "responsibilities"
        if value["source_state"][source_weight_key] is not None:
            raise ValueError("Interaction weights belong to its own state; native table slot must remain empty")
        source_load(copy.deepcopy(value["source_state"]))
        for n, module in self.model.named_modules():
            module.training = bool(value["module_training"][n])
        self._interaction_last_VALID = None
        binding.ownership(self, references)

    session.train_update, session.validate = types.MethodType(train_update, session), types.MethodType(validate, session)
    session.state, session.load_state = types.MethodType(state, session), types.MethodType(load_state, session)
    if family == "ogb":
        def metadata_method(self):
            return {**source_metadata(), "interaction": copy.deepcopy(metadata)}
        session.metadata = types.MethodType(metadata_method, session)
    return session


extend_session = transform
