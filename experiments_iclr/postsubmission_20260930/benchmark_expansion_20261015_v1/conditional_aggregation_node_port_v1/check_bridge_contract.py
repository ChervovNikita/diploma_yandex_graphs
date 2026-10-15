"""Synthetic exact NodeSession checkpoint path; no data/native constructor/fit."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent.parent
torch.set_num_threads(1)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rejects(call):
    try:
        call()
    except (ValueError, RuntimeError, TypeError):
        return
    raise AssertionError("Mismatched state/job unexpectedly accepted")


def main():
    extension = load(HERE / "transform.py", "_C6_node_bridge_test")
    redesign = PHASE / "reference_wrapper_redesign_20261013_v1"
    source = load(redesign / "smoke_20261014/node_port.py", "_C6_node_native_session_test")
    boundary = load(redesign / "runtime/boundary_core.py", "_C6_node_boundary_test")
    rules = load(redesign / "runtime/training_rules.py", "_C6_node_rules_test")

    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1, self.lin2, self.lin3 = torch.nn.Linear(5, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 3)
        def forward(self, data):
            value = self.lin1(torch.stack(data.list_mat, dim=1)).sum(1)
            value = torch.nn.functional.dropout(self.lin2(value).relu(), 0.2, self.training)
            return self.lin3(value)

    def build():
        session = source.NodeSession.__new__(source.NodeSession)
        session.torch, session.procedure, session.updates, session.device = torch, "gnnm_boundary_native", 0, "cpu"
        session.recipe = {"_redesign_root": str(redesign), "_repository_root": str(PHASE.parent), "_seed": 17,
                          "id": "synthetic_node_interface", "dataset": "synthetic", "native_fixed_config": {"hidden": 4}}
        x = torch.linspace(-1, 1, 60).reshape(12, 5)
        session.data = types.SimpleNamespace(x=x, list_mat=[x, x.cos(), x.sin()],
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]), num_nodes=12, y=torch.arange(12) % 3,
            train_mask=torch.arange(12) < 8, val_mask=(torch.arange(12) >= 8) & (torch.arange(12) < 11))
        session.source = types.SimpleNamespace(num_classes=3)
        session.train_ids = torch.arange(8)
        session.targets = session.data.y[session.train_ids]
        session.source_metadata = {"synthetic_model": True, "native_constructor_called": False}
        session.responsibility = None
        session.rules = rules
        native = Native()
        parameters = dict(native.named_parameters())
        session.optimizer = torch.optim.Adam([{"params": [p], "lr": 0.0005, "weight_decay": 1e-5}
                                             for p in native.parameters()])
        session.optimizers = [session.optimizer]
        session.core = boundary.attach_boundary_core(torch, native,
            [boundary.BoundaryBinding("lin1", "linear", "input"), boundary.BoundaryBinding("lin3", "linear", "output")],
            factor_seed=700018)
        boundary.extend_native_optimizer_groups(session.core, session.optimizer)
        session.model = session.core
        session.source_parameter_references = {"native." + name: parameter for name, parameter in parameters.items()}
        return session

    manifest = extension.receipt(HERE / "MANIFEST.json")["sha256"]
    outcomes = []
    states = {}
    for method in extension.METHODS:
        native = build()
        original_groups = tuple(native.optimizer.param_groups)
        original_options = [{k: copy.deepcopy(v) for k, v in group.items() if k != "params"} for group in original_groups]
        job = {"family": "node", "procedure": method, "base_procedure": "gnnm_boundary_native", "seed": 17,
            "recipe": copy.deepcopy(native.recipe), "policy": {"track": "engineering"}, "TEST_access": False,
            "method_settings": {"manifest_sha256": manifest}}
        native.model.eval()
        with torch.no_grad():
            reference = torch.stack([route.raw_logits[native.data.val_mask] for route in native._routes(False)]).mean(0)
        before_rng = torch.get_rng_state().clone()
        extension.transform(native, job)
        assert torch.equal(before_rng, torch.get_rng_state())
        assert torch.equal(native.validate()["outputs"], reference)
        calls, steps = [], []
        original_routes = native._routes
        def record_routes(training):
            calls.append(training)
            value = original_routes(training)
            if training:
                expected = rules.native_loss_terms(torch, [route.prediction for route in value], [route.raw_logits for route in value],
                    native_loss=lambda z: torch.nn.functional.cross_entropy(z[native.train_ids], native.targets),
                    pooled_native_loss=lambda z: torch.nn.functional.cross_entropy(z[native.train_ids], native.targets),
                    native_regularizer=value[0].raw_logits.new_zeros(()))
                steps.append({"source_expected_initial_J": float(expected["J"].detach())})
            return value
        native._routes = record_routes  # observation only after production source guard
        native.optimizer.step = lambda: steps.append({"no_op_step": True})
        result = native.train_update()
        assert calls == [True] and steps[-1]["no_op_step"] is True
        assert result["loss"] == steps[0]["source_expected_initial_J"]
        assert result["train_records"] == 8 and result["activation_checkpointing"] is True
        assert result["work"]["reverse_collections"] == 1 and native.core._active_member is None
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in native.model.parameters())
        assert all(group is old for group, old in zip(native.optimizer.param_groups, original_groups))
        assert [{k: v for k, v in group.items() if k != "params"} for group in native.optimizer.param_groups] == original_options
        gate = native.model.conditional_aggregation
        if hasattr(gate, "member_bias"):
            with torch.no_grad():
                gate.member_bias.copy_(gate.member_bias.new_tensor([-1., 0.5, 1., -0.2]))
        native.model.train()
        gate.eval()  # mixed module modes must survive the wrapped checkpoint
        state = native.state()
        states[method] = state
        valid = native.validate()
        assert valid["VALID_nodes"] == 3 and valid["VALID_node_ids"].tolist() == [8, 9, 10]
        assert valid["TEST_accessed"] is False and "actual declared served" in valid["selection_uses"]
        if method != "aggregation_mean_globalJ_v1":
            assert not torch.equal(valid["outputs"], valid["same_bank_mean_outputs"])
        if hasattr(gate, "member_bias"):
            with torch.no_grad():
                gate.member_bias.zero_()
        native.load_state(state)
        assert native.model.training and not gate.training
        assert torch.equal(native.validate()["outputs"], valid["outputs"])
        altered = copy.deepcopy(state)
        altered["method"]["source_data"]["TRAIN_ids"]["sha256"] = "incorrect"
        rejects(lambda: native.load_state(altered))
        with native.core.route(0):
            rejects(lambda: native.state())
            rejects(lambda: native.load_state(state))
        if method != "aggregation_mean_globalJ_v1":
            rejects(lambda: native.load_state(states["aggregation_mean_globalJ_v1"]))
        outcomes.append({"method": method, "parameters": gate.metadata()["parameters"],
            "source_checkpoint_path": True, "full_TRAIN_native_CE_globalJ_exact": True,
            "all_model_gradients_finite": True, "Adam_groups_options_identity": True,
            "actual_served_VALID_and_same_bank_mean": True, "state_modes_method_data_identity": True})
        native.core.remove_hooks()
    result = {"schema": "conditional-node-synthetic-bridge-check-v1", "passed": True, "torch": str(torch.__version__),
        "device": "CPU", "checks": outcomes, "frozen_C6_manifest": extension.C6_MANIFEST,
        "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                          for name in ("transform.py", "check_bridge_contract.py")},
        "native_model_constructions": 0, "native_data_reads": 0, "actual_optimizer_steps": 0,
        "synthetic_no_op_step_calls": 3, "TEST_accessed": False, "scientific_fits": 0,
        "native_qualification": "pending; synthetic model under actual frozen source route/session/boundary interfaces"}
    (HERE / "CHECKS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
