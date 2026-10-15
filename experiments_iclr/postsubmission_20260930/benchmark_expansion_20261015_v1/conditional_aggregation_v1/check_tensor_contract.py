"""Bounded synthetic tensor/interface checks; no native construction or data.

The molecular harness instantiates the frozen session class without its data
constructor and uses the actual frozen categorical boundary on a synthetic
embedding/linear model. Its evaluator is a recording stub and Adam.step is a
no-op counter. This is not native task qualification or a scientific fit.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
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


core = load(HERE / "core.py", "_conditional_core_check")
extension = load(HERE / "molecular_transform.py", "_conditional_transform_check")
CHECKS = []


def record(name, **detail):
    CHECKS.append({"name": name, "passed": True, **detail})


def close(actual, expected, *, atol=1e-12, rtol=1e-12):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def rejects(call):
    try:
        call()
    except (ValueError, TypeError, RuntimeError, KeyError):
        return
    raise AssertionError("An unsupported/mismatched input was accepted")


def values(task="binary", dtype=torch.float64, *, column=True):
    shape = (4, 7, 3) if task == "multiclass" else (4, 7, 1) if column else (4, 7)
    count = 1
    for size in shape:
        count *= size
    return (torch.arange(count, dtype=dtype).reshape(shape) * 0.71).sin().mul(3).requires_grad_()


def loss_for(task, *, column=True):
    if task == "multiclass":
        labels = torch.arange(7) % 3
        return lambda prediction: torch.nn.functional.cross_entropy(prediction, labels)
    labels = torch.tensor([0, 1, 1, 0, 0, 1, 0], dtype=torch.float64)
    if column:
        labels = labels[:, None]
    return lambda prediction: torch.nn.functional.binary_cross_entropy_with_logits(prediction, labels.to(prediction))


def perturb(gate):
    with torch.no_grad():
        if hasattr(gate, "member_bias"):
            gate.member_bias.copy_(gate.member_bias.new_tensor([-0.7, 0.2, 0.5, 1.1]))
        if hasattr(gate, "W"):
            gate.W.copy_(gate.W.new_tensor([[1, 0], [0, 1], [-1, 0.5], [0.6, -0.7]]))
            gate.a.copy_(gate.a.new_tensor([0.1, -0.2, 0.3, 0]))
            gate.v.copy_(gate.v.new_tensor([2, -0.3, 0.7, -1]))


def objective(z, gate, task="binary", *, column=True, regularizer=None, callback=None):
    loss = callback or loss_for(task, column=column)
    return core.native_objective(torch, native_member_losses=[loss(z[m]) for m in range(4)],
        raw_logits=z, aggregator=gate, served_native_loss=loss,
        native_regularizer=z.new_zeros(()) if regularizer is None else regularizer)


def initial_and_rng():
    cases = 0
    for dtype in (torch.float32, torch.float64):
        for task, column in (("multiclass", True), ("binary", True), ("binary", False)):
            z = values(task, dtype, column=column)
            loss = loss_for(task, column=column)
            regularizer = z.square().mean() * 0.013
            F = torch.stack([loss(z[m]) for m in range(4)]).mean() + regularizer
            L = loss(z.mean(0)) + regularizer
            baseline = 0.5 * F + 0.5 * L
            native_gradient = torch.autograd.grad(baseline, z, retain_graph=True)[0]
            for mode, count in (("mean", 0), ("global", 4), ("conditional", 20)):
                rng = torch.get_rng_state().clone()
                gate = core.make_aggregator(torch, mode=mode, task_kind=task, seed=17, reference=z)
                assert torch.equal(rng, torch.get_rng_state())
                assert sum(p.numel() for p in gate.parameters()) == count
                terms = objective(z, gate, task, column=column, regularizer=regularizer)
                assert torch.equal(terms["served_raw_logits"], z.mean(0))
                assert torch.equal(terms["weights"], z.new_full((4, 7), 0.25))
                assert torch.equal(terms["F"], F) and torch.equal(terms["L"], L) and torch.equal(terms["J"], baseline)
                assert torch.equal(torch.autograd.grad(terms["J"], z, retain_graph=True)[0], native_gradient)
                if mode == "conditional":
                    gradients = torch.autograd.grad(terms["J"], tuple(gate.parameters()), retain_graph=True)
                    assert all(torch.isfinite(g).all() for g in gradients)
                    named = dict(zip(dict(gate.named_parameters()), gradients))
                    assert torch.count_nonzero(named["W"]) == 0 and torch.count_nonzero(named["a"]) == 0
                    assert named["v"].abs().sum() > 0 and named["member_bias"].abs().sum() > 0
                cases += 1
    record("exact_initial_mean_F_L_J_native_gradients_and_isolated_RNG", cases=cases,
           parameter_counts={"mean": 0, "global": 4, "conditional": 20}, expected_initial_zero_gradients=["W", "a"])


def weights_and_shift():
    z = values()
    variation = {}
    for mode in core.MODES:
        gate = core.make_aggregator(torch, mode=mode, task_kind="binary", seed=17, reference=z)
        perturb(gate)
        out = gate(z)
        weights = out["weights"]
        close(weights.sum(0), torch.ones(7, dtype=z.dtype))
        assert (weights >= 0.05).all() and torch.isfinite(weights).all()
        close(out["served_raw_logits"], (weights[:, :, None] * z).sum(0))
        assert ((out["served_raw_logits"] >= z.min(0).values - 1e-12)
                & (out["served_raw_logits"] <= z.max(0).values + 1e-12)).all()
        variation[mode] = float((weights - weights[:, :1]).abs().max())
        if mode in ("mean", "global"):
            assert variation[mode] == 0
        else:
            assert variation[mode] > 0.01
            assert out["descriptors"].shape == (4, 7, 2) and not out["descriptors"].requires_grad
        rejects(lambda: gate(z, labels=torch.ones(7)))
    z = values("multiclass")
    gate = core.make_aggregator(torch, mode="conditional", task_kind="multiclass", seed=17, reference=z)
    perturb(gate)
    shifts = torch.linspace(-12, 12, 28, dtype=z.dtype).reshape(4, 7, 1)
    close(gate(z)["weights"], gate(z + shifts)["weights"])
    close(gate(z)["descriptors"], gate(z + shifts)["descriptors"])
    record("convex_positive_global_vs_conditional_and_multiclass_shift_invariance", query_weight_variation=variation)


def gradients_and_extremes():
    differences, finite_difference_errors = {}, []
    for task in core.TASKS:
        z = values(task)
        gate = core.make_aggregator(torch, mode="conditional", task_kind=task, seed=17, reference=z)
        perturb(gate)
        terms = objective(z, gate, task)
        gradient = torch.autograd.grad(terms["J"], z, retain_graph=True)[0]
        weights = terms["weights"].detach()
        fixed_served = z.mean(0) + ((weights[:, :, None] - 0.25) * z).sum(0)
        own = torch.stack([loss_for(task)(z[m]) for m in range(4)]).mean()
        fixed_objective = 0.5 * own + 0.5 * loss_for(task)(fixed_served)
        close(gradient, torch.autograd.grad(fixed_objective, z, retain_graph=True)[0])
        own_gradient = torch.autograd.grad(own, z, retain_graph=True)[0]
        differences[task] = float((gradient - own_gradient).abs().max())
        assert differences[task] > 1e-4
        parameters = dict(gate.named_parameters())
        gradients = torch.autograd.grad(terms["J"], tuple(parameters.values()), retain_graph=True)
        assert all(torch.isfinite(g).all() for g in gradients)
        for (name, parameter), derivative in zip(parameters.items(), gradients):
            index = (0, 0) if parameter.ndim == 2 else (0,)
            original = float(parameter[index].detach())
            epsilon = 1e-5
            with torch.no_grad():
                parameter[index] = original + epsilon
            plus = float(objective(z, gate, task)["J"].detach())
            with torch.no_grad():
                parameter[index] = original - epsilon
            minus = float(objective(z, gate, task)["J"].detach())
            with torch.no_grad():
                parameter[index] = original
            error = abs((plus - minus) / (2 * epsilon) - float(derivative[index]))
            assert error < 2e-9, (task, name, error)
            finite_difference_errors.append(error)
        extreme = z.detach().sign().mul(1000).requires_grad_()
        extreme_terms = objective(extreme, gate, task)
        extreme_gradients = torch.autograd.grad(extreme_terms["J"], (extreme, *gate.parameters()))
        assert all(torch.isfinite(g).all() for g in extreme_gradients)
        assert torch.isfinite(extreme_terms["descriptors"]).all()
    record("detached_descriptor_native_gradient_and_live_gate_derivatives", difference_from_own_only_gradient=differences,
           gate_finite_difference_max_error=max(finite_difference_errors),
           finite_difference_scope="gate parameters only; no stop-gradient logits finite-difference mismatch",
           extreme_logits=[-1000, 1000], all_gradients_finite=True)


def native_reductions():
    z = values()
    gate = core.make_aggregator(torch, mode="conditional", task_kind="binary", seed=17, reference=z)
    perturb(gate)
    labels = torch.tensor([1, 1, 1, 1, 1, 0, 0], dtype=z.dtype)[:, None]
    observed = torch.tensor([True, False, True, True, False, True, True])
    callbacks = {
        "molecular_observed_label_BCE": lambda p: torch.nn.functional.binary_cross_entropy_with_logits(p[observed], labels[observed]),
        "NCN_NCNC_separate_positive_negative_means": lambda p: -torch.nn.functional.logsigmoid(p[:5]).mean()
            - torch.nn.functional.logsigmoid(-p[5:]).mean(),
        "BUDDY_joint_mixed_BCE_mean": lambda p: torch.nn.functional.binary_cross_entropy_with_logits(p, labels)}
    actual = {}
    for name, callback in callbacks.items():
        regularizer = z.square().mean() * 0.13
        calls = []
        def served(p):
            calls.append(p)
            return callback(p)
        terms = core.native_objective(torch, native_member_losses=[callback(z[m]) for m in range(4)],
            raw_logits=z, aggregator=gate, served_native_loss=served, native_regularizer=regularizer)
        assert len(calls) == 1 and calls[0] is terms["served_raw_logits"]
        close(terms["F"], torch.stack([callback(z[m]) for m in range(4)]).mean() + regularizer)
        close(terms["L"], callback(terms["served_raw_logits"]) + regularizer)
        close(terms["J"], 0.5 * (terms["F"] - regularizer) + 0.5 * (terms["L"] - regularizer) + regularizer)
        actual[name] = float(terms["L"].detach())
    assert abs(actual["NCN_NCNC_separate_positive_negative_means"] - actual["BUDDY_joint_mixed_BCE_mean"]) > 0.1
    # Irregular microbatches must not silently replace the full-group mean.
    buddy = callbacks["BUDDY_joint_mixed_BCE_mean"]
    uneven_microbatch_average = 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(z[0, :2], labels[:2]) \
        + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(z[0, 2:], labels[2:])
    assert abs(float(buddy(z[0]) - uneven_microbatch_average)) > 1e-3
    record("caller_native_reductions_actual_served_callback_and_regularizer_once", values=actual,
           groups="whole native groups; NCN5positive+2negative synthetic unequal-count stress case")


def state_and_rejections():
    z = values()
    gate = core.make_aggregator(torch, mode="conditional", task_kind="binary", seed=17, reference=z)
    perturb(gate)
    payload = io.BytesIO()
    torch.save(gate.state_dict(), payload)
    payload.seek(0)
    saved = torch.load(payload, weights_only=True)
    clone = core.make_aggregator(torch, mode="conditional", task_kind="binary", seed=17, reference=z)
    clone.load_state_dict(saved, strict=True)
    assert torch.equal(gate(z)["served_raw_logits"], clone(z)["served_raw_logits"])
    for mode, task, seed in (("global", "binary", 17), ("conditional", "multiclass", 17), ("conditional", "binary", 18)):
        other = core.make_aggregator(torch, mode=mode, task_kind=task, seed=seed, reference=z)
        rejects(lambda: other.load_state_dict(saved, strict=True))
    rejects(lambda: gate(z[:3]))
    rejects(lambda: gate([z[0], z[1], z[2], z[3, :-1]]))
    rejects(lambda: gate(z.to(torch.float32)))
    rejects(lambda: gate(torch.full_like(z, float("nan"))))
    rejects(lambda: gate(z.to(torch.float16)))
    rejects(lambda: core.native_objective(torch, native_member_losses=[z.new_tensor(0.)] * 4,
        raw_logits=z, aggregator=gate, served_native_loss=lambda p: p.sum(), native_regularizer=z.new_zeros(())))
    record("strict_module_checkpoint_identity_shapes_precision_and_live_native_loss_contract")


def molecular_synthetic_interface():
    port = PHASE / "reference_wrapper_redesign_20261013_v1/smoke_20261014/ogb_port"
    source = load(port / "session.py", "_conditional_source_session_check")
    boundary = load(port / "categorical_boundary.py", "_conditional_source_boundary_check")

    class Atom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_embedding_list = torch.nn.ModuleList([torch.nn.Embedding(3, 300) for _ in range(9)])
        def forward(self, x):
            return sum(table(x[:, index]) for index, table in enumerate(self.atom_embedding_list))

    class SyntheticNative(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gnn_node = torch.nn.Module()
            self.gnn_node.atom_encoder = Atom()
            self.graph_pred_linear = torch.nn.Linear(300, 1)
        def forward(self, batch):
            return self.graph_pred_linear(self.gnn_node.atom_encoder(batch.x))

    class Batch:
        def __init__(self, ids):
            self.x = (torch.arange(len(ids) * 9).reshape(len(ids), 9) + torch.arange(len(ids))[:, None]) % 3
            self.y = torch.tensor([0., 1., float("nan"), 0.])[:len(ids), None]
            self.batch = torch.arange(len(ids))
            self.smoke_graph_id = torch.tensor(ids)
        def to(self, device):
            for name, value in vars(self).items():
                setattr(self, name, value.to(device))
            return self

    class Sampler:
        def __init__(self):
            self.cursor = 0
        def next(self):
            self.cursor += 4
            return Batch([10, 11, 12, 13]), torch.arange(4)
        def state(self):
            return {"cursor": self.cursor}
        def load_state(self, value):
            self.cursor = int(value["cursor"])

    class EvaluatorProbe:
        def __init__(self):
            self.seen = []
        def eval(self, value):
            self.seen.append(copy.deepcopy(value))
            return {"synthetic_recording_probe": float(value["y_pred"].mean())}

    def build():
        owner = source.OGBSmokeSession.__new__(source.OGBSmokeSession)
        owner.torch, owner.recipe, owner.procedure = torch, {}, "gnnm_boundary_native"
        owner.config = {"feature": "full", "gnn": "synthetic_interface_not_native_GNN"}
        owner.redesign_root, owner.device = extension.REDESIGN, torch.device("cpu")
        native = SyntheticNative()
        original = tuple(native.named_parameters())
        owner.optimizer = torch.optim.Adam(native.parameters(), lr=0.001)
        owner.optimizers = [owner.optimizer]
        owner.model = boundary.attach_ogb_boundary(torch, native, factor_seed=2000003)
        owner.optimizer_assignment = boundary.attach_native_optimizer_groups(owner.model, owner.optimizer)
        owner.source_parameter_references = {"native." + name: parameter for name, parameter in original}
        owner.train_ids, owner.valid_ids = torch.arange(10, 14), torch.arange(20, 24)
        owner.train_view, owner.valid_view = list(owner.train_ids), list(owner.valid_ids)
        owner.sampler, owner.valid_generator = Sampler(), torch.Generator().manual_seed(23)
        owner.valid_loader = [Batch([99]), Batch([20, 21, 22, 23])]
        owner.criterion = torch.nn.BCEWithLogitsLoss()
        owner.evaluator = EvaluatorProbe()
        owner.source_pins = {"synthetic_model": "no native model construction"}
        owner.rules_pin, owner.encoding_dependencies, owner.split_pins = {}, {}, {}
        owner.updates, owner.last_train_positions = 0, None
        owner.full_responsibilities, owner.responsibility_metadata = None, None
        owner.synthetic_step_calls = 0
        def no_step(optimizer, closure=None):
            owner.synthetic_step_calls += 1
        owner.optimizer.step = types.MethodType(no_step, owner.optimizer)
        return owner

    manifest_hash = hashlib.sha256((HERE / "MANIFEST.json").read_bytes()).hexdigest()
    def job(method):
        return {"procedure": method, "family": "ogb", "base_procedure": "gnnm_boundary_native", "seed": 17,
                "TEST_access": False, "policy": {"track": "extended_convergence"},
                "recipe": {"_repository_root": str(PHASE.parent)},
                "method_settings": {"manifest_sha256": manifest_hash}}
    controls = []
    for method in extension.METHODS:
        owner = build()
        refs = dict(owner.source_parameter_references)
        original_options = [{k: copy.deepcopy(v) for k, v in group.items() if k != "params"}
                            for group in owner.optimizer.param_groups]
        extension.transform(owner, job(method))
        gate = owner.model.conditional_aggregation
        perturb(gate)
        before = copy.deepcopy(owner.model.state_dict())
        result = owner.train_update()
        assert result["optimizer_updates"] == 1 and owner.synthetic_step_calls == 1
        assert result["work"]["reverse_collections"] == 1 and result["native_route_forwards"] == 4
        assert result["native_skipped_batches"] == 0 and result["TRAIN_graph_ids"] == [10, 11, 12, 13]
        for name, value in owner.model.state_dict().items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, before[name])  # no-op Adam.step, no numerical update
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in owner.model.parameters())
        after_options = [{k: v for k, v in group.items() if k != "params"} for group in owner.optimizer.param_groups]
        assert original_options == after_options
        state = owner.state()
        assert all(state["source_state"]["module_training"].values())
        output = owner.validate()
        assert output["VALID_scope"] == "complete_source_VALID" and output["VALID_graph_ids"] == [20, 21, 22, 23]
        assert output["native_skipped_batches"] == 1 and output["TEST_accessed"] is False
        batch = Batch([20, 21, 22, 23])
        with torch.no_grad():
            raw = [owner.model.run_member(m, batch)[1] for m in range(4)]
            expected = gate(raw)["served_raw_logits"]
            assert torch.equal(output["outputs"], expected)
            assert torch.equal(output["same_bank_mean_outputs"], torch.stack(raw).mean(0))
            mask = batch.y == batch.y
            close(torch.tensor(output["loss"]), torch.nn.functional.binary_cross_entropy_with_logits(
                expected[mask], batch.y[mask]), atol=1e-6, rtol=1e-6)
        assert len(owner.evaluator.seen) == 2
        assert (owner.evaluator.seen[0]["y_pred"] == output["outputs"].numpy()).all()
        if method != "aggregation_mean_globalJ_v1":
            assert not torch.equal(output["outputs"], output["same_bank_mean_outputs"])
        with torch.no_grad():
            if hasattr(gate, "member_bias"):
                gate.member_bias.add_(3)
            next(owner.model.native.parameters()).add_(1)
        owner.load_state(state)
        assert owner.updates == 1 and owner.sampler.cursor == 4
        assert all(module.training for module in owner.model.modules())
        assert all(dict(owner.model.named_parameters())[name] is parameter for name, parameter in refs.items())
        assert torch.equal(owner.validate()["outputs"], output["outputs"])
        wrong = copy.deepcopy(state)
        wrong["method"]["method_id"] = "wrong_method"
        rejects(lambda: owner.load_state(wrong))
        with owner.model.route(0):
            rejects(lambda: owner.state())
            rejects(lambda: owner.load_state(state))
        rejects(lambda: extension.transform(owner, job(method)))
        controls.append({"method": method, "parameters": gate.metadata()["parameters"],
                         "no_op_step_calls": owner.synthetic_step_calls, "served_validation_verified": True})
        owner.close()
    bad = job("aggregation_conditional_globalJ_v1")
    bad["family"] = "link"
    rejected = build()
    rejects(lambda: extension.transform(rejected, bad))
    rejected.close()
    record("molecular_transform_synthetic_frozen_interface_TRAIN_VALID_checkpoint_and_ownership", controls=controls,
           source_session_constructor_called=False, model="synthetic embeddings+linear, frozen boundary hooks",
           evaluator="recording stub; no OGB score", actual_optimizer_steps=0, no_op_step_calls=3,
           native_task_qualification=False)


def main():
    initial_and_rng()
    weights_and_shift()
    gradients_and_extremes()
    native_reductions()
    state_and_rejections()
    molecular_synthetic_interface()
    source_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                     for name in ("core.py", "molecular_transform.py", "check_tensor_contract.py", "RECIPE.md")}
    result = {"schema": "conditional-aggregation-local-checks-v1", "passed": True,
              "run_utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
              "device": "CPU", "checks": CHECKS, "package_source_sha256": source_hashes,
              "scientific_fits": 0, "native_model_constructions": 0, "native_data_reads": 0,
              "actual_optimizer_steps": 0, "synthetic_no_op_step_calls": 3, "server_operations": 0,
              "TEST_accessed": False, "native_task_qualification": "pending"}
    (HERE / "CHECKS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
