"""CPU synthetic checks of MNIST graph adapters and prospective pilot guards.

No real dataset, optimizer step, validation score, or CUDA context is used.
"""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

import mnist_adapters as adapter
import mnist_pilot as pilot


def expect_error(fn, types=(ValueError, RuntimeError)):
    try:
        fn()
    except types:
        return
    raise AssertionError("Expected guard rejection")


def synthetic_batch():
    first = Data(x=torch.randn(3, 3), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 0, 2]]),
                 edge_attr=torch.randn(4, 1), y=torch.tensor([2]))
    second = Data(x=torch.randn(5, 3), edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
                  edge_attr=torch.randn(4, 1), y=torch.tensor([7]))
    return Batch.from_data_list([first, second]), [first, second]


def check_features_readout_loss():
    graph = Data(x=torch.tensor([[0.2], [0.5]]), pos=torch.tensor([[1.0, 2.0], [4.0, 5.0]]),
                 edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.tensor([[0.3], [0.3]]))
    edges, edge_attr, pos = graph.edge_index.clone(), graph.edge_attr.clone(), graph.pos.clone()
    transformed = pilot.add_positions(graph)
    assert torch.equal(transformed.x, torch.tensor([[0.2, 1.0, 2.0], [0.5, 4.0, 5.0]]))
    assert torch.equal(edges, transformed.edge_index) and torch.equal(edge_attr, transformed.edge_attr)
    assert torch.equal(pos, transformed.pos)
    nodes = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]],
                          [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]], requires_grad=True)
    pooled = adapter.PATHS.graph_mean_logits(nodes, torch.tensor([0, 0, 1]), 2)
    assert torch.equal(pooled, torch.stack((nodes[:, :2].mean(1), nodes[:, 2]), dim=1))
    targets = torch.tensor([0, 1])
    actual = pilot.graph_member_ce(pooled, targets)
    manual = (F.cross_entropy(pooled[:, 0], targets[0].expand(2))
              + F.cross_entropy(pooled[:, 1], targets[1].expand(2))) / 2
    assert torch.allclose(actual, manual)
    actual.backward()
    assert torch.isfinite(nodes.grad).all()
    expect_error(lambda: pilot.graph_member_ce(pooled, targets.float()))
    expect_error(lambda: pilot.graph_member_ce(pooled, torch.tensor([0, 2])))


def check_adapters():
    batch, graphs = synthetic_batch()
    options = adapter.graph_options(hidden_dim=16, rank=4, layers=3, dropout=0)
    models, errors = {}, {}
    for arm in adapter.ARMS:
        torch.manual_seed(501)
        model, definition = adapter.make_graph_model(arm, options)
        model.eval()
        logits = model(batch)
        assert logits.shape == (model.members, 2, 10) and torch.isfinite(logits).all()
        separate = torch.cat([model(graph) for graph in graphs], dim=1)
        assert torch.allclose(logits, separate, atol=1e-10, rtol=1e-10)
        assert torch.allclose(model.pooled_forward(batch), logits.mean(0), atol=1e-12, rtol=1e-12)
        loss = pilot.graph_member_ce(logits, batch.y.reshape(-1))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), arm
        if arm.endswith("no_feedback"):
            assert model.node_model.expansion is None
            assert not any(name.endswith("expansion") for name, _ in model.named_parameters())
        models[arm] = model
        errors[arm] = float((logits - separate).abs().max())
    tied = models["gnnm"]
    for arm in ["untied", "first_private", "middle_private", "last_private"]:
        assert torch.allclose(tied(batch), models[arm](batch), atol=1e-10, rtol=1e-10)
    private = models["untied"].node_model
    for layer, block in enumerate(tied.node_model.model.residual_modules):
        copied = [tuple(private.private_blocks[str(layer)][member].parameters()) for member in range(4)]
        for index, parameter in enumerate(block.parameters()):
            summed = torch.stack([parameters[index].grad for parameters in copied]).sum(0)
            assert torch.allclose(parameter.grad, summed, atol=1e-9, rtol=1e-9)
    for source, packed in [("gnnm", "packed_gnnm"), ("ens", "packed_full")]:
        error = float((models[source](batch) - models[packed](batch)).abs().max())
        assert error < 1e-10
        errors[source + "_vs_" + packed] = error
    # Four-cell shared and narrow draws agree; feedback-disabled weights are absent.
    reference = models["centered_feedback"].node_model.state_dict()
    for arm in ["centered_no_feedback", "uncentered_feedback", "uncentered_no_feedback"]:
        state = models[arm].node_model.state_dict()
        for name, value in state.items():
            assert torch.equal(value, reference[name]), (arm, name)
    return errors


def check_plan():
    before = torch.get_rng_state()
    plan = adapter.arm_plan()
    assert torch.equal(before, torch.get_rng_state())
    for arm in ["centered_feedback", "uncentered_feedback"]:
        assert plan[arm]["parameters"] == 281006
    for arm in ["centered_no_feedback", "uncentered_no_feedback"]:
        assert plan[arm]["parameters"] == 278958 and plan[arm]["feedback_parameter_present"] is False
    assert plan["packed_channels"]["hidden_dim"] == 44
    assert plan["packed_channels"]["adjacency_channels_per_layer"] == 176
    assert plan["packed_parameters"]["hidden_dim"] == 67 and plan["packed_parameters"]["parameters"] == 277688
    assert plan["base_parameters"]["hidden_dim"] == 136 and plan["base_parameters"]["parameters"] == 281666
    return plan


def check_budgets():
    train = [SimpleNamespace(y=torch.tensor([i % 10])) for i in range(100)]
    official = torch.arange(100) * 3 + 7
    before = torch.get_rng_state()
    budgets, summary = pilot.train_budgets(train, official)
    repeated, other = pilot.train_budgets(train, official)
    assert torch.equal(before, torch.get_rng_state())
    assert len(budgets["0.1"]) == 10 and len(budgets["1.0"]) == 100
    assert summary["budgets"]["0.1"]["class_counts"] == [1] * 10
    assert set(budgets["0.1"].tolist()) <= set(budgets["1.0"].tolist())
    assert summary == other and torch.equal(budgets["0.1"], repeated["0.1"])
    assert summary["budgets"]["0.1"]["official_graph_indices_sha256"] == pilot.tensor_hash(official[budgets["0.1"]])
    expect_error(lambda: pilot.train_budgets([SimpleNamespace(y=torch.tensor([10]))], torch.tensor([0])))


def check_selection_and_gpu():
    metric = lambda accuracy, ce: {"accuracy": accuracy, "pooled_ce": ce, "mean_member_ce": ce + 0.1}
    assert pilot.selection_key(metric(0.81, 2.0), 10) > pilot.selection_key(metric(0.80, 0.1), 1)
    assert pilot.selection_key(metric(0.80, 0.2), 10) > pilot.selection_key(metric(0.80, 0.3), 1)
    assert pilot.selection_key(metric(0.80, 0.2), 1) > pilot.selection_key(metric(0.80, 0.2), 10)
    expect_error(lambda: pilot.selection_key(metric(float("nan"), 0.2), 1))
    assert pilot.gpu_guard(SimpleNamespace(device="cpu"))["gpu_context_created"] is False
    args = SimpleNamespace(device="cuda:0", physical_gpu=3)
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
        replies = [SimpleNamespace(stdout="GPU-synthetic, Fake, 0, 16384, 0"), SimpleNamespace(stdout="")]
        with patch.object(pilot.subprocess, "run", side_effect=replies):
            assert pilot.gpu_guard(args)["active_compute_apps_before_launch"] == 0
        with patch.object(pilot.subprocess, "run", return_value=SimpleNamespace(stdout="GPU-synthetic, Fake, 800, 16384, 0")):
            expect_error(lambda: pilot.gpu_guard(args))
        replies = [SimpleNamespace(stdout="GPU-synthetic, Fake, 0, 16384, 0"), SimpleNamespace(stdout="GPU-synthetic")]
        with patch.object(pilot.subprocess, "run", side_effect=replies):
            expect_error(lambda: pilot.gpu_guard(args))
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}):
        expect_error(lambda: pilot.gpu_guard(args))


def check_protocol(plan):
    args = SimpleNamespace(arm="centered_feedback", seed=0, label_fraction=0.1)
    env, sources, summary = {"synthetic": True}, {"synthetic.py": "hash"}, {"synthetic_budget": True}
    train, val = torch.tensor([1, 3]), torch.tensor([2, 4])
    with tempfile.TemporaryDirectory(prefix="synthetic_mnist_verify_", dir=pilot.ALLOWED) as directory:
        directory = Path(directory)
        data = directory / "data_manifest.json"
        data.write_text("{}")
        payload = {"schema": pilot.SCHEMA, "status": "prospective_protocol_not_fit",
                   "recipe": pilot.RECIPE, "source_sha256": sources,
                   "data_manifest_sha256": pilot.sha(data), "environment_sha256": pilot.json_hash(env),
                   "train_budget_manifest": summary,
                   "official_train_indices_sha256": pilot.tensor_hash(train),
                   "official_validation_indices_sha256": pilot.tensor_hash(val),
                   "arms": ["centered_feedback"], "seeds": [0],
                   "comparison_plan": {"centered_feedback": plan["centered_feedback"]}}
        path = directory / "protocol.json"
        path.write_text(json.dumps(payload))
        pilot.check_protocol(path, args, sources, env, data, summary, train, val)
        expect_error(lambda: pilot.check_protocol(path, args, {"different.py": "hash"}, env, data, summary, train, val))
        expect_error(lambda: pilot.check_protocol(path, SimpleNamespace(arm="ens", seed=0, label_fraction=0.1), sources, env, data, summary, train, val))
        changed = {**payload, "comparison_plan": {"centered_feedback": {**plan["centered_feedback"], "rank": 17}}}
        path.write_text(json.dumps(changed))
        expect_error(lambda: pilot.check_protocol(path, args, sources, env, data, summary, train, val))


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(240930)
    check_features_readout_loss()
    errors = check_adapters()
    plan = check_plan()
    check_budgets()
    check_selection_and_gpu()
    check_protocol(plan)
    print(json.dumps({"status": "passed", "checks": ["features_edges_unchanged", "mean_node_logits_equal_graph_CE",
                     "four_cells_finite_registered_gradients", "batched_vs_separate_graphs",
                     "tied_private_initial_gradient_aggregation", "source_packed_eval_equivalence",
                     "fixed_parameter_channel_plan", "nested_train_only_budgets",
                     "accuracy_CE_earliest_selection", "idle_selected_GPU_guard", "prospective_protocol_binding"],
                     "max_logit_errors": errors, "comparison_plan": plan,
                     "dataset_accessed": False, "CUDA_context_created": False,
                     "optimizer_step_or_real_scoring_launched": False}, indent=2))
