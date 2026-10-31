"""CPU synthetic checks of MolHIV pilot loss/selection/budget/launch guards.

No dataset, CUDA context, fitted model, or real validation score is used.
"""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

import molhiv_pilot as pilot


def expect_error(fn, types=(ValueError, RuntimeError)):
    try:
        fn()
    except types:
        return
    raise AssertionError("Expected a guard rejection")


def check_loss():
    logits = torch.tensor([[[0.3, -1.4, 0.8], [2.0, -0.6, 0.1]],
                           [[-0.1, 0.2, -0.5], [1.0, 0.4, -1.0]]], dtype=torch.float64,
                          requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, float("nan")], [0.0, float("nan"), float("nan")]], dtype=torch.float64)
    # Each graph receives equal weight although it has a different label count.
    first = F.binary_cross_entropy_with_logits(logits[:, 0, :2], targets[0, :2].expand(2, 2))
    second = F.binary_cross_entropy_with_logits(logits[:, 1, 0], targets[1, 0].expand(2))
    actual = pilot.graph_member_bce(logits, targets)
    assert torch.allclose(actual, (first + second) / 2)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[:, 0, 2], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(logits.grad[:, 1, 1:], torch.zeros((2, 2), dtype=torch.float64))
    expect_error(lambda: pilot.graph_member_bce(logits, torch.full_like(targets, float("nan"))))


def check_budgets():
    train = [SimpleNamespace(y=torch.tensor([[float(i % 7 == 0)]])) for i in range(100)]
    official = torch.arange(100) * 3 + 7
    before = torch.get_rng_state()
    budgets, summary = pilot.train_budgets(train, official)
    repeated, other = pilot.train_budgets(train, official)
    assert torch.equal(before, torch.get_rng_state())
    assert len(budgets["0.1"]) == 10 and len(budgets["1.0"]) == 100
    assert set(budgets["0.1"].tolist()) <= set(budgets["1.0"].tolist())
    assert summary == other and torch.equal(budgets["0.1"], repeated["0.1"])
    assert summary["budgets"]["0.1"]["positive"] > 0
    assert summary["budgets"]["0.1"]["official_graph_indices_sha256"] == pilot.tensor_hash(official[budgets["0.1"]])
    expect_error(lambda: pilot.train_budgets([SimpleNamespace(y=torch.tensor([[float("nan")]]))], torch.tensor([0])))


def check_selection():
    metric = lambda auc, bce: {"rocauc": auc, "pooled_bce": bce, "mean_member_bce": bce + 0.1}
    assert pilot.selection_key(metric(0.81, 2.0), 10) > pilot.selection_key(metric(0.80, 0.1), 1)
    assert pilot.selection_key(metric(0.80, 0.2), 10) > pilot.selection_key(metric(0.80, 0.3), 1)
    assert pilot.selection_key(metric(0.80, 0.2), 1) > pilot.selection_key(metric(0.80, 0.2), 10)
    expect_error(lambda: pilot.selection_key(metric(float("nan"), 0.2), 1))


def check_gpu_guard():
    assert pilot.gpu_guard(SimpleNamespace(device="cpu"))["gpu_context_created"] is False
    args = SimpleNamespace(device="cuda:0", physical_gpu=3)
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
        replies = [SimpleNamespace(stdout="GPU-unit-test, Fake, 0, 16384, 0"), SimpleNamespace(stdout="")]
        with patch.object(pilot.subprocess, "run", side_effect=replies) as call:
            assert pilot.gpu_guard(args)["active_compute_apps_before_launch"] == 0
            assert call.call_count == 2
        with patch.object(pilot.subprocess, "run", return_value=SimpleNamespace(stdout="GPU-unit-test, Fake, 800, 16384, 0")) as call:
            expect_error(lambda: pilot.gpu_guard(args))
            assert call.call_count == 1
        replies = [SimpleNamespace(stdout="GPU-unit-test, Fake, 0, 16384, 0"), SimpleNamespace(stdout="GPU-unit-test")]
        with patch.object(pilot.subprocess, "run", side_effect=replies):
            expect_error(lambda: pilot.gpu_guard(args))
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}):
        expect_error(lambda: pilot.gpu_guard(args))


def check_protocol():
    args = SimpleNamespace(arm="middle_private", seed=0, label_fraction=0.1)
    env, sources, summary = {"unit_test": True}, {"unit_test.py": "unit-hash"}, {"synthetic_budget": True}
    train, val = torch.tensor([1, 3]), torch.tensor([2, 4])
    with tempfile.TemporaryDirectory(prefix="synthetic_pilot_verify_", dir=pilot.ALLOWED) as directory:
        directory = Path(directory)
        data = directory / "data_manifest.json"
        data.write_text("{}")
        payload = {"schema": pilot.SCHEMA, "status": "prospective_protocol_not_fit",
                   "recipe": pilot.RECIPE, "source_sha256": sources,
                   "data_manifest_sha256": pilot.sha(data), "environment_sha256": pilot.json_hash(env),
                   "train_budget_manifest": summary,
                   "official_train_indices_sha256": pilot.tensor_hash(train),
                   "official_validation_indices_sha256": pilot.tensor_hash(val),
                   "arms": ["middle_private"], "seeds": [0]}
        path = directory / "protocol.json"
        path.write_text(json.dumps(payload))
        pilot.check_protocol(path, args, sources, env, data, summary, train, val)
        expect_error(lambda: pilot.check_protocol(path, args, {"different.py": "hash"}, env, data, summary, train, val))
        expect_error(lambda: pilot.check_protocol(path, SimpleNamespace(arm="ens", seed=0, label_fraction=0.1), sources, env, data, summary, train, val))
        changed = dict(payload)
        changed["recipe"] = {**pilot.RECIPE, "epochs": 101}
        path.write_text(json.dumps(changed))
        expect_error(lambda: pilot.check_protocol(path, args, sources, env, data, summary, train, val))


if __name__ == "__main__":
    torch.set_num_threads(1)
    check_loss()
    check_budgets()
    check_selection()
    check_gpu_guard()
    check_protocol()
    print(json.dumps({"status": "passed", "checks": ["equal_graph_masked_member_BCE", "nested_train_only_budgets",
                     "ROC_AUC_BCE_earliest_selection", "idle_selected_GPU_guard", "prospective_protocol_binding"],
                     "dataset_accessed": False, "CUDA_context_created": False, "training_or_scoring_launched": False}, indent=2))
