"""Pinned PolyFormer source/adapter engineering tests; no benchmark data."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("GNNM_REFERENCE_PROJECT", str(PACKAGE.parent)))
sys.path.insert(0, str(PACKAGE))
from adapters import polyformer
from engine import capture_checkpoint, restore_checkpoint, run_trajectory

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

SOURCES_PRESENT = all((PROJECT / row["path"]).is_file() for row in polyformer.SOURCE_FILES.values())
PYG_PRESENT = importlib.util.find_spec("torch_geometric") is not None
SMALL_NATIVE = {"K": 2, "hidden": 8, "nlayer": 1, "n_head": 2, "d_ffn": 4,
                "q": 1.6, "multi": 2., "dropout": .2, "dprate": .2}


@unittest.skipIf(torch is None or not SOURCES_PRESENT or not PYG_PRESENT,
                 "Needs existing Torch/PyG and four pinned sources; do not install a runtime")
class PolyFormerTests(unittest.TestCase):
    def tokens(self, features=3):
        x = torch.arange(4 * features, dtype=torch.float32).reshape(4, features) / max(1, features)
        return [x, x * .5, x * .25]

    def bank(self, *, members=1, factorized=False, native=SMALL_NATIVE, classes=2, features=3, seed=1009):
        return polyformer.build_bank(torch, PROJECT, self.tokens(features), native=native, classes=classes,
                                     dataset="engineering-only", members=members, factorized=factorized,
                                     initialization_seed=seed, dropout_seeds=[301010, 302019, 303028, 304037])

    def fixture(self, directory, *, method="polyformer_single", views=1, loss=None):
        x = torch.tensor([[1., 0., .2], [0., 1., .1], [1., 1., .3], [-1., 0., .1]])
        edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
        paths = {}
        for name, value in {"graph": {"x": x, "edge_index": edges},
                            "train": {"ids": torch.tensor([0, 1]), "y": torch.tensor([0, 1])},
                            "select": {"ids": torch.tensor([2, 3]), "y": torch.tensor([0, 1])}}.items():
            path = directory / (name + ".pt")
            torch.save(value, path)
            paths[name] = str(path.relative_to(PROJECT))
        config = {"native": dict(SMALL_NATIVE), "classes": 2, "node_inputs": paths,
                  "initialization_seed": 1009, "dropout_seeds": [301010, 302019, 303028, 304037],
                  "preprocessing": "pinned_mono_base", "train_views": views, "fusion": "mean_probabilities",
                  "learning_rate": .005, "attn_lr": .0005, "weight_decay": .001,
                  "attn_weight_decay": 1e-8, "factor_initialization": "unit"}
        if loss is not None:
            config["loss"] = loss
        run = {"run_id": "polyformer-engineering", "method": method, "config": config,
               "selection": {"metric": "objective", "direction": "min", "ties": "last", "save_every": 1},
               "budget": {"maximum": 3}}
        study = {"dataset": "engineering-only", "data": {"roles": "synthetic disjoint TRAIN/SELECT"},
                 "source": {"author_commit": polyformer.AUTHOR_COMMIT}}
        return polyformer.make_adapter(torch=torch, project_root=PROJECT, study=study, run=run, device="cpu")

    def test_exact_author_class_and_native_full_size_factor_counts(self):
        native = {**SMALL_NATIVE, "hidden": 256, "nlayer": 2, "n_head": 8, "d_ffn": 32,
                  "dropout": .5, "dprate": .8}
        ordinary = self.bank(native=native, classes=3, features=500)
        equipped = self.bank(native=native, classes=3, features=500, factorized=True)
        shared = self.bank(native=native, classes=3, features=500, factorized=True, members=4)
        self.assertEqual(type(ordinary.body).__name__, "PolyFormer")
        self.assertEqual(sum(p.numel() for p in ordinary.parameters()), 2069875)
        self.assertEqual(sum(p.numel() for p in equipped.parameters()), 2083818)
        self.assertEqual(sum(p.numel() for p in shared.parameters()), 2125647)
        self.assertEqual((equipped.factor_sites, shared.factor_sites), (23, 23))
        ordinary.eval()
        equipped.eval()
        shared.eval()
        with torch.no_grad():
            expected = ordinary()
            # Source FactorLinear adds bias separately, while nn.Linear may
            # fuse bias into GEMM. Algebraic containment is not bitwise parity.
            self.assertTrue(torch.allclose(equipped(), expected, atol=1e-5, rtol=1e-5))
            self.assertTrue(all(torch.allclose(route, expected[0], atol=1e-5, rtol=1e-5) for route in shared()))

    def test_private_dropout_preserves_outer_rng_and_live_hidden_gradient(self):
        model = self.bank(members=4, factorized=True)
        outside = torch.get_rng_state().clone()
        before = [state["cpu"].clone() for state in model.streams]
        model.train()
        logits = model()
        self.assertTrue(torch.equal(torch.get_rng_state(), outside))
        self.assertTrue(all(not torch.equal(a, b["cpu"]) for a, b in zip(before, model.streams)))
        self.assertEqual(tuple(logits.shape), (4, 4, 2))
        self.assertEqual(tuple(model.last_hidden.shape), (4, 4, 8))
        model.last_hidden.square().mean().backward()
        self.assertIsNotNone(model.body.lin1.weight.grad)
        self.assertIsNone(model.body.lin3.weight.grad)

    def test_mono_preprocessing_preserves_author_normalization(self):
        x = self.tokens()[0]
        directed = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        bank = polyformer.mono_tokens(torch, PROJECT, x, directed, K=2,
                                     preprocessing="author_undirected_then_mono")
        adjacency = torch.eye(4)
        for row in range(4):
            adjacency[row, (row - 1) % 4] = 1
            adjacency[row, (row + 1) % 4] = 1
        adjacency /= 3
        self.assertTrue(torch.equal(bank[0], x))
        self.assertTrue(torch.allclose(bank[1], adjacency @ x, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(bank[2], adjacency @ adjacency @ x, atol=1e-6, rtol=1e-6))
        literal = polyformer.mono_tokens(torch, PROJECT, x, directed, K=2, preprocessing="pinned_mono_base")
        self.assertFalse(torch.allclose(literal[1], bank[1]))

    def test_native_engine_bridge_complete_budget_and_named_adam_groups(self):
        with tempfile.TemporaryDirectory(prefix="polyformer-engineering-", dir=PROJECT) as name:
            directory = Path(name)
            adapter = self.fixture(directory, method="polyformer_internal_be")
            for (parameter_name, parameter), group in zip(adapter.model.named_parameters(), adapter.optimizer.param_groups):
                self.assertIs(group["params"][0], parameter)
                self.assertEqual(group["lr"], .0005 if "attnmodule" in parameter_name else .005)
                self.assertEqual(group["weight_decay"], 1e-8 if "attnmodule" in parameter_name else .001)
                self.assertEqual(group["betas"], (.9, .999))
                self.assertEqual(group["eps"], 1e-8)
            spec = {"fit_id": "polyformer-engineering", "budget_steps": 3, "budget_unit": "optimizer_updates",
                    "validate_initial": True, "validation_interval": 1, "prefixes": [],
                    "inherited_fit_ids": [], "adapter_spec": {"engineering_only": True}, "recovery_interval": 100}
            result = run_trajectory(torch, adapter, spec, directory / "run")
            self.assertEqual(result["completed_steps"], 3)
            self.assertEqual(result["operation_counts"]["optimizer_updates"], 3)
            self.assertEqual(result["operation_counts"]["backward_calls"], 12)
            self.assertEqual(result["operation_counts"]["model_forwards"], 12)
            self.assertEqual(adapter.fusion, "mean_probabilities")

    def test_runtime_static_bias_and_private_stream_checkpoint_restoration(self):
        with tempfile.TemporaryDirectory(prefix="polyformer-engineering-", dir=PROJECT) as name:
            directory = Path(name)
            adapter = self.fixture(directory, method="polyformer_factorized_single", views=4)
            metrics = adapter.validate(0)
            decision = adapter.observe_validation(0, metrics)
            payload = capture_checkpoint(torch, adapter, fit_id="polyformer-restore", step=0, validation=metrics, decision=decision)
            path = directory / "checkpoint.pt"
            torch.save(payload, path)
            runtime = adapter.model.capture_runtime_state()
            for module in adapter.model.body.modules():
                if type(module).__name__ == "PolyAttn":
                    module.bias.add_(10)
            adapter.model.streams[0]["cpu"] = torch.Generator().manual_seed(1).get_state()
            restore_checkpoint(torch, path, adapter, expected_fit_id="polyformer-restore")
            actual = adapter.model.capture_runtime_state()
            self.assertTrue(torch.equal(actual["streams"][0]["cpu"], runtime["streams"][0]["cpu"]))
            self.assertTrue(all(torch.equal(actual["poly_attn_bias"][key], value) for key, value in runtime["poly_attn_bias"].items()))
            record = adapter.train_step(1)
            self.assertEqual(record["operation_counts"]["optimizer_updates"], 1)
            self.assertEqual(record["operation_counts"]["backward_calls"], 4)

    def test_paired_scalar_callback_is_live_and_independent_serving_is_probability_mean(self):
        with tempfile.TemporaryDirectory(prefix="polyformer-engineering-", dir=PROJECT) as name:
            directory = Path(name)
            paired = self.fixture(directory, method="polyformer_internal_be", views=2,
                                  loss={"variant": "two_view_internal_be", "coefficients": {
                                      "alignment_weight": .05, "residual_weight": .05, "temperature": .2}})
            record = paired.train_step(1)
            self.assertEqual(record["operation_counts"]["model_forwards"], 8)
            self.assertEqual(record["operation_counts"]["backward_calls"], 1)
            self.assertIn("alignment", record["objective_components"])
            adapters = [self.fixture(directory) for _ in range(4)]
            served = polyformer.assemble_independent(torch, adapters)
            self.assertTrue(torch.equal(served["probabilities"], served["raw_logits"].softmax(-1).mean(0)))
            self.assertEqual(served["additional_optimizer_updates"], 0)
            with self.assertRaises(ValueError):
                polyformer.assemble_independent(torch, [adapters[0]] * 4)


if __name__ == "__main__":
    unittest.main()
