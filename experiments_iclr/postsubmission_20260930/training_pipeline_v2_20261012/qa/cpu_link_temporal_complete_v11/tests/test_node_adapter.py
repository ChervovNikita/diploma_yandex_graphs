"""CPU-only interface fixtures: all native operators and live route gradients."""
import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

TORCH_PRESENT = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_PRESENT, "Allocation Torch runtime required")
class NodeAdapterTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        package = Path(__file__).resolve().parents[1]
        source = package / "fixtures/native_models.py"
        if not source.exists():
            source = package.parent / "common_wrapper_native_source_20261010_v1/models.py"
        shutil.copyfile(source, self.root / "models.py")
        torch.save({"x": torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32,
                    "edge_index": torch.tensor([[0,1,2,3,4,5,6,7,1,2,3,4,5,6,7,0],
                                                [1,2,3,4,5,6,7,0,0,1,2,3,4,5,6,7]])}, self.root / "graph.pt")
        torch.save({"ids": torch.arange(6), "y": torch.tensor([0,0,1,1,2,2])}, self.root / "train.pt")
        torch.save({"ids": torch.tensor([6,7]), "y": torch.tensor([0,1])}, self.root / "select.pt")
        self.config = {"node_inputs": {"graph": "graph.pt", "train": "train.pt", "select": "select.pt"},
                       "native_model_source": "models.py", "backbone": "SAGE", "depth": 1,
                       "width": 8, "classes": 3, "ffn_multiplier": 1, "heads": 2,
                       "normalization": "LayerNorm", "dropout": .2, "learning_rate": .001,
                       "weight_decay": 0., "fusion": "mean_probabilities", "members": 4}
        self.run = {"run_id": "fixture", "method": "original_gnnm", "seed": 101, "body": 0,
                    "config": self.config, "budget": {"maximum": 3}, "selection": {
                        "metric": "accuracy", "direction": "max", "ties": "first"}}
        self.study = {"dataset": "synthetic_engineering_fixture", "data": {}, "source": {}}

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, run=None):
        from adapters.node import make_adapter
        return make_adapter(torch=self.torch, project_root=self.root, study=self.study,
                            run=run or self.run, device="cpu")

    def test_original_boundary_bridge_supports_all_eight_native_operators(self):
        for backbone in ("ResNet", "GCN", "SAGE", "GAT", "GAT-sep", "GT", "GT-sep", "TAG"):
            with self.subTest(backbone=backbone):
                self.config["backbone"] = backbone
                adapter = self.make()
                self.assertEqual(tuple(adapter.model().shape), (4, 8, 3))
                self.assertEqual(adapter.validate(0)["examples"], 2)

    def test_auxiliary_gradient_reaches_shared_trunk_and_boundary_factors(self):
        self.config.update(objective_variant="gncl_within_route_supcon",
                           objective_coefficients={"pool_weight": .5, "supcon_weight": .05, "temperature": .2})
        adapter = self.make()
        result = adapter.train_step(1)
        self.assertGreater(result["regularization"], 0)
        self.assertTrue(adapter.model.last_hidden.requires_grad)
        self.assertIsNotNone(adapter.model.body.input_be_block.R.grad)
        self.assertIsNotNone(adapter.model.body.residual_modules[0].module.conv.lin_l.weight.grad)
        self.assertEqual(result["operation_counts"]["route_predictions"], 4)

    def test_paired_internal_factors_use_two_live_views_and_independent_body_seeds(self):
        self.run["method"] = "internal_be"
        self.config.update(objective_variant="two_view_internal_be", fast_input_initialization="ones",
                           objective_coefficients={"alignment_weight": .05, "residual_weight": .05, "temperature": .2})
        adapter = self.make()
        result = adapter.train_step(1)
        self.assertEqual(result["operation_counts"]["model_forwards"], 8)
        self.assertTrue(all(p.grad is not None for p in adapter.model.fast_r))
        # Independent members are constructed in separate fits with distinct
        # body seeds; a coupled four-body bank is not labelled I4.
        run = copy.deepcopy(self.run)
        run.update(method="factorized_single", body=0)
        a = self.make(run)
        run["body"] = 1
        b = self.make(run)
        self.assertNotEqual(a.provenance["actual_model_seed"], b.provenance["actual_model_seed"])
        self.assertFalse(self.torch.equal(a.model.body.input_linear.weight, b.model.body.input_linear.weight))

    def test_train_select_overlap_is_rejected_before_learning(self):
        self.torch.save({"ids": self.torch.tensor([0]), "y": self.torch.tensor([0])}, self.root / "select.pt")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.make()


if __name__ == "__main__":
    unittest.main()
