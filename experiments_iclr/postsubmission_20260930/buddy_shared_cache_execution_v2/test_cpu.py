"""Dataset-free CPU numerical gate. Never downloads, opens data or uses CUDA."""
import argparse
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from cache_builder import HERE, extract_feature_function, file_sha, native_hashes, write_json
from models import FactorizedBUDDY, make_model, matched_width, parameter_count, single, verified_source
from run import batch_rows, hits50, implementation_hashes, finite_step
from guards import checkpoint_metadata
from checkpoint_io import validate_selected_checkpoint


class CPUQualification(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1701)
        self.config = json.loads((HERE / "CONFIG.json").read_text())
        self.config.update(feature_dropout=0.0, label_dropout=0.0)
        self.native = single(self.config, 13).double()
        self.factorized = FactorizedBUDDY(self.native).double()
        self.sf = torch.randn(8, 8, dtype=torch.float64)
        self.features = torch.randn(8, 2, 128, dtype=torch.float64)
        self.degree_a = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float64)
        self.degree_b = torch.tensor([2, 0, 3, 1, 5, 2, 1, 4], dtype=torch.float64)
        self.inputs = (self.sf, self.features, self.degree_a, self.degree_b)

    def test_identity_native_logits_and_gradient_pullback(self):
        copies = [copy.deepcopy(self.native).eval() for _ in range(4)]
        self.factorized.eval()
        expected = torch.cat([m(*self.inputs) for m in copies], dim=1)
        actual = self.factorized(*self.inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        labels = torch.tensor([0, 1] * 4, dtype=torch.float64)[:, None].expand(8, 4)
        F.binary_cross_entropy_with_logits(expected, labels).backward()
        F.binary_cross_entropy_with_logits(actual, labels).backward()
        for name in ["label_lin_layer", "lin_feat", "lin_out", "lin"]:
            for field in ["weight", "bias"]:
                ref = sum(getattr(getattr(m, name), field).grad for m in copies)
                torch.testing.assert_close(getattr(getattr(self.factorized, name), field).grad, ref, rtol=1e-10, atol=1e-10)
        for member in range(4):
            for name in ["bn_labels", "bn_feats"]:
                for field in ["weight", "bias"]:
                    torch.testing.assert_close(getattr(getattr(self.factorized, name)[member], field).grad, getattr(getattr(copies[member], name), field).grad, rtol=1e-10, atol=1e-10)

    def test_native_training_batchnorm_identity_and_private_state(self):
        copies = [copy.deepcopy(self.native).train() for _ in range(4)]
        self.factorized.train()
        torch.testing.assert_close(self.factorized(*self.inputs), torch.cat([m(*self.inputs) for m in copies], dim=1), rtol=1e-10, atol=1e-10)
        for name in ["bn_labels", "bn_feats", "bn_RA"]:
            states = getattr(self.factorized, name)
            self.assertEqual(len({bn.running_mean.data_ptr() for bn in states}), 4)
            self.assertEqual(len({bn.weight.data_ptr() for bn in states}), 4)
            for index, bn in enumerate(states):
                torch.testing.assert_close(bn.running_mean, getattr(copies[index], name).running_mean)
                torch.testing.assert_close(bn.running_var, getattr(copies[index], name).running_var)

    def test_parameter_counts_and_analytical_matching(self):
        config = json.loads((HERE / "CONFIG.json").read_text())
        counts = {"native1024": parameter_count(1024), "single256": parameter_count(256), "factorized4": parameter_count(256, factorized=True), "independent4": parameter_count(256, members=4), "matched_single": parameter_count(matched_width(parameter_count(256, factorized=True)))}
        for arm, expected in counts.items():
            model = make_model(config, arm, 0)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            with tempfile.TemporaryDirectory(prefix=".cpu-checkpoint-", dir=HERE) as temporary:
                path = Path(temporary) / "selected.pt"
                identity = {"arm": arm, "seed": 0, "cache_manifest_sha256": "a" * 64,
                            "config_sha256": file_sha(HERE / "CONFIG.json"), "implementation_hashes": implementation_hashes(),
                            "parameters": expected, "torch_version": str(torch.__version__)}
                record = {"epoch": 1, "train_bce": 0.7, "train_seconds": 1.0,
                          "validation_forward_seconds": 0.1, "validation_hits50": 0.5,
                          "validation_bce_sampled_pool": 0.6}
                metadata = checkpoint_metadata(identity, record)
                envelope = {"metadata": metadata, "model_state": model.state_dict()}
                torch.save(envelope, path)
                validate_selected_checkpoint(path, metadata)
                first = next(iter(envelope["model_state"]))
                original = envelope["model_state"][first]
                for invalid in (original.double(), original.flatten()[:1], torch.full_like(original, float("nan"))):
                    envelope["model_state"][first] = invalid
                    torch.save(envelope, path)
                    with self.assertRaises(RuntimeError):
                        validate_selected_checkpoint(path, metadata)
                envelope["model_state"][first] = original
                envelope["metadata"] = {**metadata, "selected_epoch": 999}
                torch.save(envelope, path)
                with self.assertRaises(RuntimeError):
                    validate_selected_checkpoint(path, metadata)
                path.write_bytes(b"opaque non-model placeholder")
                with self.assertRaises(Exception):
                    validate_selected_checkpoint(path, metadata)
        self.assertEqual(matched_width(counts["factorized4"]), 266)
        self.assertLess(abs(counts["matched_single"] / counts["factorized4"] - 1), 0.01)

    def test_batching_preserves_every_row_including_singleton_tail(self):
        for length in [1024, 1025, 2049, 2050]:
            rows = list(batch_rows(length, 1024, torch.Generator().manual_seed(5)))
            self.assertEqual(sorted(torch.cat(rows).tolist()), list(range(length)))
            self.assertGreater(min(len(x) for x in rows), 1)

    def test_hits50_matches_saved_official_strict_tie_rule(self):
        negative = torch.cat([torch.full((49,), 2.0), torch.ones(99951)])
        positive = torch.tensor([2.0, 1.0, 0.0])
        scores = torch.cat([positive, negative])
        tree = ast.parse(verified_source("ogb_evaluator.py.txt"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Evaluator")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_eval_hits")
        namespace = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "saved_ogb_eval_hits", "exec"), namespace)
        official = namespace[method.name](SimpleNamespace(K=50), positive, negative, "torch")["hits@50"]
        self.assertAlmostEqual(hits50(scores, 3), official, places=7)
        self.assertAlmostEqual(official, 1 / 3)
        scores[0] = float("nan")
        with self.assertRaises(RuntimeError):
            hits50(scores, 3)
        # Exercise loss/gradient/state guards without graph/model data.
        tiny = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(tiny.parameters(), lr=0.02)
        with self.assertRaises(RuntimeError):
            finite_step(tiny(torch.ones(2, 1)).sum() * float("inf"), tiny, optimizer)
        optimizer.zero_grad(set_to_none=True)
        finite_step(F.binary_cross_entropy_with_logits(tiny(torch.ones(2, 1)), torch.ones(2, 1)), tiny, optimizer)
        optimizer.zero_grad(set_to_none=True)
        optimizer.state[tiny.weight]["step"].fill_(float("inf"))
        with self.assertRaisesRegex(RuntimeError, "Nonfinite Adam state"):
            finite_step(F.binary_cross_entropy_with_logits(tiny(torch.ones(2, 1)), torch.ones(2, 1)), tiny, optimizer)
        optimizer.zero_grad(set_to_none=True)
        tiny.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        with self.assertRaises(RuntimeError):
            finite_step(tiny(torch.ones(2, 1)).sum(), tiny, optimizer)

    def test_native_feature_cache_against_dense_weighted_oracle(self):
        # Six nodes, one isolate, weighted reciprocal edges and a native self
        # loop insertion. No OGB dataset loader is called.
        edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
        weight = torch.tensor([2, 2, 3, 3, 1, 1])
        x = torch.randn(6, 128)
        graph = SimpleNamespace(x=x, num_nodes=6)
        result = extract_feature_function()(None, graph, edge_index, weight, 0)
        adjacency = torch.zeros(6, 6)
        adjacency[edge_index[0], edge_index[1]] = weight.float()
        adjacency += torch.eye(6)
        degree = adjacency.sum(1)
        normalized = degree.rsqrt()[:, None] * adjacency * degree.rsqrt()[None, :]
        torch.testing.assert_close(result, normalized @ x, rtol=2e-6, atol=2e-6)

    def test_native_hash_query_chunk_invariance(self):
        hasher = native_hashes(self.config)
        edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
        hashes, cards = hasher.build_hash_tables(6, edges)
        queries = torch.tensor([[0, 2], [1, 4], [3, 4], [0, 5], [5, 5]])
        small = hasher.get_subgraph_features(queries, hashes, cards, 2)
        large = hasher.get_subgraph_features(queries, hashes, cards, 128)
        self.assertEqual(tuple(small.shape), (5, 8))
        self.assertTrue(torch.isfinite(small).all())
        torch.testing.assert_close(small, large, rtol=0, atol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    implementation_hashes()  # verify every pin before any native test executes
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CPUQualification)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    write_json(args.output, {"status": "synthetic_cpu_pass", "test_count": result.testsRun, "dataset_access": False, "gpu_execution": False, "torch_version": str(torch.__version__), "implementation_hashes": implementation_hashes(), "test_script_sha256": file_sha(__file__)})
