"""Focused CPU regressions for the NCN boundary and resumable protocol."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_sparse import SparseTensor

from ablation.common.resumable import JobStore
from ablation.common.worker import Preempted
from .collect import collect, ensemble_artifact, main as collector_main
from .data import Bundle, load_bundle, masked_adjacency, tensor_identity
from .models import NCN, RoutedGCN, RoutedLinear
from .protocol import RECIPE, digest_json
from .train import metrics, predict, seed_everything, train_fit


def toy_bundle():
    nodes = 48
    source = torch.arange(nodes)
    train = torch.cat([torch.stack([source, (source + step) % nodes], dim=1)
                       for step in range(1, 6)])
    valid = torch.stack([source, (source + 6) % nodes], dim=1)
    negative = torch.cat([torch.stack([source, (source + step) % nodes], dim=1)
                          for step in range(9, 15)])
    edge_index = torch.cat([train.t(), train.t().flip(0)], dim=1)
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(nodes, 8, generator=generator)
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(nodes, nodes)).to_symmetric().coalesce()
    split = {"train": {"edge": train}, "valid": {"edge": valid, "edge_neg": negative}}
    return Bundle(x, edge_index, split, adj, None, tensor_identity([("x", x), ("edges", edge_index)]))


def config(members=1, updates=3):
    return dict(RECIPE, members=members, seed=19, cohort=0,
                method="gnnm" if members == 4 else "native0", epochs=2,
                batch_size=32, testbs=64, smoke_updates=updates, result_role="smoke_only",
                data_dir=str(Path(__file__).parent), source={"unit_test": True})


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_unit_routes_recover_native(self):
        bundle = toy_bundle()
        seed_everything(2)
        native = NCN(8, config()).eval()
        routed = copy.deepcopy(native)
        routed.members = 4
        routed.encoder.convs[0] = RoutedGCN(routed.encoder.convs[0], 4, signs=False)
        routed.predictor.lin[-1] = RoutedLinear(routed.predictor.lin[-1], 4)
        routed.eval()
        with torch.no_grad():
            native_hidden = native.encoder(bundle.x, bundle.adj)
            expected = native.predictor(native_hidden, bundle.adj, bundle.split["valid"]["edge"].t())
            for member in range(4):
                routed.select_member(member)
                hidden = routed.encoder(bundle.x, bundle.adj)
                torch.testing.assert_close(hidden, native_hidden, rtol=0, atol=0)
                actual = routed.predictor(hidden, bundle.adj, bundle.split["valid"]["edge"].t())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        routed_count = sum(p.numel() for p in routed.parameters())
        native_count = sum(p.numel() for p in native.parameters())
        expected_extra = 4 * (8 + 64 + 64 + 64 + 1 + 1) - (64 + 1)
        self.assertEqual(routed_count - native_count, expected_extra)

    def test_target_edges_removed_bidirectionally(self):
        bundle = toy_bundle()
        chosen = torch.tensor([0, 53, 109])
        adjacency = masked_adjacency(bundle, chosen).to_dense()
        for source, target in bundle.split["train"]["edge"][chosen]:
            self.assertEqual(float(adjacency[source, target]), 0)
            self.assertEqual(float(adjacency[target, source]), 0)
        self.assertEqual(float(adjacency[1, 2]), 1)

    def test_complete_valid_and_no_test_graph(self):
        bundle = toy_bundle()
        predictions = predict(NCN(8, config()).eval(), bundle, 17)
        self.assertEqual(len(predictions["train_pos"]), len(bundle.split["train"]["edge"]))
        self.assertEqual(len(predictions["valid_pos"]), len(bundle.split["valid"]["edge"]))
        self.assertEqual(len(predictions["valid_neg"]), len(bundle.split["valid"]["edge_neg"]))
        self.assertNotIn("test_pos", predictions)
        self.assertIn("valid_hits@50", metrics(predictions))
        with self.assertRaisesRegex(ValueError, "TEST adjacency"):
            predict(NCN(8, config()).eval(), bundle, 17, test_only=True)

    def test_exact_resume_at_smoke_boundary(self):
        bundle = toy_bundle()
        for members in (1, 4):
            with self.subTest(members=members), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                root = Path(directory)
                protocol = config(members)
                continuous = JobStore(root / "continuous")
                continuous.ensure_config(protocol)
                expected = train_fit(protocol, continuous, SimpleNamespace(requested=False),
                                     bundle, torch.device("cpu"))
                resumed = JobStore(root / "resumed")
                resumed.ensure_config(protocol)
                stop = SimpleNamespace(requested=False)

                def interrupt_at_bound(update):
                    stop.requested = update == protocol["smoke_updates"]

                with self.assertRaises(Preempted):
                    train_fit(protocol, resumed, stop, bundle, torch.device("cpu"),
                              on_update=interrupt_at_bound)
                state = resumed.load_state()
                self.assertEqual(state["progress"]["updates"], 3)
                self.assertIsNone(state["scheduler"])
                actual = train_fit(protocol, resumed, SimpleNamespace(requested=False),
                                   bundle, torch.device("cpu"))
                self.assertEqual(actual["updates"], expected["updates"])
                self.assertEqual(actual["best_update"], expected["best_update"])
                self.assertEqual(actual["valid_hits@50"], expected["valid_hits@50"])
                self.assertEqual(actual["resumes"], 1)
                expected_model = torch.load(continuous.dir / "selected_model.pt", weights_only=False)
                actual_model = torch.load(resumed.dir / "selected_model.pt", weights_only=False)
                for name, tensor in expected_model["model"].items():
                    torch.testing.assert_close(actual_model["model"][name], tensor, rtol=0, atol=0)
                for name, tensor in expected_model["predictions"].items():
                    torch.testing.assert_close(actual_model["predictions"][name], tensor, rtol=0, atol=0)


    def test_full_fit_scores_selected_test_once(self):
        bundle = toy_bundle()
        protocol = config(updates=None)
        complete = copy.copy(bundle)
        complete.split = dict(bundle.split)
        source = torch.arange(bundle.num_nodes)
        complete.split["test"] = {
            "edge": torch.stack([source, (source + 7) % bundle.num_nodes], dim=1),
            "edge_neg": bundle.split["valid"]["edge_neg"],
        }
        complete.test_edge_order_sha256 = tensor_identity(
            [(f"test.{name}", value) for name, value in complete.split["test"].items()])
        full_edges = torch.cat([bundle.edge_index, bundle.split["valid"]["edge"].t()], dim=1)
        complete.full_adj = SparseTensor.from_edge_index(
            full_edges, sparse_sizes=bundle.adj.sizes()).to_symmetric()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            store = JobStore(Path(directory))
            store.ensure_config(protocol)
            with patch("ablation.p12_collab_ncn.train.load_bundle", return_value=complete) as loader:
                result = train_fit(protocol, store, SimpleNamespace(requested=False),
                                   bundle, torch.device("cpu"))
            loader.assert_called_once_with(protocol["data_dir"], torch.device("cpu"), release_test=True)
            self.assertEqual(result["epochs_completed"], 2)
            self.assertEqual(result["updates"], 2 * (len(bundle.split["train"]["edge"]) // 32))
            self.assertEqual(result["test_role"], "selected_checkpoint_once")
            self.assertIn("test_hits@50", result)
            artifact = torch.load(store.dir / "selected_predictions.pt", weights_only=False)
            self.assertEqual(artifact["test_edge_order_sha256"], complete.test_edge_order_sha256)
            self.assertEqual(result["test_edge_order_sha256"], complete.test_edge_order_sha256)

    def test_convergence_continuation_and_deferred_member_scoring(self):
        import hashlib
        from ablation.common.resumable import atomic_torch_save
        from .score import score_fit
        bundle = toy_bundle()
        complete = copy.copy(bundle)
        complete.split = dict(bundle.split, test=bundle.split["valid"])
        complete.full_adj = bundle.adj
        complete.test_edge_order_sha256 = "fixture_test_order"
        protocol = dict(config(4, updates=None), epochs=8, min_epochs=3,
                        cap_schedule=[3, 8], patience=2, defer_test=True,
                        test_role="deferred", result_role="full")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            store = JobStore(Path(directory) / "training")
            store.ensure_config(protocol)
            with patch("ablation.p12_collab_ncn.train.metrics", return_value={"valid_hits@50": 0.5}):
                result = train_fit(protocol, store, SimpleNamespace(requested=False),
                                   bundle, torch.device("cpu"))
            self.assertEqual(result["epochs_completed"], 5)
            self.assertEqual(result["stop_reason"], "patience")
            self.assertEqual(len(result["continuations"]), 1)
            self.assertNotIn("test_metric", result)
            selected = torch.load(store.dir / "selected_model.pt", weights_only=False)
            self.assertEqual(selected["member_predictions"]["valid_pos"].shape[0], 4)
            # An engineering fixture exercises the reference100 serialization path.
            reference = store.dir / "reference100_model.pt"
            atomic_torch_save(selected, reference)
            result["reference100_checkpoint_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
            store.write_done(result)
            with patch("ablation.p12_collab_ncn.score.load_bundle", return_value=complete):
                scored = score_fit(store.dir, Path(directory) / "scored", "cpu")
            self.assertTrue(scored["test_evaluated"])
            self.assertIn("reference100_test_metric", scored)
            output = torch.load(Path(directory) / "scored/predictions.pt", weights_only=False)
            self.assertEqual(output["member_predictions"]["test_pos"].shape[0], 4)
            with reference.open("ab") as stream:
                stream.write(b"tamper")
            with patch("ablation.p12_collab_ncn.score.load_bundle", return_value=complete):
                with self.assertRaisesRegex(ValueError, "reference100 checkpoint"):
                    score_fit(store.dir, Path(directory) / "rejected", "cpu")

    def test_resume_rejects_identity_change(self):
        bundle = toy_bundle()
        protocol = config()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            store = JobStore(Path(directory))
            store.ensure_config(protocol)
            stop = SimpleNamespace(requested=True)
            with self.assertRaises(Preempted):
                train_fit(protocol, store, stop, bundle, torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "identity"):
                train_fit(dict(protocol, gnnlr=0.01), store, SimpleNamespace(requested=False),
                          bundle, torch.device("cpu"))

    def test_collector_smoke_opt_in_and_complete_cohort(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            for member, method in enumerate(("native0", "native1", "native2", "native3", "gnnm")):
                protocol = dict(config(4 if method == "gnnm" else 1),
                                method=method, seed=0 if method == "gnnm" else member,
                                test_role="closed_smoke")
                identity = {"config_sha256": digest_json(protocol), "data_sha256": "fixed",
                            "versions": {}, "torch_cuda": None}
                predictions = {"valid_pos": torch.arange(60).float() + member,
                               "valid_neg": torch.arange(60).float()}
                store = JobStore(root / f"cohort0/{method}")
                store.ensure_config(protocol)
                store.write_done(dict(protocol, identity=identity, train_seconds=1, num_params=10,
                                      **metrics(predictions)))
                torch.save({"config": protocol, "identity": identity, "predictions": predictions,
                            "selected_epoch": 1, "selected_update": 3, "test_role": "closed_smoke"},
                           store.dir / "selected_predictions.pt")
            excluded = collect(root, root / "excluded", [0])
            self.assertEqual(excluded["settings"], {})
            included = collect(root, root / "included", [0], include_smoke=True)
            self.assertEqual(included["incomplete_cohorts"], [])
            self.assertEqual(len(included["settings"]), 3)
            self.assertTrue((root / "included/cohort0_independent4_predictions.pt").exists())
            with contextlib.redirect_stdout(io.StringIO()):
                status = collector_main(["--root", str(root), "--out", str(root / "partial"),
                                         "--cohorts", "0", "1", "--include-smoke"])
            self.assertEqual(status, 1)
            summary = json.loads((root / "partial/summary.json").read_text())
            self.assertEqual(summary["incomplete_cohorts"], [1])
            self.assertEqual(summary["missing_groups"][0]["missing_fits"],
                             ["native0", "native1", "native2", "native3", "gnnm"])
            self.assertIn("BASE", (root / "partial/cohorts.csv").read_text())
            self.assertIn("smoke_only/independent4", summary["settings"])

    def test_empty_requested_collection_fails_with_missing_groups(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                status = collector_main(["--root", str(root / "empty"),
                                         "--out", str(root / "collected"), "--cohorts", "0"])
            self.assertEqual(status, 1)
            summary = json.loads((root / "collected/summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["missing_groups"][0]["missing_settings"],
                             ["BASE", "independent4", "GNNM4"])
            self.assertTrue((root / "collected/cohorts.csv").exists())

    def test_cross_cohort_invariants_are_enforced(self):
        for change in ("protocol", "source", "data", "runtime"):
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                def fixture(root, cohort, method):
                    if method != "native0":
                        return None
                    protocol = dict(config(), cohort=cohort, seed=cohort * 1000)
                    identity = {"data_sha256": "same", "versions": {"torch": "same"}, "torch_cuda": None}
                    if cohort == 1:
                        if change == "protocol":
                            protocol["gnnlr"] = 0.01
                        if change == "source":
                            protocol["source"] = {"changed": True}
                        if change == "data":
                            identity["data_sha256"] = "changed"
                        if change == "runtime":
                            identity["versions"] = {"torch": "changed"}
                    identity["config_sha256"] = digest_json(protocol)
                    result = dict(protocol, valid_hits50=0.5, train_seconds=1, num_params=1,
                                  test_role="closed_smoke")
                    result["valid_hits@50"] = result.pop("valid_hits50")
                    artifact = {"config": protocol, "identity": identity}
                    return result, artifact

                with patch("ablation.p12_collab_ncn.collect.read_fit", side_effect=fixture):
                    with self.assertRaisesRegex(ValueError, "Requested cohort"):
                        collect(Path(directory), Path(directory) / "collected", [0, 1], include_smoke=True)

    def test_test_edge_identity_tracks_order_and_ensemble_rejects_drift(self):
        bundle = toy_bundle()
        splits = dict(bundle.split)
        splits["test"] = {name: value.clone() for name, value in bundle.split["valid"].items()}

        class Dataset:
            def __getitem__(self, index):
                return SimpleNamespace(x=bundle.x, edge_index=bundle.edge_index)

            def get_edge_split(self):
                return splits

        with patch("ablation.p12_collab_ncn.data.PygLinkPropPredDataset", return_value=Dataset()):
            closed = load_bundle("unused", torch.device("cpu"))
            original = load_bundle("unused", torch.device("cpu"), release_test=True)
            splits["test"]["edge_neg"] = splits["test"]["edge_neg"].flip(0)
            reordered = load_bundle("unused", torch.device("cpu"), release_test=True)
        self.assertIsNone(closed.test_edge_order_sha256)
        self.assertEqual(original.identity, reordered.identity)
        self.assertNotEqual(original.test_edge_order_sha256, reordered.test_edge_order_sha256)
        fits = []
        for member in range(4):
            protocol = dict(config(updates=None), method=f"native{member}", seed=member, result_role="full")
            artifact = {
                "config": protocol,
                "identity": {"config_sha256": digest_json(protocol), "data_sha256": original.identity,
                             "versions": {}, "torch_cuda": None},
                "selected_epoch": 1, "selected_update": 7, "test_role": "selected_checkpoint_once",
                "test_edge_order_sha256": original.test_edge_order_sha256,
                "predictions": {name: torch.arange(60).float()
                                for name in ("valid_pos", "valid_neg", "test_pos", "test_neg")},
            }
            fits.append(({}, artifact))
        selected = ensemble_artifact(fits)
        self.assertEqual(selected["test_edge_order_sha256"], original.test_edge_order_sha256)
        fits[-1][1]["test_edge_order_sha256"] = reordered.test_edge_order_sha256
        with self.assertRaisesRegex(ValueError, "TEST edge ordering"):
            ensemble_artifact(fits)

    def test_independent_ensemble_averages_selected_arrays(self):
        fits = []
        for member in range(4):
            artifact = {
                "config": dict(config(), method=f"native{member}", seed=member),
                "identity": {"data_sha256": "fixed", "versions": {}, "torch_cuda": None},
                "selected_epoch": member + 1, "selected_update": member + 5,
                "test_role": "closed_smoke",
                "predictions": {"valid_pos": torch.full((60,), float(member)),
                                "valid_neg": torch.arange(60).float() + member},
            }
            artifact["identity"]["config_sha256"] = digest_json(artifact["config"])
            fits.append(({}, artifact))
        averaged = ensemble_artifact(fits)
        torch.testing.assert_close(averaged["predictions"]["valid_pos"], torch.full((60,), 1.5))
        self.assertEqual(averaged["selected_epochs"], [1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ensemble_artifact([fits[0], fits[0], fits[2], fits[3]])
        conflicted = copy.deepcopy(fits)
        conflicted[1][1]["config"]["seed"] = 0
        conflicted[1][1]["identity"]["config_sha256"] = digest_json(conflicted[1][1]["config"])
        with self.assertRaisesRegex(ValueError, "seed identity"):
            ensemble_artifact(conflicted)


if __name__ == "__main__":
    unittest.main()
