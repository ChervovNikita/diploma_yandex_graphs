"""Link custody/contracts and synthetic exact-source CPU engineering fixture."""
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from adapters.link import (AUTHOR_SOURCE, LinkAdapter, _module, bound_file, file_sha,
                           make_adapter, sampler_custody, validate_config, validate_inputs,
                           validate_roster)
from engine import AdapterStepError, _counts, _metrics, inspect_independence, restore_checkpoint, run_trajectory
from metrics import EpochMeans
from prepare_link_data import SOURCE_FILES, authenticate_cache, validate_authority

try:
    import torch
except ImportError:
    torch = None


def receipt(path):
    return {"path": path, "bytes": 0, "sha256": "0"*64}


def config(arm="native_single", predictor="NCNC"):
    return {
        "link_inputs": {name: receipt("roles/"+name+".pt") for name in ("graph", "train", "select")},
        "link_source": {name: {"path": "source/"+name+".py", **pin} for name, pin in AUTHOR_SOURCE.items()},
        "arm": arm,
        "encoder": {"hidden": 256, "layers": 1, "dropout": .1, "ln": True, "res": True,
                    "edrop": .25, "xdropout": .25, "taildropout": .05, "jk": False, "conv_fn": "gcn"},
        "predictor": {"name": predictor, "layers": 3, "dropout": .3, "edrop": 0., "ln": True, "cndeg": -1,
                      "use_xlin": True, "tailact": True, "twolayerlin": False, "beta": 1.,
                      "completion": None if predictor == "NCN" else {"alpha": 1.05, "scale": 2.5, "offset": 6.,
                          "trainresdeg": -1, "testresdeg": -1, "pt": .1, "learnablept": False, "depth": 1, "splitsize": -1}},
        "optimizer": {"name": "Adam", "encoder_lr": .0082, "predictor_lr": .0037, "weight_decay": 0.,
                      "eps": 1e-8, "betas": [.9, .999], "amsgrad": False},
        "training": {"batch_size": 65536, "eval_batch_size": 131072, "tail_policy": "native_drop_tail"},
        "support": {"graph": "train_only", "maskinput": True, "record_duplicate_policy": "native_record_multiset", "select_overlap": "reject",
                    "select_negative_self_policy": "reject"},
        "sampler": {"name": "pyg_native_defaults", "source_sha256": "0"*64, "function_sha256": "0"*64},
        "metric": {"hits_k": 50, "evaluator_source_sha256": "0"*64},
        "initialization_seed": 7127,
        **({"independence_group": "fixture-i4"} if arm == "native_independent4_body" else {}),
    }


def run(arm="native_single", predictor="NCNC"):
    return {"run_id": "link-fixture", "body": 0, "kind": "model_fit", "adapter": "adapters.link:make_adapter",
            "method": predictor, "split": 0, "config": config(arm, predictor),
            "budget": {"unit": "epochs", "maximum": 100, "policy": "full_budget"},
            "selection": {"metric": "hits@50", "direction": "max", "ties": "first", "evaluation_interval": 1,
                          "checkpoint0": False, "save_every": 1}}


def study(row, rows=None):
    return {"dataset": "ogbl-collab", "data": {"task": "link_prediction", "roles": {
        "train": copy.deepcopy(row["config"]["link_inputs"]["train"]),
        "select": copy.deepcopy(row["config"]["link_inputs"]["select"])}},
        "source": {"files": list(row["config"]["link_source"].values())}, "runs": [row] if rows is None else rows}


def native_default_sampler(edge_index, num_nodes=None, num_neg_samples=None, method="sparse", force_undirected=False):
    return None


def changed_default_sampler(edge_index, num_nodes=None, num_neg_samples=100, method="sparse", force_undirected=False):
    return None


class LinkContractTests(unittest.TestCase):
    def test_full_native_recipe_admitted_and_config_copied(self):
        row = run()
        value = validate_config(row["config"], row)
        self.assertEqual(value["training"]["tail_policy"], "native_drop_tail")
        value["encoder"]["hidden"] = 17
        self.assertEqual(row["config"]["encoder"]["hidden"], 256)
        row = run(predictor="NCN")
        validate_config(row["config"], row)

    def test_heldout_config_and_roles_rejected(self):
        for role in ("cal", "dev", "test", "evaluate"):
            row = run()
            row["config"]["link_inputs"][role] = receipt(role+".pt")
            with self.assertRaisesRegex(ValueError, "exactly GRAPH"):
                validate_config(row["config"], row)
        row = run()
        row["config"]["test_targets"] = receipt("test.pt")
        with self.assertRaisesRegex(ValueError, "heldout"):
            validate_config(row["config"], row)

    def test_source_bytes_and_receipts_must_be_exact(self):
        row = run()
        row["config"]["link_source"]["model"]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "pinned author"):
            validate_config(row["config"], row)
        for value in ("../train.pt", "/train.pt", None, ""):
            row = run()
            row["config"]["link_inputs"]["train"]["path"] = value
            with self.assertRaisesRegex(ValueError, "project-relative"):
                validate_config(row["config"], row)

    def test_malformed_nested_objects_fail_as_contract_errors(self):
        for field in ("link_inputs", "link_source", "encoder", "predictor", "optimizer", "training", "support", "sampler", "metric"):
            for value in (None, [], 12):
                row = run()
                row["config"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_config(row["config"], row)

    def test_support_mask_and_sampling_cannot_change_silently(self):
        for field, key, value in (("support", "maskinput", False), ("support", "graph", "train_plus_valid"),
                                  ("support", "select_overlap", "filter"), ("predictor", "cndeg", 20),
                                  ("encoder", "conv_fn", "cached_gcn"), ("sampler", "name", "uniform_custom")):
            row = run()
            row["config"][field][key] = value
            with self.subTest(field=field, key=key), self.assertRaises(ValueError):
                validate_config(row["config"], row)
        row = run()
        row["config"]["predictor"]["completion"]["testresdeg"] = 128
        with self.assertRaisesRegex(ValueError, "unsampled"):
            validate_config(row["config"], row)

    def test_full_budget_and_selector_declared(self):
        for key, value in (("unit", "optimizer_updates"), ("policy", "patience")):
            row = run()
            row["budget"][key] = value
            with self.assertRaisesRegex(ValueError, "full declared epoch"):
                validate_config(row["config"], row)
        row = run()
        row["selection"]["direction"] = "min"
        with self.assertRaisesRegex(ValueError, "selector"):
            validate_config(row["config"], row)
        row["selection"].update(metric="ce", direction="min")
        validate_config(row["config"], row)

    def test_role_authority_must_match_prospective_study(self):
        row = run()
        declaration = study(row)
        validate_roster(declaration, row)
        declaration["data"]["roles"]["train"] = receipt("different.pt")
        with self.assertRaisesRegex(ValueError, "prospective train"):
            validate_roster(declaration, row)

    def test_i4_four_separate_paid_seeded_same_recipe_fits(self):
        rows = []
        for body in range(4):
            row = run("native_independent4_body")
            row["body"] = body
            row["run_id"] += "-"+str(body)
            row["config"]["initialization_seed"] += body*1009
            rows.append(row)
        declaration = study(rows[0], rows)
        validate_roster(declaration, rows[0])
        with self.assertRaisesRegex(ValueError, "four separate paid"):
            validate_roster(study(rows[0], rows[:3]), rows[0])
        duplicate = copy.deepcopy(rows)
        duplicate[3]["config"]["initialization_seed"] = duplicate[0]["config"]["initialization_seed"]
        with self.assertRaisesRegex(ValueError, "distinct native fit seeds"):
            validate_roster(study(rows[0], duplicate), rows[0])
        altered = copy.deepcopy(rows)
        altered[3]["config"]["optimizer"]["predictor_lr"] *= 2
        with self.assertRaisesRegex(ValueError, "recipes differ"):
            validate_roster(study(rows[0], altered), rows[0])
        duplicate = copy.deepcopy(rows)
        duplicate[3]["run_id"] = duplicate[0]["run_id"]
        with self.assertRaisesRegex(ValueError, "distinct paid run IDs"):
            validate_roster(study(rows[0], duplicate), rows[0])

    def test_selector_own_first_tie_never_ends_full_budget(self):
        adapter = object.__new__(LinkAdapter)
        adapter.selector = {"direction": "max", "ties": "first"}
        adapter._best = adapter._best_step = None
        self.assertTrue(adapter.observe_validation(1, {"selection_metric": .6})["save_selected"])
        self.assertFalse(adapter.observe_validation(2, {"selection_metric": .6})["save_selected"])
        last = adapter.observe_validation(3, {"selection_metric": .5})
        self.assertFalse(last["stop_requested"])
        self.assertEqual(last["best_saved_step"], 1)
        with self.assertRaises(FloatingPointError):
            adapter.observe_validation(4, {"selection_metric": float("nan")})

    def test_sample_weighted_whole_epoch_engine_interface(self):
        means = EpochMeans()
        means.add({"objective": 2., "ce": 2., "regularization": 0., "positive_ce": .5, "negative_ce": 1.5}, samples=2)
        means.add({"objective": 5., "ce": 5., "regularization": 0., "positive_ce": 2., "negative_ce": 3.}, samples=1)
        result = means.finish()
        self.assertEqual(result["ce"], 3.)
        class PlainValuesOnly:
            @staticmethod
            def is_tensor(value):
                return False
        self.assertEqual(_metrics(PlainValuesOnly(), result)["sample_count"], 3)
        self.assertEqual(_counts(result["operation_counts"], positive_updates=True)["optimizer_updates"], 2)

    def test_native_sampler_defaults_and_provider_custody(self):
        pin = sampler_custody(native_default_sampler)
        self.assertEqual(pin["source_sha256"], file_sha(Path(__file__)))
        self.assertEqual(pin["function_sha256"], hashlib.sha256(inspect.getsource(native_default_sampler).encode()).hexdigest())
        with self.assertRaisesRegex(ValueError, "defaults changed"):
            sampler_custody(changed_default_sampler)

    def test_file_changed_symlink_and_module_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.py"
            path.write_text("import sys\nassert sys.modules[__name__].__name__ == __name__\nVALUE = 17\n")
            pin = {"path": "source.py", "bytes": path.stat().st_size, "sha256": file_sha(path)}
            self.assertEqual(bound_file(root, pin), path.resolve())
            self.assertEqual(_module(path, "link_registration_fixture").VALUE, 17)
            sys.modules.pop("link_registration_fixture")
            linked = root / "linked.py"
            linked.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "symlinked"):
                bound_file(root, {**pin, "path": "linked.py"})
            path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "receipt changed"):
                bound_file(root, pin)

    def test_preparation_authority_excludes_test_and_no_cache_accessor(self):
        research = Path(os.environ.get("GNNM_REFERENCE_PROJECT", str(Path(__file__).resolve().parents[2]))).resolve()
        original = json.loads((research / "graph_ncNC_valid_data_authority_root_20261003_v1/DATA_AUTHORITY.json").read_text())
        validate_authority(original)
        altered = copy.deepcopy(original)
        altered["files"]["split/time/test.pt"] = receipt("test.pt")
        with self.assertRaisesRegex(ValueError, "exactly the four"):
            validate_authority(altered)
        altered = copy.deepcopy(original)
        altered["files"]["split/time/train.pt"]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "pin changed"):
            validate_authority(altered)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = {"files": {}}
            for name in SOURCE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                authority["files"][name] = {"bytes": path.stat().st_size, "sha256": file_sha(path)}
            (root / "split/time/test.pt").write_text("unreadable decoy")
            paths = authenticate_cache(root, authority)
            self.assertEqual(set(paths), set(SOURCE_FILES))


@unittest.skipIf(torch is None, "Torch unavailable; no installation or scientific execution")
class LinkTensorTests(unittest.TestCase):
    def packs(self):
        positive = torch.tensor([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]])
        return {"x": torch.arange(36, dtype=torch.float32).reshape(12, 3)/36,
                "raw_edge_index": torch.stack((positive, positive.flip(1)), dim=1).reshape(-1, 2).T.contiguous()}, \
               {"positive": positive}, {"positive": torch.tensor([[4, 5], [6, 7], [0, 4]]), "negative": torch.tensor([[8, 9], [9, 10], [10, 11]])}

    def cfg(self):
        value = config()
        value["metric"]["hits_k"] = 2
        return value

    def test_full_raw_multiset_and_isolated_nodes_preserved(self):
        graph, train, select = self.packs()
        audit = validate_inputs(torch, graph, train, select, self.cfg())
        self.assertEqual((audit["nodes"], audit["train_records"], audit["raw_directed_records"]), (12, 5, 10))
        graph["raw_edge_index"] = graph["raw_edge_index"][:, :-1]
        with self.assertRaisesRegex(ValueError, "exact bidirectional"):
            validate_inputs(torch, graph, train, select, self.cfg())

    def test_heldout_reverse_edges_never_enter_support(self):
        graph, train, select = self.packs()
        select["positive"][0] = torch.tensor([1, 0])
        with self.assertRaisesRegex(ValueError, "leak into TRAIN"):
            validate_inputs(torch, graph, train, select, self.cfg())

    def test_declared_temporal_occurrences_preserve_recurrent_pair_queries(self):
        graph, train, select = self.packs()
        select["positive"][0] = torch.tensor([1, 0])
        select["negative"][0] = torch.tensor([2, 0])
        cfg = self.cfg()
        cfg["support"]["select_overlap"] = "official_temporal_records"
        audit = validate_inputs(torch, graph, train, select, cfg)
        self.assertEqual((audit["select_positive_train_pair_recurrences"], audit["select_negative_train_pair_recurrences"]), (1, 1))
        self.assertTrue(audit["heldout_pair_identities_already_in_train"])
        self.assertFalse(audit["heldout_records_added_to_support"])
        self.assertEqual(select["positive"][0].tolist(), [1, 0])
        graph, train, select = self.packs()
        select["negative"][0] = select["positive"][0].flip(0)
        with self.assertRaisesRegex(ValueError, "contradict"):
            validate_inputs(torch, graph, train, select, self.cfg())

    def test_native_duplicate_records_are_explicit(self):
        graph, train, select = self.packs()
        train["positive"] = torch.cat((train["positive"], train["positive"][:1]), dim=0)
        graph["raw_edge_index"] = torch.cat((train["positive"], train["positive"].flip(1)), dim=0).T.contiguous()
        validate_inputs(torch, graph, train, select, self.cfg())
        cfg = self.cfg()
        cfg["support"]["record_duplicate_policy"] = "require_unique_pairs"
        with self.assertRaisesRegex(ValueError, "unique canonical"):
            validate_inputs(torch, graph, train, select, cfg)

    def test_official_negative_self_record_requires_explicit_preservation(self):
        graph, train, select = self.packs()
        select["negative"][0] = torch.tensor([8, 8])
        cfg = self.cfg()
        with self.assertRaisesRegex(ValueError, "self-link"):
            validate_inputs(torch, graph, train, select, cfg)
        cfg["support"]["select_negative_self_policy"] = "preserve_official_records"
        audit = validate_inputs(torch, graph, train, select, cfg)
        self.assertEqual(audit["select_negative_self_pairs"], 1)
        self.assertEqual(select["negative"][0].tolist(), [8, 8])


LINK_RUNTIME = torch is not None and all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch_geometric", "torch_sparse", "ogb"))


@unittest.skipUnless(LINK_RUNTIME, "Link runtime providers unavailable; root owns allocation QA")
class LinkCPUFixtureTests(LinkTensorTests):
    def fixture(self, root, predictor="NCNC", arm="native_single"):
        from torch_geometric.utils import negative_sampling
        from ogb.linkproppred import Evaluator
        research = Path(os.environ.get("GNNM_REFERENCE_PROJECT", str(Path(__file__).resolve().parents[2]))).resolve()
        source = research / "graph_link_competing_family_source_20261003_v1/source"
        row = run(arm, predictor)
        row["config"]["encoder"]["hidden"] = 8
        row["config"]["training"].update(batch_size=2, eval_batch_size=2)
        row["config"]["metric"].update(hits_k=2, evaluator_source_sha256=file_sha(Path(inspect.getfile(Evaluator))))
        row["selection"]["metric"] = "hits@2"
        row["budget"]["maximum"] = 1
        pin = sampler_custody(negative_sampling)
        row["config"]["sampler"] = {k: pin[k] for k in ("name", "source_sha256", "function_sha256")}
        (root / "source").mkdir(exist_ok=True)
        for name in ("model", "utils"):
            path = root / "source" / (name+".py")
            shutil.copyfile(source / ("ncn__"+name+".py"), path)
            self.assertEqual(file_sha(path), AUTHOR_SOURCE[name]["sha256"])
        (root / "roles").mkdir(exist_ok=True)
        for role, pack in zip(("graph", "train", "select"), self.packs()):
            path = root / "roles" / (role+".pt")
            torch.save(pack, path)
            row["config"]["link_inputs"][role] = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": file_sha(path)}
        return row

    def test_exact_source_full_epoch_select_tail_and_strict_checkpoint_replay(self):
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for predictor in ("NCN", "NCNC"):
                with self.subTest(predictor=predictor), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    row = self.fixture(root, predictor)
                    adapter = make_adapter(torch=torch, project_root=root, study=study(row), run=row, device="cpu")
                    self.assertEqual(type(adapter.model.decoder).__name__, "CNLinkPredictor" if predictor == "NCN" else "IncompleteCN1Predictor")
                    spec = {"fit_id": row["run_id"], "budget_steps": 1, "budget_unit": "epochs", "validation_interval": 1,
                            "validate_initial": False, "prefixes": [], "inherited_fit_ids": [], "adapter_spec": {"config": row["config"]}, "recovery_interval": 1}
                    output = root / "engineering-only-output"
                    result = run_trajectory(torch, adapter, spec, output)
                    self.assertEqual(result["status"], "complete", result)
                    counts = result["operation_counts"]
                    self.assertEqual((counts["optimizer_updates"], counts["backward_calls"], counts["dropped_positive_tail"]), (2, 2, 1))
                    self.assertEqual(counts["supervised_examples"], 8)
                    self.assertEqual(counts["initial_full_support_builds"], 1)
                    self.assertEqual(counts["initial_support_input_records"], 5)
                    self.assertGreaterEqual(counts["negative_records_drawn"], 5)
                    self.assertEqual(result["selected"]["step"], 1)
                    selected = adapter.predict_queries(adapter.select["positive"], adapter.select["negative"])
                    self.assertEqual((len(selected[0]), len(selected[1])), (3, 3))
                    self.assertEqual(selected[2]["decoder_query_passes"], 4)
                    if predictor == "NCNC":
                        self.assertGreater(selected[2]["native_decoder_completed_calls"], 4)
                    with torch.no_grad():
                        for parameter in adapter.model.parameters():
                            parameter.add_(17.)
                    adapter._best = -1.
                    restore_checkpoint(torch, output / "selected.pt", adapter, expected_fit_id=row["run_id"])
                    replay = adapter.predict_queries(adapter.select["positive"], adapter.select["negative"])
                    self.assertTrue(torch.equal(selected[0], replay[0]))
                    self.assertTrue(torch.equal(selected[1], replay[1]))
                    self.assertEqual(adapter.validate(1)["selection_metric"], result["selected"]["validation"]["selection_metric"])
                    # Only the declared training tail policy changes coverage.
                    adapter.config["training"]["tail_policy"] = "complete_tail"
                    epoch = adapter.train_step(2)
                    self.assertEqual((epoch["sample_count"], epoch["positive_denominator"], epoch["operation_counts"]["optimizer_updates"]), (5, 5, 3))
                    self.assertEqual(epoch["operation_counts"]["initial_full_support_builds"], 0)
        finally:
            torch.set_num_threads(original_threads)

    def test_genuine_i4_storage_private_rng_and_partial_failure_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = self.fixture(root, "NCN", "native_independent4_body")
            rows = []
            for body in range(4):
                row = copy.deepcopy(template)
                row["body"], row["run_id"] = body, "paid-body-"+str(body)
                row["config"]["initialization_seed"] += body*1009
                rows.append(row)
            declaration = study(rows[0], rows)
            adapters = [make_adapter(torch=torch, project_root=root, study=declaration, run=row, device="cpu") for row in rows]
            self.assertEqual(inspect_independence(adapters)["independent_adapters"], 4)
            before = adapters[1].capture_state()["fit_rng"]["torch_cpu"].clone()
            adapters[0].train_step(1)
            self.assertTrue(torch.equal(before, adapters[1].capture_state()["fit_rng"]["torch_cpu"]))
            def failure(*args):
                raise RuntimeError("synthetic sampler failure")
            adapters[2].sampler = failure
            with self.assertRaises(AdapterStepError) as caught:
                adapters[2].train_step(1)
            self.assertEqual(caught.exception.operation_counts["sampler_calls"], 1)
            self.assertEqual(caught.exception.operation_counts["optimizer_updates"], 0)
            self.assertIsNone(adapters[2]._active_counts)

    def test_exact_source_declared_temporal_and_negative_self_query_serving(self):
        for predictor in ("NCN", "NCNC"):
            with self.subTest(predictor=predictor), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                row = self.fixture(root, predictor)
                pack = self.packs()[2]
                pack["positive"][0] = torch.tensor([1, 0])
                pack["negative"][0] = torch.tensor([8, 8])
                path = root/"roles/select.pt"
                torch.save(pack, path)
                row["config"]["link_inputs"]["select"] = {"path": "roles/select.pt", "bytes": path.stat().st_size, "sha256": file_sha(path)}
                row["config"]["support"].update(select_overlap="official_temporal_records", select_negative_self_policy="preserve_official_records")
                adapter = make_adapter(torch=torch, project_root=root, study=study(row), run=row, device="cpu")
                audit = adapter.provenance["input_audit"]
                self.assertEqual((audit["select_positive_train_pair_recurrences"], audit["select_negative_self_pairs"]), (1, 1))
                self.assertFalse(audit["heldout_records_added_to_support"])
                metrics = adapter.validate(0)
                self.assertEqual((metrics["positive_count"], metrics["negative_count"], metrics["operation_counts"]["optimizer_updates"]), (3, 3, 0))


if __name__ == "__main__":
    unittest.main()
