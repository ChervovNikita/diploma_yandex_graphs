"""Stdlib synthetic ledger/source checks; never a numerical CPU certificate."""
import copy
import json
from pathlib import Path
import tempfile
import sys
from types import ModuleType
import unittest

from guards import (HERE, file_sha, read_json, strict_json, implementation_hashes,
                    expected_parameters, selection_record_sha, validate_history,
                    validate_run_metadata, verify_family_metadata)
from cache_builder import checked_split_loader_source, official_split_file, load_registered_module


class GuardChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".guard-fixture-", dir=HERE)
        self.root = Path(self.temporary.name)
        self.config = read_json(HERE / "CONFIG.json")
        self.sources = implementation_hashes()
        self.cache_sha = "a" * 64

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, path, value):
        path.write_text(json.dumps(value, allow_nan=False) + "\n")

    def fixture(self, arm="single256", seed=0):
        folder = self.root / f"{arm}_seed{seed}"
        folder.mkdir()
        identity = {"arm": arm, "seed": seed, "cache_manifest_sha256": self.cache_sha,
                    "config_sha256": file_sha(HERE / "CONFIG.json"), "implementation_hashes": self.sources,
                    "parameters": expected_parameters(arm), "optimizer_fits": 4 if arm == "independent4" else 1,
                    "pooling": "mean_raw_logits", "device": "cpu", "torch_version": "synthetic-no-torch"}
        self.write_json(folder / "identity.json", identity)
        # Opaque fixture bytes intentionally satisfy metadata only. Production
        # wrappers must additionally pass the real Torch checkpoint validator.
        (folder / "selected.pt").write_bytes(b"not a model; metadata-only fixture")
        digest = file_sha(folder / "selected.pt")
        records = [{"epoch": epoch, "train_bce": 0.7, "train_seconds": 1.0,
                    "validation_forward_seconds": 0.1, "validation_hits50": 0.1 if epoch == 1 else 0.5,
                    "validation_bce_sampled_pool": 0.6} for epoch in range(1, 101)]
        records[0]["selected_checkpoint_sha256"] = "b" * 64
        records[1]["selected_checkpoint_sha256"] = digest
        self.write_records(folder, records)
        summary = {**identity, "status": "training_complete", "epochs_completed": 100,
                   "total_seconds": 110.0, "peak_cuda_allocated": None, "peak_cuda_reserved": None,
                   "test_loaded_or_scored": False, "identity_sha256": file_sha(folder / "identity.json"),
                   "epochs_sha256": file_sha(folder / "epochs.jsonl"), "selected_epoch": 2,
                   "selected_validation_hits50": 0.5, "selected_record_sha256": selection_record_sha(records[1]),
                   "selected_checkpoint_sha256": digest}
        self.write_json(folder / "completion.json", summary)
        return folder, records, summary

    def write_records(self, folder, records):
        (folder / "epochs.jsonl").write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))

    def validate(self, folder):
        return validate_run_metadata(folder, "single256", 0, self.cache_sha, self.config, self.sources)

    def test_complete_ledger_derives_earliest_maximum_and_binds_bytes(self):
        folder, _, _ = self.fixture()
        run, metadata = self.validate(folder)
        self.assertEqual(run["selected_epoch"], 2)
        self.assertEqual(metadata["selected_epoch"], 2)
        self.assertEqual(run["epochs_sha256"], file_sha(folder / "epochs.jsonl"))

    def test_missing_short_duplicate_and_nonconsecutive_history_rejected(self):
        folder, records, _ = self.fixture()
        (folder / "epochs.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate(folder)
        for bad in (records[:-1], [records[0]] * 100, list(reversed(records))):
            self.write_records(folder, bad)
            with self.assertRaises(RuntimeError):
                self.validate(folder)

    def test_wrong_selection_epoch_score_and_test_access_rejected(self):
        folder, _, summary = self.fixture()
        for changes in ({"selected_epoch": 999}, {"selected_epoch": 3}, {"selected_validation_hits50": 999}, {"test_loaded_or_scored": True}):
            self.write_json(folder / "completion.json", {**summary, **changes})
            with self.assertRaises(RuntimeError):
                self.validate(folder)

    def test_nonfinite_negative_and_out_of_range_records_rejected(self):
        folder, records, _ = self.fixture()
        for field, value in (("train_bce", -1), ("train_seconds", -1), ("validation_bce_sampled_pool", -1), ("validation_hits50", 1.01)):
            bad = copy.deepcopy(records)
            bad[50][field] = value
            self.write_records(folder, bad)
            with self.assertRaises(RuntimeError):
                validate_history(folder / "epochs.jsonl", 100)
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e400}', '{"x":1,"x":2}'):
            with self.assertRaises(RuntimeError):
                strict_json(text)

    def test_tie_checkpoint_and_unbound_history_checkpoint_rejected(self):
        folder, records, _ = self.fixture()
        bad = copy.deepcopy(records)
        bad[2]["selected_checkpoint_sha256"] = "c" * 64
        self.write_records(folder, bad)
        with self.assertRaises(RuntimeError):
            self.validate(folder)
        self.write_records(folder, records)
        (folder / "selected.pt").write_bytes(b"different opaque bytes")
        with self.assertRaises(RuntimeError):
            self.validate(folder)

    def test_actual_vendor_mutation_rejected(self):
        copy_root = self.root / "sources"
        (copy_root / "vendor").mkdir(parents=True)
        (copy_root / "SOURCE_PINS.json").write_bytes((HERE / "SOURCE_PINS.json").read_bytes())
        for name in read_json(HERE / "SOURCE_PINS.json")["files"]:
            (copy_root / "vendor" / name).write_bytes((HERE / "vendor" / name).read_bytes())
        (copy_root / "vendor" / "hashing.py").write_text("changed vendor source")
        with self.assertRaises(RuntimeError):
            implementation_hashes(copy_root)
        self.assertIn("vendor/hashing.py", self.sources)
        self.assertIn("guards.py", self.sources)
        self.assertIn("checkpoint_io.py", self.sources)

    def test_full_family_metadata_then_mutation_rejected(self):
        runs = []
        for arm in self.config["arms"]:
            for seed in self.config["seeds"]:
                folder, _, _ = self.fixture(arm, seed)
                run, _ = validate_run_metadata(folder, arm, seed, self.cache_sha, self.config, self.sources)
                runs.append(run)
        lock = self.root / "family.json"
        self.write_json(lock, {"schema": "buddy-family-lock-v2", "status": "family_locked",
                              "cache_manifest_sha256": self.cache_sha, "config_sha256": file_sha(HERE / "CONFIG.json"),
                              "implementation_hashes": self.sources, "runs": runs})
        _, checkpoints = verify_family_metadata(lock, self.cache_sha, file_sha(HERE / "CONFIG.json"))
        self.assertEqual(len(checkpoints), 15)
        Path(runs[0]["run_directory"], "epochs.jsonl").write_text("{}\n")
        with self.assertRaises(RuntimeError):
            verify_family_metadata(lock, self.cache_sha, file_sha(HERE / "CONFIG.json"))

    def test_loader_mismatch_and_missing_expected_digest_fail_before_torch(self):
        class SyntheticLoader:
            def get_edge_split(self):
                return "train.pt", "valid.pt", "test.pt"
        dataset = SyntheticLoader()
        with self.assertRaises(RuntimeError):
            checked_split_loader_source(dataset, "0" * 64)
        with self.assertRaises(RuntimeError):
            official_split_file(dataset, "test")
        with self.assertRaises(RuntimeError):
            official_split_file(dataset, "test", expected_loader_sha="0" * 64)

    def test_dynamic_module_registered_before_execution_and_reused(self):
        name = f"buddy_stdlib_fixture_{id(self)}"
        path = self.root / "fixture_module.py"
        path.write_text("import sys\nregistered_at_execution = sys.modules[__name__]\nclass Fixture: pass\n")
        try:
            module = load_registered_module(name, path)
            self.assertIs(module.registered_at_execution, module)
            self.assertIs(sys.modules[module.Fixture.__module__], module)
            self.assertIs(load_registered_module(name, path), module)
            path.write_text("changed bytes")
            with self.assertRaisesRegex(RuntimeError, "identity collision"):
                load_registered_module(name, path)
            self.assertIs(sys.modules[name], module)
        finally:
            sys.modules.pop(name, None)

    def test_installed_ogb_metadata_split_basename_without_dataset_split_attribute(self):
        class SyntheticLoader:
            def get_edge_split(self):
                return "train.pt", "valid.pt", "test.pt"
        dataset = SyntheticLoader()
        dataset.root = str(self.root)
        dataset.meta_info = {"split": "time"}
        self.assertFalse(hasattr(dataset, "split"))
        with self.assertRaisesRegex(RuntimeError, "Official train.pt absent"):
            official_split_file(dataset, "train")
        for split in (None, "", ".", "..", "/time", "time/other", "time\\other", "time\x00other"):
            dataset.meta_info = {"split": split}
            with self.assertRaisesRegex(RuntimeError, "Unexpected official OGB split type"):
                official_split_file(dataset, "train")

    def test_dynamic_module_collision_preserves_existing_entry(self):
        name = f"buddy_stdlib_fixture_{id(self)}"
        path = self.root / "fixture_module.py"
        path.write_text("raise AssertionError('must not execute on collision')\n")
        existing = ModuleType(name)
        sys.modules[name] = existing
        try:
            with self.assertRaisesRegex(RuntimeError, "identity collision"):
                load_registered_module(name, path)
            self.assertIs(sys.modules[name], existing)
        finally:
            sys.modules.pop(name, None)

    def test_dynamic_module_failure_restores_absent_entry(self):
        name = f"buddy_stdlib_fixture_{id(self)}"
        path = self.root / "fixture_module.py"
        for body in ("raise RuntimeError('fixture failure')\n",
                     "import sys\nsys.modules[__name__] = object()\nraise RuntimeError('fixture failure')\n"):
            path.write_text(body)
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                load_registered_module(name, path)
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
