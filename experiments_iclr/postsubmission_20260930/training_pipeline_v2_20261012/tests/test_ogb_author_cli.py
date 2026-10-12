"""Focused pure metadata/parser/process fixtures; no OGB data/model fits or numerical imports."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from author_cli_spec import author_counts, register_author_spec, validate_author_spec
from ogb_author_cli_spec import (AUTHOR_FAMILY, BUDGET, DATASETS, ENVIRONMENT, MODELS,
                                 RAW_FILES, SEEDS, SEED_POLICY, SELECTION, SOURCE_COMMIT,
                                 SOURCE_PINS, seeded_argv, validate_ogb_author_spec)
from registry import Registry
from run_ogb_author_cli import OGBProgress, execute_claimed_ogb_group
from spec import canonical, fingerprint, validate


def fake_receipt(path, size=1, sha="a" * 64):
    return {"path": str(path), "bytes": size, "sha256": sha}


def fixture():
    wrapper = fake_receipt("runtime/ogb_declared_seed_entry.py")
    custody = {}
    for dataset in DATASETS:
        custody[dataset] = {"receipt": fake_receipt("custody/" + dataset + ".json"),
            "raw_files": [fake_receipt(Path("provider") / dataset.replace("-", "_") / path) for path in sorted(RAW_FILES)],
            "processed_cache_policy": "fresh_empty_per_process"}
    runs = []
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                identifier = dataset + "-" + model + "-seed" + str(seed)
                work = Path("packet/workdirs") / identifier
                source = [fake_receipt(work / name, size, sha) for name, (size, sha) in SOURCE_PINS.items()]
                source.append(fake_receipt(work / "declared_seed_entry.py"))
                runs.append({"run_id": identifier, "kind": "author_cli_group", "method": "synthetic-log-observation-fixture",
                    "split": "official_scaffold", "seed": seed, "body": 0, "parents": [],
                    "config": {"stage": "author_cli", "author_family": AUTHOR_FAMILY, "dataset": dataset, "source_model": model,
                        "reservation_mib": 1, "estimated_seconds": 1, "wall_seconds": 10,
                        "resource_evidence": {"status": "SYNTHETIC_NOT_SOURCE_RESOURCE_QUALIFIED"},
                        "max_train_batches_per_epoch": DATASETS[dataset]["max_train_batches"],
                        "source_code_path_replication": True, "published_score_equality_claim": False},
                    "budget": dict(BUDGET), "selection": dict(SELECTION),
                    "cli": {"cwd": str(work), "argv": seeded_argv(dataset, model, seed), "environment": dict(ENVIRONMENT),
                        "protocol": copy.deepcopy(SEED_POLICY), "python_executable": sys.executable,
                        "execution_repository": "/synthetic-opaque-repository", "runtime_versions": {"opaque": "fixture"},
                        "runtime_provider_files": [fake_receipt("runtime/provider.py")], "source_files": source,
                        "raw_files": [fake_receipt(work / "dataset" / dataset.replace("-", "_") / path) for path in sorted(RAW_FILES)],
                        "fresh_processed_dir": str(work / "dataset" / dataset.replace("-", "_") / "processed"),
                        "quarantine": {"stdout": "packet/quarantine/" + identifier + ".stdout.log",
                                       "stderr": "packet/quarantine/" + identifier + ".stderr.log"}}})
    return {"schema_version": 1, "study_id": "synthetic-ogb-parser-spec-fixture", "stage": "engineering",
        "hypothesis": "Opaque metadata and source-stream observation only", "dataset": "ogb-two-molecule-native-author-roster",
        "source": {"author_family": AUTHOR_FAMILY, "repository": "snap-stanford/ogb", "git_commit": SOURCE_COMMIT,
                   "replication_seed_policy": copy.deepcopy(SEED_POLICY), "published_score_equality_claim": False,
                   "native_CLI_modified": False, "files": [wrapper], "seed_entry": wrapper, "packet_root": "packet"},
        "data": {"roles": {"train": "opaque", "select": "opaque", "evaluate": "opaque"},
                 "role_protocol": "official_ogb_scaffold_train_whole_valid_test_no_cal", "prior_exposure": "engineering",
                 "evidence": ["synthetic opaque custody"], "train_graph_count_authority": copy.deepcopy(DATASETS),
                 "dataset_custody": custody},
        "runs": runs, "comparisons": [{"engineering_only": True}],
        "release": {"policy": "whole_roster", "global_plan_gates_required": True,
                    "required_before_launch": ["whole_search"], "author_output_gate": "whole_author"},
        "resources": {"wall_seconds": 800}}


def stream(seed=SEEDS[0], count=100):
    yield b"DECLARED_SEED_PROCESS_START " + str(seed).encode()
    for epoch in range(1, count + 1):
        yield b"=====Epoch " + str(epoch).encode()
        yield b"{'Train': {'rocauc': 0.1}, 'Validation': {'rocauc': 0.2}, 'Test': {'rocauc': NEVER_PARSE_TEST_FLOAT_SENTINEL}}"
    yield b"Finished training!"
    yield b"Best validation score: 0.2"
    yield b"Test score: FINAL_TEST_FLOAT_SENTINEL"
    yield b"DECLARED_SEED_PROCESS_RETURN 0"


class OGBAuthorTests(unittest.TestCase):
    def test_finite_roster_dispatch_counts_and_exact_numeric_source_rejection(self):
        value = fixture()
        self.assertIs(validate_ogb_author_spec(value), value)
        self.assertIs(validate_author_spec(value), value)
        self.assertIs(validate(value), value)
        counts = author_counts(value)
        self.assertEqual(counts["author_cli_groups"], 80)
        self.assertEqual(counts["author_independent_seed_trajectories"], 80)
        self.assertEqual(counts["author_registered_maximum_optimizer_updates"], 4268000)
        self.assertEqual(counts["model_fit_groups"], 0)
        for mutation in ["missing_seed", "seed", "batch", "source", "cache", "TEST_path", "CAL"]:
            wrong = copy.deepcopy(value)
            if mutation == "missing_seed": wrong["runs"].pop()
            elif mutation == "seed": wrong["runs"][0]["seed"] = 42
            elif mutation == "batch": wrong["runs"][0]["cli"]["argv"][wrong["runs"][0]["cli"]["argv"].index("--batch_size") + 1] = "64"
            elif mutation == "source": wrong["runs"][0]["cli"]["source_files"][0]["sha256"] = "b" * 64
            elif mutation == "cache": wrong["runs"][0]["cli"]["raw_files"][0]["sha256"] = "b" * 64
            elif mutation == "TEST_path": wrong["runs"][0]["cli"]["quarantine"]["stdout"] = "../outside.log"
            else: wrong["data"]["roles"]["cal"] = "invented"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_author_spec(wrong)

    def test_TEST_opaque_and_earliest_validation_tie(self):
        observer = OGBProgress(SEEDS[0], 1029)
        events = [x for line in stream() if (x := observer.feed(line)) is not None]
        public = canonical(events) + canonical(observer.costs(child_started=True, exit_code=0))
        self.assertNotIn("SENTINEL", public)
        self.assertEqual(observer.best_epoch, 1)
        self.assertTrue(observer.costs(child_started=True, exit_code=0)["complete_native_process_observation"])
        self.assertIsNone(observer.costs(child_started=True, exit_code=0)["actual_optimizer_updates"])
        self.assertEqual(observer.costs(child_started=True, exit_code=0)["registered_maximum_optimizer_updates"], 102900)

    def test_missing_epoch_nonfinite_valid_and_wrong_seed_fail_closed(self):
        for variant in ["missing", "nan", "wrong_seed", "duplicate"]:
            observer = OGBProgress(SEEDS[0], 38)
            lines = list(stream())
            if variant == "missing": lines.pop(4)
            elif variant == "nan": lines[2] = lines[2].replace(b"'Validation': {'rocauc': 0.2}", b"'Validation': {'rocauc': nan}")
            elif variant == "wrong_seed": lines[0] = b"DECLARED_SEED_PROCESS_START 42"
            else: lines.insert(3, lines[2])
            for line in lines: observer.feed(line)
            with self.subTest(variant=variant):
                self.assertFalse(observer.costs(child_started=True, exit_code=0)["complete_native_process_observation"])
                self.assertTrue(observer.anomalies)

    def test_pure_synthetic_subprocess_registry_terminal_and_quarantine(self):
        value = fixture(); run = value["runs"][0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work = root / "fake"; work.mkdir()
            lines = [line.decode() for line in stream()]
            source = "import time,sys\ntime.sleep(.03)\n" + "\n".join("print(" + repr(line) + ",flush=True)" for line in lines) + "\nsys.stderr.write('STDERR_TEST_SENTINEL\\n')\n"
            (work / "fake.py").write_text(source)
            registry = Registry(root / "registry.sqlite")
            try:
                register_author_spec(registry, value)
                owner = {"pid": os.getpid(), "hostname": "synthetic-fixture"}
                registry.claim(value["study_id"], run["run_id"], owner)
                result = execute_claimed_ogb_group(root, value, run, registry, root / "result", owner,
                    ([sys.executable, "-u", "fake.py"], work, dict(os.environ)), cuda_absence_check=lambda _: True)
                self.assertEqual(result["terminal_status"], "COMPLETE_EXIT_0")
                self.assertTrue(result["child_pid_absent"])
                self.assertTrue(result["owned_cuda_pid_absent"])
                self.assertEqual(result["cost"]["completed_native_epochs_observed"], 100)
                self.assertIsNone(result["cost"]["actual_optimizer_updates"])
                safe = canonical(result) + (root / "result/TRAIN_VALID_PROGRESS.jsonl").read_text()
                self.assertNotIn("SENTINEL", safe)
                self.assertIn("SENTINEL", (root / run["cli"]["quarantine"]["stdout"]).read_text())
                start = json.loads((root / "result/OWNER_START.json").read_text())
                end = json.loads((root / "result/OWNER_END.json").read_text())
                for field in ["owner", "child_owner", "command"]: self.assertEqual(start[field], end[field])
                with self.assertRaisesRegex(ValueError, "already claimed"):
                    registry.claim(value["study_id"], run["run_id"], owner)
            finally:
                registry.close()


if __name__ == "__main__":
    unittest.main()
