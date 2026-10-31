"""Metadata/scoped provider/synthetic launcher checks; no actual-input fits."""
import copy
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "runtime"))
from sync_common import bound_master, canonical, fingerprint, read_json, receipt, safe, verify_counts
from runtime_requirements import metadata_environment, source_imports
from make_final_master_bundle import write_bundle


class FinalSyncTests(unittest.TestCase):
    def test_exact_master_summary_bindings_reject_stale_hash_or_cross_plan(self):
        with tempfile.TemporaryDirectory(dir=HERE / "tests") as directory:
            root = Path(directory)
            plan = {"studies": [], "scientific_launch_enabled": False, "human_GO_required": True}
            (root / "PLAN.json").write_text(canonical(plan))
            summary = {"plan_hash": fingerprint(plan), "counts": {}}
            (root / "SUMMARY.json").write_text(canonical(summary))
            digest = receipt(root, "SUMMARY.json")["sha256"]
            bound_master(root, "PLAN.json", fingerprint(plan), "SUMMARY.json", digest)
            with self.assertRaisesRegex(ValueError, "Canonical"):
                bound_master(root, "PLAN.json", "0" * 64, "SUMMARY.json", digest)
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                bound_master(root, "PLAN.json", fingerprint(plan), "SUMMARY.json", "0" * 64)
            summary["plan_hash"] = "0" * 64
            (root / "SUMMARY.json").write_text(canonical(summary))
            with self.assertRaisesRegex(ValueError, "another master"):
                bound_master(root, "PLAN.json", fingerprint(plan), "SUMMARY.json", receipt(root, "SUMMARY.json")["sha256"])

    def test_bounded_optimizer_totals_remain_unknown_and_counts_are_dynamic(self):
        studies = [{"runs": [{"kind": "model_fit"}, {"kind": "calibration"}, {"kind": "model_fit"}]}]
        counts = {"totals": {"run_records": 3, "atomic_fit_run_groups": 2},
                  "model_optimizer_step_call_total_fully_determined": False,
                  "model_optimizer_step_call_total": None, "model_optimizer_step_call_minimum": 31,
                  "model_optimizer_step_call_maximum": 700,
                  "actual_model_optimizer_updates": 0, "actual_scalar_CAL_updates": 0, "actual_predictive_scores": 0}
        compiler = SimpleNamespace(typed_counts=lambda _: copy.deepcopy(counts))
        plan = {"studies": [{}], "immutable_source_bundle": {"files": [{}] * 4}}
        actual = verify_counts(plan, {"counts": counts}, studies, compiler)
        self.assertEqual((actual["studies"], actual["registered_records"], actual["model_groups"], actual["controller_files"]), (1, 3, 2, 4))
        counts["model_optimizer_step_call_total"] = 700
        with self.assertRaisesRegex(ValueError, "remain null"):
            verify_counts(plan, {"counts": counts}, studies, compiler)

    def test_overlay_dependency_metadata_recurses_without_importing_provider(self):
        with tempfile.TemporaryDirectory(dir=HERE / "tests") as directory:
            root = Path(directory)
            for name, version, requires in (("SyntheticDGL", "2.4.0", "Requires-Dist: SyntheticTorchData (==0.7.0)\n"),
                                             ("SyntheticTorchData", "0.7.0", "")):
                path = root / (name + "-" + version + ".dist-info")
                path.mkdir()
                (path / "METADATA").write_text("Metadata-Version: 2.1\nName: " + name + "\nVersion: " + version + "\n" + requires)
            result = metadata_environment([str(root)], ["SyntheticDGL"])
            self.assertEqual(result["distributions"]["synthetictorchdata"]["version"], "0.7.0")
            self.assertFalse(result["missing_distribution_metadata"])
            self.assertEqual(result["scientific_imports"], 0)
            self.assertNotIn("SyntheticDGL", sys.modules)

    def test_AST_preserves_dynamic_provider_and_optional_native_requirements(self):
        with tempfile.TemporaryDirectory(dir=HERE / "tests") as directory:
            root = Path(directory)
            (root / "port.py").write_text("import torch\nimport importlib\nimportlib.import_module('torch_householder')\ntry:\n import dgl\nexcept ImportError:\n pass\n")
            result = source_imports(root, [receipt(root, "port.py")])
            self.assertEqual(set(result["external_module_source_paths"]), {"torch", "torch_householder", "dgl"})
            self.assertEqual(result["module_distribution_candidates"]["torch_householder"], ["torch-householder"])

    def test_opaque_archive_stream_validates_changed_input_and_preserves_failure(self):
        with tempfile.TemporaryDirectory(dir=HERE / "tests") as directory:
            root = Path(directory)
            (root / "opaque.pt").write_bytes(b"opaque bytes, never deserialized")
            row = receipt(root, "opaque.pt")
            result = write_bundle(root, [row], [], root / "valid.tar")
            self.assertEqual(result["input_stream_hash_passes_per_path"], 1)
            with tarfile.open(root / "valid.tar") as archive:
                self.assertEqual(archive.getnames(), ["opaque.pt"])
                self.assertEqual(archive.extractfile("opaque.pt").read(), (root / "opaque.pt").read_bytes())
            (root / "opaque.pt").write_bytes(b"x" * row["bytes"])
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                write_bundle(root, [row], [], root / "failed.tar")
            self.assertTrue((root / "failed.tar").exists())

    def test_actual_frozen_launcher_metadata_audit_on_synthetic_opaque_roles(self):
        original = Path(os.environ.get("FINAL_SYNC_TEST_CONTROLLER", HERE.parent / "native_hgt_mixed_newdata_packet_20261012_v2/controller_v16_base"))
        if not (original / "launcher.py").is_file():
            self.skipTest("Explicit frozen metadata controller required")
        with tempfile.TemporaryDirectory(dir=HERE / "tests") as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            for path in original.iterdir():
                if path.is_file() and path.suffix in (".py", ".sh"):
                    shutil.copyfile(path, runtime / path.name)
            if not (runtime / "start_massive.sh").exists():
                (runtime / "start_massive.sh").write_text("# synthetic startup is never executed\n")
            sys.path.insert(0, str(runtime))
            sync = importlib.import_module("prepare_sync_bundle")
            compiler = importlib.import_module("compile_master_plan")
            generic = importlib.import_module("generic_manifest")
            (root / "AGENTS.md").write_text("synthetic phase fixture")
            (root / "adapter.py").write_text("# synthetic source is never executed\n")
            for role in ("train", "select", "cal", "evaluate"):
                (root / (role + ".pt")).write_bytes(b"opaque synthetic role; not a tensor pickle")
            run = {"run_id": "fit", "kind": "model_fit", "adapter": "adapter.py:make_adapter", "method": "single",
                   "split": 0, "seed": 1, "body": 0, "parents": [],
                   "config": {"stage": "search", "architecture": "single", "members": 1, "reservation_mib": 1, "estimated_seconds": .01,
                              "node_inputs": {"train": "train.pt", "select": "select.pt"},
                              "endpoint_inputs": {"cal_targets": "cal.pt", "evaluate_targets": "evaluate.pt"}},
                   "budget": {"unit": "optimizer_updates", "policy": "full_budget", "maximum": 7},
                   "selection": {"metric": "accuracy", "direction": "max", "ties": "first", "evaluation_interval": 1, "checkpoint0": False}}
            study = {"schema_version": 1, "study_id": "synthetic-small", "stage": "engineering", "hypothesis": "source access fixture",
                     "dataset": "synthetic-compatible", "data": {"task": "node_classification", "preprocessing": {}, "prior_exposure": "engineering_only",
                         "roles": {r: r + ".pt" for r in ("train", "select", "cal", "evaluate")},
                         "evidence": {"provider_files": [{"path": str(root / "absent-provider.pt"), "bytes": 999, "sha256": "0" * 64}]}},
                     "source": {"git_commit": "synthetic", "files": [receipt(root, "adapter.py")]}, "runs": [run],
                     "comparisons": [{"references": ["single"]}], "release": {"policy": "whole_roster", "primary_endpoint": "synthetic"}, "resources": {"wall_seconds": 1}}
            (root / "study.json.gz").write_bytes(gzip.compress(canonical(study).encode(), mtime=0))
            environment = {"packages": dict(zip(sync.PACKAGES, ["synthetic"] * len(sync.PACKAGES)))}
            plan = {"schema_version": 1, "slots_per_gpu": 2, "scientific_launch_enabled": False, "human_GO_required": True,
                    "runtime_package": "runtime", "runtime_launcher": "runtime/launcher.py", "runtime_requirements": environment,
                    "studies": [{"path": "study.json.gz", "spec_hash": fingerprint(study)}],
                    "groups": [{"id": "tiny", "run_keys": ["synthetic-small/fit"], "gpu_assignment": "allocation/gpu0"}],
                    "phase_rendezvous": {"remote_name": "origin", "remote_url_sha256": "1" * 64},
                    "immutable_source_bundle": {"files": [receipt(root, "runtime/" + p.name) for p in runtime.iterdir()]}}
            (root / "PLAN.json").write_text(canonical(plan))
            summary = {"plan_hash": fingerprint(plan), "counts": compiler.typed_counts([study])}
            (root / "SUMMARY.json").write_text(canonical(summary))
            manifest = generic.manifest(root, "PLAN.json", environment, plan["immutable_source_bundle"]["files"])
            manifest["exact_summary_binding"] = receipt(root, "SUMMARY.json")
            manifest["expected_counts"] = verify_counts(plan, summary, [study], compiler)
            manifest["files"].append(dict(receipt(root, "SUMMARY.json"), reasons=["bound_summary"]))
            (root / "MANIFEST.json").write_text(canonical(manifest))
            command = [sys.executable, "-B", "-c", "import sync_common;sync_common.allocation_guard=lambda:None;import audit_final_master_source_loading as a;a.main()",
                       "--project-root", str(root), "--manifest", "MANIFEST.json", "--manifest-sha256", receipt(root, "MANIFEST.json")["sha256"],
                       "--master", "PLAN.json", "--master-hash", fingerprint(plan), "--summary", "SUMMARY.json", "--summary-sha256", receipt(root, "SUMMARY.json")["sha256"], "--output", "AUDIT.json"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(HERE / "runtime"))
            result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            audit = json.loads((root / "AUDIT.json").read_text())
            self.assertEqual((audit["loaded_studies"], audit["registered_records"]), (1, 1))
            self.assertEqual(audit["tensor_arrays_or_targets_deserialized"], 0)
            self.assertTrue(audit["optional_provider_paths_denied_and_unused"])
            # The old manifest expected Polyformer-only directory keys. A
            # synthetic OGB author contract must preserve its native keys.
            (root / "author_processed").mkdir()
            (root / "provider.py").write_text("# immutable synthetic provider\n")
            author = copy.deepcopy(run)
            author.update(run_id="author", kind="author_cli_group", cli={
                "source_files": [receipt(root, "adapter.py")], "raw_files": [receipt(root, "train.pt")],
                "runtime_provider_files": [receipt(root, "provider.py")], "fresh_processed_dir": "author_processed",
                "quarantine": {"stdout": "new_quarantine/stdout.log"}, "argv": ["main.py"],
                "execution_repository": "/synthetic/repo", "python_executable": "/synthetic/repo/.venv/bin/python",
                "environment": {"OMP_NUM_THREADS": "2"}, "runtime_versions": {"torch": "synthetic"}})
            author_study = copy.deepcopy(study)
            author_study["runs"] = [author]
            (root / "study.json.gz").write_bytes(gzip.compress(canonical(author_study).encode(), mtime=0))
            plan["studies"][0]["spec_hash"] = fingerprint(author_study)
            plan["groups"][0]["run_keys"] = ["synthetic-small/author"]
            (root / "PLAN.json").write_text(canonical(plan))
            final = generic.manifest(root, "PLAN.json", environment, plan["immutable_source_bundle"]["files"])
            self.assertEqual(final["empty_directories"], ["author_processed"])
            self.assertIn("provider.py", {row["path"] for row in final["files"]})
            self.assertIsNone(final["allocation_only_author_contracts"][0]["private_dependency_dir"])


if __name__ == "__main__":
    unittest.main()
