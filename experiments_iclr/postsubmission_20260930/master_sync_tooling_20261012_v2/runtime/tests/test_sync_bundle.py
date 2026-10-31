"""Stdlib opaque artifacts only; no tensor loading, model or remote operation."""
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import prepare_sync_bundle as sync


class SyncTests(unittest.TestCase):
    def fixtures(self, root, *, author=False):
        for path, payload in {"AGENTS.md": b"phase marker", "runtime/launcher.py": b"# frozen launcher",
            "runtime/start_massive.sh": b"# frozen startup", "packet/model.py": b"# source",
            "data/train.pt": b"opaque-not-a-torch-pickle", "data/select.pt": b"opaque-select",
            "data/cal.pt": b"opaque-cal-target", "data/evaluate.pt": b"opaque-evaluation-target"}.items():
            target = root / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
        row = {"run_id": "fit", "kind": "model_fit", "adapter": "packet/model.py:make_adapter",
            "config": {"node_inputs": {"train": "data/train.pt", "select": "data/select.pt"},
                       "endpoint_inputs": {"cal_targets": "data/cal.pt", "evaluate_targets": "data/evaluate.pt"}}}
        if author:
            for name in ("author/bases", "author/processed"): (root / name).mkdir(parents=True)
            row.update(kind="author_cli_group", cli={"source_files": [sync.receipt(root, "packet/model.py")],
                "raw_files": [sync.receipt(root, "data/train.pt")], "basis_dir": "author/bases", "processed_dir": "author/processed",
                "quarantine": {"stdout": "author/quarantine/stdout.log"}, "execution_repository": "/allocation/repository",
                "python_executable": "/allocation/repository/.venv/bin/python", "environment": {"PYTHONPATH": "/allocation/overlay"},
                "runtime_versions": {"torch": "source-build"}, "private_dependency_dir": "author/overlay"})
        study = {"study_id": "tiny-study", "dataset": "new-compatible-task", "data": {"roles": {}},
            "source": {"files": [sync.receipt(root, "packet/model.py")]}, "runs": [row]}
        path = root / "master/study.json.gz"; path.parent.mkdir(); path.write_bytes(gzip.compress(sync.canonical(study).encode(), mtime=0))
        plan = {"scientific_launch_enabled": False, "human_GO_required": True, "runtime_package": "runtime",
            "runtime_launcher": "runtime/launcher.py", "host_bindings": {"server77": {"status": "pending"}},
            "phase_rendezvous": {"remote_name": "origin", "remote_url_sha256": "1" * 64},
            "studies": [{"path": "master/study.json.gz", "spec_hash": sync.fingerprint(study)}],
            "groups": [{"run_keys": ["tiny-study/fit"], "gpu_assignment": "allocation/gpu0"}]}
        (root / "master/PLAN.json").write_text(json.dumps(plan))
        return plan, study

    def test_minimal_receipt_union_preserves_opaque_data_specs_and_paths(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
            root = Path(name); self.fixtures(root)
            (root / "unrelated-history.txt").write_text("never included")
            result = sync.manifest(root, "master/PLAN.json", {"packages": {}})
            names = {r["path"] for r in result["files"]}
            self.assertEqual(names, {"AGENTS.md", "runtime/launcher.py", "runtime/start_massive.sh", "packet/model.py",
                "data/train.pt", "data/select.pt", "data/cal.pt", "data/evaluate.pt", "master/PLAN.json", "master/study.json.gz"})
            self.assertEqual(result["statistics"]["files"], 10)
            self.assertFalse(result["tensor_deserialization_performed"])
            self.assertFalse(result["environment_ready_on_server77"])
            sync.verify_manifest(root, result)
            bundle = root / "bundle.tar"; receipt = sync.make_bundle(root, result, bundle)
            self.assertEqual(receipt["bytes"], bundle.stat().st_size)
            self.assertTrue(sync.verify_bundle(result, bundle)["opaque_hashes_verified"])
            with tarfile.open(bundle) as tar:
                self.assertTrue(all(m.isfile() for m in tar.getmembers()))
            destination = root / "destination"; destination.mkdir()
            self.assertEqual(sync.install_bundle(destination, result, bundle)["new_files_installed"], 10)
            self.assertEqual(sync.install_bundle(destination, result, bundle)["new_files_installed"], 0)
            (destination / "data/train.pt").chmod(0o644); (destination / "data/train.pt").write_bytes(b"preserve-conflict")
            with self.assertRaisesRegex(ValueError, "preserve it"): sync.install_bundle(destination, result, bundle)
            (root / "data/cal.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"): sync.verify_manifest(root, result)

    def test_author_absolute_environment_pinned_allocation_and_emptydirs_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
            root = Path(name); plan, _ = self.fixtures(root, author=True)
            result = sync.manifest(root, "master/PLAN.json", {})
            self.assertEqual(result["empty_directories"], ["author/bases", "author/processed"])
            author = result["allocation_only_author_contracts"][0]
            self.assertEqual(author["absolute_python_executable"], "/allocation/repository/.venv/bin/python")
            self.assertFalse(author["execute_or_rewrite_author_contract_on_server77"])
            sync.make_bundle(root, result, root / "author.tar")
            plan["groups"][0]["gpu_assignment"] = "server77/gpu0"; (root / "master/PLAN.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "remain allocated"): sync.manifest(root, "master/PLAN.json", {})

    def test_forbidden_escape_symlink_and_conflicting_receipts_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
            root = Path(name); plan, study = self.fixtures(root)
            for path in ("../outside", "/absolute", ".git/config", "execution/checkpoint.pt", ".ssh/id_rsa", "private_key.pem", "logs/out.log", "a/./b"):
                with self.assertRaises(ValueError): sync.safe(root, path, file=False)
            (root / "link").symlink_to(root / "data", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinks"): sync.safe(root, "link/train.pt")
            bad = sync.receipt(root, "packet/model.py"); bad["sha256"] = "0" * 64
            study["source"]["files"].append(bad)
            (root / "master/study.json.gz").write_bytes(gzip.compress(sync.canonical(study).encode(), mtime=0))
            plan["studies"][0]["spec_hash"] = sync.fingerprint(study); (root / "master/PLAN.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "Conflicting"): sync.manifest(root, "master/PLAN.json", {})

    def test_original_five_hard_excluded_and_no_extra_archive_members(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
            root = Path(name); plan, study = self.fixtures(root)
            result = sync.manifest(root, "master/PLAN.json", {})
            path = root / "bad.tar"
            with tarfile.open(path, "w") as tar:
                member = tarfile.TarInfo(".ssh/id_rsa"); member.size = 3; tar.addfile(member, io.BytesIO(b"bad"))
            with self.assertRaisesRegex(ValueError, "archive member"): sync.verify_bundle(result, path)
            for dataset in ("Roman Empire", "Amazon Ratings", "Minesweeper", "Questions", "Tolokers"):
                study["dataset"] = dataset
                (root / "master/study.json.gz").write_bytes(gzip.compress(sync.canonical(study).encode(), mtime=0))
                plan["studies"][0]["spec_hash"] = sync.fingerprint(study); (root / "master/PLAN.json").write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "hard excluded"): sync.manifest(root, "master/PLAN.json", {})

    def test_exact_environment_and_hardware_no_pending_host_assumption(self):
        observed = {"hostname": "anogena-2-0", "gpu_uuids": [sync.ALLOCATION_UUID]}
        self.assertEqual(sync.host_verify("allocation", {}, observed=observed), observed)
        with self.assertRaisesRegex(ValueError, "requires later"): sync.host_verify("server77", {"server77": {"status": "pending"}}, observed={})
        expected = {key: "synthetic" for key in ("implementation", "python_version", "python_major_minor", "platform_system", "machine", "libc", "stdlib", "packages")}
        sync.environment_verify(expected, expected)
        changed = copy.deepcopy(expected); changed["packages"] = {"torch": "different-ABI"}
        with self.assertRaisesRegex(ValueError, "not ready"): sync.environment_verify(expected, changed)

    def test_absolute_provider_provenance_mirrors_within_phase_without_spec_rewrite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
            root = Path(name); plan, study = self.fixtures(root)
            original = sync.receipt(root, "data/train.pt"); original["path"] = str(root / "data/train.pt")
            study["data"]["evidence"] = {"provider_files": [original]}
            raw = gzip.compress(sync.canonical(study).encode(), mtime=0); (root / "master/study.json.gz").write_bytes(raw)
            plan["studies"][0]["spec_hash"] = sync.fingerprint(study); (root / "master/PLAN.json").write_text(json.dumps(plan))
            result = sync.manifest(root, "master/PLAN.json", {})
            self.assertEqual((root / "master/study.json.gz").read_bytes(), raw)
            self.assertEqual(result["absolute_data_provenance_aliases"][0]["phase_relative_mirror_receipt"]["path"], "data/train.pt")
            self.assertFalse(result["absolute_data_provenance_aliases"][0]["spec_rewritten"])
            study["data"]["evidence"]["provider_files"][0]["path"] = "/outside-authorized-phase/provider.pt"
            (root / "master/study.json.gz").write_bytes(gzip.compress(sync.canonical(study).encode(), mtime=0))
            plan["studies"][0]["spec_hash"] = sync.fingerprint(study); (root / "master/PLAN.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "outside authorized"): sync.manifest(root, "master/PLAN.json", {})


if __name__ == "__main__": unittest.main()
