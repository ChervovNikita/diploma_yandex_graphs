"""Stdlib-only tiny archive fixtures; no official archive or Torch access."""
import ast
import gzip
import hashlib
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from prepare_cache import HERE, check_archive_members, confined_path, member_metadata, stage_archive


class StagingChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".stage-fixture-", dir=HERE)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self):
        archive = self.root / "synthetic.zip"
        payloads = {"collab/RELEASE_v1.txt": b"fixture release", "collab/split/time/train.pt": b"fixture train bytes",
                    "collab/split/time/valid.pt": b"fixture valid bytes", "collab/raw/num-node-list.csv.gz": gzip.compress(b"235868\n"),
                    "collab/split/time/test.pt": b"closed fixture; must never be opened"}
        with zipfile.ZipFile(archive, "w") as packed:
            for name, payload in payloads.items():
                packed.writestr(name, payload)
        with zipfile.ZipFile(archive) as packed:
            metadata = [member_metadata(member) for member in packed.infolist()]
        contract = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "archive_bytes": archive.stat().st_size,
                    "archive_metadata": metadata, "stage_members": [name for name in payloads if "test.pt" not in name]}
        return archive, payloads, contract

    def test_exact_public_train_valid_extraction_never_opens_test(self):
        archive, payloads, contract = self.fixture()
        opened = []
        original = zipfile.ZipFile.open
        def tracked(packed, member, *args, **kwargs):
            opened.append(member.filename if isinstance(member, zipfile.ZipInfo) else member)
            return original(packed, member, *args, **kwargs)
        with patch.object(zipfile.ZipFile, "open", tracked):
            evidence = stage_archive(archive, self.root / "dataset", contract)
        self.assertEqual(opened, contract["stage_members"])
        self.assertNotIn("collab/split/time/test.pt", opened)
        self.assertFalse((self.root / "dataset/ogbl_collab/split/time/test.pt").exists())
        for item in evidence:
            self.assertEqual(item["sha256"], hashlib.sha256(payloads[item["archive_member"]]).hexdigest())
        public = next(item for item in evidence if item["archive_member"].endswith(".gz"))
        self.assertEqual(public["csv_sha256"], hashlib.sha256(b"235868\n").hexdigest())

    def test_wrong_archive_identity_and_existing_destination_refused(self):
        archive, _, contract = self.fixture()
        destination = self.root / "dataset"
        with self.assertRaisesRegex(RuntimeError, "archive bytes changed"):
            stage_archive(archive, destination, {**contract, "archive_sha256": "0" * 64})
        self.assertFalse(destination.exists())
        destination.mkdir()
        with self.assertRaisesRegex(RuntimeError, "fresh"):
            stage_archive(archive, destination, contract)

    def test_unsafe_duplicate_and_nonregular_archive_metadata_refused(self):
        for name in ("../escape", "/absolute", "collab/../escape", "collab\\escape", "collab//escape"):
            item = zipfile.ZipInfo(name)
            with self.assertRaises(RuntimeError):
                check_archive_members([item], [])
        item = zipfile.ZipInfo("collab/raw/safe")
        with self.assertRaises(RuntimeError):
            check_archive_members([item, item], [])
        for mode in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFSOCK):
            item = zipfile.ZipInfo("collab/raw/nonregular")
            item.create_system = 3
            item.external_attr = (mode | 0o777) << 16
            with self.assertRaises(RuntimeError):
                check_archive_members([item], [])

    def test_metadata_and_closed_allowlist_mismatch_refused(self):
        archive, _, contract = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "directory metadata differs"):
            stage_archive(archive, self.root / "dataset", {**contract, "archive_metadata": []})
        with self.assertRaisesRegex(RuntimeError, "closed-test"):
            stage_archive(archive, self.root / "dataset", {**contract, "stage_members": ["collab/split/time/test.pt"]})
        with self.assertRaisesRegex(RuntimeError, "closed-test"):
            stage_archive(archive, self.root / "dataset", {**contract, "stage_members": ["collab/split/time/split_dict.pt"]})

    def test_symlink_and_outside_repository_paths_refused(self):
        outside = self.root.parent.parent
        with self.assertRaises(RuntimeError):
            confined_path(outside, self.root)
        target = self.root / "target"
        target.mkdir()
        link = self.root / "link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            confined_path(link / "new", self.root)

    def test_frozen_wrapper_calls_only_direct_train_valid_before_build(self):
        tree = ast.parse((HERE / "prepare_cache.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(call.func, ast.Attribute) and call.func.attr == "get_edge_split" for call in calls))
        split_calls = [call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == "official_split_file"]
        self.assertEqual([call.args[1].value for call in split_calls], ["train", "valid"])
        train_qualification = next(call.lineno for call in calls if isinstance(call.func, ast.Name) and call.func.id == "qualify_train_split")
        build_line = next(call.lineno for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == "build")
        self.assertLess(train_qualification, build_line)
        for assignment in (node for node in ast.walk(tree) if isinstance(node, ast.Assign)):
            for target in assignment.targets:
                self.assertFalse(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id in {"builder", "torch", "dataset"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
