"""CPU-only regression tests using the real frozen witness and tempfile data."""

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import reporting_patch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReportingRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reporting = reporting_patch._reporting_directory()
        cls.original_sources = reporting_patch._verify_frozen_sources(cls.reporting)
        cls.producer = reporting_patch.install()
        cls.composed = sys.modules["composed_authority"]
        cls.observer = sys.modules["observe_exports"]
        cls.witness = staticmethod(cls.composed.runtime_witness)
        cls.original = staticmethod(cls.witness.__wrapped__)
        cls.readme_source = (
            Path(__file__).resolve().parent.parent
            / "link_expansion_ports_BUDDY_production_v4" / "README.md"
        )
        if sha256(cls.readme_source) != sha256(cls.readme_source):
            raise AssertionError("Test README source differs from the required campaign pin")

    @classmethod
    def tearDownClass(cls):
        if reporting_patch._verify_frozen_sources(cls.reporting) != cls.original_sources:
            raise AssertionError("Tests changed a frozen reporting source")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="campaign-reporting-repair-", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "project"
        self.root.mkdir()
        self.readme = self.root / "declared_source/README.md"
        self.readme.parent.mkdir(parents=True)
        self.readme.write_bytes(self.readme_source.read_bytes())
        self.python = self.root / "ordinary.py"
        self.python.write_text("VALUE = 1\n")
        self.json = self.root / "ordinary.json"
        self.json.write_text('{"value": 1}\n')
        self.job = {"id": "cpu-fixture", "normalization_composition": "fixture-composition"}
        authority = {
            "schema": "normalization-reporting-runtime-authority-v1",
            "normalization_composition": self.job["normalization_composition"],
            "project_root": str(self.root),
            "producer_manifest": {"path": "fixture-manifest"},
            "entrypoint": "fixture-entrypoint",
            "normalization_source": "fixture-normalization",
            "link_extension_authority": "fixture-link-authority",
            "runtime_sha256": "composed-runtime",
            "helper_runtime_sha256": "helper-runtime",
            "callback_sha256": "callback",
            "extension_sha256": "extension",
            "policy_sha256": "policy",
        }
        dependencies = {str(path): sha256(path) for path in (self.python, self.json, self.readme)}
        helper = {key: authority[key] for key in ("callback_sha256", "extension_sha256", "policy_sha256")}
        helper.update(runtime_sha256=authority["helper_runtime_sha256"], dependency_sha256=dependencies)
        self.actual = {
            "schema": "server51-composed-normalization-genuine-runtime-binding-v2",
            "runtime_sha256": authority["runtime_sha256"],
            "callback_sha256": authority["callback_sha256"],
            "extension_sha256": authority["extension_sha256"],
            "policy_sha256": authority["policy_sha256"],
            "dependency_sha256": dependencies,
            "composition_producer": {
                "source_manifest": authority["producer_manifest"],
                "entrypoint": authority["entrypoint"],
                "normalization_source": authority["normalization_source"],
                "unchanged_helper_runtime": helper,
                "link_extension_authority": authority["link_extension_authority"],
            },
        }
        self.spec = {"job_hash": self.composed.digest(self.job), "composed_runtime_authority": authority}

    def replace_dependencies(self, dependencies):
        self.actual["dependency_sha256"] = dependencies
        self.actual["composition_producer"]["unchanged_helper_runtime"]["dependency_sha256"] = dependencies

    def test_accepts_exact_pinned_readme_and_preserves_original_witness(self):
        with self.assertRaisesRegex(ValueError, "Only genuine project source"):
            self.original(self.job, self.spec, self.actual)
        before = copy.deepcopy((self.job, self.spec, self.actual))
        observed = self.witness(self.job, self.spec, self.actual)
        expected = self.original(self.job, self.spec, self.actual, verify_dependency_bytes=False)
        repair = observed.pop("reporting_repair")
        self.assertEqual(observed, expected)
        self.assertEqual(repair["producer_receipt"], self.producer["producer_receipt"])
        self.assertTrue(repair["dependency_bytes_verified"])
        self.assertEqual((self.job, self.spec, self.actual), before)

    def test_documentation_dependencies_are_verified_like_python(self):
        for name in ("SMOKE_PLAN.md", "DELTA.diff.txt", "MANIFEST.data"):
            member = self.root / name
            member.write_text("Declared source metadata.\n")
            self.replace_dependencies({str(member): sha256(member)})
            self.witness(self.job, self.spec, self.actual)
            member.write_text("Changed source metadata.\n")
            with self.assertRaisesRegex(ValueError, "source dependency bytes changed"):
                self.witness(self.job, self.spec, self.actual)

    def test_rejects_internal_symlink_alias(self):
        alias = self.root / "alias.md"
        alias.symlink_to(self.readme)
        self.replace_dependencies({str(alias): sha256(self.readme)})
        with self.assertRaisesRegex(ValueError, "regular canonical"):
            self.witness(self.job, self.spec, self.actual)

    def test_rejects_nonregular_dependency(self):
        self.replace_dependencies({str(self.root): "0" * 64})
        with self.assertRaisesRegex(ValueError, "regular canonical"):
            self.witness(self.job, self.spec, self.actual)

    def test_rejects_absolute_traversal_and_symlink_paths_outside_project(self):
        outside = self.base / "outside.py"
        outside.write_text("VALUE = 2\n")
        symlink = self.root / "escape.py"
        symlink.symlink_to(outside)
        for path in (outside, self.root / ".." / outside.name, symlink):
            with self.subTest(path=str(path)):
                self.replace_dependencies({str(path): sha256(outside)})
                with self.assertRaisesRegex(ValueError, "Only genuine project source"):
                    self.witness(self.job, self.spec, self.actual)

    def test_ordinary_python_and_json_hashes_are_verified_as_before(self):
        dependencies = {str(path): sha256(path) for path in (self.python, self.json)}
        self.replace_dependencies(dependencies)
        expected = self.original(self.job, self.spec, self.actual)
        observed = self.witness(self.job, self.spec, self.actual)
        observed.pop("reporting_repair")
        self.assertEqual(observed, expected)
        for path in (self.python, self.json):
            with self.subTest(path=path.name):
                saved = path.read_bytes()
                path.write_bytes(saved + b" ")
                try:
                    with self.assertRaisesRegex(ValueError, "source dependency bytes changed"):
                        self.witness(self.job, self.spec, self.actual)
                finally:
                    path.write_bytes(saved)
        dependencies[str(self.python)] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source dependency bytes changed"):
            self.witness(self.job, self.spec, self.actual)

    def test_original_structural_checks_remain_active(self):
        self.spec["job_hash"] = "different-job"
        with self.assertRaisesRegex(ValueError, "job authority differs"):
            self.witness(self.job, self.spec, self.actual)

    def test_structural_only_mode_matches_original_and_reports_its_mode(self):
        self.readme.unlink()
        expected = self.original(self.job, self.spec, self.actual, verify_dependency_bytes=False)
        observed = self.witness(self.job, self.spec, self.actual, verify_dependency_bytes=False)
        repair = observed.pop("reporting_repair")
        self.assertEqual(observed, expected)
        self.assertFalse(repair["dependency_bytes_verified"])

    def test_install_is_idempotent_and_both_module_references_are_patched(self):
        before = self.composed.runtime_witness
        self.assertEqual(reporting_patch.install(), self.producer)
        self.assertIs(self.composed.runtime_witness, before)
        self.assertIs(self.observer.runtime_witness, before)

    def test_collector_recomputation_and_receipt_copies_are_stable(self):
        first = self.witness(self.job, self.spec, self.actual)
        self.assertEqual(first, self.observer.runtime_witness(self.job, self.spec, self.actual))
        first["reporting_repair"]["producer_receipt"]["sha256"] = "mutated-copy"
        self.assertEqual(self.witness(self.job, self.spec, self.actual)["reporting_repair"][
            "producer_receipt"], self.producer["producer_receipt"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
