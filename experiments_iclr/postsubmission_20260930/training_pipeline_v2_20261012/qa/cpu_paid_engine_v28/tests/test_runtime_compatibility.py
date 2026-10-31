"""Required platform/ABI gates versus provenance; no science or installation."""
import copy
import importlib.util
import unittest

from prepare_sync_bundle import (PACKAGES, STDLIB_MODULES, compatibility_requirements,
    runtime_compatibility_report, binary_compatibility_probe)


def fixture():
    versions = dict(zip(PACKAGES, ("2.1.2+cu118", "1.26.4", "1.14.1", "1.5.2", "2.7.0", "2.1.2+pt21cu118", "0.6.18+pt21cu118", "1.3.6")))
    return {"implementation": "CPython", "python_major_minor": [3, 11], "python_version": "3.11.14",
        "platform_system": "Linux", "machine": "x86_64", "libc": ["glibc", "2.35"],
        "stdlib": {"sqlite": "3.50.4", "OpenSSL": "OpenSSL 3.5.5"}, "packages": versions,
        "stdlib_capabilities": {key: True for key in (*STDLIB_MODULES, "sqlite_transactions", "ssl_context")}}


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_micro_SQLite_SSL_glibc_changes_pass_and_are_reported(self):
        source = fixture(); target = copy.deepcopy(source)
        target.update(python_version="3.11.8", libc=["glibc", "2.31"])
        target["stdlib"].update(sqlite="3.34.1", OpenSSL="OpenSSL 1.1.1")
        report = runtime_compatibility_report(source, target)
        self.assertEqual(set(report["observed_provenance_differences"]), {"python_version", "libc", "stdlib.sqlite", "stdlib.OpenSSL"})
        self.assertEqual(report["observed_provenance_differences"]["python_version"], {"source": "3.11.14", "observed": "3.11.8"})
        self.assertTrue(report["numerical_package_locks_checked"])
        self.assertFalse(report["bitwise_environment_replication_claimed"])
        self.assertFalse(report["native_binary_compatibility_verified"])

    def test_wrong_Python_implementation_major_minor_OS_or_architecture_fails(self):
        source = fixture()
        for key, value in (("implementation", "PyPy"), ("python_major_minor", [3, 10]),
                           ("platform_system", "Windows"), ("machine", "aarch64")):
            target = copy.deepcopy(source); target[key] = value
            with self.assertRaisesRegex(ValueError, key): runtime_compatibility_report(source, target)

    def test_mandatory_numerical_and_compiled_build_versions_remain_exact(self):
        source = fixture()
        for package in PACKAGES:
            target = copy.deepcopy(source); target["packages"][package] += ".changed"
            with self.assertRaisesRegex(ValueError, "locks differ"): runtime_compatibility_report(source, target)
        target = copy.deepcopy(source); target["packages"]["torch"] = "2.1.2"
        with self.assertRaisesRegex(ValueError, "locks differ"): runtime_compatibility_report(source, target)
        target = copy.deepcopy(source); target["packages"]["unreferenced-editor-helper"] = "9.9"
        runtime_compatibility_report(source, target)

    def test_required_stdlib_capabilities_fail_even_when_versions_match(self):
        source = fixture()
        for capability in (*STDLIB_MODULES, "sqlite_transactions", "ssl_context"):
            target = copy.deepcopy(source); target["stdlib_capabilities"][capability] = False
            with self.assertRaisesRegex(ValueError, "capability absent"): runtime_compatibility_report(source, target)

    def test_base_interpreter_gate_can_precede_installation_and_keeps_source_record(self):
        source = fixture(); unchanged = copy.deepcopy(source); base = copy.deepcopy(source); base.pop("packages")
        base["python_version"] = "3.11.5"
        report = runtime_compatibility_report(source, base, require_packages=False)
        self.assertFalse(report["numerical_package_locks_checked"])
        self.assertEqual(source, unchanged)
        self.assertEqual(compatibility_requirements(source)["numerical_package_versions"], source["packages"])

    @unittest.skipIf(any(importlib.util.find_spec(module) is None for module in ("torch", "torch_geometric", "torch_scatter", "torch_sparse")),
                         "Uses only the existing allocation numerical stack; do not install")
    def test_existing_native_CPU_extension_operations_without_CUDA_or_fit(self):
        result = binary_compatibility_probe()
        self.assertTrue(result["native_CPU_imports_and_operations_passed"])
        self.assertFalse(result["CUDA_initialized"])
        self.assertEqual(result["scientific_fits"], 0)


if __name__ == "__main__": unittest.main()
