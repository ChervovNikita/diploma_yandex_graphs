"""Stdlib synthetic regressions for paired inputs, paths and device admission."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from admission import create_output_directory, reference_device

import native_driver as DRIVER
import bank as HELPERS
import views as VIEWS
from runtime import ROOT, file_descriptor as describe, verify_descriptor as verify_record
BANK_MANIFEST_SHA256 = json.loads((ROOT / "SOURCE_BINDINGS.json").read_text())["bank_source_manifest"]["sha256"]
OWN_MASTER_FIELD = "native_gnnm_manifest_sha256"
OWN_CONTEXT_FIELD = "manifest_sha256"
PROTOCOL_CONTEXT_FIELD = "protocol_sha256"
CONTEXT_SCHEMA = "accuracy-first-native-gnnm-context-v1"
FEATURE_HASHER = "bank_driver.feature_identity"


@contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class FakeDevice:
    def __init__(self, value, index=None):
        text = str(value)
        self.type = text.split(":")[0]
        self.index = int(text.split(":")[1]) if ":" in text else index

    def __str__(self):
        return self.type if self.index is None else self.type + ":" + str(self.index)


def fake_device(value, index=None):
    return value if isinstance(value, FakeDevice) and index is None else FakeDevice(value, index)


class AdmissionRegressions(unittest.TestCase):
    def fixture(self, directory, split=0):
        role = VIEWS.TrainRole(24, split, tuple(range(22)), (0,) * 11 + (1,) * 11, (22,), (23,))
        bundle = VIEWS.construct_views(role, VIEWS.native_edges(24,
                  ((a, b) for a in range(24) for b in range(a + 1, 24))), (17, 29, 43)[split])
        feature = {"dtype": "synthetic-fixture", "shape": [24, 3], "logical_sha256": "fixture-only"}
        labels = (0,)
        rt = SimpleNamespace(manifest_sha256="1" * 64, protocol_sha256="2" * 64, views=VIEWS)
        expected = {"role": role.identity(), "native_edges_sha256": bundle["coverage"]["native_edges_sha256"],
                    "feature_identity": feature, "validation_labels_sha256": VIEWS.digest(labels)}
        paired = {"reference_manifest_sha256": "3" * 64, "native_gnnm_manifest_sha256": "4" * 64,
                  "bank_manifest_sha256": BANK_MANIFEST_SHA256,
                  "input_bindings_by_split": {str(split): deepcopy(expected)}}
        paired[OWN_MASTER_FIELD] = rt.manifest_sha256
        context = {"schema": CONTEXT_SCHEMA, OWN_CONTEXT_FIELD: rt.manifest_sha256,
                   PROTOCOL_CONTEXT_FIELD: rt.protocol_sha256, "features": feature,
                   "validation_labels_sha256": expected["validation_labels_sha256"]}
        context["role"] = expected["role"]
        context["native_edges_sha256"] = expected["native_edges_sha256"]
        paired_path = Path(directory) / "synthetic-paired.json"
        def validate(value):
            paired_path.write_text(json.dumps(value))
            context["paired_protocol"] = describe(paired_path)
            with patch(FEATURE_HASHER, return_value=feature):
                DRIVER.validate_context(rt, context, bundle, object(), labels)
        return paired, expected, validate

    def test_missing_split_mapping_entry_and_fields_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for split in (0, 1, 2):
                paired, expected, validate = self.fixture(directory, split)
                validate(paired)
                invalid = deepcopy(paired)
                del invalid["input_bindings_by_split"]
                with self.assertRaises(ValueError):
                    validate(invalid)
                for mapping in ([], {}, {str((split + 1) % 3): expected}, {str(split): None}):
                    invalid = dict(paired, input_bindings_by_split=mapping)
                    with self.subTest(split=split, mapping=mapping), self.assertRaises(ValueError):
                        validate(invalid)
                for field in expected:
                    invalid = deepcopy(paired)
                    del invalid["input_bindings_by_split"][str(split)][field]
                    with self.subTest(split=split, absent=field), self.assertRaises(ValueError):
                        validate(invalid)

    def test_conflicting_split_identities_and_source_keys_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for split in (0, 1, 2):
                paired, expected, validate = self.fixture(directory, split)
                for field in expected:
                    invalid = deepcopy(paired)
                    invalid["input_bindings_by_split"][str(split)][field] = "conflicting synthetic identity"
                    with self.subTest(split=split, conflict=field), self.assertRaises(ValueError):
                        validate(invalid)
                invalid = deepcopy(paired)
                invalid["input_bindings_by_split"][str(split)]["role"]["split"] = (split + 1) % 3
                with self.assertRaises(ValueError):
                    validate(invalid)
                for field in (OWN_MASTER_FIELD, "bank_manifest_sha256"):
                    invalid = deepcopy(paired)
                    del invalid[field]
                    with self.subTest(split=split, missing_source=field), self.assertRaises(ValueError):
                        validate(invalid)
                    invalid = dict(paired, **{field: "wrong-source"})
                    with self.subTest(split=split, wrong_source=field), self.assertRaises(ValueError):
                        validate(invalid)

    def test_relative_output_checkpoint_roundtrip_and_conflicts(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            first.mkdir()
            second.mkdir()
            rt = SimpleNamespace(torch=SimpleNamespace(save=lambda value, stream:
                                  stream.write(json.dumps(value).encode())))
            with working_directory(first):
                output = create_output_directory("relative-output")
                self.assertEqual(output, (first / "relative-output").resolve())
                saved = DRIVER.save_image(rt, Path("relative-output") / "selected.pt", {"synthetic": True})
                self.assertTrue(Path(saved["path"]).is_absolute())
                with self.assertRaises(FileExistsError):
                    create_output_directory("relative-output")
                with self.assertRaises(FileExistsError):
                    DRIVER.save_image(rt, Path("relative-output") / "selected.pt", {"synthetic": False})
            with working_directory(second):
                self.assertEqual(verify_record(saved), describe(output / "selected.pt"))
                with self.assertRaises(ValueError):
                    verify_record(dict(saved, sha256="wrong-digest"))
                with self.assertRaises(FileNotFoundError):
                    verify_record(dict(saved, path=str(output / "absent.pt")))

    def test_cpu_cuda_aliases_and_unsupported_backend_reject_before_fit(self):
        cuda = SimpleNamespace(current_device=Mock(return_value=3))
        resolver = Mock(side_effect=HELPERS.resolve_device)
        rt = SimpleNamespace(torch=SimpleNamespace(device=fake_device, cuda=cuda),
                             helpers=SimpleNamespace(resolve_device=resolver, seed_all=Mock()),
                             native=SimpleNamespace(Polynormer=Mock()))
        self.assertEqual(str(reference_device(rt, "cpu", resolver)), "cpu")
        for value in ("cuda", "cuda:3", FakeDevice("cuda"), FakeDevice("cuda:3")):
            self.assertEqual(str(reference_device(rt, value, resolver)), "cuda:3")
        self.assertEqual(str(reference_device(rt, "cuda:2", resolver)), "cuda:2")
        resolver.reset_mock()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "must-not-be-created"
            for backend in ("mps", "xpu", "meta", FakeDevice("mps")):
                with self.assertRaises(ValueError):
                    reference_device(rt, backend, resolver)
                with self.assertRaises(ValueError):
                    DRIVER.run_native_reference(rt, execute=True, split=0, x=None, bundle={},
                                                validation_labels=(), context={}, output=output, device=backend)
                # The native entry retains the unchanged bound bank builder behind its guard.
            self.assertFalse(output.exists())
        resolver.assert_not_called()
        rt.helpers.seed_all.assert_not_called()
        rt.native.Polynormer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
