"""Independent source-only review fixtures; writes only temporary review files."""
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

REVIEW_ROOT = Path(__file__).resolve().parent
PACKAGE = Path(sys.argv[1]).resolve()
ORDINARY = PACKAGE.name.startswith("accuracy_first_graph_view_reference_")
sys.path.insert(0, str(PACKAGE))
admission = importlib.import_module("admission")
runtime = importlib.import_module("runtime")
driver = importlib.import_module("driver" if ORDINARY else "native_driver")
if ORDINARY:
    source_bindings = json.loads((PACKAGE / "SOURCE_BINDINGS.json").read_text())["files"]
    views, _ = runtime.bound_module("_independent_review_views", source_bindings["views"])
    helpers, _ = runtime.bound_module("_independent_review_helpers", source_bindings["helpers"])
    bank_sha = source_bindings["bank_manifest"]["sha256"]
    own_field, schema = "reference_manifest_sha256", "accuracy-first-reference-context-v1"
    own_context, protocol_context = own_field, "reference_protocol_sha256"
    describe, verify = runtime.descriptor, runtime.verify
else:
    views, helpers = importlib.import_module("views"), importlib.import_module("bank")
    bank_sha = json.loads((PACKAGE / "SOURCE_BINDINGS.json").read_text())["bank_source_manifest"]["sha256"]
    own_field, schema = "native_gnnm_manifest_sha256", "accuracy-first-native-gnnm-context-v1"
    own_context, protocol_context = "manifest_sha256", "protocol_sha256"
    describe, verify = runtime.file_descriptor, runtime.verify_descriptor


class FakeFeatures:
    dtype = "synthetic-review-dtype"
    shape = (24, 3)

    def __init__(self, data=b"synthetic-review-features"):
        self.data = data

    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self
    def tobytes(self): return self.data


class FakeDevice:
    def __init__(self, value, index=None):
        text = str(value)
        self.type = text.split(":")[0]
        self.index = int(text.split(":")[1]) if ":" in text else index

    def __str__(self):
        return self.type if self.index is None else self.type + ":" + str(self.index)


def fake_device(value, index=None):
    return value if isinstance(value, FakeDevice) and index is None else FakeDevice(value, index)


class ReviewChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=REVIEW_ROOT)
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def fixture(self, split):
        role = views.TrainRole(24, split, tuple(range(22)), (0,) * 11 + (1,) * 11, (22,), (23,))
        edges = views.native_edges(24, ((a, b) for a in range(24) for b in range(a + 1, 24)))
        bundle = views.construct_views(role, edges, (17, 29, 43)[split])
        x, labels = FakeFeatures(), (0,)
        hasher = driver.feature_identity if ORDINARY else driver.bank_driver.feature_identity
        feature = hasher(x)
        expected = {"role": role.identity(), "native_edges_sha256": bundle["coverage"]["native_edges_sha256"],
                    "feature_identity": feature, "validation_labels_sha256": views.digest(labels)}
        rt = SimpleNamespace(manifest_sha256="1" * 64, protocol_sha256="2" * 64, views=views)
        context = {"schema": schema, own_context: rt.manifest_sha256, protocol_context: rt.protocol_sha256,
                   "features": feature, "validation_labels_sha256": expected["validation_labels_sha256"]}
        if ORDINARY:
            coverage = dict(bundle["coverage"], coverage_origin="official_TRAIN",
                            scientific_freeze_eligible=bundle["coverage"]["coverage_eligible"])
            coverage_path = self.directory / (str(split) + "-synthetic-coverage.json")
            coverage_path.write_text(json.dumps(coverage))
            context["coverage"] = describe(coverage_path)
        else:
            context.update(role=expected["role"], native_edges_sha256=expected["native_edges_sha256"])
        paired = {own_field: rt.manifest_sha256, "bank_manifest_sha256": bank_sha,
                  "input_bindings_by_split": {str(split): deepcopy(expected)}}
        paired_path = self.directory / (str(split) + "-synthetic-paired.json")
        def validate(master=paired, live_bundle=bundle, live_x=x, live_labels=labels):
            paired_path.write_text(json.dumps(master))
            context["paired_protocol"] = describe(paired_path)
            driver.validate_context(rt, context, live_bundle, live_x, live_labels)
        return paired, expected, validate, bundle, x, labels, context, rt, paired_path

    def test_all_splits_bind_exact_live_inputs(self):
        for split in (0, 1, 2):
            paired, expected, validate, *_ = self.fixture(split)
            validate()
            for mapping in (None, [], {}, {str((split + 1) % 3): expected}, {str(split): None}):
                with self.subTest(split=split, mapping=mapping), self.assertRaises(ValueError):
                    validate(dict(paired, input_bindings_by_split=mapping))
            invalid = deepcopy(paired); del invalid["input_bindings_by_split"]
            with self.assertRaises(ValueError): validate(invalid)
            for field in expected:
                invalid = deepcopy(paired); del invalid["input_bindings_by_split"][str(split)][field]
                with self.subTest(split=split, missing=field), self.assertRaises(ValueError): validate(invalid)
                invalid = deepcopy(paired)
                invalid["input_bindings_by_split"][str(split)][field] = "different-synthetic-identity"
                with self.subTest(split=split, conflict=field), self.assertRaises(ValueError): validate(invalid)

    def test_stale_live_features_labels_native_graph_and_role_fail(self):
        _, _, validate, bundle, _, _, _, _, _ = self.fixture(0)
        with self.assertRaises(ValueError): validate(live_x=FakeFeatures(b"changed-live-bytes"))
        with self.assertRaises(ValueError): validate(live_labels=(1,))
        invalid = dict(bundle, native=bundle["native"][:-1])
        with self.assertRaises(ValueError): validate(live_bundle=invalid)
        role = bundle["role"]
        changed_role = views.TrainRole(24, 0, role.train_ids, (1,) + role.train_labels[1:], role.val_ids, role.test_ids)
        with self.assertRaises(ValueError): validate(live_bundle=dict(bundle, role=changed_role))

    def test_exact_applicable_source_bindings_and_descriptor_fail_closed(self):
        paired, _, validate, bundle, x, labels, context, rt, paired_path = self.fixture(0)
        for field in (own_field, "bank_manifest_sha256"):
            invalid = deepcopy(paired); invalid.pop(field)
            with self.assertRaises(ValueError): validate(invalid)
            invalid = dict(paired, **{field: "wrong-source"})
            with self.assertRaises(ValueError): validate(invalid)
        validate()
        paired_path.write_text(json.dumps({"changed": True}))
        with self.assertRaises(ValueError): driver.validate_context(rt, context, bundle, x, labels)

    def test_absolute_checkpoint_paths_survive_cwd_change_before_and_after_save(self):
        original = Path.cwd()
        self.addCleanup(os.chdir, original)
        first, second = self.directory / "first", self.directory / "second"
        first.mkdir(); second.mkdir(); os.chdir(first)
        output = admission.create_output_directory("relative-fit")
        self.assertEqual(output, first / "relative-fit")
        with self.assertRaises(FileExistsError): admission.create_output_directory("relative-fit")
        rt = SimpleNamespace(torch=SimpleNamespace(save=lambda image, stream: stream.write(json.dumps(image).encode())))
        direct = driver.save_image(rt, Path("relative-fit") / "direct.pt", {"synthetic": True})
        os.chdir(second)
        self.assertEqual(Path(verify(direct)["path"]), output / "direct.pt")
        for name in ("SELECTED_LOCAL.pt", "SELECTED_FINAL.pt", "CONSTRUCTION.pt"):
            record = driver.save_image(rt, output / name, {"synthetic": name})
            self.assertTrue(Path(record["path"]).is_absolute())
            self.assertEqual(Path(verify(record)["path"]), output / name)
            with self.assertRaises(FileExistsError): driver.save_image(rt, output / name, {})
        with self.assertRaises(ValueError): verify(dict(direct, sha256="wrong-digest"))
        with self.assertRaises(FileNotFoundError): verify(dict(direct, path=str(output / "absent.pt")))

    def test_cpu_and_selected_cuda_aliases_use_inherited_resolver(self):
        cuda = SimpleNamespace(current_device=Mock(return_value=3))
        rt = SimpleNamespace(torch=SimpleNamespace(device=fake_device, cuda=cuda))
        resolver = Mock(side_effect=helpers.resolve_device)
        for value, expected in (("cpu", "cpu"), ("cuda", "cuda:3"), ("cuda:0", "cuda:0"),
                                (FakeDevice("cuda"), "cuda:3"), (FakeDevice("cuda:2"), "cuda:2")):
            self.assertEqual(str(admission.reference_device(rt, value, resolver)), expected)
        self.assertEqual(resolver.call_count, 5)
        self.assertEqual(cuda.current_device.call_count, 2)

    def test_unsupported_backends_reject_before_resolver_seed_builder_and_output(self):
        seed, constructor = Mock(), Mock()
        resolver = Mock(side_effect=helpers.resolve_device)
        rt = SimpleNamespace(torch=SimpleNamespace(device=fake_device),
                             helpers=SimpleNamespace(resolve_device=resolver, seed_all=seed),
                             native=SimpleNamespace(Polynormer=constructor))
        output = self.directory / "rejected-output"
        old_resolver = helpers.resolve_device
        if not ORDINARY: helpers.resolve_device = resolver
        try:
            for value in ("mps", "xpu", "meta", "privateuseone", FakeDevice("mps")):
                with self.assertRaises(ValueError): admission.reference_device(rt, value, resolver)
                with self.assertRaises(ValueError):
                    if ORDINARY:
                        driver.run_reference(rt, execute=True, kind="native_member", split=0, member=0,
                                             x=None, bundle={}, validation_labels=(), context={}, output=output, device=value)
                    else:
                        driver.run_native_reference(rt, execute=True, split=0, x=None, bundle={},
                                                    validation_labels=(), context={}, output=output, device=value)
                if ORDINARY:
                    with self.assertRaises(ValueError): driver.build(rt, "native_member", 0, 0, device=value)
        finally:
            helpers.resolve_device = old_resolver
        self.assertFalse(output.exists())
        resolver.assert_not_called(); seed.assert_not_called(); constructor.assert_not_called()

    def test_no_numerical_imports(self):
        self.assertFalse(any(name.split(".")[0] in ("torch", "numpy", "torch_geometric", "torch_scatter")
                             for name in sys.modules))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReviewChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {"package": PACKAGE.name, "tests_run": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(),
               "numerical_imports": False, "dataset_access": False, "server_access": False,
               "fixture_directory": str(REVIEW_ROOT), "source_mutation": False}
    print(json.dumps(summary, sort_keys=True))
    sys.exit(0 if result.wasSuccessful() else 1)
