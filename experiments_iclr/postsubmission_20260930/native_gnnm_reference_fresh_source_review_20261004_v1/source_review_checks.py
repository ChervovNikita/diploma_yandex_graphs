"""Independent source-only checks. No numerical packages or dataset access.

The only deliberate temporary fixture is under this review directory.
The reviewed source package is never written, and bytecode writes are disabled.
"""
import ast
import hashlib
import importlib.abc
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
REVIEW = Path(__file__).resolve().parent
PACKAGE = REVIEW.parent / "accuracy_first_native_gnnm_reference_source_preparation_20261004_v1"
BASE = PACKAGE.parent
FORBIDDEN = ("numpy", "torch", "torch_geometric")


class ExcludeNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise AssertionError("Numerical import attempted: " + fullname)


sys.meta_path.insert(0, ExcludeNumerics())
sys.path.insert(0, str(PACKAGE))
import bank
import bank_driver
import native_driver
import runtime
import schedule
import views


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_descriptor(record, base):
    path = Path(record["path"])
    path = path if path.is_absolute() else base / path
    data = path.read_bytes()
    assert len(data) == record["bytes"], str(path)
    assert hashlib.sha256(data).hexdigest() == record["sha256"], str(path)
    return path


def check_custody():
    manifest = json.loads((PACKAGE / "MANIFEST.json").read_text())
    bindings = json.loads((PACKAGE / "SOURCE_BINDINGS.json").read_text())
    seal = json.loads((PACKAGE / "SEAL.json").read_text())
    for record in manifest["payload"]:
        check_descriptor(record, PACKAGE)
    check_descriptor(seal["manifest"], BASE)
    scalar = ("bank_source_manifest", "bank_source_seal", "block_source", "native_recipe_source",
              "v6_manifest", "v6_seal")
    for key in scalar:
        check_descriptor(bindings[key], BASE)
    for key in ("context", "immutable_neural_files"):
        for record in bindings[key]:
            check_descriptor(record, BASE)
    for row in bindings["reused_source_files"]:
        current = check_descriptor(row["current"], BASE)
        original = check_descriptor(row["original"], BASE)
        assert current.read_bytes() == original.read_bytes()
    for row in bindings["exact_line_bindings"]:
        source = check_descriptor(row["source"], BASE)
        excerpt = "".join(source.read_text().splitlines(keepends=True)[row["start_line"] - 1:row["end_line"]])
        assert excerpt == row["excerpt_utf8"]
        assert hashlib.sha256(excerpt.encode()).hexdigest() == row["excerpt_sha256"]
    for path in PACKAGE.glob("*.py"):
        ast.parse(path.read_bytes(), filename=str(path))
    verified = runtime.verify_package(sha(PACKAGE / "MANIFEST.json"), sha(PACKAGE / "PROTOCOL.json"))
    assert verified == bindings
    return {"payload_descriptors": len(manifest["payload"]),
            "exact_reuse_copies": len(bindings["reused_source_files"]),
            "exact_line_bindings": len(bindings["exact_line_bindings"]),
            "external_context_descriptors": len(bindings["context"]),
            "immutable_neural_descriptors": len(bindings["immutable_neural_files"]),
            "AST_files": len(list(PACKAGE.glob("*.py"))),
            "all_passed": True}


def check_default_and_schedule():
    try:
        runtime.load_runtime()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Default numerical runtime did not reject")
    # The import blocker also applies inside the subprocess.
    program = """import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path, target=None):
  if fullname.split('.')[0] in ('numpy','torch','torch_geometric'):
   raise AssertionError('Numerical import attempted: ' + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
runpy.run_path(sys.argv[1] + '/native_driver.py', run_name='__main__')
assert not any(n in sys.modules for n in ('numpy','torch','torch_geometric'))
"""
    result = subprocess.run([sys.executable, "-B", "-c", program, str(PACKAGE)],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["scientific_training"] is False
    assert native_driver.BLOCKS == ((0, 17), (1, 29), (2, 43))
    assert sum(schedule.stage_at(u) == "local" for u in range(1, 2701)) == 200
    assert sum(schedule.stage_at(u) == "global" for u in range(1, 2701)) == 2500
    selector = schedule.Selector()
    assert selector.observe(1, 3, 1)
    assert not selector.observe(1, 3, 201)
    assert selector.selected_update == 1
    assert selector.observe(2, 3, 202)
    assert not any(n in sys.modules for n in FORBIDDEN)
    return {"source_only_CLI_import_blocker": "passed", "default_runtime_rejects": True,
            "fixed_blocks": [list(b) for b in native_driver.BLOCKS],
            "local_updates": 200, "global_updates": 2500,
            "per_block_training_and_backward_passes": 4 * 2700,
            "per_block_scheduled_selection_passes": 4 * 2700,
            "strict_earliest_tie_selector": "passed"}


def check_paired_protocol_gate():
    role = views.TrainRole(6, 0, (0, 1), (0, 1), (2, 3), (4, 5))
    edges = views.native_edges(6, [(0, 1), (2, 3)])
    bundle = views.construct_views(role, edges, 0)
    rt = SimpleNamespace(manifest_sha256=sha(PACKAGE / "MANIFEST.json"),
                         protocol_sha256=sha(PACKAGE / "PROTOCOL.json"))
    feature_record = {"synthetic_fixture_feature_identity": True}
    labels = (0, 1)
    predecessor = json.loads((PACKAGE / "SOURCE_BINDINGS.json").read_text())["bank_source_manifest"]
    pair = {"native_gnnm_manifest_sha256": rt.manifest_sha256,
            "bank_manifest_sha256": predecessor["sha256"],
            "role": {"synthetic_mismatch": True},
            "features": {"synthetic_mismatch": True},
            "native_edges_sha256": "synthetic_mismatch",
            "validation_labels_sha256": "synthetic_mismatch"}
    # Explicit fixture directory under the project-local review sibling.
    with tempfile.TemporaryDirectory(prefix="paired_gate_fixture_", dir=REVIEW) as directory:
        path = Path(directory) / "PAIRED_PROTOCOL.json"
        path.write_text(json.dumps(pair))
        context = {"schema": "accuracy-first-native-gnnm-context-v1",
                   "manifest_sha256": rt.manifest_sha256, "protocol_sha256": rt.protocol_sha256,
                   "role": role.identity(), "native_edges_sha256": bundle["coverage"]["native_edges_sha256"],
                   "features": feature_record, "validation_labels_sha256": views.digest(labels),
                   "paired_protocol": runtime.file_descriptor(path)}
        with patch.object(bank_driver, "feature_identity", return_value=feature_record):
            native_driver.validate_context(rt, context, bundle, object(), labels)
        # Removing every paired input identity also passes.
        path.write_text(json.dumps({key: pair[key] for key in
                                   ("native_gnnm_manifest_sha256", "bank_manifest_sha256")}))
        context["paired_protocol"] = runtime.file_descriptor(path)
        with patch.object(bank_driver, "feature_identity", return_value=feature_record):
            native_driver.validate_context(rt, context, bundle, object(), labels)
    return {"mismatched_paired_input_identity_accepted": True,
            "paired_input_identity_omitted_accepted": True,
            "numerical_feature_hasher": "patched to explicit synthetic identity",
            "fixture_location": "inside this review directory; removed"}


class CloneToken:
    def clone(self):
        return self


class SyntheticDevice:
    def __init__(self, name, index=None):
        self.type = name.type if isinstance(name, SyntheticDevice) else str(name)
        self.index = name.index if isinstance(name, SyntheticDevice) else index
    def __str__(self):
        return self.type if self.index is None else self.type + ":" + str(self.index)


def check_device_rng_gate():
    calls = []
    keys = SimpleNamespace(tolist=lambda: [0])
    np = SimpleNamespace(random=SimpleNamespace(get_state=lambda: ("synthetic", keys, 0, 0, 0.0),
                                               set_state=lambda value: None),
                         asarray=lambda keys, dtype: keys, uint32="synthetic")
    torch = SimpleNamespace(device=SyntheticDevice, get_rng_state=lambda: CloneToken(),
                            set_rng_state=lambda value: calls.append("CPU restore"),
                            mps=SimpleNamespace(get_rng_state=lambda: calls.append("MPS capture"),
                                                set_rng_state=lambda value: calls.append("MPS restore")))
    rt = SimpleNamespace(torch=torch, numpy=np)
    device = bank.resolve_device(rt, "mps")
    state = bank.rng_state(rt, device)
    assert state["selected_device"]["type"] == "mps" and state["cuda"] is None
    bank.restore_rng(rt, device, state)
    assert calls == ["CPU restore"]
    return {"MPS_device_accepted": True, "MPS_rng_capture_called": False,
            "MPS_rng_restore_called": False,
            "method": "stdlib stubs; no numerical backend imported or executed"}


if __name__ == "__main__":
    before = {str(p.relative_to(PACKAGE)): sha(p) for p in PACKAGE.iterdir() if p.is_file()}
    result = {"schema": "native-gnnm-independent-source-checks-v1",
              "custody": check_custody(), "default_and_schedule": check_default_and_schedule(),
              "paired_protocol_gate": check_paired_protocol_gate(),
              "device_rng_gate": check_device_rng_gate(),
              "numerical_packages_imported": False,
              "scientific_execution": False, "dataset_or_server_access": False}
    after = {str(p.relative_to(PACKAGE)): sha(p) for p in PACKAGE.iterdir() if p.is_file()}
    assert before == after
    assert not any(n in sys.modules for n in FORBIDDEN)
    result["reviewed_package_file_hashes_unchanged"] = True
    result["reviewed_package_directory_inventory"] = sorted(p.name for p in PACKAGE.iterdir())
    print(json.dumps(result, indent=2, sort_keys=True))
