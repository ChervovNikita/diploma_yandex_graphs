"""Simple explicit numerical-entry flag and source/protocol binding. No sandbox."""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def file_descriptor(path):
    path = Path(path)
    value = path.read_bytes()
    return {"path": str(path), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def verify_descriptor(record):
    path = Path(record["path"])
    resolved = path if path.is_absolute() else ROOT.parent / path
    actual = file_descriptor(resolved.resolve())
    if actual["bytes"] != record["bytes"] or actual["sha256"] != record["sha256"]:
        raise ValueError("File binding changed: " + record["path"])
    return actual


def verify_package(manifest_sha256, protocol_sha256):
    manifest_path = ROOT / "MANIFEST.json"
    if file_descriptor(manifest_path)["sha256"] != manifest_sha256:
        raise ValueError("Explicit source-manifest binding required")
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["payload"]:
        verify_descriptor(dict(row, path=str(ROOT / row["path"])))
    if file_descriptor(ROOT / "PROTOCOL.json")["sha256"] != protocol_sha256:
        raise ValueError("Explicit protocol binding required")
    bindings = json.loads((ROOT / "SOURCE_BINDINGS.json").read_text())
    for record in bindings["immutable_neural_files"]:
        verify_descriptor(record)
    verify_descriptor(bindings["v6_manifest"])
    verify_descriptor(bindings["v6_seal"])
    for key in ("native_recipe_source", "block_source"):
        verify_descriptor(bindings[key])
    for record in bindings["context"]:
        verify_descriptor(record)
    return bindings


@dataclass(frozen=True)
class Runtime:
    torch: object
    numpy: object
    native: object
    boundary: object
    manifest_sha256: str
    protocol_sha256: str
    source_provenance: tuple


def load_runtime(*, execute=False, manifest_sha256=None, protocol_sha256=None):
    """Call only after root admission. Default fails before numerical imports."""
    if execute is not True:
        raise RuntimeError("Source-only default: numerical entry requires deliberate root admission")
    bindings = verify_package(manifest_sha256, protocol_sha256)
    import numpy
    import torch
    modules, provenance = [], []
    for index, record in enumerate(bindings["immutable_neural_files"]):
        actual = verify_descriptor(record)
        provenance.append(dict(actual, declared_phase_path=record["path"]))
        spec = importlib.util.spec_from_file_location("_accuracy_view_pinned_" + str(index), actual["path"])
        module = importlib.util.module_from_spec(spec)
        source_bytes = Path(actual["path"]).read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != record["sha256"]:
            raise ValueError("Neural source changed before loading")
        # Compile exactly the bound bytes, avoiding bytecode-cache writes inside immutable v6.
        exec(compile(source_bytes, actual["path"], "exec"), module.__dict__)
        modules.append(module)
    return Runtime(torch, numpy, modules[0], modules[1], manifest_sha256, protocol_sha256, tuple(provenance))
