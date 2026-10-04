"""Caller-driven source/protocol entry, portable descriptors; no acquisition."""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def descriptor(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def verify(record):
    path = Path(record["path"])
    actual = descriptor((path if path.is_absolute() else ROOT.parent / path).resolve())
    if actual["bytes"] != record["bytes"] or actual["sha256"] != record["sha256"]:
        raise ValueError("Bound file changed: " + record["path"])
    return actual


def verify_package(manifest_sha256, protocol_sha256):
    if descriptor(ROOT / "MANIFEST.json")["sha256"] != manifest_sha256 or \
            descriptor(ROOT / "PROTOCOL.json")["sha256"] != protocol_sha256:
        raise ValueError("Exact caller source/protocol binding required")
    for row in json.loads((ROOT / "MANIFEST.json").read_text())["payload"]:
        verify(dict(row, path=str(ROOT / row["path"])))
    binding = json.loads((ROOT / "SOURCE_BINDINGS.json").read_text())
    for record in binding["files"].values():
        verify(record)
    return binding


def bound_module(name, record):
    actual = verify(record)
    source = Path(actual["path"]).read_bytes()
    if hashlib.sha256(source).hexdigest() != record["sha256"]:
        raise ValueError("Source changed before compilation")
    spec = importlib.util.spec_from_file_location(name, actual["path"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required by the bound stdlib dataclass in views.
    exec(compile(source, actual["path"], "exec"), module.__dict__)
    return module, dict(actual, declared_phase_path=record["path"])


@dataclass(frozen=True)
class Runtime:
    torch: object
    numpy: object
    native: object
    helpers: object
    views: object
    manifest_sha256: str
    protocol_sha256: str
    provenance: tuple


def load_runtime(*, execute=False, manifest_sha256=None, protocol_sha256=None):
    if execute is not True:
        raise RuntimeError("Source-only default; deliberate root-admitted call required")
    binding = verify_package(manifest_sha256, protocol_sha256)
    import numpy
    import torch
    loaded = [bound_module("_graph_reference_bound_" + name, binding["files"][name])
              for name in ("native", "helpers", "views")]
    return Runtime(torch, numpy, *(item[0] for item in loaded), manifest_sha256,
                   protocol_sha256, tuple(item[1] for item in loaded))
