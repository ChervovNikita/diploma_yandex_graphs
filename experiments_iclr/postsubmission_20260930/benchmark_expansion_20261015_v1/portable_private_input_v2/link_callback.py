"""Explicit link runner entrypoint for the portable private-input candidates.

The frozen native link integration constructs its actual base session. This
new callback applies the separately source-bound input residual extension.
No generic runtime policy, link source program or native callback is edited.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent


def _load(path):
    path = Path(path).resolve()
    name = "portable_input_callback_"+hashlib.sha256(path.read_bytes()).hexdigest()
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def create_adapter(torch, job, policy, root):
    root = Path(root).resolve()
    job = copy.deepcopy(job)
    declared = job.get("recipe", {}).get("_repository_root")
    if declared is not None and Path(declared).resolve() != root:
        raise ValueError("Declared recipe repository root differs from the runner binding")
    job["recipe"]["_repository_root"] = str(root)
    if job.get("family") != "link" or job.get("TEST_access") is not False \
            or getattr(policy, "family", None) != "link" or getattr(policy, "track", None) == "source_reference":
        raise ValueError("A declared matched link policy and closed TEST role are required")
    expected = job.get("method_settings", {}).get("manifest_sha256")
    manifest_path = HERE/"MANIFEST.json"
    if not isinstance(expected, str) or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Pin the actual portable method manifest before loading its helper source")
    rows = {row["path"]: row for row in json.loads(manifest_path.read_text())["files"]}
    for name in ("extension.py", "core.py"):
        raw = (HERE/name).read_bytes()
        if len(raw) != rows[name]["bytes"] or hashlib.sha256(raw).hexdigest() != rows[name]["sha256"]:
            raise ValueError("Portable helper source changed: "+name)
    extension = _load(HERE/"extension.py")
    if job.get("procedure") not in extension.METHODS:
        raise ValueError("This callback admits only the two portable private-input candidates")
    extension._source_bindings("link", root, job.get("method_settings", {}).get("manifest_sha256"))
    actual = Path(job["callback_module"])
    actual = (actual if actual.is_absolute() else Path(root)/actual).resolve()
    if actual != Path(__file__).resolve() or job.get("callback_sha256") != hashlib.sha256(actual.read_bytes()).hexdigest():
        raise ValueError("Pin this actual portable link entrypoint explicitly")
    native_path = HERE.parent/"link_native_integration_v1/session.py"
    native = _load(native_path)
    base_job = copy.deepcopy(job)
    base_job.update(procedure="gnnm_boundary_native", base_procedure="gnnm_boundary_native",
        callback_module=str(native_path), callback_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        method_settings={})
    for key in ("session_transform", "extension_factory", "session_extension"):
        base_job.pop(key, None)
    adapter = native.create_adapter(torch, base_job, policy, root)
    adapter.session = extension.transform(adapter.session, job)
    adapter.model, adapter.optimizers = adapter.session.model, adapter.session.optimizers
    outer_pin = extension._receipt(actual)
    inner_pin = extension._receipt(native_path)
    adapter.callback_receipt = outer_pin
    extension_pin = extension._receipt(HERE/"extension.py")
    adapter.extension_receipt = extension_pin
    original_metadata = adapter.metadata
    def metadata(self):
        return {**original_metadata(), "portable_input_entrypoint": {
            "outer_job_callback_receipt": outer_pin, "inner_native_callback_receipt": inner_pin,
            "extension_receipt": extension_pin,
            "repository_root": str(root), "inner_native_binding_is_not_outer_job_callback": True}}
    adapter.metadata = types.MethodType(metadata, adapter)
    return adapter
