"""Deferred exact-byte native source loading; no installs or data access.

Importing this module itself is stdlib only. Its future heavy import caller
must already hold the separately issued root qualification/science release.
This preparation issues no such release and calls no loader.
"""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

NATIVE_PACKET = "graph_ncNC_member_completion_qualification_preparation_20261003_v2"
NATIVE_MANIFEST_SHA256 = "a99b0e3b0e8ec597b03e2c390ac176597b5b6422d365fc20624ab98a113d42f9"
PLAN_MANIFEST_SHA256 = "aa70f5a4d89a394ef059a431c78d333af95e6ab1c0c4e8c8dbbd028ee4b8d2f9"


def check_file(path, descriptor):
    data = path.read_bytes()
    if len(data) != descriptor["bytes"] or sha256(data).hexdigest() != descriptor["sha256"]:
        raise RuntimeError("Bound source bytes changed: " + str(path))


def verify_dependencies():
    here = Path(__file__).resolve().parent
    bindings = json.loads((here / "SOURCE_BINDINGS.json").read_text())
    for entry in bindings["external_sources"]:
        check_file(here.parent / entry["path"], entry)
    for entry in bindings["copied_sources"]:
        check_file(here / entry["copied_path"], entry)
    plan = here / "FROZEN_PLAN" / "MANIFEST.json"
    if sha256(plan.read_bytes()).hexdigest() != PLAN_MANIFEST_SHA256:
        raise RuntimeError("Frozen source plan manifest changed")
    for entry in json.loads(plan.read_text())["payload"]:
        check_file(plan.parent / entry["path"], entry)
    root = here.parent / NATIVE_PACKET
    manifest = root / "MANIFEST.json"
    if sha256(manifest.read_bytes()).hexdigest() != NATIVE_MANIFEST_SHA256:
        raise RuntimeError("Native prototype dependency manifest changed")
    for entry in json.loads(manifest.read_text())["files"]:
        check_file(root / entry["path"], entry)
    return root


def load_native():
    root = verify_dependencies()
    spec = importlib.util.spec_from_file_location("ncnc_cardinality_qualified_native_reference", root / "native_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.native_modules()
