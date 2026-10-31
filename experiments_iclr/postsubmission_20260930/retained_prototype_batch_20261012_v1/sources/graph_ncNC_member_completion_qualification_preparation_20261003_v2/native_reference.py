"""Private native reference loader, used only by later engineering QA.

No third-party source is included in this public module. Exact preserved bytes
remain in Git-ignored private_evidence; no dependency installation is attempted.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def verify_seal():
    here=Path(__file__).resolve().parent
    manifest_path=here/"MANIFEST.json"
    manifest=json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        path=here/entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"] or path.stat().st_size!=entry["bytes"]:
            raise RuntimeError("Sealed source changed: "+entry["path"])
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def native_modules():
    here = Path(__file__).resolve().parent
    pins = json.loads((here / "PRIVATE_SOURCE_PINS.json").read_text())
    for entry in pins["files"]:
        path = here / entry["private_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("Preserved native source mismatch: " + entry["private_path"])
    source_dir = here / "private_evidence" / "native"
    prior = sys.modules.get("utils")
    try:
        spec = importlib.util.spec_from_file_location("ncnc_private_reference_utils", source_dir / "utils.py")
        utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils)
        sys.modules["utils"] = utils
        spec = importlib.util.spec_from_file_location("ncnc_private_reference_model", source_dir / "model.py")
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
    finally:
        if prior is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = prior
    return model, utils
