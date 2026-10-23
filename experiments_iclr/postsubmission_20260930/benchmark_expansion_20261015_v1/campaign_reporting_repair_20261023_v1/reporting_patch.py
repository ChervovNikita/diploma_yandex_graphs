"""Install the additive, pinned BUDDY README reporting repair.

Call ``install()`` before importing the frozen reporting runtime entrypoint or
calling the frozen collector's ``main()``. No native runtime module is imported
or changed. The frozen reporting sources remain unchanged on disk.
"""

import copy
import functools
import hashlib
import importlib
import json
from pathlib import Path
import sys


REPAIR_SCHEMA = "campaign-reporting-pinned-buddy-readme-repair-v1"
README_PROJECT_RELATIVE = (
    "experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/"
    "link_expansion_ports_BUDDY_production_v4/README.md"
)
README_SHA256 = "b8786ea8a3fc170dbafbaa1bfa06896bd3b1e2879894f2b104950deaa3ec96f0"
FROZEN_REPORTING_PINS = {
    "MANIFEST.json": {
        "bytes": 1203,
        "sha256": "e00621e598396d1a0ed8e3fc10ea195fbde5c710e8945ae26071a20452b77213",
    },
    "composed_authority.py": {
        "bytes": 4921,
        "sha256": "225a9b28eb662fd510b533cf340744e1a1d1ae7f3756e2fa64385b1ecc8f30d6",
    },
    "observe_exports.py": {
        "bytes": 9994,
        "sha256": "ec52994f566f7c744677035a8613a71215f41caf30b77c02063fd3f0c22fe9a9",
    },
}
_STATE_ATTRIBUTE = "_campaign_reporting_readme_repair_20261023_v1"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _receipt(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _reporting_directory():
    return Path(__file__).resolve().parent.parent / "server51_campaign_port_v6" / "reporting"


def _verify_frozen_sources(reporting):
    """Pin bytes before importing or using either frozen reporting module."""
    receipts = {}
    for name, expected in FROZEN_REPORTING_PINS.items():
        observed = _receipt(reporting / name)
        _require(all(observed[key] == expected[key] for key in ("bytes", "sha256")),
                 "Frozen reporting source differs: " + name)
        receipts[name] = observed
    manifest = json.loads((reporting / "MANIFEST.json").read_text())
    _require(manifest["schema"] == "allocation-server51-reporting-source-v6",
             "Frozen reporting manifest schema differs")
    members = {row["path"]: row for row in manifest["files"]}
    for name in ("composed_authority.py", "observe_exports.py"):
        _require(name in members and all(
            members[name][key] == FROZEN_REPORTING_PINS[name][key]
            for key in ("bytes", "sha256")),
            "Frozen reporting manifest member differs: " + name)
    return receipts


def _check_module_path(module, reporting, name):
    _require(getattr(module, "__file__", None) is not None
             and Path(module.__file__).resolve() == (reporting / (name + ".py")).resolve(),
             "Refusing a reporting module from another source: " + name)


def _import_reporting_module(name, reporting):
    existing = sys.modules.get(name)
    if existing is not None:
        _check_module_path(existing, reporting, name)
        return existing
    # These are ordinary imports of the reviewed, SHA-pinned plaintext sources.
    location = str(reporting)
    if not sys.path or sys.path[0] != location:
        sys.path.insert(0, location)
    importlib.invalidate_caches()
    module = importlib.import_module(name)
    _check_module_path(module, reporting, name)
    return module


def _verify_dependencies(original_module, authority, actual):
    root = Path(authority["project_root"]).resolve()
    allowed_readme = root / README_PROJECT_RELATIVE
    for name, sha in actual["dependency_sha256"].items():
        path = Path(name).resolve()
        original_module.require(path.is_relative_to(root),
                                "Only genuine project source dependency bytes may be verified")
        if path.suffix not in (".py", ".json"):
            # Restrict the exception to this literal project member as well as
            # its resolved target. A different .md alias is not an allowance.
            original_module.require(
                path == allowed_readme and Path(name).absolute() == allowed_readme
                and sha == README_SHA256,
                "Only the exact pinned BUDDY README dependency is additionally allowed")
        original_module.require(original_module.receipt(path)["sha256"] == sha,
                                "Observed runtime source dependency bytes changed")


def install():
    """Patch both reporting references once and return the producer receipt.

    The original ``verify_dependency_bytes=False`` API remains structural only.
    Production observation and collector calls use the default ``True`` and
    verify every dependency. Both modes explicitly record their verification
    mode in the additive witness field.
    """
    reporting = _reporting_directory()
    frozen_receipts = _verify_frozen_sources(reporting)
    composed = _import_reporting_module("composed_authority", reporting)
    observer = _import_reporting_module("observe_exports", reporting)
    producer_receipt = {
        "schema": REPAIR_SCHEMA,
        "producer_receipt": _receipt(__file__),
        "frozen_reporting_sources": frozen_receipts,
        "allowed_additional_dependency": {
            "path_relative_to_project": README_PROJECT_RELATIVE,
            "sha256": README_SHA256,
        },
        "native_calls_added": 0,
    }
    installed = getattr(composed, _STATE_ATTRIBUTE, None)
    if installed is not None:
        _require(installed["receipt"] == producer_receipt
                 and composed.runtime_witness is installed["wrapper"]
                 and observer.runtime_witness is installed["wrapper"],
                 "Installed reporting repair or its source changed")
        return copy.deepcopy(producer_receipt)

    original = composed.runtime_witness
    _require(observer.runtime_witness is original,
             "Original reporting witness references differ before repair")
    _require(getattr(original, "__module__", None) == "composed_authority"
             and getattr(original, "__name__", None) == "runtime_witness"
             and Path(original.__code__.co_filename).resolve()
             == (reporting / "composed_authority.py").resolve(),
             "Original composed reporting witness was replaced before repair")

    @functools.wraps(original)
    def repaired_runtime_witness(job, spec, actual, *, verify_dependency_bytes=True):
        _verify_frozen_sources(reporting)
        _require(_receipt(__file__) == producer_receipt["producer_receipt"],
                 "Reporting repair producer source changed after installation")
        witness = original(job, spec, actual, verify_dependency_bytes=False)
        if verify_dependency_bytes:
            _verify_dependencies(composed, spec["composed_runtime_authority"], actual)
        witness["reporting_repair"] = copy.deepcopy(producer_receipt)
        witness["reporting_repair"]["dependency_bytes_verified"] = bool(verify_dependency_bytes)
        return witness

    state = {"receipt": producer_receipt, "wrapper": repaired_runtime_witness}
    composed.runtime_witness = repaired_runtime_witness
    observer.runtime_witness = repaired_runtime_witness
    setattr(composed, _STATE_ATTRIBUTE, state)
    return copy.deepcopy(producer_receipt)
