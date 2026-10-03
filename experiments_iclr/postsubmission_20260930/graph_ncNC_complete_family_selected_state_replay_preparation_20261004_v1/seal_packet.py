"""Verify and seal source preparation only; no numerical runtime or outcomes."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import sys
from replay_gate import verify_manifest

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent


def desc(path, base=ROOT):
    value = path.read_bytes()
    return {"path": str(path.relative_to(base)), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    assert not (ROOT / "MANIFEST.json").exists() and not (ROOT / "SEAL.json").exists()
    source = json.loads((ROOT / "SOURCE_BINDINGS.json").read_text())
    for row in source["original_manifests"].values():
        verify_manifest((PHASE / row["local_root"]).resolve(), row["manifest_sha256"])
    check = json.loads((ROOT / "STDLIB_PREPARATION_CHECK.json").read_text())
    assert check["status"] == "PASS" and check["case_count"] == len(check["cases"]) and check["case_count"] >= 24
    assert check["scientific_modules_imported"] is False and check["numerical_qualification"] is False
    template = json.loads((ROOT / "ROOT_RELEASE_TEMPLATE.json").read_text())
    assert template["execution_enabled"] is False and template["authorized_invocations"] == [] and template["unit_custody"] == []
    assert template["family_lock"] == source["original_lock_provenance"]["remote"]
    for key in ("data_authority", "runtime_authority", "sealed_minimal_plan"):
        row = source[key]
        assert desc(PHASE / row["path"], PHASE) == row
    custody_pin = source["original_lock_provenance"]["root_custody"]
    assert desc(PHASE / custody_pin["path"], PHASE) == custody_pin
    custody = json.loads((PHASE / custody_pin["path"]).read_text())
    # Read the transport custody descriptor, never the original outcome file.
    lock_row = next(r for r in custody["files"] if r["path"].endswith("/family_lock/FAMILY_LOCK.json"))
    assert lock_row["bytes"] == template["family_lock"]["bytes"] and lock_row["sha256"] == template["family_lock"]["sha256"]
    for key in ("manifest", "seal"):
        row = source["independently_verified_monitor"][key]
        assert desc(PHASE / row["path"], PHASE) == row
    for path in ROOT.glob("*.py"):
        tree = ast.parse(path.read_text())
        compile(tree, str(path), "exec")
    for path in ROOT.glob("*.json"):
        json.loads(path.read_text())
    assert not any(n in sys.modules for n in ("torch", "numpy", "pandas", "pilot_fit", "pilot_train", "pilot_data"))
    stamp = datetime.now(timezone.utc).isoformat()
    files = [p for p in sorted(ROOT.iterdir()) if p.is_file()]
    write(ROOT / "MANIFEST.json", {"schema": "ncnc-selected-state-replay-source-manifest-v1", "UTC": stamp, "files": [desc(p) for p in files], "payload_count": len(files), "payload_bytes": sum(p.stat().st_size for p in files), "scientific_execution": False, "numerical_qualification": False, "execution_release": False})
    write(ROOT / "SEAL.json", {"schema": "ncnc-selected-state-replay-source-seal-v1", "UTC": stamp, "manifest": desc(ROOT / "MANIFEST.json"), "stdlib_check": desc(ROOT / "STDLIB_PREPARATION_CHECK.json"), "source_bindings": desc(ROOT / "SOURCE_BINDINGS.json"), "report": desc(ROOT / "REPORT.md"), "execution_enabled": False, "numerical_qualification": False})
    for path in ROOT.iterdir():
        if path.is_file():
            path.chmod(0o444)
    ROOT.chmod(0o555)
    print(json.dumps({"manifest": desc(ROOT / "MANIFEST.json", PHASE), "seal": desc(ROOT / "SEAL.json", PHASE), "payload_count": len(files), "payload_bytes": sum(p.stat().st_size for p in files), "execution_release": False}, indent=2))


if __name__ == "__main__":
    main()
