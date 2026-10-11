"""Read-only verification of this sealed source bundle; stdlib only."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest_path = root / "MANIFEST.json"
manifest = json.loads(manifest_path.read_text())
seal = json.loads((root / "SEAL.json").read_text())
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == seal["manifest_sha256"]
for row in manifest["files"]:
    path = root / row["path"]
    assert path.stat().st_size == row["bytes"], row["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"]
print(f"PASS: {len(manifest['files'])} sealed files; no compilation or model execution")
