"""Install a pinned chemical grouping dependency in the research phase only."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

PHASE = Path(__file__).resolve().parents[1]
if os.environ.get("GNNM_PHASE_ROOT") != str(PHASE):
    raise ValueError("Require explicit repository wrapper")
TARGET = PHASE / "dependencies/rdkit_2025_3_6"
RECEIPT = PHASE / "dependencies/rdkit_2025_3_6_install.json"
if TARGET.exists() or RECEIPT.exists():
    raise ValueError("Preserve existing dependency attempt; do not reinstall over it")
TARGET.parent.mkdir(parents=True, exist_ok=True)
argv = [sys.executable, "-m", "pip", "--isolated", "install", "--no-deps", "--only-binary=:all:",
        "--index-url", "https://pypi.org/simple", "--cache-dir", str(PHASE / "cache/pip"),
        "--target", str(TARGET), "rdkit==2025.3.6"]
result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout, flush=True)
hashes = {str(p.relative_to(TARGET)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(TARGET.rglob("*")) if p.is_file()}
value = {"status": "installed" if result.returncode == 0 else "failed_preserved", "argv": argv,
         "exit_code": result.returncode, "files_sha256": hashes, "existing_venv_modified": False,
         "dependency": "rdkit==2025.3.6", "usage": "training-only molecule/scaffold grouping export",
         "predictive_evidence": False}
with RECEIPT.open("x") as stream:
    json.dump(value, stream, indent=2)
    stream.write("\n")
print(json.dumps({k: v for k, v in value.items() if k != "files_sha256"}), flush=True)
sys.exit(result.returncode)
