"""Verified phase-local wheel extraction; existing research venv stays unchanged."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

PHASE = Path(__file__).resolve().parents[1]
if os.environ.get("GNNM_PHASE_ROOT") != str(PHASE):
    raise ValueError("Require explicit repository wrapper")
TARGET = PHASE / "dependencies/rdkit_2025_3_6_wheel_v2"
RECEIPT = PHASE / "dependencies/rdkit_wheel_unpack_v2.json"
if TARGET.exists() or RECEIPT.exists():
    raise ValueError("Preserve prior attempt; use a new version rather than overwrite")
TARGET.parent.mkdir(parents=True, exist_ok=True)
URL = "https://pypi.org/pypi/rdkit/2025.3.6/json"
with urllib.request.urlopen(URL, timeout=90) as response:
    metadata_bytes = response.read()
metadata = json.loads(metadata_bytes)
tags = {tag: i for i, tag in enumerate(sys_tags())}
compatible = []
for row in metadata["urls"]:
    if row["packagetype"] != "bdist_wheel" or row.get("yanked"):
        continue
    _, version, _, wheel_tags = parse_wheel_filename(row["filename"])
    ranks = [tags[tag] for tag in wheel_tags if tag in tags]
    if str(version) == "2025.3.6" and ranks:
        compatible.append((min(ranks), row["filename"], row))
if not compatible:
    raise ValueError("No compatible pinned RDKit wheel; do not build or change venv")
row = min(compatible)[2]
wheel = TARGET.parent / row["filename"]
if wheel.exists():
    raise ValueError("Preserve existing downloaded wheel")
with urllib.request.urlopen(row["url"], timeout=90) as response, wheel.open("xb") as output:
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        output.write(chunk)
digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
if digest != row["digests"]["sha256"]:
    raise ValueError("PyPI wheel digest differs")
with zipfile.ZipFile(wheel) as archive:
    for item in archive.infolist():
        rel = Path(item.filename)
        if rel.is_absolute() or ".." in rel.parts or ".data" in rel.parts:
            raise ValueError("Unsupported wheel installation path")
        if ((item.external_attr >> 16) & 0o170000) == 0o120000:
            raise ValueError("Refuse archive symlinks")
    TARGET.mkdir()
    archive.extractall(TARGET)
env = os.environ.copy()
env["PYTHONPATH"] = str(TARGET)
check = subprocess.run([sys.executable, "-c",
                        "from rdkit import Chem, __version__; assert __version__ == '2025.03.6'; "
                        "assert Chem.MolFromSmiles('CCO') is not None; print(__version__)"],
                       text=True, capture_output=True, env=env)
files = {str(p.relative_to(TARGET)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(TARGET.rglob("*")) if p.is_file()}
value = {"status": "installed_and_import_checked" if check.returncode == 0 else "import_failure_preserved",
         "pypi_metadata_url": URL, "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
         "wheel_filename": row["filename"], "wheel_url": row["url"], "wheel_sha256": digest,
         "target": str(TARGET.relative_to(PHASE)), "files_sha256": files,
         "import_check_exit": check.returncode, "import_check_stdout": check.stdout,
         "import_check_stderr": check.stderr, "existing_venv_modified": False,
         "previous_pip_failure": "dependencies/rdkit_2025_3_6_install.json; existing venv has no pip",
         "scientific_results_or_old_runtime_modified": False}
with RECEIPT.open("x") as stream:
    json.dump(value, stream, indent=2)
    stream.write("\n")
print(json.dumps({k: v for k, v in value.items() if k != "files_sha256"}), flush=True)
sys.exit(check.returncode)
