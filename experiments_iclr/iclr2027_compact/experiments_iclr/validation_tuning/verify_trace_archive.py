"""Verify the exact primary 432-cell result/trace archive, then replay selection.

The anonymous upload stores every original result JSON and validation trace in
one lossless XZ tar. The complete author stage may instead keep plain files.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def unpack(target: Path) -> None:
    original = json.loads((ROOT / "COMPACT_BUNDLE_MANIFEST.json").read_text())
    expected = {name: digest for name, digest in original["file_sha256"].items()
                if name.startswith("results/")}
    require(len(expected) == 864, "Primary result/trace manifest must have 864 files")
    packing = json.loads((ROOT / "PACKED_RESULTS_MANIFEST.json").read_text())
    archive_path = ROOT / "RESULTS_RECORDS.tar.xz"
    require(sha(archive_path.read_bytes()) == packing["archive_sha256"],
            "Packed primary archive SHA-256 differs")
    require(packing["records"] == expected, "Packed primary record map differs")
    found = set()
    with tarfile.open(archive_path, "r:xz") as source:
        for item in source:
            path = Path(item.name)
            require(item.isfile() and not path.is_absolute() and
                    ".." not in path.parts and item.name in expected and
                    item.name not in found,
                    f"Unsafe or unknown primary archive member: {item.name}")
            stream = source.extractfile(item)
            require(stream is not None, f"Missing primary archive bytes: {item.name}")
            data = stream.read()
            require(sha(data) == expected[item.name],
                    f"Primary result/trace hash differs: {item.name}")
            out = target / item.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            found.add(item.name)
    require(found == set(expected), "Missing primary archive records")


def main() -> int:
    if (ROOT / "results").exists():
        return subprocess.run([sys.executable,
                               str(ROOT / "verify_compact_tuning.py")]).returncode
    with tempfile.TemporaryDirectory(prefix="primary_trace_check_") as name:
        target = Path(name) / "study"
        shutil.copytree(ROOT, target,
                        ignore=shutil.ignore_patterns("RESULTS_RECORDS.tar.xz",
                                                       "__pycache__"))
        unpack(target)
        return subprocess.run([sys.executable,
                               str(target / "verify_compact_tuning.py")]).returncode


if __name__ == "__main__":
    sys.exit(main())
