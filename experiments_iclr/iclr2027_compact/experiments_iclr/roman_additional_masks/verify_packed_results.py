"""Verify exact Roman additional-mask records, then run unchanged verifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unpack(target: Path) -> None:
    packing = json.loads((ROOT / "PACKED_RESULTS_MANIFEST.json").read_text())
    expected = packing["records"]
    require(packing["protocol"] == "roman_additional_exact_result_transport_v1"
            and len(expected) == 210 and
            all(name.startswith("results/") for name in expected),
            "Roman packed-record manifest scope differs")
    archive_path = ROOT / "RESULTS_RECORDS.tar.xz"
    require(sha(archive_path.read_bytes()) == packing["archive_sha256"],
            "Roman packed archive SHA-256 differs")
    found = set()
    with tarfile.open(archive_path, "r:xz") as archive:
        for item in archive:
            path = Path(item.name)
            require(item.isfile() and not path.is_absolute() and
                    ".." not in path.parts and item.name in expected and
                    item.name not in found,
                    f"Unsafe or unknown Roman archive member: {item.name}")
            stream = archive.extractfile(item)
            require(stream is not None, f"Missing Roman member bytes: {item.name}")
            data = stream.read()
            require(sha(data) == expected[item.name],
                    f"Roman packed record hash differs: {item.name}")
            out = target / item.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            found.add(item.name)
    require(found == set(expected), "Roman packed archive coverage differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-npz", type=Path)
    args = parser.parse_args()
    extra = (["--public-npz", str(args.public_npz.resolve())]
             if args.public_npz else [])
    if (ROOT / "results").exists():
        return subprocess.run([sys.executable,
                               str(ROOT / "verify_roman_multimask_compact.py"),
                               *extra]).returncode
    with tempfile.TemporaryDirectory(prefix="roman_multimask_records_") as name:
        target = Path(name) / "stage"
        shutil.copytree(ROOT, target,
                        ignore=shutil.ignore_patterns("RESULTS_RECORDS.tar.xz",
                                                       "__pycache__"))
        unpack(target)
        return subprocess.run([sys.executable,
                               str(target / "verify_roman_multimask_compact.py"),
                               *extra]).returncode


if __name__ == "__main__":
    sys.exit(main())
