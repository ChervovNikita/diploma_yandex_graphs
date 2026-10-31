"""Quarantined import of the GPU77 validation-only result archive on A100.

Prepared before the archive exists. This transport audit never loads labels,
predictions, traces, or checkpoints. The frozen independent verifier performs
the complete 432-cell scientific audit after import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
DATASETS = ("actor", "chameleon_filtered")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def normalized(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    parts = tuple(p for p in path.parts if p != ".")
    require(parts and not path.is_absolute() and ".." not in parts and "\\" not in name,
            f"Unsafe archive member: {name!r}")
    return PurePosixPath(*parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("expected_sha256")
    args = parser.parse_args()
    archive = args.archive.resolve()
    require(archive.is_file() and archive.parent == ROOT, "Archive must be inside the study folder")
    digest = sha(archive)
    require(digest == args.expected_sha256, "Archive SHA-256 mismatch")
    incoming = ROOT / "GPU77_IMPORT_QUARANTINE"
    require(not incoming.exists(), "Quarantine already exists")
    for dataset in DATASETS:
        require(not (ROOT / "results" / dataset).exists(),
                f"Destination results/{dataset} already exists")
    members, metadata, names = [], [], set()
    with tarfile.open(archive, "r:gz") as stream:
        total_size = 0
        for member in stream:
            path = normalized(member.name)
            require(str(path) not in names, f"Duplicate archive path: {path}")
            names.add(str(path))
            require(member.isfile() or member.isdir(), f"Unsupported archive member type: {path}")
            total_size += member.size
            require(total_size <= 10_000_000_000, "Uncompressed archive exceeds 10 GB cap")
            if path.parts[0] == "results":
                require(len(path.parts) >= 2 and path.parts[1] in DATASETS,
                        f"Unexpected results path: {path}")
                require(".inprogress" not in str(path), f"Interrupted cell in archive: {path}")
            else:
                require(len(path.parts) == 1 and (
                    path.name in ("FROZEN_STUDY.json", "preflight.json") or
                    path.name.startswith("gpu77_") or path.name.startswith("GPU77_")),
                    f"Unexpected metadata path: {path}")
                if member.isfile():
                    metadata.append(path.name)
            members.append(member)
        require(members, "Empty archive")
        incoming.mkdir()
        stream.extractall(incoming, members=members)
    freeze = incoming / "FROZEN_STUDY.json"
    require(freeze.is_file() and sha(freeze) == sha(ROOT / "FROZEN_STUDY.json"),
            "GPU77 archive freeze differs from A100")
    counts = {}
    for dataset in DATASETS:
        path = incoming / "results" / dataset
        require(path.is_dir(), f"Missing results/{dataset}")
        count = len(list(path.rglob("result.json")))
        require(count == 108, f"Expected 108 cells for {dataset}, got {count}")
        counts[dataset] = count
    for dataset in DATASETS:
        shutil.move(str(incoming / "results" / dataset), str(ROOT / "results" / dataset))
    manifest = {
        "archive_sha256": digest, "archive_bytes": archive.stat().st_size,
        "source_freeze_sha256": sha(ROOT / "FROZEN_STUDY.json"),
        "imported_result_counts": counts,
        "metadata_in_quarantine": sorted(metadata),
        "quarantine_path": str(incoming),
        "import_script_sha256": sha(Path(__file__)),
    }
    (ROOT / "GPU77_IMPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
