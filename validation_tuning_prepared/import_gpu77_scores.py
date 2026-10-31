"""Quarantined import of GPU77 selected/default score artifacts after lock.

This checks only archive paths, hashes, and allowlisted score file presence.
The frozen verifier performs fresh metric/logit replay after the import.
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
METADATA = {"VALIDATION_SELECTION_LOCK.json", "FROZEN_STUDY.json",
            "GPU77_SUBSET_VALIDATION_AUDIT.json"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    parts = tuple(x for x in path.parts if x != ".")
    require(parts and not path.is_absolute() and ".." not in parts and "\\" not in name,
            f"Unsafe archive member: {name!r}")
    return PurePosixPath(*parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("expected_sha256")
    args = parser.parse_args()
    archive = args.archive.resolve()
    require(archive.is_file() and archive.parent == ROOT, "Archive must be inside study")
    digest = sha(archive)
    require(digest == args.expected_sha256, "Archive digest mismatch")
    lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text())
    require(len(lock["cells"]) == 432 and lock["freeze_sha256"] == sha(ROOT / "FROZEN_STUDY.json"),
            "Complete global validation lock required before score import")
    expected = set()
    for dataset in DATASETS:
        for arm, group in lock["selections"][dataset].items():
            chosen = tuple(group["selected_candidate"])
            for lr, wd in dict.fromkeys((chosen, (0.001, 0.0))):
                for seed in (0, 1, 2):
                    expected.add(f"scores/{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}")
    incoming = ROOT / "GPU77_SCORE_IMPORT_QUARANTINE"
    require(not incoming.exists(), "Score quarantine exists")
    for dataset in DATASETS:
        require(not (ROOT / "scores" / dataset).exists(),
                f"Destination scores/{dataset} already exists")
    members, names = [], set()
    with tarfile.open(archive, "r:gz") as stream:
        total = 0
        for member in stream:
            path = safe_name(member.name)
            require(str(path) not in names, f"Duplicate member {path}")
            names.add(str(path))
            require(member.isfile() or member.isdir(), f"Unexpected member type {path}")
            total += member.size
            require(total <= 10_000_000_000, "Archive exceeds 10 GB uncompressed cap")
            if path.parts[0] == "scores":
                require(len(path.parts) >= 2 and path.parts[1] in DATASETS,
                        f"Unexpected score path {path}")
                if member.isfile():
                    require(path.name in ("score.json", "predictions.npz") and
                            str(path.parent) in expected,
                            f"Undeclared score artifact {path}")
            else:
                require(len(path.parts) == 1 and path.name in METADATA,
                        f"Unexpected metadata path {path}")
            members.append(member)
        require(members, "Empty score archive")
        incoming.mkdir()
        stream.extractall(incoming, members=members)
    require(METADATA.issubset(names) and
            sha(incoming / "VALIDATION_SELECTION_LOCK.json") == sha(lock_path) and
            sha(incoming / "FROZEN_STUDY.json") == sha(ROOT / "FROZEN_STUDY.json"),
            "Score archive lock/freeze bytes differ from this study")
    subset = json.loads((incoming / "GPU77_SUBSET_VALIDATION_AUDIT.json").read_text())
    subset_keys = {key for key in lock["cells"]
                   if key.split("/", 1)[0] in DATASETS}
    require(subset["freeze_sha256"] == lock["freeze_sha256"] and
            subset["status"] == "GPU77_216_CELL_VALIDATION_ARTIFACT_AUDIT_PASS" and
            set(subset["cells"]) == subset_keys and len(subset_keys) == 216 and
            all(subset["cells"][key][field] == lock["cells"][key][field]
                for key in subset_keys
                for field in ("result_sha256", "trace_sha256", "checkpoint_sha256")),
            "GPU77 validation subset audit differs from global lock")
    actual = {str(path.parent.relative_to(incoming))
              for path in (incoming / "scores").rglob("score.json")}
    require(actual == expected, f"Score set differs from selected/default allowlist: {len(actual)} vs {len(expected)}")
    for key in expected:
        require((incoming / key / "predictions.npz").is_file(), f"Missing predictions for {key}")
    for dataset in DATASETS:
        shutil.move(str(incoming / "scores" / dataset), str(ROOT / "scores" / dataset))
    report = {"archive_sha256": digest, "archive_bytes": archive.stat().st_size,
              "lock_sha256": sha(lock_path), "score_groups_imported": len(expected),
              "import_script_sha256": sha(Path(__file__))}
    (ROOT / "GPU77_SCORE_IMPORT_MANIFEST.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
