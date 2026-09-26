"""Export only the 24 locked seed-0 checkpoints for off-host timing.

This transport script is post-freeze. It requires complete independent test
score replay before exporting and never changes the training or selection
source. Archive paths are relative to the tuning study folder.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def add_file(stream: tarfile.TarFile, path: Path, relative: Path) -> None:
    info = stream.gettarinfo(str(path), arcname=relative.as_posix())
    require(info.isfile(), f"Unexpected nonfile in profile transfer: {relative}")
    info.uid = info.gid = info.mtime = 0
    info.uname = info.gname = ""
    with path.open("rb") as source:
        stream.addfile(info, source)


def main() -> None:
    lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    final_path = ROOT / "FINAL_SCORE_AUDIT.json"
    frozen_path = ROOT / "FROZEN_STUDY.json"
    profile_source = ROOT / "profile_selected_inference.py"
    tools_manifest = ROOT / "POSTFREEZE_TOOLS_MANIFEST.json"
    for path in (lock_path, final_path, frozen_path, profile_source,
                 tools_manifest):
        require(path.is_file(), f"Required profile transfer input missing: {path.name}")
    lock = json.loads(lock_path.read_text())
    final = json.loads(final_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    tools = json.loads(tools_manifest.read_text())
    require(lock["protocol"] == final["protocol"] == frozen["protocol"] ==
            "validation_tuning_sensitivity_v1" and
            lock["freeze_sha256"] == final["freeze_sha256"] == sha(frozen_path) and
            final["selection_lock_sha256"] == sha(lock_path) and
            len(lock["cells"]) == 432 and len(lock["selections"]) == 4,
            "Complete frozen validation lock and final score audit required")
    require(tools["source_sha256"]["profile_selected_inference.py"] ==
            sha(profile_source), "Profiler source differs from postfreeze manifest")

    expected_scores = set()
    for dataset in DATASETS:
        for arm in ARMS:
            selected_candidate = tuple(
                lock["selections"][dataset][arm]["selected_candidate"])
            for lr, wd in dict.fromkeys((selected_candidate, (0.001, 0.0))):
                for seed in (0, 1, 2):
                    expected_scores.add(
                        f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}")
    require(set(final["scores"]) == expected_scores,
            "Final score audit does not cover the exact selected/default allowlist")

    selected = {}
    files = [frozen_path, lock_path, final_path, profile_source, tools_manifest]
    for dataset in DATASETS:
        for arm in ARMS:
            lr, wd = lock["selections"][dataset][arm]["selected_candidate"]
            key = f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed0"
            require(key in lock["cells"] and key in final["scores"],
                    f"Selected seed-0 cell lacks lock/score audit: {key}")
            cell = ROOT / "results" / key
            result, checkpoint = cell / "result.json", cell / "checkpoint.pt"
            require(result.is_file() and checkpoint.is_file() and
                    sha(result) == lock["cells"][key]["result_sha256"] and
                    sha(checkpoint) == lock["cells"][key]["checkpoint_sha256"],
                    f"Selected profile checkpoint differs from lock: {key}")
            selected[key] = {"result_sha256": sha(result),
                             "checkpoint_sha256": sha(checkpoint),
                             "checkpoint_bytes": checkpoint.stat().st_size}
            files.extend((result, checkpoint))
    require(len(selected) == 24, "Expected 24 selected graph/arm seed-0 cells")
    manifest_path = ROOT / "PROFILE_SELECTED_CHECKPOINTS_MANIFEST.json"
    archive_path = ROOT / "PROFILE_SELECTED_CHECKPOINTS.tar.gz"
    require(not manifest_path.exists() and not archive_path.exists(),
            "Refusing to overwrite a prior profile transfer")
    manifest = {
        "protocol": "selected_inference_profile_transfer_v1",
        "source_script_sha256": sha(Path(__file__)),
        "frozen_study_sha256": sha(frozen_path),
        "validation_selection_lock_sha256": sha(lock_path),
        "final_score_audit_sha256": sha(final_path),
        "profile_source_sha256": sha(profile_source),
        "selected_seed0_cells": selected,
        "file_sha256": {str(path.relative_to(ROOT)): sha(path) for path in files},
        "purpose": "off-host inference timing after complete score audit; no score selection or retraining",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with tarfile.open(archive_path, "w:gz") as stream:
        for path in sorted(files + [manifest_path]):
            add_file(stream, path, path.relative_to(ROOT))
    print(json.dumps({"archive": archive_path.name,
                      "archive_sha256": sha(archive_path),
                      "archive_bytes": archive_path.stat().st_size,
                      "manifest_sha256": sha(manifest_path),
                      "selected_cells": len(selected)}), flush=True)


if __name__ == "__main__":
    main()
