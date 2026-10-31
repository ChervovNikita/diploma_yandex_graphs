"""Package every validation-selected HPO test-logit archive, without filtering outcomes.

Run in the original audited 432-cell study. The validation lock determines
exactly 72 keys (4 graphs × 6 arms × 3 seeds), and the independent final
score audit binds every original prediction and score file by SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")
EXPECTED_LOCK = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"
EXPECTED_AUDIT = "63da49d3d797fa7856717d4b21a530c74419bfbdfa92614a03b036aa9950bdde"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    audit_path = ROOT / "FINAL_SCORE_AUDIT.json"
    assert sha(lock_path) == EXPECTED_LOCK and sha(audit_path) == EXPECTED_AUDIT
    lock = json.loads(lock_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert len(lock["cells"]) == 432 and len(audit["scores"]) == 141
    assert audit["selection_lock_sha256"] == EXPECTED_LOCK
    paths, rows = [lock_path, audit_path], {}
    for dataset in DATASETS:
        for arm in ARMS:
            lr, wd = lock["selections"][dataset][arm]["selected_candidate"]
            for seed in (0, 1, 2):
                key = f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"
                score_path = ROOT / "scores" / key / "score.json"
                pred_path = ROOT / "scores" / key / "predictions.npz"
                assert score_path.is_file() and pred_path.is_file()
                recorded = audit["scores"][key]
                assert sha(score_path) == recorded["score_sha256"]
                assert sha(pred_path) == recorded["predictions_sha256"]
                score = json.loads(score_path.read_text())
                assert score["checkpoint_sha256"] == lock["cells"][key]["checkpoint_sha256"]
                assert score["validation_selection_lock_sha256"] == EXPECTED_LOCK
                assert score["selected_candidate"] is True
                paths.extend((score_path, pred_path))
                rows[key] = {
                    "score_sha256": sha(score_path),
                    "predictions_sha256": sha(pred_path),
                    "predictions_bytes": pred_path.stat().st_size,
                }
    assert len(rows) == 72 and len(paths) == 146
    manifest_path = ROOT / "SELECTED_HPO72_LOGITS_MANIFEST.json"
    archive_path = ROOT / "SELECTED_HPO72_ORIGINAL_NPZ.tar.gz"
    assert not manifest_path.exists() and not archive_path.exists()
    manifest = {
        "protocol": "all_validation_selected_hpo72_original_logits_v1",
        "validation_selection_lock_sha256": EXPECTED_LOCK,
        "independent_final_score_audit_sha256": EXPECTED_AUDIT,
        "selection_rule": "Every selected candidate for every graph/arm and all three seeds; no test-outcome filtering",
        "cells": rows,
        "original_predictions_npz_bytes": sum(r["predictions_bytes"] for r in rows.values()),
        "original_predictions_npz_count": len(rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    paths.append(manifest_path)
    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        for path in sorted(paths):
            info = archive.gettarinfo(str(path), arcname=path.relative_to(ROOT).as_posix())
            assert info.isfile()
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    print(json.dumps({"archive": archive_path.name,
                      "archive_sha256": sha(archive_path),
                      "archive_bytes": archive_path.stat().st_size,
                      "manifest_sha256": sha(manifest_path),
                      "selected_cells": len(rows),
                      "original_npz_bytes": manifest["original_predictions_npz_bytes"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
