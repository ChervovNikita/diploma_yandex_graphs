"""Assemble an anonymous, independently checkable HPO study folder.

Run only after the complete selection, score, strict-decision, and compact
hard-class audits. Checkpoints, raw logits, datasets, and host logs stay out.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "compact_stage"
ARCHIVE = ROOT / "VALIDATION_TUNING_COMPACT.tar.gz"
SOURCE_FILES = (
    "tuning.py", "models.py", "verify_tuning.py", "STUDY_PROTOCOL.md",
    "PRETRAIN_REPAIR.md", "FROZEN_STUDY.json",
    "FROZEN_STUDY_ABORTED_PRETRAIN.json",
    "STUDY_EXECUTION_CHRONOLOGY.md", "POSTFREEZE_TOOLS_MANIFEST.json",
    "COMPACT_TRANSPORT_TOOLS_MANIFEST.json", "COMPACT_SCOPE.md",
    "import_gpu77_results.py", "import_gpu77_scores.py",
    "strict_score_decision_audit.py", "summarize_tuning.py",
    "profile_selected_inference.py", "audit_selected_inference.py",
    "export_profile_checkpoint_archive.py",
    "export_compact_hard_decisions.py", "HARD_DECISION_EXPORT_PROTOCOL.md",
    "verify_compact_tuning.py",
    "VALIDATION_SELECTION_LOCK.json", "FINAL_SCORE_AUDIT.json",
    "STRICT_DECISION_AUDIT.json", "HARD_DECISION_EXPORT_MANIFEST.json",
)
PRIVATE_PATTERN = re.compile(
    r"/Users/|/home/|jovyan|shmelev|GENLINK|Chervov|github\.com/Chervov",
    re.IGNORECASE,
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def copy(relative: Path) -> None:
    source, target = ROOT / relative, STAGE / relative
    require(source.is_file(), f"Compact source file missing: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    require(not STAGE.exists() and not ARCHIVE.exists(),
            "Refusing to overwrite prior compact stage/archive")
    lock = json.loads((ROOT / "VALIDATION_SELECTION_LOCK.json").read_text())
    final = json.loads((ROOT / "FINAL_SCORE_AUDIT.json").read_text())
    strict = json.loads((ROOT / "STRICT_DECISION_AUDIT.json").read_text())
    hard = json.loads((ROOT / "HARD_DECISION_EXPORT_MANIFEST.json").read_text())
    require(len(lock["cells"]) == 432 and
            strict["final_score_audit_sha256"] ==
            sha(ROOT / "FINAL_SCORE_AUDIT.json") and
            hard["strict_decision_audit_sha256"] ==
            sha(ROOT / "STRICT_DECISION_AUDIT.json") and
            set(final["scores"]) == set(strict["per_checkpoint"]) ==
            set(hard["cells"]),
            "Complete independent score/decision/hard-class audit required")
    STAGE.mkdir()
    for name in SOURCE_FILES:
        copy(Path(name))
    for key in sorted(lock["cells"]):
        copy(Path("results") / key / "result.json")
        copy(Path("results") / key / "validation_trace.csv")
    for key in sorted(final["scores"]):
        copy(Path("scores") / key / "score.json")
    for record in hard["graph_references"].values():
        copy(Path(record["reference_path"]))
    for record in hard["cells"].values():
        copy(Path(record["hard_path"]))
    for name in ("selected_default.csv", "validation_grid.csv",
                 "paired_seed_differences.csv", "provenance.json"):
        copy(Path("summary") / name)

    verified = subprocess.run(
        [sys.executable, str(STAGE / "verify_compact_tuning.py")],
        check=True, capture_output=True, text=True)
    require('"status": "PASS"' in verified.stdout,
            "Compact selection/hard-score verifier did not report PASS")
    paths = sorted(path for path in STAGE.rglob("*") if path.is_file())
    for path in paths:
        relative = path.relative_to(STAGE)
        require(path.suffix not in (".pt", ".pth", ".ckpt") and
                path.name != "predictions.npz" and
                "cache" not in relative.parts and
                "data" not in relative.parts,
                f"Private/raw file entered compact stage: {relative}")
        if path.suffix in (".py", ".md", ".json", ".csv", ".txt"):
            require(PRIVATE_PATTERN.search(path.read_text(errors="replace")) is None,
                    f"Host/identity string entered compact stage: {relative}")
    file_hashes = {str(path.relative_to(STAGE)): sha(path) for path in paths}
    bundle = {
        "protocol": "validation_tuning_compact_stage_v1",
        "frozen_study_sha256": sha(ROOT / "FROZEN_STUDY.json"),
        "selection_lock_sha256": sha(ROOT / "VALIDATION_SELECTION_LOCK.json"),
        "final_score_audit_sha256": sha(ROOT / "FINAL_SCORE_AUDIT.json"),
        "strict_decision_audit_sha256": sha(ROOT / "STRICT_DECISION_AUDIT.json"),
        "hard_decision_export_manifest_sha256":
            sha(ROOT / "HARD_DECISION_EXPORT_MANIFEST.json"),
        "author_stage_builder_sha256": sha(Path(__file__)),
        "compact_verifier_stdout": verified.stdout.strip(),
        "validation_cells": len(lock["cells"]),
        "audited_selected_default_scores": len(final["scores"]),
        "file_sha256": file_hashes,
        "omitted": "training checkpoints, source float logits, raw datasets, host logs and GPU context",
    }
    manifest = STAGE / "COMPACT_BUNDLE_MANIFEST.json"
    manifest.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    with tarfile.open(ARCHIVE, "w:gz") as stream:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                relative = path.relative_to(STAGE)
                info = stream.gettarinfo(str(path), arcname=relative.as_posix())
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                with path.open("rb") as source:
                    stream.addfile(info, source)
    print(json.dumps({
        "compact_archive_sha256": sha(ARCHIVE),
        "compact_archive_bytes": ARCHIVE.stat().st_size,
        "bundle_manifest_sha256": sha(manifest),
        "validation_cells": len(lock["cells"]),
        "selected_default_scores": len(final["scores"]),
        "files": len(file_hashes),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
