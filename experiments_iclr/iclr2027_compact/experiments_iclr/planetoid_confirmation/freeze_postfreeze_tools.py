"""Lock the independent exact-decision test audit and compact exporter pretest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "POSTFREEZE_AUDIT_EXPORT_SOURCE_LOCK.json"
SOURCES = ("strict_score_decision_audit.py",
           "export_compact_hard_decisions.py",
           "HARD_DECISION_EXPORT_PROTOCOL.md")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    assert not OUT.exists()
    assert not (ROOT / "scores").exists()
    assert not (ROOT / "FINAL_SCORE_AUDIT.json").exists()
    assert not (ROOT / "VALIDATION_SELECTION_LOCK.json").exists()
    freeze = ROOT / "FROZEN_STUDY.json"
    frozen = json.loads(freeze.read_text())
    assert frozen["protocol"] == "planetoid_two_graph_confirmation_v1"
    assert frozen["matrix"]["datasets"] == ["citeseer", "pubmed"]
    payload = {
        "status": "PRETEST_INDEPENDENT_AUDIT_AND_EXPORT_SOURCE_LOCK",
        "study_freeze_sha256": sha(freeze),
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "test_scores_opened_at_lock": False,
        "scope": "post-lock exact pooled/member test decision replay and class export only",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("POSTFREEZE_TOOLS_LOCKED", sha(OUT), flush=True)


if __name__ == "__main__":
    main()
