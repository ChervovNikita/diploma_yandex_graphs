"""Post-audit compact projection of the frozen WikiCS matched54 study."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch

import tuning as primary
import wikics_matched54 as study


OUT = study.OUT
STAGE = OUT / "compact_stage"
ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def make_manifest(folder, excluded=()):
    return {str(path.relative_to(folder)): study.sha(path)
            for path in sorted(folder.rglob("*")) if path.is_file() and
            str(path.relative_to(folder)) not in excluded}


def main():
    freeze_sha = study.check_freeze()
    require(not STAGE.exists(), "Compact stage overwrite refused")
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    audit_path = OUT / "FINAL_SCORE_AUDIT.json"
    summary_path = OUT / "SUMMARY.json"
    require((OUT / "finalize.exit").read_text().strip() == "0" and
            lock_path.is_file() and audit_path.is_file() and summary_path.is_file(),
            "Complete independent audit required")
    lock = json.loads(lock_path.read_text())
    audit = json.loads(audit_path.read_text())
    require(len(lock["cells"]) == 54 and 6 <= len(audit["scores"]) <= 18 and
            lock["freeze_sha256"] == freeze_sha and audit["freeze_sha256"] == freeze_sha and
            audit["validation_selection_lock_sha256"] == study.sha(lock_path),
            "Frozen lock or score audit differs")
    bundle, _ = primary.load_graph("wikics", torch.device("cpu"), include_test=True)
    valid_idx = bundle.valid_idx.cpu().numpy()
    valid_y = bundle.valid_y.cpu().numpy()
    test_idx = bundle.test_idx.cpu().numpy()
    test_y = bundle.test_y.cpu().numpy()

    for name in ("FROZEN_WIKICS_MATCHED54.json", "PREFLIGHT_CUDA.json",
                 "VALIDATION_SELECTION_LOCK.json", "FINAL_SCORE_AUDIT.json", "SUMMARY.json"):
        copy(OUT / name, STAGE / name)
    for name in ("wikics_matched54.py", "verify_wikics_matched54.py",
                 "WIKICS_MATCHED54_PROTOCOL.md", "WIKICS_MATCHED54_WIDTH_SCAN.json",
                 "build_wikics_matched54_compact.py", "verify_wikics_matched54_compact.py"):
        copy(ROOT / name, STAGE / "source" / name)
    save(STAGE / "references" / "validation.npz", indices=valid_idx, labels=valid_y)
    save(STAGE / "references" / "test.npz", indices=test_idx, labels=test_y)

    for arm in study.ARMS:
        for lr, wd in study.CANDIDATES:
            for seed in study.SEEDS:
                identity = study.key(arm, lr, wd, seed)
                cell = study.cell_dir(arm, lr, wd, seed)
                frozen = lock["cells"][identity]
                require(study.sha(cell / "result.json") == frozen["result_sha256"] and
                        study.sha(cell / "validation_trace.csv") == frozen["trace_sha256"],
                        f"Locked cell differs: {identity}")
                copy(cell / "result.json", STAGE / "results" / identity / "result.json")
                copy(cell / "validation_trace.csv", STAGE / "results" / identity / "validation_trace.csv")
                if frozen["failure"] is not None:
                    continue
                companion = cell / "validation_companion.npz"
                require(study.sha(companion) == frozen["validation_companion_sha256"],
                        f"Validation companion differs: {identity}")
                with np.load(companion, allow_pickle=False) as data:
                    require(np.array_equal(data["valid_indices"], valid_idx) and
                            np.array_equal(data["valid_labels"], valid_y),
                            f"Validation reference differs: {identity}")
                    save(STAGE / "hard_decisions" / "validation" / (identity + ".npz"),
                         pooled=data["valid_pooled_logits"].argmax(-1).astype(np.uint8),
                         members=data["valid_member_logits"].argmax(-1).astype(np.uint8))

    expected = set()
    for arm in study.ARMS:
        selected = tuple(lock["selections"][arm]["selected_candidate"])
        for lr, wd in dict.fromkeys((selected, study.DEFAULT)):
            for seed in study.SEEDS:
                expected.add(study.key(arm, lr, wd, seed))
    require(set(audit["scores"]) == expected, "Test allowlist differs")
    for identity in sorted(expected):
        folder = OUT / "scores" / identity
        score_path = folder / "score.json"
        prediction_path = folder / "predictions.npz"
        require(study.sha(score_path) == audit["scores"][identity]["score_sha256"] and
                study.sha(prediction_path) == audit["scores"][identity]["predictions_sha256"],
                f"Audited test artifacts differ: {identity}")
        copy(score_path, STAGE / "scores" / identity / "score.json")
        with np.load(prediction_path, allow_pickle=False) as data:
            pooled = np.asarray(data["test_pooled_logits"], dtype=np.float32)
            pclass = pooled.argmax(-1).astype(np.uint8)
            mclass = data["test_member_logits"].argmax(-1).astype(np.uint8)
            row = json.loads(score_path.read_text())
            require(pooled.shape == (len(test_y), bundle.classes) and
                    abs(float(np.mean(pclass == test_y)) - row["test_accuracy"]) <= 1e-7,
                    f"Test decision or accuracy differs: {identity}")
            save(STAGE / "hard_decisions" / "test" / (identity + ".npz"),
                 pooled=pclass, members=mclass)
            save(STAGE / "pooled_float_companion" / (identity + ".npz"), logits=pooled)

    readme = ("# WikiCS parameter-matched post hoc evidence\n\n"
              "This compact folder contains all 54 validation traces, frozen selection and score audits, "
              "and exact pooled/member hard decisions for every allowed test checkpoint. "
              "Run `python3 source/verify_wikics_matched54_compact.py .` with NumPy. "
              "The `pooled_float_companion` subfolder contains exact float32 pooled test logits and "
              "allows cross-entropy replay. The original weights and full per-member float logits "
              "are omitted for size; independent forward replay of the original checkpoints "
              "requires those omitted author weights. The server audit verified all original "
              "checkpoints before test scoring.\n")
    (STAGE / "README.md").write_text(readme)
    base_manifest = make_manifest(STAGE, excluded=("COMPACT_MANIFEST.json",))
    primary.write_json(STAGE / "COMPACT_MANIFEST.json", {
        "protocol": study.PROTOCOL, "freeze_sha256": freeze_sha,
        "validation_lock_sha256": study.sha(lock_path),
        "score_audit_sha256": study.sha(audit_path), "files": base_manifest})
    with tarfile.open(OUT / "WIKICS_MATCHED54_COMPACT.tar.xz", "w:xz", preset=6) as tar:
        tar.add(STAGE, arcname="wikics_matched54")
    print(json.dumps({"stage": str(STAGE), "archive": str(OUT / "WIKICS_MATCHED54_COMPACT.tar.xz"),
                      "archive_bytes": (OUT / "WIKICS_MATCHED54_COMPACT.tar.xz").stat().st_size,
                      "archive_sha256": study.sha(OUT / "WIKICS_MATCHED54_COMPACT.tar.xz")}), flush=True)


if __name__ == "__main__":
    main()
