"""Build a small anonymous Roman narrow-control evidence package after audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

DEPTHS = (2, 5)
SEEDS = (0, 1, 2)
SOURCE = ("roman_narrow_untied.py", "verify_roman_narrow_untied.py",
          "ROMAN_NARROW_UNTIED_PROTOCOL.md")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--roman24", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    src, roman24, dst = (x.resolve() for x in
                         (args.source, args.roman24, args.output))
    assert not dst.exists()
    study = src / "roman_narrow_untied_prepared"
    freeze = read(study / "SOURCE_FREEZE.json")
    source_sha = sha(study / "SOURCE_FREEZE.json")
    assert source_sha == "51626e3443341695f59890b832483a28ae2ad73189a08c25cba261d5793dceff"
    assert freeze["source_sha256"] == {name: sha(src / name) for name in SOURCE}
    assert sha(src / "ROMAN_NARROW_UNTIED_WIDTH_LOCK.json") == freeze["width_lock_sha256"]
    assert sha(roman24 / "validation_lock.json") == freeze["original_validation_lock_sha256"]
    assert sha(roman24 / "complete_score_audit.json") == freeze["original_test_audit_sha256"]
    amendment = read(study / "INITIAL_DECISION_AMENDMENT_FREEZE.json")
    lock = read(study / "validation_lock.json")
    audit = read(study / "complete_score_audit.json")
    assert amendment["status"] == "PROSPECTIVE_NARROW_INITIAL_DECISION_VERIFICATION_AMENDMENT"
    assert amendment["original_narrow_source_freeze_sha256"] == source_sha
    assert amendment["amended_verifier_sha256"] == sha(src / "verify_roman_narrow_initial_amendment.py")
    assert amendment["all_six_validation_only_diagnosis_sha256"] == sha(study / "initial_replay_all6_diagnostic.json")
    assert amendment["original_validation_failure_log_sha256"] == sha(study / "validate.log")
    assert lock["status"] == "COMPLETE_6_CELL_NARROW_VALIDATION_REPLAY_LOCK"
    assert lock["source_freeze_sha256"] == source_sha
    assert lock["initial_decision_amendment_sha256"] == sha(study / "INITIAL_DECISION_AMENDMENT_FREEZE.json")
    assert lock["test_scored"] is False and len(lock["records"]) == 6
    assert audit["status"] == "COMPLETE_6_CELL_NARROW_CUDA_TEST_REPLAY_PASS"
    assert audit["validation_lock_sha256"] == sha(study / "validation_lock.json")
    assert len(audit["records"]) == len(audit["paired"]) == 6
    with np.load(roman24 / "official_labels_mask0.npz", allow_pickle=False) as p:
        labels = p["node_labels"].copy()
        indices = {part: np.flatnonzero(p[f"{part}_mask"])
                   for part in ("val", "test")}
    assert labels.shape == (22662,)
    assert (len(indices["val"]), len(indices["test"])) == (5665, 5666)
    copied = ("roman_narrow_untied.py", "verify_roman_narrow_untied.py",
              "verify_roman_narrow_initial_amendment.py",
              "roman_narrow_initial_replay_diag.py",
              "roman_narrow_untied_width_lock.py",
              "ROMAN_NARROW_UNTIED_PROTOCOL.md",
              "ROMAN_NARROW_UNTIED_WIDTH_LOCK.json")
    for name in copied:
        copy(src / name, dst / name)
    for name in ("SOURCE_FREEZE.json", "preflight.json",
                 "INITIAL_DECISION_AMENDMENT_FREEZE.json",
                 "initial_replay_all6_diagnostic.json", "validation_lock.json",
                 "complete_score_audit.json"):
        copy(study / name, dst / name)
    failed = (study / "validate.log").read_text()
    failed = re.sub(r"/disk/" + r"10tb/home/[^/]+/gnnm_iclr_multimask/",
                    "<redacted repository>/", failed)
    (dst / "original_validation_failure_redacted.log").write_text(failed)
    copy(Path(__file__).resolve(), dst / "prepare_roman_narrow_compact.py")
    copy(Path(__file__).with_name("verify_roman_narrow_compact.py"),
         dst / "verify_roman_narrow_compact.py")
    copy(Path(__file__).with_name("roman_narrow_compact_README.md"), dst / "README.md")
    derivations = []
    for depth in DEPTHS:
        for seed in SEEDS:
            run = Path("results/roman") / f"depth{depth}" / f"seed{seed}" / "untied"
            test = Path("test_scores") / f"depth{depth}" / f"seed{seed}" / "untied"
            row = read(study / run / "result.json")
            score = read(study / test / "test_result.json")
            assert row["source_freeze_sha256"] == score["source_freeze_sha256"] == source_sha
            assert score["validation_lock_sha256"] == sha(study / "validation_lock.json")
            assert score["selected_checkpoint_sha256"] == row["artifact_sha256"]["checkpoint.pt"]
            for name, digest in row["artifact_sha256"].items():
                assert sha(study / run / name) == digest
            assert sha(study / test / "test_predictions.npz") == score["test_predictions_sha256"]
            assert sha(study / run / "result.json") == amendment["six_trained_result_sha256"][f"{depth}/{seed}"]
            baseline = freeze["baseline_tied128_artifact_sha256"][f"{depth}/{seed}"]
            tied = roman24 / run.parent / "tied"
            tied_test = roman24 / test.parent / "tied" / "test_result.json"
            assert sha(tied / "result.json") == baseline["result_sha256"]
            assert sha(tied_test) == baseline["test_result_sha256"]
            assert score["test_accuracy"] == next(r["narrow_test_accuracy"] for r in audit["records"] if r["depth"] == depth and r["seed"] == seed)
            data = {}
            for part, path, result in (("valid", study / run / "selected_validation.npz", row),
                                       ("test", study / test / "test_predictions.npz", score)):
                with np.load(path, allow_pickle=False) as p:
                    idx, y, logits = (p[k].copy() for k in
                                      ("node_index", "y_true", "member_logits"))
                assert np.array_equal(idx, indices["val" if part == "valid" else "test"])
                assert np.array_equal(y, labels[idx])
                assert logits.shape == (4, len(idx), 18) and np.isfinite(logits).all()
                pooled = logits.mean(0)
                assert abs(float(np.mean(pooled.argmax(-1) == y)) - result[f"{part}_accuracy"]) < 1e-6
                if part == "test":
                    data["test_pool_logits"] = pooled
                else:
                    data["valid_pool_pred"] = pooled.argmax(-1).astype(np.uint8)
                data[f"{part}_member_pred"] = logits.argmax(-1).astype(np.uint8)
            for name in ("result.json", "validation_trace.csv"):
                copy(study / run / name, dst / run / name)
            copy(study / test / "test_result.json", dst / test / "test_result.json")
            output = dst / run / "selected_evidence.npz"
            np.savez_compressed(output, **data)
            derivations.append({"depth": depth, "seed": seed,
                                "result_sha256": sha(study / run / "result.json"),
                                "test_result_sha256": sha(study / test / "test_result.json"),
                                "original_validation_sha256": row["artifact_sha256"]["selected_validation.npz"],
                                "original_test_sha256": score["test_predictions_sha256"],
                                "derived_evidence_sha256": sha(output)})
    manifest = {"status": "COMPLETE_6_CELL_NARROW_COMPACT_DERIVATION_PASS",
                "source_freeze_sha256": source_sha,
                "width_lock_sha256": sha(src / "ROMAN_NARROW_UNTIED_WIDTH_LOCK.json"),
                "validation_lock_sha256": sha(study / "validation_lock.json"),
                "complete_score_audit_sha256": sha(study / "complete_score_audit.json"),
                "original_failure_log_sha256": amendment["original_validation_failure_log_sha256"],
                "redacted_failure_log_sha256": sha(dst / "original_validation_failure_redacted.log"),
                "amendment_source_sha256": sha(src / "verify_roman_narrow_initial_amendment.py"),
                "roman24_anchor_sha256": sha(roman24 / "official_labels_mask0.npz"),
                "derivation_script_sha256": sha(dst / "prepare_roman_narrow_compact.py"),
                "records": derivations}
    (dst / "derivation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in dst.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".md", ".json", ".csv", ".log"):
            lower = path.read_bytes().lower()
            for marker in (b"/" + b"users/", b"/home/" + b"jovyan",
                           b"/disk/" + b"10tb/home", b"gen" + b"link",
                           b"ssh-" + b"sr003", b"ai000" + b"1053",
                           b"cher" + b"vov", b"niki" + b"ta",
                           b"shm" + b"elev"):
                assert marker not in lower, (path, marker)
    print("NARROW_COMPACT_STAGE_PASS", sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
