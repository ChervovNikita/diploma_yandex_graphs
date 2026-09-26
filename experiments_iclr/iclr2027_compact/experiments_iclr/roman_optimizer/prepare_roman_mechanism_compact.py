"""Derive an anonymous decision-level Roman optimizer supplement after full audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

DEPTHS = (2, 5)
SEEDS = (0, 1, 2)
ARMS = ("tied", "untied", "sync", "norm_sync")
SOURCE = ("roman_mechanism.py", "verify_roman_mechanism.py", "tuning.py",
          "ROMAN_MECHANISM_PROTOCOL.md", "ROMAN_MECHANISM_DESIGN_FREEZE.json",
          "roman_multimask.py", "models.py", "mechanism.py", "norm_sync_v2.py")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def ce(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(axis=-1)) -
                         shifted[np.arange(len(labels)), labels]))


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    src, dst = args.source.resolve(), args.output.resolve()
    assert not dst.exists(), "Refusing an existing compact stage"
    freeze_path = src / "ROMAN_MECHANISM_SOURCE_FREEZE.json"
    freeze = read(freeze_path)
    assert freeze["protocol"] == "roman_optimizer_history_mask0_depth2_5_v3"
    assert freeze["depths"] == list(DEPTHS) and freeze["seeds"] == list(SEEDS)
    assert freeze["arms"] == list(ARMS) and freeze["expected_cells"] == 24
    assert set(freeze["source_sha256"]) == set(SOURCE) | {"data/roman_empire.npz"}
    for name, expected in freeze["source_sha256"].items():
        assert sha(src / name) == expected, name
    preflight = read(src / "preflight_cuda.json")
    lock = read(src / "validation_lock.json")
    audit = read(src / "complete_score_audit.json")
    source_sha = sha(freeze_path)
    assert preflight["status"] == "ALL_SEED_DEPTH_CUDA_PREFLIGHT_PASS"
    assert preflight["source_freeze_sha256"] == source_sha
    assert len(preflight["records"]) == 6
    assert lock["status"] == "COMPLETE_24_CELL_VALIDATION_REPLAY_LOCK"
    assert lock["source_freeze_sha256"] == source_sha
    assert lock["preflight_sha256"] == sha(src / "preflight_cuda.json")
    assert lock["test_scored"] is False and len(lock["records"]) == 24
    amendment = read(src / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json")
    amendment_source = src.parent / "roman_mechanism_v3_verification_amendment.py"
    diagnosis = src.parent / "ROMAN_MECHANISM_V3_COLLAPSE_DIAG_DEPTH2_SEED2_SYNC.json"
    transport = src / "transport_interruptions/TRANSPORT_INTERRUPT_NOTE.json"
    assert amendment["status"] == "PROSPECTIVE_VALIDATION_ONLY_VERIFICATION_AMENDMENT_V1"
    assert amendment["original_v3_source_freeze_sha256"] == source_sha
    assert amendment["original_preflight_sha256"] == sha(src / "preflight_cuda.json")
    assert amendment["amendment_source_sha256"] == sha(amendment_source)
    assert amendment["diagnosis_sha256"] == sha(diagnosis)
    assert lock["verification_amendment_sha256"] == sha(src / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json")
    assert lock["amendment_source_sha256"] == sha(amendment_source)
    assert len(amendment["prior_complete_result_sha256"]) == 10
    assert read(transport)["amendment_freeze_sha256"] == sha(src / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json")
    for item in lock["records"]:
        collapse = item["collapse_replay"]
        if item["arm"] in ("sync", "norm_sync"):
            assert collapse["cpu_member_logits_bitwise_equal"] is True
            assert collapse["cpu_collapsed_state_exactly_mapped"] is True
            assert collapse["gpu_collapsed_state_exactly_mapped"] is True
            assert collapse["member_validation_decision_mismatches"] == 0
            assert collapse["pooled_validation_decision_mismatches"] == 0
            assert collapse["max_abs_validation_member_logit_difference"] <= 1e-4
        else:
            assert collapse is None
    assert audit["status"] == "COMPLETE_24_CELL_CUDA_TEST_REPLAY_PASS"
    assert audit["source_freeze_sha256"] == source_sha
    assert audit["validation_lock_sha256"] == sha(src / "validation_lock.json")
    assert len(audit["records"]) == 24 and len(audit["paired"]) == 6
    test_collapse = read(src / "post_score_test_collapse_audit.json")
    test_collapse_source = src.parent / "roman_v3_postscore_test_collapse_audit.py"
    assert test_collapse["status"] == "COMPLETE_12_CELL_POST_SCORE_TEST_COLLAPSE_AUDIT_PASS"
    assert test_collapse["source_freeze_sha256"] == source_sha
    assert test_collapse["verification_amendment_sha256"] == sha(
        src / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json")
    assert test_collapse["validation_lock_sha256"] == sha(src / "validation_lock.json")
    assert test_collapse["complete_score_audit_sha256"] == sha(src / "complete_score_audit.json")
    assert test_collapse["audit_source_sha256"] == sha(test_collapse_source)
    assert test_collapse["gpu_diagnostic_tolerance"] == 1e-4
    assert len(test_collapse["records"]) == 12
    assert all(r["cpu_member_logits_bitwise_equal"] is True and
               r["gpu_member_decision_mismatches"] == 0 and
               r["gpu_pooled_decision_mismatches"] == 0 and
               r["gpu_max_abs_member_logit_difference"] <= 1e-4
               for r in test_collapse["records"])
    for name in SOURCE:
        copy(src / name, dst / name)
    design = read(src / "ROMAN_MECHANISM_DESIGN_FREEZE.json")
    revision_evidence = (
        ("ROMAN_MECHANISM_V2_FIRST_STEP_DIAG_DEPTH2_SEED1.json",
         "v2_failing_first_step_diagnostic_sha256"),
        ("roman_mechanism_v2_all6_first_step_diag.log",
         "all_six_training_only_calibration_log_sha256"),
    )
    for filename, hash_key in revision_evidence:
        origin = src.parent / filename
        assert sha(origin) == design[hash_key]
        copy(origin, dst / "numeric_revision_evidence" / filename)
    for name in ("ROMAN_MECHANISM_SOURCE_FREEZE.json", "preflight_cuda.json",
                 "validation_lock.json", "complete_score_audit.json"):
        copy(src / name, dst / name)
    for origin in (src / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json",
                   amendment_source, diagnosis, transport,
                   src / "post_score_test_collapse_audit.json", test_collapse_source):
        copy(origin, dst / "verification_amendment" / origin.name)
    note = src.parent / "ROMAN_V3_VERIFICATION_AMENDMENT_NOTE.md"
    if note.exists():
        copy(note, dst / "verification_amendment" / note.name)
    helper = Path(__file__).resolve()
    copy(helper, dst / helper.name)
    copy(helper.with_name("verify_roman_mechanism_compact.py"),
         dst / "verify_roman_mechanism_compact.py")
    copy(helper.with_name("roman_mechanism_compact_README.md"), dst / "README.md")
    with np.load(src / "data/roman_empire.npz", allow_pickle=False) as p:
        labels = p["node_labels"].copy()
        masks = {part: p[f"{part}_masks"][0].copy()
                 for part in ("train", "val", "test")}
    assert labels.shape == (22662,)
    assert tuple(int(masks[k].sum()) for k in ("train", "val", "test")) == (
        11331, 5665, 5666)
    anchor = dst / "official_labels_mask0.npz"
    np.savez_compressed(anchor, node_labels=labels,
                        train_mask=masks["train"], val_mask=masks["val"],
                        test_mask=masks["test"])
    derivations = []
    for depth in DEPTHS:
        for seed in SEEDS:
            for arm in ARMS:
                run = Path("results/roman") / f"depth{depth}" / f"seed{seed}" / arm
                test = Path("test_scores") / f"depth{depth}" / f"seed{seed}" / arm
                row = read(src / run / "result.json")
                score = read(src / test / "test_result.json")
                assert row["protocol"] == freeze["protocol"]
                assert row["source_freeze_sha256"] == source_sha
                assert (row["depth"], row["seed"], row["arm"]) == (depth, seed, arm)
                assert row["official_mask"] == 0 and row["epochs_completed"] == 1000
                assert score["protocol"] == freeze["protocol"]
                assert score["source_freeze_sha256"] == source_sha
                assert score["validation_lock_sha256"] == sha(src / "validation_lock.json")
                assert (score["depth"], score["seed"], score["arm"]) == (
                    depth, seed, arm)
                assert score["selected_checkpoint_sha256"] == row["artifact_sha256"]["checkpoint.pt"]
                for artifact, expected in row["artifact_sha256"].items():
                    assert sha(src / run / artifact) == expected
                assert sha(src / test / "test_predictions.npz") == score["test_predictions_sha256"]
                for artifact in ("result.json", "validation_trace.csv"):
                    copy(src / run / artifact, dst / run / artifact)
                copy(src / test / "test_result.json", dst / test / "test_result.json")
                decisions = {}
                for part, original, key in (
                    ("valid", src / run / "selected_validation.npz", "selected_validation.npz"),
                    ("test", src / test / "test_predictions.npz", "test_predictions.npz"),
                ):
                    with np.load(original, allow_pickle=False) as p:
                        idx, y = p["node_index"].copy(), p["y_true"].copy()
                        logits = p["member_logits"].copy()
                    assert np.array_equal(idx, np.flatnonzero(masks["val" if part == "valid" else "test"]))
                    assert np.array_equal(y, labels[idx])
                    assert logits.shape == (4, len(idx), 18) and np.isfinite(logits).all()
                    pooled = logits.mean(axis=0)
                    decisions[f"{part}_indices"] = idx
                    decisions[f"{part}_labels"] = y
                    decisions[f"{part}_member_pred"] = logits.argmax(axis=-1)
                    decisions[f"{part}_pool_pred"] = pooled.argmax(axis=-1)
                    result = row if part == "valid" else score
                    assert abs(float(np.mean(decisions[f"{part}_pool_pred"] == y)) -
                               result[f"{part}_accuracy"]) < 1e-6
                    assert abs(ce(pooled, y) - result[f"{part}_ce"]) < 1e-4
                    if part == "valid":
                        assert sha(original) == row["artifact_sha256"][key]
                    else:
                        assert sha(original) == score["test_predictions_sha256"]
                out = dst / run / "selected_decisions.npz"
                np.savez_compressed(out, **decisions)
                derivations.append({"depth": depth, "seed": seed, "arm": arm,
                                    "result_sha256": sha(src / run / "result.json"),
                                    "test_result_sha256": sha(src / test / "test_result.json"),
                                    "original_validation_sha256": row["artifact_sha256"]["selected_validation.npz"],
                                    "original_test_sha256": score["test_predictions_sha256"],
                                    "derived_decisions_sha256": sha(out)})
    assert len(derivations) == 24
    manifest = {"status": "COMPLETE_24_CELL_DECISION_DERIVATION_PASS",
                "source_freeze_sha256": source_sha,
                "preflight_sha256": sha(src / "preflight_cuda.json"),
                "validation_lock_sha256": sha(src / "validation_lock.json"),
                "complete_score_audit_sha256": sha(src / "complete_score_audit.json"),
                "original_public_npz_sha256": sha(src / "data/roman_empire.npz"),
                "official_label_anchor_sha256": sha(anchor),
                "derivation_script_sha256": sha(helper),
                "verification_amendment_freeze_sha256": sha(src / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json"),
                "verification_amendment_source_sha256": sha(amendment_source),
                "verification_diagnosis_sha256": sha(diagnosis),
                "transport_interrupt_note_sha256": sha(transport),
                "post_score_test_collapse_audit_sha256": sha(src / "post_score_test_collapse_audit.json"),
                "post_score_test_collapse_source_sha256": sha(test_collapse_source),
                "records": derivations}
    (dst / "decision_derivation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in dst.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".md", ".json", ".csv", ".log"):
            contents = path.read_bytes().lower()
            for word in (b"/" + b"users/", b"/home/" + b"jovyan",
                         b"/disk/" + b"10tb/home", b"gen" + b"link",
                         b"ssh-" + b"sr003", b"ai000" + b"1053",
                         b"cher" + b"vov", b"niki" + b"ta",
                         b"shm" + b"elev"):
                assert word not in contents, (path, word)
            assert not re.search(rb"gpu-[0-9a-f]{8}-[0-9a-f-]{20,}", contents), path
    print("COMPACT_24_STAGE_PASS", dst,
          "files", sum(p.is_file() for p in dst.rglob("*")),
          "bytes", sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
