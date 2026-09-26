"""Verify the early GNNM pilot against the current control runner.

Run only after the separate split-0 rerun has finished. The adoption rule is
fixed before the rerun: keep the pilot only when its selected validation
checkpoint and model tensors reproduce within the tolerances below. Otherwise
use the fresh row, regardless of either row's test score. Preserve the pilot.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
MAIN_ROOT = REPO / "experiments_iclr" / "results"
RERUN_ROOT = REPO / "experiments_iclr" / "gnnm_split0_verification"
KEY = ("roman-empire", "SAGE", "gnnm", "0")
MAIN_VARIANTS = (
    "gnnm", "independent_projectors", "heads_only", "input_only",
    "output_only", "base", "gnnm_m1",
)
CONFIG_FIELDS = (
    "dataset", "model", "variant", "split", "seed", "num_layers",
    "hidden_dim", "lr", "m", "num_steps", "num_params",
)
VALIDATION_TOL = 1e-7
STATE_TOL = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def selected_row(rows: list[dict[str, str]]) -> dict[str, str]:
    found = [r for r in rows if tuple(r[k] for k in KEY_FIELDS) == KEY]
    if len(found) != 1:
        raise ValueError(f"Expected one {KEY} row, found {len(found)}")
    return found[0]


KEY_FIELDS = ("dataset", "model", "variant", "split")


def row_digest(row: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def safe_record_path(row: dict[str, str], field: str, parent: Path) -> Path:
    relative = Path(row[field])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe {field} path: {relative}")
    path = REPO / relative
    if not path.is_relative_to(parent) or not path.resolve().is_relative_to(parent.resolve()):
        raise ValueError(f"{field} is outside {parent}: {relative}")
    return path


def replace_row(main_csv: Path, fields: list[str], rows: list[dict[str, str]],
                replacement: dict[str, str]) -> None:
    updated = [replacement if tuple(row[k] for k in KEY_FIELDS) == KEY else row
               for row in rows]
    temp = main_csv.with_suffix(".verified.tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(updated)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, main_csv)


def write_json_atomic(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def max_state_difference(left_path: Path, right_path: Path) -> tuple[float, bool]:
    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    for state_label, state in (("pilot", left), ("fresh", right)):
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Non-tensor checkpoint value in {state_label}: {key}")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Nonfinite {state_label} checkpoint tensor: {key}")
    if left.keys() != right.keys():
        return float("inf"), False
    maximum = 0.0
    for key in left:
        x, y = left[key], right[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            return float("inf"), False
        if x.numel():
            difference = (x.double() - y.double()).abs()
            maximum = max(maximum, difference.max().item())
    return maximum, True


def finite_validation(row: dict[str, str], label: str) -> float:
    value = float(row["val_metric"])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {label} validation metric")
    return value


def check_artifacts(row: dict[str, str], root: Path) -> tuple[Path, Path]:
    checkpoint = safe_record_path(row, "checkpoint", root)
    prediction = safe_record_path(row, "prediction_file", root)
    for path in (checkpoint, prediction):
        if not path.is_file():
            raise FileNotFoundError(path)
    return checkpoint, prediction


def validate_protocol_rows(rows: list[dict[str, str]], fresh: dict[str, str] | None) -> None:
    """Reject a complete-looking matrix with mixed settings or missing files."""
    for row in rows:
        split = int(row["split"])
        variant = row["variant"]
        expected_m = 1 if variant in ("base", "gnnm_m1") else 4
        if (int(row["seed"]) != split or int(row["num_layers"]) != 5
                or int(row["hidden_dim"]) != 512
                or not math.isclose(float(row["lr"]), 3e-5, rel_tol=0, abs_tol=1e-12)
                or int(row["m"]) != expected_m
                or int(row["num_steps"]) != 5000
                or not 1 <= int(row["best_step"]) <= 5000):
            raise ValueError(f"Mixed protocol in {variant} split {split}")
        finite_validation(row, f"{variant} split {split}")
        if not math.isfinite(float(row["test_metric"])):
            raise ValueError(f"Nonfinite test metric in {variant} split {split}")
        root = (RERUN_ROOT if fresh is not None and
                tuple(row[k] for k in KEY_FIELDS) == KEY and
                row_digest(row) == row_digest(fresh) else MAIN_ROOT)
        check_artifacts(row, root)


def main() -> None:
    main_csv = MAIN_ROOT / "projector_controls.csv"
    rerun_csv = RERUN_ROOT / "projector_controls.csv"
    audit_path = MAIN_ROOT / "gnnm_split0_verification_audit.json"
    runner_hash = sha256(REPO / "experiments_iclr" / "projector_controls.py")
    verifier_hash = sha256(Path(__file__))
    recorded_hash = (RERUN_ROOT / "runner.sha256").read_text().strip()
    recorded_verifier_hash = (RERUN_ROOT / "verifier.sha256").read_text().strip()
    matrix_hash = (RERUN_ROOT / "matrix.sha256").read_text().strip()
    if runner_hash != recorded_hash:
        raise RuntimeError("Verification run was made with a different source file")
    if verifier_hash != recorded_verifier_hash:
        raise RuntimeError("Verification decision script differs from its frozen source")
    fields, main_rows = read_rows(main_csv)
    fresh_fields, rerun_rows = read_rows(rerun_csv)
    if fields != fresh_fields:
        raise ValueError("Main and fresh CSV headers differ")
    if len(main_rows) != 35 or len(rerun_rows) != 1:
        raise ValueError("Expected 35 control rows and exactly one verification row")
    expected = {("roman-empire", "SAGE", variant, str(split))
                for variant in MAIN_VARIANTS for split in range(5)}
    keys = [tuple(row[k] for k in KEY_FIELDS) for row in main_rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected exactly seven variants on each of five masks")
    fresh = selected_row(rerun_rows)
    validate_protocol_rows(main_rows, fresh)
    validate_protocol_rows([fresh], fresh)
    fresh_ckpt, fresh_pred = check_artifacts(fresh, RERUN_ROOT)
    fresh_validation = finite_validation(fresh, "fresh")

    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit["runner_sha256"] != runner_hash:
            raise RuntimeError("Existing verification audit uses different source")
        if audit["verifier_sha256"] != verifier_hash:
            raise RuntimeError("Existing verification audit uses different decision script")
        if audit["matrix_sha256"] != matrix_hash:
            raise RuntimeError("Existing verification audit uses a different matrix")
        archive = RERUN_ROOT / "pilot_archive"
        archived_csv = archive / "projector_controls_before_verification.csv"
        if sha256(archived_csv) != matrix_hash:
            raise RuntimeError("Archived pilot matrix differs from recorded matrix")
        archived_fields, archived_rows = read_rows(archived_csv)
        if archived_fields != fields:
            raise RuntimeError("Archived matrix header changed")
        validate_protocol_rows(archived_rows, None)
        original = selected_row(archived_rows)
        finite_validation(original, "archived pilot")
        if row_digest(original) != audit["pilot_row_sha256"]:
            raise RuntimeError("Archived pilot row changed after its audit")
        if any(original[field] != fresh[field] for field in CONFIG_FIELDS):
            raise RuntimeError("Audited pilot and fresh configurations differ")
        original_ckpt, original_pred = check_artifacts(original, MAIN_ROOT)
        if (sha256(original_ckpt) != audit["pilot_checkpoint_sha256"]
                or sha256(original_pred) != audit["pilot_prediction_sha256"]):
            raise RuntimeError("Original pilot artifact changed after its audit")
        archived_ckpt = archive / "pilot_checkpoint.pt"
        archived_pred = archive / "pilot_prediction.npz"
        if (sha256(archived_ckpt) != audit["pilot_checkpoint_sha256"]
                or sha256(archived_pred) != audit["pilot_prediction_sha256"]):
            raise RuntimeError("Archived pilot artifact changed after its audit")
        if audit["fresh_row_sha256"] != row_digest(fresh):
            raise RuntimeError("Fresh verification row changed after its audit")
        if (audit["fresh_checkpoint_sha256"] != sha256(fresh_ckpt)
                or audit["fresh_prediction_sha256"] != sha256(fresh_pred)):
            raise RuntimeError("Fresh verification artifact changed after its audit")
        current = selected_row(main_rows)
        archived_others = [row for row in archived_rows
                           if tuple(row[k] for k in KEY_FIELDS) != KEY]
        current_others = [row for row in main_rows
                          if tuple(row[k] for k in KEY_FIELDS) != KEY]
        if current_others != archived_others:
            raise RuntimeError("Nonverification rows changed after the audit")
        if audit["adopted"] == "fresh":
            if row_digest(current) == audit["pilot_row_sha256"]:
                if sha256(main_csv) != matrix_hash:
                    raise RuntimeError("Pending matrix switch differs from recorded matrix")
                replace_row(main_csv, fields, main_rows, fresh)
            elif row_digest(current) != audit["fresh_row_sha256"]:
                raise RuntimeError("Main row matches neither audited version")
        elif audit["adopted"] == "pilot":
            if row_digest(current) != audit["pilot_row_sha256"]:
                raise RuntimeError("Pilot row changed after its audit")
        else:
            raise RuntimeError("Unknown audited adoption decision")
        print(json.dumps(audit, sort_keys=True))
        return

    if sha256(main_csv) != matrix_hash:
        raise RuntimeError("Main matrix differs from preverification manifest")
    pilot = selected_row(main_rows)
    if any(pilot[field] != fresh[field] for field in CONFIG_FIELDS):
        raise ValueError("Pilot and fresh run have different configurations")
    pilot_ckpt, pilot_pred = check_artifacts(pilot, MAIN_ROOT)
    pilot_validation = finite_validation(pilot, "pilot")
    maximum, same_keys = max_state_difference(pilot_ckpt, fresh_ckpt)
    same_step = pilot["best_step"] == fresh["best_step"]
    validation_difference = abs(pilot_validation - fresh_validation)
    keep_pilot = (
        same_keys and same_step and validation_difference <= VALIDATION_TOL
        and maximum <= STATE_TOL
    )
    audit = {
        "selection_rule": (
            "Keep pilot only if checkpoint keys and selected step match, "
            "validation metric differs by at most 1e-7, and the maximum "
            "absolute state-tensor difference is at most 1e-6. Otherwise "
            "adopt the fresh run regardless of test score."
        ),
        "pilot_checkpoint_sha256": sha256(pilot_ckpt),
        "fresh_checkpoint_sha256": sha256(fresh_ckpt),
        "pilot_best_step": int(pilot["best_step"]),
        "fresh_best_step": int(fresh["best_step"]),
        "validation_difference": validation_difference,
        "maximum_state_difference": maximum if math.isfinite(maximum) else None,
        "same_state_keys": same_keys,
        "adopted": "pilot" if keep_pilot else "fresh",
        "runner_sha256": runner_hash,
        "verifier_sha256": verifier_hash,
        "matrix_sha256": matrix_hash,
        "pilot_row_sha256": row_digest(pilot),
        "fresh_row_sha256": row_digest(fresh),
        "pilot_prediction_sha256": sha256(pilot_pred),
        "fresh_prediction_sha256": sha256(fresh_pred),
    }

    archive = RERUN_ROOT / "pilot_archive"
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(main_csv, archive / "projector_controls_before_verification.csv")
    shutil.copy2(pilot_ckpt, archive / "pilot_checkpoint.pt")
    shutil.copy2(pilot_pred, archive / "pilot_prediction.npz")
    if (sha256(archive / "pilot_checkpoint.pt") != audit["pilot_checkpoint_sha256"]
            or sha256(archive / "pilot_prediction.npz") != audit["pilot_prediction_sha256"]):
        raise RuntimeError("Archived pilot artifact copy failed verification")
    write_json_atomic(archive / "pilot_row.json", pilot)
    write_json_atomic(audit_path, audit)
    if not keep_pilot:
        replace_row(main_csv, fields, main_rows, fresh)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
