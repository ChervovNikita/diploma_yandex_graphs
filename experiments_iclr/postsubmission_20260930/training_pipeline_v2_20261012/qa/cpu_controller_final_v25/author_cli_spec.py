"""Explicit native-policy author CLI contracts; no numerical imports."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

from registry import utc
from spec import canonical, confined, fingerprint

SPLIT_SEEDS = [1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]
AUTHOR_RUN_FIELDS = {"run_id", "kind", "method", "split", "seed", "body", "cli", "config", "budget", "selection", "parents"}
AUTHOR_SPEC_FIELDS = {"schema_version", "study_id", "stage", "hypothesis", "dataset", "data", "source", "runs", "comparisons", "release", "resources"}


def receipt(root, path):
    target = confined(root, path)
    if not target.is_file() or target.is_symlink():
        raise ValueError("Missing or symlinked bound file: " + str(path))
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return {"path": str(target.relative_to(Path(root).resolve())), "bytes": target.stat().st_size, "sha256": digest.hexdigest()}


def validate_receipt(row):
    if (set(row) != {"path", "bytes", "sha256"} or type(row["bytes"]) is not int or row["bytes"] < 0
            or not isinstance(row["sha256"], str) or len(row["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in row["sha256"])):
        raise ValueError("An exact project-relative file receipt is required")
    path = Path(row["path"])
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Bound file must be project-relative")


def verify_receipt(root, row):
    validate_receipt(row)
    if receipt(root, row["path"]) != row:
        raise ValueError("Author CLI bound file changed: " + row["path"])
    return confined(root, row["path"])


def validate_author_spec(value):
    if set(value) != AUTHOR_SPEC_FIELDS or value["schema_version"] != 1:
        raise ValueError("Incomplete author CLI study")
    if value["stage"] not in {"development", "engineering"} or not value["hypothesis"]:
        raise ValueError("Author source replication is development or synthetic engineering work")
    for name in [value["study_id"], *[r["run_id"] for r in value["runs"]]]:
        if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Use safe study/group identifiers")
    if not value["runs"] or len({r["run_id"] for r in value["runs"]}) != len(value["runs"]):
        raise ValueError("Nonempty unique author groups are required")
    data = value["data"]
    if (set(data.get("roles", {})) != {"train", "select", "evaluate"}
            or data.get("role_protocol") != "author_native_train_select_test_no_cal"
            or not data.get("evidence") or not data.get("prior_exposure")):
        raise ValueError("Declare native author roles and no invented CAL authority")
    source = value["source"]
    if not source.get("git_commit") or not source.get("files"):
        raise ValueError("Bind frozen author/runtime source bytes")
    for row in source["files"]:
        validate_receipt(row)
    release = value["release"]
    if (release.get("policy") != "whole_roster" or not release.get("global_plan_gates_required")
            or release.get("author_output_gate") != "whole_author"
            or release.get("required_before_launch") != ["whole_search"]):
        raise ValueError("Author groups require whole-master HPO and whole-author quarantine gates")
    if not value["comparisons"] or not value["resources"].get("wall_seconds"):
        raise ValueError("Declare reporting rule and finite group resource budgets")
    for run in value["runs"]:
        if set(run) != AUTHOR_RUN_FIELDS or run["kind"] != "author_cli_group" or run["parents"]:
            raise ValueError("Author CLI group is an explicit native process, without adapters or fake checkpoints")
        if run["seed"] != 42 or run["body"] != 0 or run["config"].get("stage") != "author_cli":
            raise ValueError("Bind the original process seed and explicit author stage")
        if run["budget"] != {"unit": "optimizer_updates_per_repeat", "policy": "author_native_validation_patience",
                            "maximum": 2000, "patience": 250, "repeats": 10, "resume": False}:
            raise ValueError("Do not convert native stopping or sequential repeats to a full-budget trajectory")
        selection = run["selection"]
        if selection != {"metric": "accuracy", "direction": "max", "ties": "first_strict_above_initial_zero",
                         "evaluation_interval": 1, "checkpoint0": False, "artifact": "source_in_memory_no_checkpoint"}:
            raise ValueError("Retain original native selector and no invented checkpoint")
        cli = run["cli"]
        for key in ["cwd", "argv", "python_executable", "execution_repository", "source_files", "raw_files", "environment", "runtime_versions", "quarantine", "protocol", "basis_dir", "processed_dir", "private_dependency_dir"]:
            if key not in cli:
                raise ValueError("Missing author CLI binding: " + key)
        if (cli["argv"][:2] != ["-u", "training.py"] or not all(isinstance(v, str) for v in cli["argv"])
                or any(v in cli["argv"] for v in ["--runs", "--epochs", "--early_stopping", "--seed", "--device_idx"])
                or cli["protocol"].get("split_seeds") != SPLIT_SEEDS
                or cli["protocol"].get("repeat_grouping") != "ten_sequential_repeats_one_process"):
            raise ValueError("Preserve literal source arguments, defaults and sequential RNG order")
        for key in ["cwd", "basis_dir", "processed_dir", *cli["quarantine"].values()]:
            relative = Path(cli[key] if key in cli else key)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("Author work/output paths must be confined")
        for row in cli["source_files"] + cli["raw_files"]:
            validate_receipt(row)
        if not cli["source_files"] or not cli["raw_files"]:
            raise ValueError("Bind real workdir source and provider raw files")
        for key in ["reservation_mib", "wall_seconds", "estimated_seconds"]:
            number = run["config"].get(key)
            if type(number) not in {int, float} or not math.isfinite(number) or number <= 0:
                raise ValueError("Positive finite author scheduling resources required")
        if type(run["config"]["reservation_mib"]) is not int:
            raise ValueError("Author memory reservation uses integer MiB")
    return value


def bind_author_files(root, value):
    for row in value["source"]["files"]:
        verify_receipt(root, row)
    for run in value["runs"]:
        for row in run["cli"]["source_files"] + run["cli"]["raw_files"]:
            verify_receipt(root, row)


def load_author_spec(path, root, *, bind=True):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    value = validate_author_spec(json.loads(raw))
    if bind:
        bind_author_files(root, value)
    return value, fingerprint(value)


def register_author_spec(registry, value):
    """Use the real registry schema/lifecycle before shared validator integration.

    Only this explicit author schema is admitted; existing model validation is
    not replaced or monkeypatched. Claim/event/finish stay Registry methods.
    """
    validate_author_spec(value)
    body, digest, now = canonical(value), fingerprint(value), utc()
    with registry.db:
        prior = registry.db.execute("SELECT spec_hash FROM studies WHERE id=?", (value["study_id"],)).fetchone()
        if prior:
            if prior[0] != digest:
                raise ValueError("Author study changed; declare a new study")
            return digest
        registry.db.execute("INSERT INTO studies VALUES (?,?,?,?)", (value["study_id"], digest, body, now))
        for run in value["runs"]:
            registry.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                value["study_id"], run["run_id"], run["kind"], run["method"], str(run["split"]),
                run["seed"], run["body"], canonical(run), "registered", now, now, None, None))
    return digest


def author_counts(value):
    validate_author_spec(value)
    groups = len(value["runs"])
    return {"author_cli_groups": groups, "author_sequential_repeat_trajectories": groups*10,
            "author_registered_maximum_optimizer_updates": groups*10*2000,
            "author_registered_maximum_TRAIN_forwards": groups*10*2000,
            "author_registered_maximum_validation_TEST_forwards": groups*10*2000*4,
            "author_native_actual_updates": None, "author_known_actual_update_subtotal": 0,
            "author_actual_update_counts_unresolved_before_execution": True,
            "actual_author_optimizer_updates_executed_so_far": 0,
            "model_fit_groups": 0, "fake_model_checkpoints": 0,
            "actual_author_groups_started": 0, "actual_author_fits": 0}
