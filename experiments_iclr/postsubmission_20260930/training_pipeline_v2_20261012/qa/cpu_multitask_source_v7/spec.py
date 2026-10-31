"""Prospective study contracts. No numerical libraries or data are imported."""
from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path

STAGES = {"development", "confirmation", "engineering"}
KINDS = {"model_fit", "selection", "calibration", "ensemble_assembly", "evaluation"}
ORIGINAL_PAPER_DATASETS = {"romanempire", "amazonratings", "minesweeper", "questions", "tolokers"}


def normalized_id(value):
    return "".join(c for c in str(value).lower() if c.isalnum())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def confined(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Use a project-relative path")
    target = root / rel
    if not target.resolve().is_relative_to(root):
        raise ValueError("Path escapes the project")
    return target


def validate(spec):
    """Validate decisions, not their scientific adequacy or data provenance."""
    required = {
        "schema_version", "study_id", "stage", "hypothesis", "dataset",
        "data", "source", "runs", "comparisons", "release", "resources",
    }
    if set(spec) != required or spec["schema_version"] != 1:
        raise ValueError("Unsupported or incomplete study contract")
    if spec["stage"] not in STAGES or not spec["hypothesis"]:
        raise ValueError("Declare the experimental stage and falsifiable hypothesis")
    if not spec["study_id"] or not spec["dataset"] or not spec["runs"]:
        raise ValueError("Study, dataset and nonempty run roster are required")
    data = spec["data"]
    for key in ("task", "preprocessing", "roles", "prior_exposure", "evidence"):
        if key not in data:
            raise ValueError("Missing data declaration: " + key)
    if set(data["roles"]) != {"train", "select", "cal", "evaluate"}:
        raise ValueError("Declare four separate role authorities")
    if spec["stage"] == "confirmation" and not data.get("confirmation_eligibility"):
        raise ValueError("Confirmation requires explicit unused-evidence eligibility")
    source = spec["source"]
    if not source.get("git_commit") or not source.get("files"):
        raise ValueError("Bind actual source commit and file fingerprints")
    seen = set()
    for run in spec["runs"]:
        fields = {"run_id", "kind", "method", "split", "seed", "body", "adapter",
                  "config", "budget", "selection", "parents"}
        if set(run) != fields or run["kind"] not in KINDS:
            raise ValueError("Incomplete run contract")
        if run["run_id"] in seen:
            raise ValueError("Duplicate run ID")
        seen.add(run["run_id"])
        method_id = normalized_id(run["method"])
        original = (method_id in {"originalgnnm", "gnnmoriginal", "boundarygnnm"}
                    or method_id.startswith("originalboundarygnnm")
                    or run["config"].get("is_original_gnnm", False))
        if original and normalized_id(spec["dataset"]) in ORIGINAL_PAPER_DATASETS:
            raise ValueError("Original GNNM paper experiments are excluded by the user")
        if type(run["seed"]) is not int or type(run["body"]) is not int:
            raise ValueError("Use explicit seed and independent-body identifiers")
        if run["kind"] == "model_fit":
            budget = run["budget"]
            if budget.get("unit") not in {"optimizer_updates", "epochs"}:
                raise ValueError("Declare the training unit")
            if type(budget.get("maximum")) is not int or budget["maximum"] < 1:
                raise ValueError("Declare a positive fixed maximum budget")
            if budget.get("policy") not in {"full_budget", "minimum_then_patience"}:
                raise ValueError("Declare the training-budget policy")
            if budget["policy"] == "minimum_then_patience":
                minimum, patience = budget.get("minimum"), budget.get("patience")
                if (type(minimum) is not int or not 0 <= minimum <= budget["maximum"]
                        or type(patience) is not int or patience < 1):
                    raise ValueError("Invalid minimum/patience")
            selection = run["selection"]
            for key in ("metric", "direction", "ties", "evaluation_interval", "checkpoint0"):
                if key not in selection:
                    raise ValueError("Missing selection decision: " + key)
            if selection["direction"] not in {"min", "max"} or selection["ties"] not in {"first", "last"}:
                raise ValueError("Declare selector direction and tie handling")
            if (type(selection["evaluation_interval"]) is not int
                    or selection["evaluation_interval"] < 1):
                raise ValueError("Invalid selector interval")
    release = spec["release"]
    if release.get("policy") != "whole_roster" or not release.get("primary_endpoint"):
        raise ValueError("Declare whole-roster release and the primary selected endpoint")
    if not spec["comparisons"]:
        raise ValueError("Specify comparisons and decision rules before fitting")
    if not spec["resources"].get("wall_seconds"):
        raise ValueError("A finite resource budget is required")
    return spec


def bind_source(root, spec):
    for row in spec["source"]["files"]:
        path = confined(root, row["path"])
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing or symlinked source: " + row["path"])
        payload = path.read_bytes()
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("Source changed: " + row["path"])


def load(path, project_root):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    spec = validate(json.loads(raw))
    bind_source(project_root, spec)
    return spec, fingerprint(spec)
