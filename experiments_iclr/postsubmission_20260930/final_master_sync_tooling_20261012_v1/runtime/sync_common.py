"""Exact immutable metadata bindings; never load arrays or scientific states."""
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess


def allocation_guard():
    if socket.gethostname() != "anogena-2-0":
        raise ValueError("Allocation hostname verification failed")
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                     text=True, timeout=10).splitlines()
    if output != ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise ValueError("Allocation sole GPU UUID verification failed")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def safe(root, relative, file=True):
    root = Path(root).absolute()
    rel = PurePosixPath(str(relative))
    if (not str(relative) or str(rel) != str(relative) or rel.is_absolute()
            or ".." in rel.parts or "\\" in str(relative) or str(rel) == "."):
        raise ValueError("Canonical project-relative path required: " + str(relative))
    path = root.joinpath(*rel.parts)
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError("No symlink traversal: " + str(relative))
    if file and not path.is_file():
        raise ValueError("Missing regular file: " + str(relative))
    return path


def receipt(root, relative):
    path = safe(root, relative)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return {"path": str(relative), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked(row):
    if (not isinstance(row, dict) or not {"path", "bytes", "sha256"} <= row.keys()
            or type(row["bytes"]) is not int or row["bytes"] < 0
            or not isinstance(row["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
        raise ValueError("Exact path/bytes/SHA256 receipt required")
    return {k: row[k] for k in ("path", "bytes", "sha256")}


def nested_receipts(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield checked(value)
        else:
            for item in value.values():
                yield from nested_receipts(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_receipts(item)


def read_json(root, relative, sha256=None, canonical_hash=None):
    path = safe(root, relative)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError("Bound metadata bytes changed: " + relative)
    value = json.loads(gzip.decompress(raw) if relative.endswith(".gz") else raw)
    if canonical_hash is not None and fingerprint(value) != canonical_hash:
        raise ValueError("Canonical metadata binding differs: " + relative)
    return value, {"path": relative, "bytes": len(raw), "sha256": digest}


def fresh(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    Path(path).chmod(0o444)


def bound_master(root, master, master_hash, summary, summary_sha256):
    plan, plan_receipt = read_json(root, master, canonical_hash=master_hash)
    summary_value, summary_receipt = read_json(root, summary, sha256=summary_sha256)
    if plan.get("scientific_launch_enabled") is not False or plan.get("human_GO_required") is not True:
        raise ValueError("Only a disabled master with explicit human GO may be prepared")
    if summary_value.get("plan_hash") != master_hash:
        raise ValueError("Summary belongs to another master")
    if summary_value.get("files", {}).get(Path(master).name, plan_receipt) != plan_receipt:
        raise ValueError("Summary raw plan receipt differs")
    return plan, summary_value, plan_receipt, summary_receipt


def verify_counts(plan, summary, studies, compiler):
    """Recompute the frozen controller's typed quantities, including unknown totals."""
    actual = compiler.typed_counts(studies)
    expected = summary["counts"]
    if actual != expected:
        keys = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise ValueError("Typed summary differs from frozen contracts: " + ", ".join(keys))
    records = sum(len(s["runs"]) for s in studies)
    fits = sum(r["kind"] == "model_fit" for s in studies for r in s["runs"])
    authors = sum(r["kind"] == "author_cli_group" for s in studies for r in s["runs"])
    if actual["totals"].get("run_records", 0) != records or actual["totals"].get("atomic_fit_run_groups", 0) != fits:
        raise ValueError("Registered count mismatch")
    if len(plan["studies"]) != len(studies):
        raise ValueError("Study count mismatch")
    if type(summary.get("studies")) is int and summary["studies"] != len(studies):
        raise ValueError("Summary study count mismatch")
    determined = actual["model_optimizer_step_call_total_fully_determined"]
    if not determined and actual["model_optimizer_step_call_total"] is not None:
        raise ValueError("Bounded/unresolved optimizer total must remain null")
    for field in ("actual_model_optimizer_updates", "actual_scalar_CAL_updates", "actual_predictive_scores"):
        if actual.get(field) != 0:
            raise ValueError("Preparation must not claim execution")
    return {"studies": len(studies), "registered_records": records, "model_groups": fits,
            "author_groups": authors, "controller_files": len(plan["immutable_source_bundle"]["files"]),
            "typed_counts": actual, "optimizer_totals_are_registered_contracts_not_execution": True}
