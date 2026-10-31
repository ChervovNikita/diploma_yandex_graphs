"""Confined outputs and exact-source receipt binding for the chemical amendment."""
import hashlib
import json
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
PACKET = next(p for p in SOURCE_ROOT.parents if p.name == "gine_port_v1")


def output_inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PACKET) or path == PACKET or path.is_relative_to(PACKET / "source") or path.is_relative_to(PACKET / "before_amendment"):
        raise ValueError("New outputs must stay in the chemical amendment outside source/preserved trees")
    return path


def source_paths(root=SOURCE_ROOT):
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and "runtime_classes" not in p.parts and p.suffix != ".pyc"}


def source_hashes(root=SOURCE_ROOT):
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths(root).items()}


def source_gate(args, sources):
    if args.source_gate is None:
        raise ValueError("Prospective freeze requires the completed exact-source CPU gate")
    path = output_inside(args.source_gate)
    value = json.loads(path.read_text())
    if (value.get("status") != "PASS" or value.get("source_sha256") != sources
            or value.get("dataset_loads") != 0 or value.get("optimizer_steps") != 0
            or value.get("performance_evidence") is not False):
        raise ValueError("Source gate is not a completed exact-source synthetic engineering PASS")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
