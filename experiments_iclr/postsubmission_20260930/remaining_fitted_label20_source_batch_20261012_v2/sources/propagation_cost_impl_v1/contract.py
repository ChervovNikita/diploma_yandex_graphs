"""Stdlib contract. This packet supplies no authorization for real execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_ONLY = True
MAX_FIXTURE_NODES = 2048
MAX_FIXTURE_EDGES = 16384
MAX_FIXTURE_INPUT_WIDTH = 64
SCHEMA = "gnnm-current-member-mean-propagation-source-only-v1"
PINS = {
    "_pinned/frozen_models.py": "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303",
    "_pinned/frozen_centered_paths.py": "60224869cc7e08f4d35c776a1339b285dd38fe01be08ce117e8cad88b9784130",
}
RECIPE = {
    "members": 4, "width": 64, "layers": 2, "dropout": 0.2,
    "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
    "betas": [0.9, 0.999], "eps": 1e-8, "epochs": 100,
    "tasks": ["Reddit2", "ogbn-products"], "seeds": [0, 1, 2],
    "arms": ["candidate", "exact_member", "single_base", "naive_shared_sample"],
    "eval_colors": [0, 2], "float_dtype": "float32", "amp": False,
    "tf32": False, "gpu_hours_total": 8, "forecast_max_gpu_hours": 6.5,
    "peak_reserved_GiB_max": 60, "cpu_preparation_hours_max": 16,
    "disk_GiB_extra_max": 200,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_pins():
    for relative, expected in PINS.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Pinned source changed: {relative}")
    return dict(PINS)


def check_seal():
    manifest_path = ROOT / "SOURCE_MANIFEST_local_run01.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "source_only_sealed":
        raise RuntimeError("Missing or invalid source-only seal")
    for row in manifest["files"]:
        path = confined_output(ROOT / row["relative_path"])
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"Sealed packet changed: {row['relative_path']}")
    return sha256(manifest_path)


def confined_output(path):
    """All fixture writes must remain in this implementation directory."""
    resolved = Path(path).resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise ValueError("Output must be a descendant of the implementation directory")
    return resolved


def fixture_shape(nodes, edges, input_width=None):
    if not (0 < nodes <= MAX_FIXTURE_NODES and 0 <= edges <= MAX_FIXTURE_EDGES):
        raise ValueError("Source-only fixture size exceeded; real-data execution is closed")
    if input_width is not None and not (0 < input_width <= MAX_FIXTURE_INPUT_WIDTH):
        raise ValueError("Source-only fixture feature width exceeded")


def refuse_real_pilot(*_args, **_kwargs):
    raise RuntimeError(
        "REAL_PILOT_CLOSED: unadopted source-only packet. A separate frozen "
        "admission, complete-graph resource gate and authorized account are required."
    )


if __name__ == "__main__":
    refuse_real_pilot()
