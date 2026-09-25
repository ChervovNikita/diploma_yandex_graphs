"""Fail-closed, read-only CPU audit of the frozen ogbn-arxiv propagation control.

The result root must contain all three matched-initialization seed pairs and
every declared artifact. No incomplete or resumed result is accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "experiments_iclr" / "ogbn_arxiv_untied_results"
SOURCE_LOCK = REPO / "experiments_iclr" / "ogbn_arxiv_untied_source_lock.json"
REFERENCE_CONFIG = REPO / "experiments_iclr" / "ogbn_arxiv_300_results" / "run_config.json"
REFERENCE_DATA = REPO / "experiments_iclr" / "ogbn_arxiv_300_results" / "dataset_manifest.json"
SEEDS = (0, 1, 2)
ARMS = ("tied", "untied_propagation")
MEMBERS = 4
EPOCHS = 300
SELECTED_RULE = "max pooled accuracy, then min pooled CE, then earliest epoch"
EPOCH_COLUMNS = (
    "epoch", "train_mean_member_ce", "valid_pooled_accuracy",
    "valid_pooled_ce", "train_step_seconds", "validation_seconds",
    "elapsed_seconds", "is_selected_so_far",
)
SUMMARY_COLUMNS = (
    "seed", "tied_test_accuracy", "untied_test_accuracy", "tied_minus_untied",
    "tied_selected_epoch", "untied_selected_epoch",
)
NPZ_KEYS = {
    "valid_member_logits", "valid_labels", "valid_indices",
    "test_member_logits", "test_labels", "test_indices",
}
PILOT_SELECTED_KEYS = {
    "seed", "variant", "selected_epoch", "validation_selection",
    "valid_accuracy", "valid_ce", "test_accuracy", "test_ce",
    "parameter_count", "trainable_parameter_count", "epochs_run",
    "train_step_seconds_total", "validation_seconds_total", "total_seconds",
    "mean_train_step_ms", "median_train_step_ms", "peak_allocated_mib",
    "checkpoint_sha256", "finished_utc", "device",
}
SELECTED_KEYS = PILOT_SELECTED_KEYS | {
    "arm", "rng_start_sha256", "initialization_sha256",
}
INIT_KEYS = {
    "seed", "tied_initial_state_sha256", "untied_initial_state_sha256",
    "tied_stack_sha256", "untied_stack_sha256", "shared_nonprop_sha256",
    "synthetic_member_max_abs_diff", "all_initial_functions_equal",
    "all_parameters_equal_to_tied", "untied_stacks_disjoint",
    "cross_arm_storage_disjoint", "parameter_count",
    "rng_after_construction_sha256", "tied_rng_start_sha256",
    "untied_rng_start_sha256",
}
CONFIG_KEYS = {
    "protocol", "seeds", "arms", "members", "configuration", "optimizer",
    "train_objective", "selection_rule", "source_lock_sha256",
    "reference_300_config_sha256", "reference_300_dataset_manifest_sha256",
    "torch_version", "torch_geometric_version",
}
EXPECTED_CONFIGURATION = {
    "backbone": "SAGE", "layers": 2, "hidden_dim": 128, "dropout": 0.2,
    "lr": 0.001, "weight_decay": 0.0, "max_epochs": 300,
    "min_epochs": 300, "patience": 20, "eval_every": 1,
}
SOURCE_PATHS = {
    "models.py",
    "experiments_iclr/ogbn_arxiv_pilot.py",
    "experiments_iclr/ogbn_arxiv_300_results/run_config.json",
    "experiments_iclr/ogbn_arxiv_300_results/dataset_manifest.json",
    "experiments_iclr/ogbn_arxiv_untied_protocol.md",
    "experiments_iclr/ogbn_arxiv_untied_control.py",
    "experiments_iclr/verify_ogbn_arxiv_untied.py",
}
FROZEN_SOURCE_HASHES = {
    "models.py": "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303",
    "experiments_iclr/ogbn_arxiv_pilot.py": "7abf975fcf900f52a01a51652605b2d2519aaee02918948732ee5401952dd89b",
    "experiments_iclr/ogbn_arxiv_300_results/run_config.json": "ee26cd93647d77c964a554960ab4893fdc95e5bd7d73fd9479e0a8c32a1bde97",
}
EXPECTED_DATA = {
    "dataset": "ogbn-arxiv", "dataset_package": "ogb",
    "split_source": "PygNodePropPredDataset.get_idx_split()",
    "graph_processing": (
        "torch_geometric.utils.to_undirected on the OGB edge_index; "
        "no explicit self loops"
    ),
    "num_nodes": 169343, "num_features": 128, "num_classes": 40,
    "raw_directed_edges": 1166243, "training_undirected_edges": 2315598,
    "split_sizes": {"train": 90941, "valid": 29799, "test": 48603},
}
REQUIRED_DATA_FILES = (
    "RELEASE_v1.txt", "processed/geometric_data_processed.pt",
    "raw/edge.csv.gz", "raw/node-feat.csv.gz", "raw/node-label.csv.gz",
    "split/time/train.csv.gz", "split/time/valid.csv.gz",
    "split/time/test.csv.gz",
)


def fail(message: str) -> None:
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    if path.is_symlink():
        fail(f"Linked path is forbidden: {path}")
    resolved = path.resolve()
    if resolved != REPO and REPO not in resolved.parents:
        fail(f"Path leaves repository: {path}")
    return resolved


def checked_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        fail(f"Missing, empty, or linked file: {path}")
    return path


def checked_dir(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        fail(f"Missing or linked directory: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with checked_file(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        fail(f"Invalid SHA256 in {name}")
    return value


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> Any:
    fail(f"Nonstandard JSON constant: {value}")


def read_json(path: Path) -> dict[str, Any]:
    with checked_file(path).open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=unique_object,
                           parse_constant=reject_constant)
    if not isinstance(result, dict):
        fail(f"Expected a JSON object: {path}")
    return result


def exact_keys(value: dict[str, Any], keys: set[str], name: str) -> None:
    if set(value) != keys:
        fail(f"{name} fields differ: missing={sorted(keys - set(value))}, extra={sorted(set(value) - keys)}")


def integer(value: Any, name: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        fail(f"Invalid integer in {name}: {value!r}")
    return value


def csv_integer(value: str, name: str, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer in {name}: {value!r}") from exc
    if str(result) != value or (minimum is not None and result < minimum):
        fail(f"Invalid integer in {name}: {value!r}")
    return result


def finite_number(value: Any, name: str, minimum: float | None = None,
                  maximum: float | None = None) -> float:
    if isinstance(value, bool):
        fail(f"Invalid Boolean in {name}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number in {name}: {value!r}") from exc
    if (not math.isfinite(result)
            or (minimum is not None and result < minimum)
            or (maximum is not None and result > maximum)):
        fail(f"Nonfinite or out-of-range {name}: {value!r}")
    return result


def json_number(value: Any, name: str, minimum: float | None = None,
                maximum: float | None = None) -> float:
    if type(value) not in (int, float):
        fail(f"Expected JSON number in {name}: {value!r}")
    return finite_number(value, name, minimum, maximum)


def close(actual: float, expected: Any, name: str, tolerance: float) -> None:
    recorded = finite_number(expected, name)
    if abs(actual - recorded) > tolerance:
        fail(f"{name} differs: recomputed={actual:.10g}, recorded={recorded:.10g}")


def tensor_fingerprint(value: torch.Tensor | np.ndarray) -> str:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def expected_artifact_files() -> set[str]:
    files = {"run_config.json", "dataset_manifest.json", "summary.csv", "summary.json"}
    for seed in SEEDS:
        seed_dir = f"seed_{seed}"
        files.add(f"{seed_dir}/initialization.json")
        for arm in ARMS:
            prefix = f"{seed_dir}/{arm}"
            files.update({
                f"{prefix}/epochs.csv", f"{prefix}/selected_checkpoint.pt",
                f"{prefix}/selected.json", f"{prefix}/selected_predictions.npz",
            })
    return files


def verify_tree_layout(root: Path) -> None:
    checked_dir(root)
    root_names = {"run_config.json", "dataset_manifest.json",
                  "summary.csv", "summary.json", "artifact_manifest.json"}
    root_names.update(f"seed_{seed}" for seed in SEEDS)
    if {entry.name for entry in root.iterdir()} != root_names:
        fail("Result root has missing or extra entries")
    for seed in SEEDS:
        seed_dir = checked_dir(root / f"seed_{seed}")
        if {entry.name for entry in seed_dir.iterdir()} != {"initialization.json", *ARMS}:
            fail(f"Seed directory has missing or extra entries: {seed_dir}")
        checked_file(seed_dir / "initialization.json")
        for arm in ARMS:
            arm_dir = checked_dir(seed_dir / arm)
            required = {"epochs.csv", "selected_checkpoint.pt",
                        "selected.json", "selected_predictions.npz"}
            if {entry.name for entry in arm_dir.iterdir()} != required:
                fail(f"Arm directory has missing or extra entries: {arm_dir}")
            for filename in required:
                checked_file(arm_dir / filename)
    for filename in root_names - {f"seed_{seed}" for seed in SEEDS}:
        checked_file(root / filename)


def verify_artifact_manifest(root: Path) -> None:
    manifest = read_json(root / "artifact_manifest.json")
    exact_keys(manifest, {"sha256"}, "artifact manifest")
    hashes = manifest["sha256"]
    if not isinstance(hashes, dict) or set(hashes) != expected_artifact_files():
        fail("Artifact manifest path set differs from the complete canonical tree")
    for relative in sorted(hashes):
        expected = valid_hash(hashes[relative], relative)
        observed = sha256_file(root / relative)
        if observed != expected:
            fail(f"Artifact SHA256 differs: {relative}")


def verify_sources(config: dict[str, Any]) -> None:
    valid_hash(config["source_lock_sha256"], "source_lock_sha256")
    if sha256_file(SOURCE_LOCK) != config["source_lock_sha256"]:
        fail("Source lock changed since run_config.json was written")
    lock = read_json(SOURCE_LOCK)
    exact_keys(lock, {"sha256"}, "source lock")
    hashes = lock["sha256"]
    if not isinstance(hashes, dict) or set(hashes) != SOURCE_PATHS:
        fail("Source lock path set differs from the frozen protocol")
    for relative, expected in hashes.items():
        valid_hash(expected, relative)
        if sha256_file(REPO / relative) != expected:
            fail(f"Source changed since lock: {relative}")
    for relative, expected in FROZEN_SOURCE_HASHES.items():
        if hashes[relative] != expected:
            fail(f"Predeclared source fingerprint differs: {relative}")
    if config["reference_300_config_sha256"] != hashes["experiments_iclr/ogbn_arxiv_300_results/run_config.json"]:
        fail("Run config does not reference locked 300-epoch configuration")
    if config["reference_300_dataset_manifest_sha256"] != hashes["experiments_iclr/ogbn_arxiv_300_results/dataset_manifest.json"]:
        fail("Run config does not reference locked OGB manifest")


def verify_config(root: Path) -> dict[str, Any]:
    config = read_json(root / "run_config.json")
    exact_keys(config, CONFIG_KEYS, "run configuration")
    if config["protocol"] != "ogbn_arxiv_untied_v1":
        fail("Unknown protocol version")
    if (config["seeds"] != list(SEEDS)
            or not all(type(seed) is int for seed in config["seeds"])
            or config["arms"] != list(ARMS)):
        fail("Run configuration has wrong seeds or arms")
    if integer(config["members"], "members") != MEMBERS:
        fail("Run configuration does not use four members")
    if config["configuration"] != EXPECTED_CONFIGURATION:
        fail("Run configuration differs from frozen 300-epoch design")
    if config["optimizer"] != {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0}:
        fail("Optimizer differs from frozen design")
    if config["train_objective"] != "mean_member_ce":
        fail("Training objective differs from frozen design")
    if config["selection_rule"] != "pooled_validation_accuracy_then_ce_then_earliest":
        fail("Checkpoint selection rule differs from frozen design")
    if config["torch_version"] != torch.__version__:
        fail("PyTorch version differs from run configuration")
    import torch_geometric
    if config["torch_geometric_version"] != torch_geometric.__version__:
        fail("PyG version differs from run configuration")
    verify_sources(config)
    return config


def verify_dataset_manifest(root: Path, data_root: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], Any, torch.Tensor]:
    if sha256_file(root / "dataset_manifest.json") != sha256_file(REFERENCE_DATA):
        fail("Result dataset manifest differs byte-for-byte from frozen 300-epoch manifest")
    manifest = read_json(root / "dataset_manifest.json")
    exact_keys(manifest, set(EXPECTED_DATA) | {"ogb_version", "fingerprints_sha256"},
               "dataset manifest")
    for key, expected in EXPECTED_DATA.items():
        if manifest[key] != expected:
            fail(f"Dataset manifest differs at {key}")
    fingerprints = manifest["fingerprints_sha256"]
    fingerprint_names = {"x", "raw_edge_index", "y", "train_index", "valid_index", "test_index"}
    if not isinstance(fingerprints, dict) or set(fingerprints) != fingerprint_names:
        fail("Dataset fingerprint set differs from protocol")
    for name, digest in fingerprints.items():
        valid_hash(digest, name)
    checked_dir(data_root)
    dataset_dir = checked_dir(data_root / "ogbn_arxiv")
    for relative in REQUIRED_DATA_FILES:
        checked_file(dataset_dir / relative)
    import ogb
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.utils import to_undirected
    if manifest["ogb_version"] != getattr(ogb, "__version__", "unknown"):
        fail("OGB version differs from dataset manifest")
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(data_root))
    data = dataset[0]
    split = dataset.get_idx_split()
    if set(split) != {"train", "valid", "test"}:
        fail("Official OGB split keys differ")
    indices = {name: split[name].view(-1).long().cpu().numpy()
               for name in ("train", "valid", "test")}
    node_count = int(data.num_nodes)
    joined = np.concatenate(tuple(indices.values()))
    if (len(joined) != node_count or len(np.unique(joined)) != node_count
            or joined.min() < 0 or joined.max() >= node_count):
        fail("Official OGB split is not a partition")
    raw_edges = data.edge_index.cpu()
    undirected_edges = to_undirected(raw_edges, num_nodes=node_count)
    observed = {
        "dataset": "ogbn-arxiv", "dataset_package": "ogb",
        "ogb_version": getattr(ogb, "__version__", "unknown"),
        "split_source": EXPECTED_DATA["split_source"],
        "graph_processing": EXPECTED_DATA["graph_processing"],
        "num_nodes": node_count, "num_features": int(data.x.size(1)),
        "num_classes": int(dataset.num_classes),
        "raw_directed_edges": int(raw_edges.size(1)),
        "training_undirected_edges": int(undirected_edges.size(1)),
        "split_sizes": {name: int(len(indices[name])) for name in ("train", "valid", "test")},
        "fingerprints_sha256": {
            "x": tensor_fingerprint(data.x), "raw_edge_index": tensor_fingerprint(raw_edges),
            "y": tensor_fingerprint(data.y),
            **{name + "_index": tensor_fingerprint(indices[name]) for name in indices},
        },
    }
    if observed != manifest:
        fail("Local OGB tensors or official split differ from frozen manifest")
    labels = data.y.view(-1).long().cpu().numpy()
    if labels.shape != (node_count,) or labels.min() < 0 or labels.max() >= manifest["num_classes"]:
        fail("Invalid local OGB labels")
    graph = SimpleNamespace(edge_index=undirected_edges)
    return indices, labels, manifest, graph, data.x.float().cpu()


def verify_initialization(path: Path, seed: int) -> dict[str, Any]:
    audit = read_json(path)
    exact_keys(audit, INIT_KEYS, "initialization audit")
    if integer(audit["seed"], "initialization seed") != seed:
        fail(f"Initialization seed differs from path: {path}")
    scalar_hashes = (
        "tied_initial_state_sha256", "untied_initial_state_sha256",
        "tied_stack_sha256", "rng_after_construction_sha256",
        "tied_rng_start_sha256", "untied_rng_start_sha256",
    )
    for name in scalar_hashes:
        valid_hash(audit[name], name)
    stacks = audit["untied_stack_sha256"]
    if not isinstance(stacks, list) or len(stacks) != MEMBERS:
        fail(f"Initialization audit has wrong stack count: {path}")
    for member, digest in enumerate(stacks):
        valid_hash(digest, f"untied_stack_sha256[{member}]")
        if digest != audit["tied_stack_sha256"]:
            fail(f"Untied propagation stack {member} differs initially from tied stack")
    shared = audit["shared_nonprop_sha256"]
    if not isinstance(shared, dict) or set(shared) != set(ARMS):
        fail(f"Initialization audit has wrong nonprop hash map: {path}")
    for arm in ARMS:
        valid_hash(shared[arm], f"shared_nonprop_sha256[{arm}]")
    if shared["tied"] != shared["untied_propagation"]:
        fail(f"Initial projector/factor/output values differ between arms: {path}")
    diffs = audit["synthetic_member_max_abs_diff"]
    if not isinstance(diffs, list) or len(diffs) != MEMBERS:
        fail(f"Initialization audit has wrong synthetic-logit count: {path}")
    for member, difference in enumerate(diffs):
        json_number(difference, f"synthetic_member_max_abs_diff[{member}]", 0, 1e-7)
    for field in ("all_initial_functions_equal", "all_parameters_equal_to_tied",
                  "untied_stacks_disjoint", "cross_arm_storage_disjoint"):
        if audit[field] is not True:
            fail(f"Initialization equality/isolation assertion failed: {field}")
    counts = audit["parameter_count"]
    if not isinstance(counts, dict) or set(counts) != set(ARMS):
        fail(f"Initialization audit has wrong parameter-count map: {path}")
    for arm in ARMS:
        integer(counts[arm], f"initial {arm} parameter count", 1)
    if counts["untied_propagation"] <= counts["tied"]:
        fail(f"Untied propagation has no extra trainable parameters: {path}")
    if not (audit["rng_after_construction_sha256"]
            == audit["tied_rng_start_sha256"]
            == audit["untied_rng_start_sha256"]):
        fail(f"Arms did not start training from the same recorded RNG state: {path}")
    return audit


def verify_selected(path: Path, seed: int, arm: str,
                    initialization: dict[str, Any], init_hash: str) -> dict[str, Any]:
    selected = read_json(path)
    exact_keys(selected, SELECTED_KEYS, "selected result")
    if (integer(selected["seed"], "selected seed") != seed
            or selected["arm"] != arm or selected["variant"] != arm):
        fail(f"Selected identity differs from canonical path: {path}")
    if selected["validation_selection"] != SELECTED_RULE:
        fail(f"Selected result has wrong validation rule: {path}")
    if integer(selected["epochs_run"], "epochs_run") != EPOCHS:
        fail(f"Selected result is not a complete 300-epoch run: {path}")
    epoch = integer(selected["selected_epoch"], "selected_epoch", 1)
    if epoch > EPOCHS:
        fail(f"Selected epoch exceeds 300: {path}")
    for name in ("valid_accuracy", "test_accuracy"):
        json_number(selected[name], name, 0, 1)
    for name in ("valid_ce", "test_ce", "train_step_seconds_total",
                 "validation_seconds_total", "total_seconds",
                 "mean_train_step_ms", "median_train_step_ms"):
        json_number(selected[name], name, 0)
    for name in ("parameter_count", "trainable_parameter_count"):
        integer(selected[name], name, 1)
    if selected["parameter_count"] != initialization["parameter_count"][arm]:
        fail(f"Selected parameter count differs from initialization audit: {path}")
    peak = selected["peak_allocated_mib"]
    if peak is not None:
        json_number(peak, "peak_allocated_mib", 0)
    device = selected["device"]
    if not isinstance(device, str) or not (device == "cpu" or device.startswith("cuda")):
        fail(f"Invalid recorded device: {path}")
    if (device == "cpu" and peak is not None) or (device.startswith("cuda") and peak is None):
        fail(f"Peak memory is inconsistent with recorded device: {path}")
    if not isinstance(selected["finished_utc"], str):
        fail(f"Invalid finish timestamp: {path}")
    try:
        finished = datetime.fromisoformat(selected["finished_utc"])
    except ValueError as exc:
        raise ValueError(f"Invalid finish timestamp: {path}") from exc
    if finished.tzinfo is None:
        fail(f"Finish timestamp has no timezone: {path}")
    valid_hash(selected["checkpoint_sha256"], "checkpoint_sha256")
    valid_hash(selected["initialization_sha256"], "initialization_sha256")
    valid_hash(selected["rng_start_sha256"], "rng_start_sha256")
    if selected["initialization_sha256"] != init_hash:
        fail(f"Selected result refers to another initialization audit: {path}")
    if selected["rng_start_sha256"] != initialization[f"{arm}_rng_start_sha256"]:
        fail(f"Selected RNG does not match paired start state: {path}")
    return selected


def better_validation(acc: float, ce: float, best_acc: float, best_ce: float) -> bool:
    return acc > best_acc + 1e-12 or (
        abs(acc - best_acc) <= 1e-12 and ce < best_ce - 1e-12
    )


def read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with checked_file(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != columns:
            fail(f"Unexpected CSV header: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        fail(f"Truncated or extended CSV row: {path}")
    return rows


def verify_epochs(path: Path, selected: dict[str, Any]) -> None:
    rows = read_csv(path, EPOCH_COLUMNS)
    if len(rows) != EPOCHS:
        fail(f"Epoch history must contain exactly 300 rows: {path}")
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    train_times: list[float] = []
    validation_times: list[float] = []
    previous_elapsed = 0.0
    for epoch, row in enumerate(rows, 1):
        if csv_integer(row["epoch"], "epoch", 1) != epoch:
            fail(f"Epoch history is not contiguous: {path}")
        finite_number(row["train_mean_member_ce"], "train_mean_member_ce", 0)
        train_times.append(finite_number(row["train_step_seconds"], "train_step_seconds", 0))
        validation_times.append(finite_number(row["validation_seconds"], "validation_seconds", 0))
        elapsed = finite_number(row["elapsed_seconds"], "elapsed_seconds", 0)
        if elapsed < previous_elapsed:
            fail(f"Elapsed epoch time moved backwards: {path}")
        previous_elapsed = elapsed
        acc = finite_number(row["valid_pooled_accuracy"], "valid_pooled_accuracy", 0, 1)
        ce = finite_number(row["valid_pooled_ce"], "valid_pooled_ce", 0)
        improved = better_validation(acc, ce, best_acc, best_ce)
        if improved:
            best_acc, best_ce, best_epoch = acc, ce, epoch
        flag = csv_integer(row["is_selected_so_far"], "is_selected_so_far", 0)
        if flag not in (0, 1) or flag != int(improved):
            fail(f"Selected-so-far flag differs from validation replay at epoch {epoch}: {path}")
    if best_epoch != selected["selected_epoch"]:
        fail(f"Selected epoch differs from validation-only replay: {path}")
    close(best_acc, selected["valid_accuracy"], "selected validation accuracy", 1e-7)
    close(best_ce, selected["valid_ce"], "selected validation CE", 1e-6)
    close(sum(train_times), selected["train_step_seconds_total"], "training time total", 1e-3)
    close(sum(validation_times), selected["validation_seconds_total"], "validation time total", 1e-3)
    close(1000 * statistics.mean(train_times), selected["mean_train_step_ms"], "mean train step ms", 1e-3)
    close(1000 * statistics.median(train_times), selected["median_train_step_ms"], "median train step ms", 1e-3)
    if finite_number(selected["total_seconds"], "total_seconds", 0) + 1e-6 < previous_elapsed:
        fail(f"Selected total time precedes last epoch: {path}")


def verify_checkpoint(path: Path, selected: dict[str, Any], config: dict[str, Any],
                      dataset_manifest: dict[str, Any]) -> torch.nn.Module:
    if sha256_file(path) != selected["checkpoint_sha256"]:
        fail(f"Checkpoint SHA256 differs from selected.json: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"state_dict", "epoch", "arm", "seed"}:
        fail(f"Checkpoint payload differs from protocol: {path}")
    if (payload["epoch"] != selected["selected_epoch"]
            or payload["arm"] != selected["arm"]
            or payload["seed"] != selected["seed"]):
        fail(f"Checkpoint identity differs from selected result: {path}")
    state = payload["state_dict"]
    if not isinstance(state, dict) or not state:
        fail(f"Checkpoint lacks state_dict: {path}")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            fail(f"Invalid checkpoint tensor: {name}")
        if not bool(torch.isfinite(tensor).all()):
            fail(f"Nonfinite checkpoint tensor: {name}")
    sys.path.insert(0, str(REPO))
    from experiments_iclr import ogbn_arxiv_pilot as pilot
    from experiments_iclr import ogbn_arxiv_untied_control as control
    cfg = pilot.Configuration(**config["configuration"])
    model = control.build_arm_model(
        selected["arm"], cfg, dataset_manifest["num_features"],
        dataset_manifest["num_classes"], torch.device("cpu"),
    )
    model.load_state_dict(state, strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    if parameters != selected["parameter_count"] or trainable != selected["trainable_parameter_count"]:
        fail(f"Checkpoint model parameter count differs from selected result: {path}")
    return model


@torch.no_grad()
def replay_member_logits(model: torch.nn.Module, graph: Any, x: torch.Tensor,
                         manifest: dict[str, Any]) -> list[torch.Tensor]:
    model.eval()
    output = []
    for member in range(MEMBERS):
        full = model(graph, x, tabm_seed=member)
        if (full.shape != (manifest["num_nodes"], manifest["num_classes"])
                or not bool(torch.isfinite(full).all())):
            fail(f"Checkpoint replay produced invalid full-graph logits for member {member}")
        output.append(full.detach().cpu())
    return output


def score_split(archive: Any, split: str, selected: dict[str, Any],
                manifest: dict[str, Any], indices: dict[str, np.ndarray],
                official_labels: np.ndarray, replayed: list[torch.Tensor],
                path: Path) -> dict[str, float]:
    logits = archive[f"{split}_member_logits"]
    labels = archive[f"{split}_labels"]
    saved_indices = archive[f"{split}_indices"]
    count = manifest["split_sizes"][split]
    classes = manifest["num_classes"]
    if logits.shape != (MEMBERS, count, classes) or logits.dtype != np.float32:
        fail(f"Invalid {split} member logits shape or dtype: {path}")
    if labels.shape != (count,) or labels.dtype != np.int64:
        fail(f"Invalid {split} labels shape or dtype: {path}")
    if saved_indices.shape != (count,) or saved_indices.dtype != np.int64:
        fail(f"Invalid {split} indices shape or dtype: {path}")
    if not np.isfinite(logits).all():
        fail(f"Nonfinite {split} member logits: {path}")
    if (not np.array_equal(saved_indices, indices[split])
            or tensor_fingerprint(saved_indices) != manifest["fingerprints_sha256"][split + "_index"]):
        fail(f"{split} indices differ from official OGB split order: {path}")
    if not np.array_equal(labels, official_labels[saved_indices]):
        fail(f"{split} labels differ from official OGB labels: {path}")
    index_tensor = torch.from_numpy(np.ascontiguousarray(indices[split]))
    for member, full in enumerate(replayed):
        regenerated = full.index_select(0, index_tensor).numpy()
        if not np.allclose(regenerated, logits[member], atol=5e-5, rtol=1e-5):
            max_error = float(np.max(np.abs(regenerated - logits[member])))
            fail(f"{split} member {member} saved logits differ from checkpoint replay "
                 f"(max absolute error {max_error:.6g}): {path}")
    pooled = torch.from_numpy(np.ascontiguousarray(logits)).mean(dim=0)
    target = torch.from_numpy(np.ascontiguousarray(labels))
    accuracy = float((pooled.argmax(dim=-1) == target).float().mean().item())
    ce = float(F.cross_entropy(pooled, target).item())
    close(accuracy, selected[f"{split}_accuracy"], f"{split} pooled accuracy", 1e-7)
    close(ce, selected[f"{split}_ce"], f"{split} pooled CE", 2e-5)
    return {"accuracy": accuracy, "ce": ce}


def verify_predictions(path: Path, selected: dict[str, Any],
                       manifest: dict[str, Any], indices: dict[str, np.ndarray],
                       official_labels: np.ndarray, model: torch.nn.Module,
                       graph: Any, x: torch.Tensor) -> dict[str, dict[str, float]]:
    replayed = replay_member_logits(model, graph, x, manifest)
    with np.load(checked_file(path), allow_pickle=False) as archive:
        if set(archive.files) != NPZ_KEYS or len(archive.files) != len(NPZ_KEYS):
            fail(f"Prediction archive keys differ from protocol: {path}")
        valid = score_split(archive, "valid", selected, manifest, indices,
                            official_labels, replayed, path)
        # Test arrays and labels are accessed only after selection and restored
        # validation metrics have passed.
        test = score_split(archive, "test", selected, manifest, indices,
                           official_labels, replayed, path)
    return {"valid": valid, "test": test}


def verify_summary(root: Path, results: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    rows = read_csv(root / "summary.csv", SUMMARY_COLUMNS)
    if len(rows) != len(SEEDS):
        fail("Summary CSV does not have exactly three seed rows")
    differences: list[float] = []
    for seed, row in zip(SEEDS, rows):
        if csv_integer(row["seed"], "summary seed") != seed:
            fail("Summary CSV seed order differs from protocol")
        tied = results[seed]["tied"]
        untied = results[seed]["untied_propagation"]
        tied_accuracy = finite_number(tied["test_accuracy"], "tied test accuracy", 0, 1)
        untied_accuracy = finite_number(untied["test_accuracy"], "untied test accuracy", 0, 1)
        difference = tied_accuracy - untied_accuracy
        close(tied_accuracy, row["tied_test_accuracy"], "summary tied accuracy", 1e-9)
        close(untied_accuracy, row["untied_test_accuracy"], "summary untied accuracy", 1e-9)
        close(difference, row["tied_minus_untied"], "summary difference", 1e-9)
        if csv_integer(row["tied_selected_epoch"], "tied selected epoch") != tied["selected_epoch"]:
            fail("Summary CSV tied epoch differs from selected result")
        if csv_integer(row["untied_selected_epoch"], "untied selected epoch") != untied["selected_epoch"]:
            fail("Summary CSV untied epoch differs from selected result")
        differences.append(difference)
    summary = read_json(root / "summary.json")
    exact_keys(summary, {"contrast", "seeds", "differences", "mean", "sample_sd"},
               "summary JSON")
    if (summary["contrast"] != "tied_minus_untied_propagation"
            or summary["seeds"] != list(SEEDS)
            or not all(type(seed) is int for seed in summary["seeds"])):
        fail("Summary JSON identity differs from protocol")
    saved_differences = summary["differences"]
    if not isinstance(saved_differences, list) or len(saved_differences) != len(SEEDS):
        fail("Summary JSON differences do not cover three seeds")
    for seed, (actual, recorded) in enumerate(zip(differences, saved_differences)):
        json_number(recorded, f"summary JSON difference seed {seed}")
        close(actual, recorded, f"summary JSON difference seed {seed}", 1e-9)
    json_number(summary["mean"], "summary mean")
    json_number(summary["sample_sd"], "summary sample SD", 0)
    close(statistics.mean(differences), summary["mean"], "summary mean", 1e-9)
    close(statistics.stdev(differences), summary["sample_sd"], "summary sample SD", 1e-9)
    return summary


def verify_tree(root: Path, data_root: Path) -> dict[str, Any]:
    verify_tree_layout(root)
    verify_artifact_manifest(root)
    config = verify_config(root)
    indices, labels, dataset_manifest, graph, x = verify_dataset_manifest(root, data_root)
    results: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        seed_dir = root / f"seed_{seed}"
        init_path = seed_dir / "initialization.json"
        audit = verify_initialization(init_path, seed)
        init_hash = sha256_file(init_path)
        results[seed] = {}
        for arm in ARMS:
            arm_dir = seed_dir / arm
            selected = verify_selected(arm_dir / "selected.json", seed, arm, audit, init_hash)
            verify_epochs(arm_dir / "epochs.csv", selected)
            model = verify_checkpoint(arm_dir / "selected_checkpoint.pt", selected,
                                      config, dataset_manifest)
            verify_predictions(arm_dir / "selected_predictions.npz", selected,
                               dataset_manifest, indices, labels, model, graph, x)
            results[seed][arm] = selected
    summary = verify_summary(root, results)
    return {
        "status": "complete", "verified_arms": len(SEEDS) * len(ARMS),
        "verified_seed_pairs": len(SEEDS), "summary": summary,
        "artifact_manifest_sha256": sha256_file(root / "artifact_manifest.json"),
        "source_lock_sha256": config["source_lock_sha256"],
        "dataset_manifest_sha256": sha256_file(root / "dataset_manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--data-root", default="experiments_iclr/data")
    args = parser.parse_args()
    report = verify_tree(repo_path(args.result_root), repo_path(args.data_root))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as exc:
        print(f"ogbn-arxiv untied verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
