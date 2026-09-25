"""Read-only CPU audit of the fixed nine-run ogbn-arxiv pilot.

Run from the repository root after (or during) the queued pilot. A partial
result set passes by default, but every present result must be intact.
--require-complete requires all three variants and all three seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("base", "ens", "gnnm")
SEEDS = (0, 1, 2)
PAIRS = tuple(itertools.product(SEEDS, VARIANTS))
SELECTION = "max pooled accuracy, then min pooled CE, then earliest epoch"
EPOCH_COLUMNS = (
    "epoch", "train_mean_member_ce", "valid_pooled_accuracy",
    "valid_pooled_ce", "train_step_seconds", "validation_seconds",
    "elapsed_seconds", "is_selected_so_far",
)
NPZ_KEYS = {
    "valid_member_logits", "valid_labels", "valid_indices",
    "test_member_logits", "test_labels", "test_indices",
}
SELECTED_KEYS = {
    "seed", "variant", "selected_epoch", "validation_selection",
    "valid_accuracy", "valid_ce", "test_accuracy", "test_ce",
    "parameter_count", "trainable_parameter_count", "epochs_run",
    "train_step_seconds_total", "validation_seconds_total", "total_seconds",
    "mean_train_step_ms", "median_train_step_ms", "peak_allocated_mib",
    "checkpoint_sha256", "finished_utc", "device",
}
SOURCE_FILES = {
    "ogbn_arxiv_pilot.py": REPO / "experiments_iclr" / "ogbn_arxiv_pilot.py",
    "models.py": REPO / "models.py",
}
EXPECTED_CONFIG = {
    "backbone": "SAGE", "layers": 2, "hidden_dim": 128, "dropout": 0.2,
    "lr": 0.001, "weight_decay": 0.0, "max_epochs": 100,
    "min_epochs": 20, "patience": 20, "eval_every": 1,
}
# The repeat is a fixed 300-epoch sensitivity run, not an extension of the
# original checkpoints. Requiring min_epochs=300 disables early stopping while
# retaining the same validation and pooled-checkpoint rule on every epoch.
EXPECTED_CONFIG_300 = {**EXPECTED_CONFIG, "max_epochs": 300, "min_epochs": 300}
PROFILES = {
    "pilot-100": (EXPECTED_CONFIG, "experiments_iclr/ogbn_arxiv_results"),
    "fixed-300": (EXPECTED_CONFIG_300, "experiments_iclr/ogbn_arxiv_300_results"),
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
    "RELEASE_v1.txt",
    "processed/geometric_data_processed.pt",
    "raw/edge.csv.gz", "raw/node-feat.csv.gz", "raw/node-label.csv.gz",
    "split/time/train.csv.gz", "split/time/valid.csv.gz",
    "split/time/test.csv.gz",
)


def fail(message: str) -> None:
    raise ValueError(message)


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    resolved = path.resolve()
    if resolved != REPO and REPO not in resolved.parents:
        fail(f"Path leaves repository: {path}")
    return resolved


def checked_file(root: Path, path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        fail(f"Missing, empty, or linked artifact: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        fail(f"Artifact path leaves result root: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_fingerprint(value: torch.Tensor | np.ndarray) -> str:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = value
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def read_json(root: Path, path: Path) -> dict[str, Any]:
    checked_file(root, path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        fail(f"Expected a JSON object: {path}")
    return value


def finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        fail(f"Invalid Boolean in {name}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number in {name}: {value!r}") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        fail(f"Nonfinite or out-of-range {name}: {value!r}")
    return result


def integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        fail(f"Invalid Boolean in {name}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer in {name}: {value!r}") from exc
    if str(result) != str(value) or (minimum is not None and result < minimum):
        fail(f"Invalid integer in {name}: {value!r}")
    return result


def close(actual: float, expected: Any, name: str, tolerance: float) -> None:
    recorded = finite_number(expected, name)
    if abs(actual - recorded) > tolerance:
        fail(f"{name} mismatch: recomputed {actual:.9g}, recorded {recorded:.9g}")


def verify_config(root: Path, expected_config: dict[str, Any] = EXPECTED_CONFIG) -> dict[str, Any]:
    config = read_json(root, root / "run_config.json")
    required = {
        "configuration", "seeds", "variants", "members_for_ens_and_gnnm",
        "selection_rule", "checkpoint_frequency", "train_objective",
        "torch_version", "torch_geometric_version", "source_fingerprints_sha256",
    }
    if set(config) != required:
        fail("run_config.json fields differ from the pilot protocol")
    if config["configuration"] != expected_config:
        fail("Run configuration differs from the selected fixed OGB protocol")
    if config["seeds"] != list(SEEDS) or config["variants"] != list(VARIANTS):
        fail("Run config does not declare all 3 seeds and 3 variants")
    if config["members_for_ens_and_gnnm"] != 4:
        fail("Run config member count differs from four")
    if config["selection_rule"] != (
        "joint pooled validation accuracy, then pooled CE, then earliest epoch"
    ):
        fail("Run config selection rule changed")
    if config["checkpoint_frequency"] != "every 1 epoch(s) and final epoch":
        fail("Run config checkpoint frequency changed")
    if config["train_objective"] != (
        "mean member CE for GNNM and ENS; ordinary CE for BASE"
    ):
        fail("Run config training objective changed")
    if config["torch_version"] != torch.__version__:
        fail("Current PyTorch version differs from run_config.json")
    import torch_geometric
    if config["torch_geometric_version"] != torch_geometric.__version__:
        fail("Current PyG version differs from run_config.json")
    recorded = config["source_fingerprints_sha256"]
    if not isinstance(recorded, dict) or set(recorded) != set(SOURCE_FILES):
        fail("Run config source fingerprint set is incomplete")
    for name, path in SOURCE_FILES.items():
        if sha256_file(checked_file(REPO, path)) != recorded[name]:
            fail(f"Pilot source changed since run config was written: {name}")
    return config


def verify_manifest_shape(manifest: dict[str, Any]) -> None:
    if set(manifest) != set(EXPECTED_DATA) | {"ogb_version", "fingerprints_sha256"}:
        fail("Dataset manifest fields differ from the pilot protocol")
    for key, expected in EXPECTED_DATA.items():
        if manifest[key] != expected:
            fail(f"Dataset manifest {key} differs from official ogbn-arxiv")
    fingerprints = manifest["fingerprints_sha256"]
    expected_keys = {"x", "raw_edge_index", "y", "train_index", "valid_index", "test_index"}
    if not isinstance(fingerprints, dict) or set(fingerprints) != expected_keys:
        fail("Dataset manifest fingerprint set is incomplete")
    for name, value in fingerprints.items():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            fail(f"Invalid dataset fingerprint: {name}")


def verify_local_dataset(manifest: dict[str, Any], data_root: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Reload only existing repository-local OGB files, with no download path."""
    if not data_root.is_dir() or data_root.is_symlink():
        fail(f"Missing local OGB data root: {data_root}")
    dataset_dir = data_root / "ogbn_arxiv"
    for relative in REQUIRED_DATA_FILES:
        checked_file(data_root, dataset_dir / relative)
    import ogb
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.utils import to_undirected

    if manifest["ogb_version"] != getattr(ogb, "__version__", "unknown"):
        fail("Current OGB version differs from dataset manifest")
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(data_root))
    data = dataset[0]
    split = dataset.get_idx_split()
    if set(split) != {"train", "valid", "test"}:
        fail("Unexpected OGB split keys")
    indices = {name: split[name].view(-1).long().cpu().numpy()
               for name in ("train", "valid", "test")}
    n = int(data.num_nodes)
    joined = np.concatenate(list(indices.values()))
    if (len(joined) != n or len(np.unique(joined)) != n
            or joined.min() < 0 or joined.max() >= n):
        fail("Local OGB split is not a disjoint partition")
    raw_edges = data.edge_index.cpu()
    computed = {
        "dataset": "ogbn-arxiv", "dataset_package": "ogb",
        "ogb_version": getattr(ogb, "__version__", "unknown"),
        "split_source": EXPECTED_DATA["split_source"],
        "graph_processing": EXPECTED_DATA["graph_processing"],
        "num_nodes": n, "num_features": int(data.x.size(1)),
        "num_classes": int(dataset.num_classes),
        "raw_directed_edges": int(raw_edges.size(1)),
        "training_undirected_edges": int(to_undirected(raw_edges, num_nodes=n).size(1)),
        "split_sizes": {name: int(len(indices[name])) for name in ("train", "valid", "test")},
        "fingerprints_sha256": {
            "x": tensor_fingerprint(data.x),
            "raw_edge_index": tensor_fingerprint(raw_edges),
            "y": tensor_fingerprint(data.y),
            **{name + "_index": tensor_fingerprint(indices[name]) for name in indices},
        },
    }
    if computed != manifest:
        fail("Local OGB data or split differs from dataset_manifest.json")
    labels = data.y.view(-1).long().cpu().numpy()
    if labels.shape != (n,) or labels.min() < 0 or labels.max() >= manifest["num_classes"]:
        fail("Invalid local OGB labels")
    return indices, labels


def better_validation(acc: float, ce: float, best_acc: float, best_ce: float) -> bool:
    return acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and ce < best_ce - 1e-12)


def verify_epochs(path: Path, selected: dict[str, Any], config: dict[str, Any]) -> None:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EPOCH_COLUMNS:
            fail(f"Unexpected epochs.csv header: {path}")
        rows = list(reader)
    count = integer(selected.get("epochs_run"), "epochs_run", minimum=1)
    if len(rows) != count or count > config["max_epochs"]:
        fail(f"Epoch row count differs from selected.json: {path}")
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    bad_evaluations = 0
    train_times: list[float] = []
    validation_times: list[float] = []
    previous_elapsed = 0.0
    for epoch, row in enumerate(rows, 1):
        if None in row or any(value is None for value in row.values()):
            fail(f"Truncated or extended epoch row: {path}")
        if integer(row["epoch"], "epoch", minimum=1) != epoch:
            fail(f"Epochs are not contiguous: {path}")
        finite_number(row["train_mean_member_ce"], "train_mean_member_ce", minimum=0)
        train_times.append(finite_number(row["train_step_seconds"], "train_step_seconds", minimum=0))
        validation_times.append(finite_number(row["validation_seconds"], "validation_seconds", minimum=0))
        elapsed = finite_number(row["elapsed_seconds"], "elapsed_seconds", minimum=0)
        if elapsed < previous_elapsed:
            fail(f"Epoch elapsed time moved backward: {path}")
        previous_elapsed = elapsed
        eval_expected = epoch % config["eval_every"] == 0 or epoch == config["max_epochs"]
        observed = row["valid_pooled_accuracy"] != "" and row["valid_pooled_ce"] != ""
        if observed != eval_expected:
            fail(f"Missing or unexpected validation measurement at epoch {epoch}: {path}")
        improved = False
        if observed:
            acc = finite_number(row["valid_pooled_accuracy"], "valid_pooled_accuracy", minimum=0)
            ce = finite_number(row["valid_pooled_ce"], "valid_pooled_ce", minimum=0)
            if acc > 1:
                fail(f"Validation accuracy above one: {path}")
            improved = better_validation(acc, ce, best_acc, best_ce)
            if improved:
                best_acc, best_ce, best_epoch = acc, ce, epoch
                bad_evaluations = 0
            else:
                bad_evaluations += 1
        elif row["valid_pooled_accuracy"] != "" or row["valid_pooled_ce"] != "":
            fail(f"Incomplete validation measurement at epoch {epoch}: {path}")
        if integer(row["is_selected_so_far"], "is_selected_so_far") != int(improved):
            fail(f"Selected flag differs from pooled validation rule at epoch {epoch}: {path}")
        should_stop = epoch >= config["min_epochs"] and bad_evaluations >= config["patience"]
        if should_stop and epoch != count:
            fail(f"Epoch history continued after patience stop: {path}")
    if not best_epoch or best_epoch != integer(selected.get("selected_epoch"), "selected_epoch", minimum=1):
        fail(f"Selected epoch differs from validation-only replay: {path}")
    if count < config["max_epochs"] and not (
        count >= config["min_epochs"] and bad_evaluations >= config["patience"]
    ):
        fail(f"Epoch history stops before max epochs without patience stop: {path}")
    close(best_acc, selected.get("valid_accuracy"), "selected validation accuracy", 1e-7)
    close(best_ce, selected.get("valid_ce"), "selected validation CE", 1e-6)
    close(sum(train_times), selected.get("train_step_seconds_total"), "training time total", 1e-3)
    close(sum(validation_times), selected.get("validation_seconds_total"), "validation time total", 1e-3)
    close(1000 * float(np.mean(train_times)), selected.get("mean_train_step_ms"), "mean train step ms", 1e-3)
    close(1000 * float(np.median(train_times)), selected.get("median_train_step_ms"), "median train step ms", 1e-3)
    if finite_number(selected.get("total_seconds"), "total_seconds", minimum=0) + 1e-6 < previous_elapsed:
        fail(f"Total runtime precedes last epoch: {path}")


def verify_checkpoint(path: Path, selected: dict[str, Any], config: dict[str, Any],
                      manifest: dict[str, Any], *, check_model: bool = True) -> str:
    observed_hash = sha256_file(path)
    if selected.get("checkpoint_sha256") != observed_hash:
        fail(f"Checkpoint SHA256 differs from selected.json: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"state_dict", "epoch", "variant", "seed"}:
        fail(f"Invalid checkpoint payload: {path}")
    for key, field in (("epoch", "selected_epoch"), ("variant", "variant"), ("seed", "seed")):
        if payload[key] != selected[field]:
            fail(f"Checkpoint {key} differs from selected.json: {path}")
    state = payload["state_dict"]
    if not isinstance(state, dict) or not state:
        fail(f"Missing checkpoint state_dict: {path}")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all()):
            fail(f"Invalid checkpoint tensor {name}: {path}")
    if not check_model:
        return observed_hash
    sys.path.insert(0, str(REPO))
    from models import Model, TABMModel
    common = dict(
        model_name=config["backbone"], num_layers=config["layers"],
        input_dim=manifest["num_features"], hidden_dim=config["hidden_dim"],
        output_dim=manifest["num_classes"], hidden_dim_multiplier=1,
        num_heads=8, normalization="LayerNorm", dropout=config["dropout"],
    )
    variant = selected["variant"]
    if variant == "base":
        model = Model(**common)
    elif variant == "ens":
        model = torch.nn.ModuleList([Model(**common) for _ in range(4)])
    else:
        model = TABMModel(**common, tabm_inits=4, device=torch.device("cpu"))
    model.load_state_dict(state, strict=True)
    param_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if selected.get("parameter_count") != param_count or selected.get("trainable_parameter_count") != trainable_count:
        fail(f"Recorded parameter count differs from checkpoint model: {path}")
    return observed_hash


def diagnostics(member_logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    predictions = member_logits.argmax(axis=-1)
    correct = predictions == labels[None, :]
    pooled = torch.from_numpy(np.ascontiguousarray(member_logits)).mean(dim=0).numpy()
    pooled_correct = pooled.argmax(axis=-1) == labels
    counts = correct.sum(axis=0)
    members = member_logits.shape[0]
    pairwise = [float(np.mean(predictions[a] != predictions[b]))
                for a, b in itertools.combinations(range(members), 2)]
    return {
        "member_accuracies": [float(x) for x in correct.mean(axis=1)],
        "mean_member_accuracy": float(correct.mean()),
        "pooled_accuracy": float(pooled_correct.mean()),
        "pool_minus_mean_member_accuracy": float(pooled_correct.mean() - correct.mean()),
        "pairwise_prediction_disagreement": float(np.mean(pairwise)) if pairwise else None,
        "all_members_wrong_fraction": float(np.mean(counts == 0)),
        "all_members_correct_fraction": float(np.mean(counts == members)),
        "pooled_correct_with_no_correct_member_fraction": float(np.mean((counts == 0) & pooled_correct)),
        "correct_member_counts": [int(np.sum(counts == count)) for count in range(members + 1)],
    }


def verify_predictions(path: Path, selected: dict[str, Any], manifest: dict[str, Any],
                       official_indices: dict[str, np.ndarray], official_labels: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != NPZ_KEYS:
            fail(f"Prediction archive keys differ from protocol: {path}")
        arrays = {name: archive[name] for name in NPZ_KEYS}
    output: dict[str, Any] = {}
    for split in ("valid", "test"):
        logits = arrays[f"{split}_member_logits"]
        labels = arrays[f"{split}_labels"]
        indices = arrays[f"{split}_indices"]
        members = 1 if selected["variant"] == "base" else 4
        n = manifest["split_sizes"][split]
        classes = manifest["num_classes"]
        if logits.shape != (members, n, classes) or logits.dtype != np.float32:
            fail(f"Invalid {split} logits shape or dtype: {path}")
        if labels.shape != (n,) or labels.dtype != np.int64:
            fail(f"Invalid {split} labels shape or dtype: {path}")
        if indices.shape != (n,) or indices.dtype != np.int64:
            fail(f"Invalid {split} indices shape or dtype: {path}")
        if not np.isfinite(logits).all():
            fail(f"Nonfinite {split} logits: {path}")
        if (not np.array_equal(indices, official_indices[split])
                or tensor_fingerprint(indices) != manifest["fingerprints_sha256"][split + "_index"]):
            fail(f"{split} indices differ from official split: {path}")
        if not np.array_equal(labels, official_labels[indices]):
            fail(f"{split} labels differ from local official labels: {path}")
        pooled_logits = torch.from_numpy(np.ascontiguousarray(logits)).mean(dim=0)
        target = torch.from_numpy(np.ascontiguousarray(labels))
        accuracy = float((pooled_logits.argmax(dim=-1) == target).float().mean().item())
        ce = float(F.cross_entropy(pooled_logits, target).item())
        close(accuracy, selected.get(f"{split}_accuracy"), f"{split} pooled accuracy", 1e-7)
        close(ce, selected.get(f"{split}_ce"), f"{split} pooled CE", 2e-5)
        stats = diagnostics(logits, labels)
        if abs(stats["pooled_accuracy"] - accuracy) > 1e-7:
            fail(f"Independent {split} pooled accuracy differs: {path}")
        output[split] = {"accuracy": accuracy, "ce": ce, **stats}
    output["sha256"] = sha256_file(path)
    return output


def verify_pair(root: Path, seed: int, variant: str, manifest: dict[str, Any],
                official_indices: dict[str, np.ndarray], official_labels: np.ndarray,
                config: dict[str, Any], *, check_model: bool = True) -> dict[str, Any]:
    directory = root / f"seed_{seed}" / variant
    selected_path = checked_file(root, directory / "selected.json")
    selected = read_json(root, selected_path)
    if set(selected) != SELECTED_KEYS:
        fail(f"Selected result fields differ from pilot schema: {selected_path}")
    if selected.get("seed") != seed or selected.get("variant") != variant:
        fail(f"Selected result identity differs from its canonical path: {selected_path}")
    if selected.get("validation_selection") != SELECTION:
        fail(f"Selected result used an unexpected checkpoint rule: {selected_path}")
    if not isinstance(selected["device"], str) or not selected["device"]:
        fail(f"Invalid selected device: {selected_path}")
    if selected["peak_allocated_mib"] is not None:
        finite_number(selected["peak_allocated_mib"], "peak_allocated_mib", minimum=0)
    if not isinstance(selected["finished_utc"], str):
        fail(f"Invalid finish timestamp: {selected_path}")
    try:
        finished = datetime.fromisoformat(selected["finished_utc"])
    except ValueError as exc:
        raise ValueError(f"Invalid finish timestamp: {selected_path}") from exc
    if finished.tzinfo is None:
        fail(f"Finish timestamp has no timezone: {selected_path}")
    for key in ("parameter_count", "trainable_parameter_count"):
        integer(selected[key], key, minimum=1)
    epoch_path = checked_file(root, directory / "epochs.csv")
    checkpoint_path = checked_file(root, directory / "selected_checkpoint.pt")
    prediction_path = checked_file(root, directory / "selected_predictions.npz")
    verify_epochs(epoch_path, selected, config["configuration"])
    checkpoint_sha = verify_checkpoint(checkpoint_path, selected, config["configuration"],
                                       manifest, check_model=check_model)
    predictions = verify_predictions(prediction_path, selected, manifest, official_indices, official_labels)
    return {
        "seed": seed, "variant": variant, "selected_epoch": selected["selected_epoch"],
        "valid": predictions["valid"], "test": predictions["test"],
        "artifact_sha256": {
            "selected.json": sha256_file(selected_path),
            "epochs.csv": sha256_file(epoch_path),
            "selected_checkpoint.pt": checkpoint_sha,
            "selected_predictions.npz": predictions["sha256"],
        },
    }


def result_pairs(root: Path) -> tuple[list[tuple[int, str]], list[str]]:
    expected_paths = {root / f"seed_{seed}" / variant / "selected.json": (seed, variant)
                      for seed, variant in PAIRS}
    present: list[tuple[int, str]] = []
    partial: list[str] = []
    for selected_path in root.rglob("selected.json"):
        if selected_path not in expected_paths:
            fail(f"Unexpected selected.json outside the nine canonical result paths: {selected_path}")
    for path, pair in expected_paths.items():
        if path.exists() or path.is_symlink():
            present.append(pair)
        elif path.parent.exists() and any(path.parent.iterdir()):
            partial.append(str(path.parent.relative_to(root)))
    return present, partial


def verify_tree(root: Path, data_root: Path,
                expected_config: dict[str, Any] = EXPECTED_CONFIG) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        fail(f"Missing or linked result root: {root}")
    config = verify_config(root, expected_config)
    manifest = read_json(root, root / "dataset_manifest.json")
    verify_manifest_shape(manifest)
    indices, labels = verify_local_dataset(manifest, data_root)
    present, partial = result_pairs(root)
    if partial:
        fail("Result directories contain artifacts without selected.json: " + ", ".join(partial))
    results = [verify_pair(root, seed, variant, manifest, indices, labels, config)
               for seed, variant in present]
    missing = [{"seed": seed, "variant": variant} for seed, variant in PAIRS
               if (seed, variant) not in present]
    return {
        "status": "complete" if not missing else "incomplete",
        "verified_runs": len(results), "required_runs": len(PAIRS),
        "missing": missing, "results": results,
        "run_config_sha256": sha256_file(root / "run_config.json"),
        "dataset_manifest_sha256": sha256_file(root / "dataset_manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="pilot-100")
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--data-root", default="experiments_iclr/data")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    expected_config, default_root = PROFILES[args.profile]
    result_root = args.result_root if args.result_root is not None else default_root
    report = verify_tree(repo_path(result_root), repo_path(args.data_root),
                         expected_config)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_complete and report["missing"]:
        print(f"Incomplete pilot: {len(report['missing'])} of 9 results are absent", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"ogbn-arxiv verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
