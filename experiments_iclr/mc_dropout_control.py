"""Inference-only MC Dropout comparison using the selected Roman Empire BASE runs.

The protocol is fixed in this source before examining MC Dropout test results:
official masks 0--4, SAGE depth 5 and width 512, dropout probability 0.2,
exactly four stochastic full-graph forward passes per selected BASE checkpoint,
and arithmetic averaging of raw logits before argmax. No parameters are trained,
no checkpoint is reselected, and no MC Dropout hyperparameter is tuned.

Run from the repository root after the SAGE training queue has finished:
  .venv/bin/python experiments_iclr/mc_dropout_control.py --device cuda:0
The CPU-only synthetic check is:
  .venv/bin/python experiments_iclr/mc_dropout_control.py --smoke
All run outputs go to experiments_iclr/mc_dropout_results.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "mc_dropout_results"
SOURCE_CSV = HERE / "results" / "projector_controls.csv"
DATA_FILE = ROOT / "data" / "roman_empire.npz"
SPLITS = tuple(range(5))
PASSES = 4
SEED_BASE = 750_000
SEED_STRIDE = 1_000
MODEL_NAME = "SAGE"
DEPTH = 5
WIDTH = 512
LR = 3e-5
DROPOUT_P = 0.2
ACC_TOL = 1e-6
LOSS_TOL = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checked_checkpoint(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=True)
    if ROOT not in path.parents or not path.is_file():
        raise RuntimeError(f"Checkpoint is not a file inside the repository: {raw}")
    return path


def source_rows() -> dict[int, dict[str, str]]:
    with SOURCE_CSV.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["dataset"] == "roman-empire"
                and row["model"] == MODEL_NAME and row["variant"] == "base"]
    selected = {}
    for row in rows:
        split = int(row["split"])
        if split in selected:
            raise RuntimeError(f"Duplicate BASE checkpoint row for split {split}")
        selected[split] = row
    if tuple(sorted(selected)) != SPLITS:
        raise RuntimeError(f"Expected exactly BASE splits {SPLITS}, found {sorted(selected)}")
    for split, row in selected.items():
        expected = {
            "seed": split, "num_layers": DEPTH, "hidden_dim": WIDTH,
            "m": 1, "num_steps": 5000,
        }
        for field, value in expected.items():
            if int(row[field]) != value:
                raise RuntimeError(f"BASE split {split} has {field}={row[field]}")
        if not math.isclose(float(row["lr"]), LR, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"BASE split {split} has a different learning rate")
        if not 1 <= int(row["best_step"]) <= 5000:
            raise RuntimeError(f"BASE split {split} has an invalid selected step")
        for field in ("val_metric", "test_metric", "test_acc", "test_loss"):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"BASE split {split} has nonfinite {field}")
        checked_checkpoint(row["checkpoint"])
    return selected


def activate_only_dropout(model: nn.Module, num_layers: int) -> None:
    """Keep all modules in eval mode except their existing dropout layers."""
    model.eval()
    dropout_layers = [module for module in model.modules()
                      if isinstance(module, nn.modules.dropout._DropoutNd)]
    if len(dropout_layers) != 1 + 2 * num_layers:
        raise RuntimeError(f"Unexpected number of dropout layers: {len(dropout_layers)}")
    if any(not math.isclose(layer.p, DROPOUT_P) for layer in dropout_layers):
        raise RuntimeError("Checkpoint architecture has a different dropout rate")
    for layer in dropout_layers:
        layer.train()
    if any(module.training for module in model.modules()
           if not isinstance(module, nn.modules.dropout._DropoutNd)):
        raise RuntimeError("A non-dropout module entered training mode")


def stochastic_logits(model: nn.Module, graph, split: int,
                      device: torch.device) -> torch.Tensor:
    """Run the fixed four passes and return [4, num_nodes, num_classes]."""
    if device.type == "cuda" and device.index is None:
        raise ValueError("Use an explicit CUDA index, for example cuda:0")
    cuda_devices = [device.index] if device.type == "cuda" else []
    outputs = []
    for pass_index in range(PASSES):
        seed = SEED_BASE + SEED_STRIDE * split + pass_index
        # fork_rng saves CPU and the selected CUDA generator. Seed exactly
        # those generators so every state changed here is restored afterward.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            outputs.append(model(graph, graph.x))
    return torch.stack(outputs, dim=0)


def check_close(label: str, actual: float, archived: str, atol: float) -> None:
    reference = float(archived)
    if not math.isfinite(actual) or abs(actual - reference) > atol:
        raise RuntimeError(f"Selected BASE checkpoint does not reproduce {label}: "
                           f"recomputed={actual}, CSV={reference}, tolerance={atol}")


def atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
               + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending_",
                                     suffix=".json", delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending_",
                                     suffix=".npz", delete=False) as stream:
        temp = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def runtime_manifest(device: torch.device) -> dict[str, object]:
    """Record the environment that turns fixed seeds into actual predictions."""
    from importlib.metadata import version

    runtime: dict[str, object] = {
        "device": str(device),
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "torch_geometric_version": version("torch-geometric"),
        "dgl_version": version("dgl"),
        "numpy_version": np.__version__,
        "scikit_learn_version": version("scikit-learn"),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        runtime["cuda_device_name"] = torch.cuda.get_device_name(device)
        runtime["cuda_device_capability"] = list(torch.cuda.get_device_capability(device))
    return runtime


@contextmanager
def output_lock(directory: Path):
    """Allow only one writer for this result directory."""
    if not directory.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("MC Dropout output directory escaped the repository")
    with (directory / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another MC Dropout run holds the output lock") from exc
        yield


def input_manifest(row: dict[str, str], checkpoint: Path,
                   data_hash: str, runtime: dict[str, object]) -> dict[str, object]:
    return {
        "row": row,
        "runtime": runtime,
        "checkpoint_sha256": sha256(checkpoint),
        "data_sha256": data_hash,
        "script_sha256": sha256(Path(__file__)),
        "models_sha256": sha256(ROOT / "models.py"),
        "datasets_sha256": sha256(ROOT / "datasets.py"),
        "control_runner_sha256": sha256(HERE / "projector_controls.py"),
        "protocol": {
            "dataset": "roman-empire", "model": MODEL_NAME,
            "splits": list(SPLITS), "passes": PASSES,
            "seed_formula": "750000 + 1000 * split + pass_index (pass_index=0,1,2,3)",
            "pooling": "arithmetic mean of raw logits, then argmax",
            "dropout_probability": DROPOUT_P,
            "training": "none; validation-selected BASE state_dict restored",
            "inference_mode": "all non-dropout modules eval; existing dropout modules train",
        },
    }


def existing_record(split: int, fingerprint: str) -> dict | None:
    path = OUT / f"split{split}.json"
    prediction = OUT / f"split{split}_predictions.npz"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("input_fingerprint") != fingerprint:
        raise RuntimeError(f"Split {split} output has a different input fingerprint")
    if not prediction.is_file() or sha256(prediction) != record.get("prediction_sha256"):
        raise RuntimeError(f"Split {split} prediction archive is absent or changed")
    return record


def run_real(device: torch.device) -> None:
    if device.type != "cuda" or device.index is None:
        raise RuntimeError("Full-graph run requires explicit --device cuda:N; --smoke uses CPU")
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device is unavailable: {device}")
    if not DATA_FILE.is_file():
        raise RuntimeError(f"Dataset is absent: {DATA_FILE}")
    rows = source_rows()
    data_hash = sha256(DATA_FILE)
    if not OUT.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("MC Dropout output directory escaped the repository")
    OUT.mkdir(parents=True, exist_ok=True)
    with output_lock(OUT):
        run_real_locked(device, rows, data_hash)


def run_real_locked(device: torch.device, rows: dict[int, dict[str, str]],
                    data_hash: str) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE))
    from datasets import compute_metrics, load_dataset
    from projector_controls import make_model, test_analysis

    graph, train_masks, val_masks, test_masks, _, output_dim, is_binary = load_dataset(
        "roman-empire", add_self_loops=True, device=device, data_dir=str(ROOT / "data"))
    if is_binary or output_dim <= 2 or train_masks.shape[1] != 10:
        raise RuntimeError("Unexpected Roman Empire dataset or official masks")
    if any(mask.shape != train_masks.shape for mask in (val_masks, test_masks)):
        raise RuntimeError("Official mask shapes differ")
    runtime = runtime_manifest(device)
    records = []
    for split in SPLITS:
        row = rows[split]
        checkpoint = checked_checkpoint(row["checkpoint"])
        manifest = input_manifest(row, checkpoint, data_hash, runtime)
        fingerprint = stable_hash(manifest)
        previous = existing_record(split, fingerprint)
        if previous is not None:
            records.append(previous)
            print(f"SKIP split {split}: verified existing result", flush=True)
            continue

        args = SimpleNamespace(model=MODEL_NAME, num_layers=DEPTH,
                               hidden_dim=WIDTH, m=1)
        model = make_model(args, "base", graph.x.size(1), output_dim, device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if actual_params != int(row["num_params"]):
            raise RuntimeError(f"BASE split {split} parameter count differs")
        model.eval()
        graph.train_mask = train_masks[:, split].to(device)
        graph.val_mask = val_masks[:, split].to(device)
        graph.test_mask = test_masks[:, split].to(device)
        if (graph.train_mask & graph.val_mask).any() or (graph.train_mask & graph.test_mask).any() or (graph.val_mask & graph.test_mask).any():
            raise RuntimeError(f"Official split {split} masks overlap")
        if int(graph.test_mask.sum()) != int(row["n_test"]):
            raise RuntimeError(f"BASE split {split} test-mask count differs")

        # The fixed checkpoint must reproduce both the archived validation
        # selection metric and its original deterministic test measurement.
        with torch.inference_mode():
            deterministic = model(graph, graph.x)
            val_base = compute_metrics(deterministic, graph.y, graph.val_mask,
                                       False, "roman-empire")
            test_base = compute_metrics(deterministic, graph.y, graph.test_mask,
                                        False, "roman-empire")
        check_close("validation metric", val_base["metric"], row["val_metric"], ACC_TOL)
        check_close("test metric", test_base["metric"], row["test_metric"], ACC_TOL)
        check_close("test accuracy", test_base["acc"], row["test_acc"], ACC_TOL)
        check_close("test loss", test_base["loss"], row["test_loss"], LOSS_TOL)

        activate_only_dropout(model, DEPTH)
        with torch.inference_mode():
            member_logits = stochastic_logits(model, graph, split, device)
            pooled = member_logits.mean(dim=0)
            val_mc = compute_metrics(pooled, graph.y, graph.val_mask,
                                     False, "roman-empire")
            test_mc = compute_metrics(pooled, graph.y, graph.test_mask,
                                      False, "roman-empire")
            descriptive, arrays = test_analysis(member_logits, graph.y,
                                                graph.test_mask, graph.edge_index)
        prediction = OUT / f"split{split}_predictions.npz"
        atomic_npz(prediction, arrays)
        record = {
            "split": split,
            "input_fingerprint": fingerprint,
            "input_manifest": manifest,
            "device": runtime["device"],
            "torch_version": torch.__version__,
            "deterministic_base_val_metric": val_base["metric"],
            "deterministic_base_test_metric": test_base["metric"],
            "deterministic_base_test_loss": test_base["loss"],
            "mc_val_metric": val_mc["metric"],
            "mc_test_metric": test_mc["metric"],
            "mc_test_acc": test_mc["acc"],
            "mc_test_loss": test_mc["loss"],
            "delta_vs_same_checkpoint_pp": 100 * (test_mc["acc"] - test_base["acc"]),
            "descriptive_test": descriptive,
            "prediction_file": str(prediction.relative_to(ROOT)),
            "prediction_sha256": sha256(prediction),
        }
        atomic_json(OUT / f"split{split}.json", record)
        records.append(record)
        print("RESULT", json.dumps({"split": split,
                                    "base_test_acc": test_base["acc"],
                                    "mc_test_acc": test_mc["acc"]}, sort_keys=True),
              flush=True)
        del model, deterministic, member_logits, pooled
        torch.cuda.empty_cache()

    if len(records) != len(SPLITS):
        raise RuntimeError("A split is missing")
    code_keys = ("script_sha256", "models_sha256", "datasets_sha256",
                 "control_runner_sha256", "data_sha256", "runtime")
    first_inputs = records[0]["input_manifest"]
    if any(any(record["input_manifest"][key] != first_inputs[key]
               for key in code_keys) for record in records[1:]):
        raise RuntimeError("Code or dataset changed between split results")
    summary = {
        "protocol": "Four seeded full-graph passes of each unchanged BASE checkpoint; raw-logit mean; no validation selection or training",
        "splits": list(SPLITS),
        "per_split": [{k: record[k] for k in (
            "split", "deterministic_base_val_metric", "deterministic_base_test_metric",
            "mc_val_metric", "mc_test_metric", "mc_test_loss",
            "delta_vs_same_checkpoint_pp")}
            for record in records],
        "mean_base_test_acc_percent": 100 * float(np.mean([
            r["deterministic_base_test_metric"] for r in records])),
        "mean_mc_test_acc_percent": 100 * float(np.mean([
            r["mc_test_metric"] for r in records])),
        "mean_paired_delta_pp": float(np.mean([
            r["delta_vs_same_checkpoint_pp"] for r in records])),
        "population_sd_mc_test_acc_percent": 100 * float(np.std([
            r["mc_test_metric"] for r in records], ddof=0)),
    }
    atomic_json(OUT / "summary.json", summary)
    print("SUMMARY", json.dumps(summary, sort_keys=True), flush=True)


def smoke() -> None:
    """Exercise dropout mode, seeds, pooling, and unchanged weights on CPU."""
    sys.path.insert(0, str(ROOT))
    from models import Model

    torch.manual_seed(37)
    graph = SimpleNamespace(
        x=torch.randn(8, 3),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6],
                                 [1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 0]],
                                dtype=torch.long),
    )
    model = Model(MODEL_NAME, 1, 3, 8, 3, 1, 8, "LayerNorm", DROPOUT_P)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.eval()
    with torch.inference_mode():
        deterministic = model(graph, graph.x)
        activate_only_dropout(model, 1)
        torch.manual_seed(137)
        rng_before = torch.random.get_rng_state().clone()
        first = stochastic_logits(model, graph, 0, torch.device("cpu"))
        rng_after_first = torch.random.get_rng_state().clone()
        second = stochastic_logits(model, graph, 0, torch.device("cpu"))
        rng_after_second = torch.random.get_rng_state().clone()
    if not (torch.equal(rng_before, rng_after_first)
            and torch.equal(rng_before, rng_after_second)):
        raise RuntimeError("MC passes changed the surrounding CPU RNG state")
    if first.shape != (PASSES, 8, 3) or not torch.equal(first, second):
        raise RuntimeError("Four CPU stochastic passes did not reproduce exactly")
    if torch.equal(first[0], first[1]) or torch.equal(first.mean(0), deterministic):
        raise RuntimeError("Smoke check did not activate dropout")
    if any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items()):
        raise RuntimeError("MC inference changed a model parameter")

    runtime = runtime_manifest(torch.device("cpu"))
    manifest = input_manifest({"split": "0"}, Path(__file__), "smoke-data", runtime)
    changed_device = {**runtime, "device": "cuda:0"}
    changed_version = {**runtime, "torch_version": "changed-version"}
    if (stable_hash(manifest) == stable_hash(input_manifest(
            {"split": "0"}, Path(__file__), "smoke-data", changed_device))
            or stable_hash(manifest) == stable_hash(input_manifest(
                {"split": "0"}, Path(__file__), "smoke-data", changed_version))):
        raise RuntimeError("Runtime or device changes did not change the input fingerprint")

    with tempfile.TemporaryDirectory(dir=HERE, prefix=".mc_dropout_lock_smoke_") as temporary:
        directory = Path(temporary)
        with output_lock(directory):
            try:
                with output_lock(directory):
                    raise RuntimeError("A second writer acquired the output lock")
            except RuntimeError as exc:
                if "Another MC Dropout run holds" not in str(exc):
                    raise
        with output_lock(directory):
            pass
    print("CPU synthetic smoke check passed: reproducible stochastic passes, "
          "restored RNG, immutable runtime fingerprint, exclusive output lock, "
          "unchanged weights", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None,
                        help="Explicit CUDA device for the five-mask full-graph run")
    parser.add_argument("--smoke", action="store_true",
                        help="Run synthetic CPU checks; remove temporary lock files")
    args = parser.parse_args()
    if args.smoke:
        if args.device is not None:
            parser.error("--smoke must not specify a device")
        smoke()
        return
    if args.device is None:
        parser.error("Specify --device cuda:N for the full-graph run")
    run_real(torch.device(args.device))


if __name__ == "__main__":
    main()
