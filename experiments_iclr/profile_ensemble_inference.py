"""Measure selected checkpoint bytes and split-0 pooled inference latency.

All three models use the same resident Roman Empire graph, A100, and
CUDA-synchronized warmup/timing loop. No label is used for this measurement.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_iclr"))
from datasets import load_dataset  # noqa: E402
from projector_controls import all_logits, make_model  # noqa: E402

STANDARD = ROOT / "experiments_iclr" / "results"
MATCHED = ROOT / "experiments_iclr" / "parameter_matched_ens_results"
OUT = ROOT / "experiments_iclr" / "parameter_matched_ens_results" / "inference_profile.json"
WARMUP = 20
TIMED = 100


def selected_rows(root: Path, variant: str, width: int) -> dict[int, dict[str, str]]:
    with (root / "projector_controls.csv").open(newline="") as f:
        matching = [row for row in csv.DictReader(f)
                    if row["dataset"] == "roman-empire" and row["model"] == "SAGE"
                    and row["variant"] == variant and int(row["num_layers"]) == 5
                    and int(row["hidden_dim"]) == width and float(row["lr"]) == 3e-5]
    by_split = {int(row["split"]): row for row in matching}
    if len(matching) != 5 or tuple(sorted(by_split)) != tuple(range(5)):
        raise RuntimeError(f"Expected five complete {variant} rows at width {width}")
    return by_split


def checkpoint_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if ROOT not in path.parents or not path.is_file():
        raise RuntimeError(f"Invalid checkpoint path: {raw}")
    return path


def time_one(variant: str, width: int, row: dict[str, str], graph,
             input_dim: int, output_dim: int, device: torch.device) -> dict:
    args = SimpleNamespace(model="SAGE", num_layers=5, hidden_dim=width, m=4)
    model = make_model(args, variant, input_dim, output_dim, device)
    state = torch.load(checkpoint_path(row["checkpoint"]),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    expected = int(row["num_params"])
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual != expected:
        raise RuntimeError(f"Parameter count differs for {variant}: {actual} vs {expected}")
    torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(WARMUP):
            pooled = all_logits(model, variant, graph, 4).mean(0)
        torch.cuda.synchronize(device)
        milliseconds = []
        for _ in range(TIMED):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            pooled = all_logits(model, variant, graph, 4).mean(0)
            torch.cuda.synchronize(device)
            milliseconds.append((time.perf_counter() - started) * 1000)
    if pooled.shape != (graph.x.size(0), output_dim):
        raise RuntimeError(f"Unexpected pooled logit shape: {pooled.shape}")
    result = {
        "split_for_inference": 0,
        "trainable_parameters": actual,
        "mean_inference_ms": float(np.mean(milliseconds)),
        "median_inference_ms": float(np.median(milliseconds)),
        "sample_sd_inference_ms": float(np.std(milliseconds, ddof=1)),
        "min_inference_ms": float(np.min(milliseconds)),
        "max_inference_ms": float(np.max(milliseconds)),
    }
    del model, state, pooled
    torch.cuda.empty_cache()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This profile requires the same CUDA device for all variants")
    selection = json.loads((MATCHED / "selection.json").read_text())
    width = int(selection["selected_width"])
    device = torch.device("cuda:0")
    graph, _, _, _, _, output_dim, _ = load_dataset(
        "roman-empire", add_self_loops=True, device=device, data_dir="data")
    configurations = (
        ("gnnm", 512, STANDARD),
        ("ens_pooled", 512, STANDARD),
        ("ens_pooled", width, MATCHED),
    )
    results = []
    for variant, model_width, root in configurations:
        rows = selected_rows(root, variant, model_width)
        checkpoint_bytes = [checkpoint_path(rows[split]["checkpoint"]).stat().st_size
                            for split in range(5)]
        metric = time_one(variant, model_width, rows[0], graph,
                          graph.x.size(1), output_dim, device)
        result = {
            "variant": variant,
            "hidden_dim": model_width,
            "checkpoint_bytes_by_split": checkpoint_bytes,
            "mean_checkpoint_bytes": float(np.mean(checkpoint_bytes)),
            **metric,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    output = {
        "dataset": "roman-empire",
        "model": "SAGE",
        "graph_split": "Full transductive graph for all variants; selected checkpoints from official split 0 for inference timing",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "protocol": "eval mode and torch.inference_mode; 20 warmup then 100 CUDA-synchronized forward-and-logit-pool measurements per variant, in listed order; model weights and graph resident on the same device; data loading and checkpoint loading excluded",
        "units": "milliseconds per full-graph pooled prediction; checkpoint bytes are serialized selected state_dict file sizes",
        "results": results,
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
