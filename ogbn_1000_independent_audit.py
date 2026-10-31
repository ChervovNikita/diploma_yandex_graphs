"""Independent, validation-first audit of the frozen ogbn-arxiv 1,000-epoch run.

Run only after the one-shot launcher has written its completion marker and its
full verifier has passed. The script checks the frozen source lock, replays
checkpoint selection, reloads the official OGB split, scores saved logits,
and performs fresh CPU inference from all nine selected checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "experiments_iclr/ogbn_arxiv_1000_results"
LOCK = ROOT / "experiments_iclr/ogbn_arxiv_1000_source_lock.json"
PAIRS = [(seed, variant) for seed in (0, 1, 2) for variant in ("base", "ens", "gnnm")]
TOL = 1e-12


def demand(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(tensor: torch.Tensor | np.ndarray) -> str:
    array = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(str(array.dtype).encode())
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def score(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    pooled = torch.from_numpy(np.ascontiguousarray(logits)).mean(0)
    target = torch.from_numpy(np.ascontiguousarray(labels))
    return (float((pooled.argmax(-1) == target).float().mean()),
            float(F.cross_entropy(pooled, target)))


def member_diagnostics(logits: np.ndarray, labels: np.ndarray) -> dict:
    decisions = logits.argmax(-1)
    correct = decisions == labels[None, :]
    pooled_decisions = torch.from_numpy(np.ascontiguousarray(logits)).mean(0).argmax(-1).numpy()
    pooled_accuracy = float(np.mean(pooled_decisions == labels))
    disagreement = [float(np.mean(decisions[a] != decisions[b]))
                    for a in range(len(decisions)) for b in range(a + 1, len(decisions))]
    return {
        "member_accuracies": [float(v) for v in correct.mean(1)],
        "mean_member_accuracy": float(correct.mean()),
        "pool_minus_mean_member_accuracy": pooled_accuracy - float(correct.mean()),
        "all_members_wrong_fraction": float(np.mean(~correct.any(0))),
        "pairwise_prediction_disagreement": float(np.mean(disagreement)) if disagreement else None,
    }


def replay_epoch_history(path: Path, selected: dict) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    demand(len(rows) == 1000 and selected["epochs_run"] == 1000,
           f"Incomplete 1,000-epoch trace: {path}")
    best_acc, best_ce, best_epoch = -math.inf, math.inf, 0
    best_by_900, best_by_950 = -math.inf, -math.inf
    for expected_epoch, row in enumerate(rows, 1):
        demand(int(row["epoch"]) == expected_epoch, f"Noncontiguous trace: {path}")
        acc, ce = float(row["valid_pooled_accuracy"]), float(row["valid_pooled_ce"])
        demand(math.isfinite(acc) and 0 <= acc <= 1 and math.isfinite(ce) and ce >= 0,
               f"Invalid validation metric: {path}, epoch {expected_epoch}")
        improved = acc > best_acc + TOL or (abs(acc - best_acc) <= TOL and ce < best_ce - TOL)
        demand(int(row["is_selected_so_far"]) == int(improved),
               f"Trace checkpoint flag mismatch: {path}, epoch {expected_epoch}")
        if improved:
            best_acc, best_ce, best_epoch = acc, ce, expected_epoch
        if expected_epoch <= 900:
            best_by_900 = max(best_by_900, acc)
        if expected_epoch <= 950:
            best_by_950 = max(best_by_950, acc)
    demand(best_epoch == selected["selected_epoch"], f"Validation selection mismatch: {path}")
    demand(abs(best_acc - selected["valid_accuracy"]) < 1e-7 and
           abs(best_ce - selected["valid_ce"]) < 1e-6,
           f"Selected validation metric mismatch: {path}")
    return {
        "selected_epoch": best_epoch,
        "best_validation_accuracy": best_acc,
        "best_validation_ce": best_ce,
        "accuracy_gain_after_900_pp": 100 * (best_acc - best_by_900),
        "accuracy_gain_after_950_pp": 100 * (best_acc - best_by_950),
    }


def build_model(variant: str):
    # Construct directly from frozen model definitions, independent of the runner.
    from models import Model, TABMModel

    common = dict(model_name="SAGE", num_layers=2, input_dim=128,
                  hidden_dim=128, output_dim=40, hidden_dim_multiplier=1,
                  num_heads=8, normalization="LayerNorm", dropout=0.2)
    if variant == "base":
        return Model(**common)
    if variant == "ens":
        return torch.nn.ModuleList([Model(**common) for _ in range(4)])
    return TABMModel(**common, tabm_inits=4, device=torch.device("cpu"))


def fresh_logits(model, variant: str, graph, features: torch.Tensor,
                 indices: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        if variant == "base":
            members = [model(graph, features)]
        elif variant == "ens":
            members = [member(graph, features) for member in model]
        else:
            members = [model(graph, features, tabm_seed=k) for k in range(4)]
        return torch.stack([value.index_select(0, indices) for value in members]).numpy()


def main() -> None:
    demand((ROOT / "experiments_iclr/logs/ogbn_arxiv_1000.complete").is_file(),
           "One-shot completion marker is absent")
    lock = json.loads(LOCK.read_text())
    source_hashes = {}
    for relative, expected in lock["files"].items():
        actual = sha(ROOT / relative)
        demand(actual == expected, f"Frozen source hash mismatch: {relative}")
        source_hashes[relative] = actual

    # Replay every validation choice before opening any saved test logits.
    selected = {}
    replay = {}
    for seed, variant in PAIRS:
        path = RESULT / f"seed_{seed}" / variant
        record = json.loads((path / "selected.json").read_text())
        demand(record["seed"] == seed and record["variant"] == variant,
               f"Pair identity mismatch: {seed} {variant}")
        replay[(seed, variant)] = replay_epoch_history(path / "epochs.csv", record)
        demand(sha(path / "selected_checkpoint.pt") == record["checkpoint_sha256"],
               f"Selected checkpoint hash mismatch: {seed} {variant}")
        selected[(seed, variant)] = record

    manifest = json.loads((RESULT / "dataset_manifest.json").read_text())
    demand(sha(RESULT / "dataset_manifest.json") ==
           lock["files"]["experiments_iclr/ogbn_arxiv_300_results/dataset_manifest.json"],
           "Official data manifest changed across budgets")
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(ROOT / "experiments_iclr/data"))
    data = dataset[0]
    official = dataset.get_idx_split()
    labels = data.y.view(-1).long().cpu()
    indices = {part: official[part].view(-1).long().cpu() for part in ("train", "valid", "test")}
    graph = SimpleNamespace(edge_index=to_undirected(data.edge_index.cpu(), num_nodes=data.num_nodes))
    x = data.x.float().cpu()
    computed_fingerprints = {
        "x": fingerprint(data.x), "raw_edge_index": fingerprint(data.edge_index),
        "y": fingerprint(data.y),
        **{part + "_index": fingerprint(indices[part]) for part in indices},
    }
    demand(computed_fingerprints == manifest["fingerprints_sha256"],
           "Official OGB graph, labels, or temporal split fingerprint mismatch")
    demand(int(data.num_nodes) == 169343 and len(indices["valid"]) == 29799 and
           len(indices["test"]) == 48603 and len(indices["train"]) == 90941,
           "Official OGB dataset dimensions changed")

    torch.set_num_threads(4)
    results = []
    artifact_hashes = {}
    for seed, variant in PAIRS:
        directory = RESULT / f"seed_{seed}" / variant
        record = selected[(seed, variant)]
        with np.load(directory / "selected_predictions.npz", allow_pickle=False) as saved:
            parts = {name: saved[name] for name in saved.files}
        row = {"seed": seed, "variant": variant, **replay[(seed, variant)]}
        for part in ("valid", "test"):
            expected_idx = indices[part].numpy()
            expected_labels = labels[indices[part]].numpy()
            demand(np.array_equal(parts[part + "_indices"], expected_idx),
                   f"Official {part} indices mismatch: {seed} {variant}")
            demand(np.array_equal(parts[part + "_labels"], expected_labels),
                   f"Official {part} labels mismatch: {seed} {variant}")
            saved_acc, saved_ce = score(parts[part + "_member_logits"], expected_labels)
            demand(abs(saved_acc - record[part + "_accuracy"]) <= 1e-7 and
                   abs(saved_ce - record[part + "_ce"]) <= 2e-5,
                   f"Recorded {part} score mismatch: {seed} {variant}")
            row[part + "_accuracy"] = saved_acc
            row[part + "_ce"] = saved_ce
            if part == "test":
                row["test_member_diagnostics"] = member_diagnostics(
                    parts[part + "_member_logits"], expected_labels)

        checkpoint = torch.load(directory / "selected_checkpoint.pt", map_location="cpu", weights_only=True)
        demand((checkpoint["seed"], checkpoint["variant"], checkpoint["epoch"]) ==
               (seed, variant, row["selected_epoch"]), f"Checkpoint identity mismatch: {seed} {variant}")
        model = build_model(variant)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        combined_indices = torch.cat((indices["valid"], indices["test"]))
        fresh_combined = fresh_logits(model, variant, graph, x, combined_indices)
        split_at = len(indices["valid"])
        for part in ("valid", "test"):
            computed = (fresh_combined[:, :split_at] if part == "valid"
                        else fresh_combined[:, split_at:])
            stored = parts[part + "_member_logits"]
            demand(computed.shape == stored.shape, f"Fresh CPU shape mismatch: {seed} {variant} {part}")
            max_abs = float(np.max(np.abs(computed - stored)))
            cpu_acc, cpu_ce = score(computed, labels[indices[part]].numpy())
            row[part + "_cpu_replay"] = {
                "max_absolute_logit_difference": max_abs,
                "accuracy": cpu_acc,
                "ce": cpu_ce,
                "accuracy_difference_from_saved": cpu_acc - row[part + "_accuracy"],
            }
            demand(max_abs <= 1e-3 and abs(cpu_acc - row[part + "_accuracy"]) <= 1e-7,
                   f"Fresh CPU inference disagrees with saved {part} logits/score: {seed} {variant}")
        del model, fresh_combined, checkpoint
        results.append(row)

    for path in RESULT.rglob("*"):
        if path.is_file():
            artifact_hashes[str(path.relative_to(ROOT))] = sha(path)
    demand(len(artifact_hashes) == 40, "Unexpected result artifact count")
    test = {(r["seed"], r["variant"]): r["test_accuracy"] for r in results}
    summary = {}
    for variant in ("base", "ens", "gnnm"):
        values = np.array([test[(seed, variant)] for seed in (0, 1, 2)])
        summary[variant] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
    pairs = {}
    for left, right in (("gnnm", "ens"), ("gnnm", "base"), ("ens", "base")):
        values = np.array([test[(seed, left)] - test[(seed, right)] for seed in (0, 1, 2)])
        pairs[f"{left}_minus_{right}"] = {
            "by_seed": [float(v) for v in values],
            "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
        }
    reported = json.loads((RESULT / "summary.json").read_text())
    for variant in summary:
        demand(abs(summary[variant]["mean"] - reported["variants"][variant]["test_accuracy_mean"]) <= 1e-7 and
               abs(summary[variant]["sample_sd"] - reported["variants"][variant]["test_accuracy_sample_sd"]) <= 1e-7,
               f"Reported summary mismatch: {variant}")
    for name in pairs:
        demand(abs(pairs[name]["mean"] - reported["paired_differences"][name]["mean"]) <= 1e-7 and
               abs(pairs[name]["sample_sd"] - reported["paired_differences"][name]["sample_sd"]) <= 1e-7,
               f"Reported paired summary mismatch: {name}")
    report = {
        "status": "independent_audit_passed", "profile": "fixed-1000",
        "audit_script_sha256": sha(Path(__file__)),
        "official_split": {part: len(indices[part]) for part in indices},
        "source_hashes_sha256": source_hashes,
        "source_lock_sha256": sha(LOCK),
        "artifact_hashes_sha256": artifact_hashes,
        "results": results, "test_accuracy_summary": summary,
        "paired_test_accuracy_differences": pairs,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"independent 1,000-epoch audit failed: {exc}", file=sys.stderr)
        raise
