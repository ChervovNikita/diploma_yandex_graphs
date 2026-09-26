"""Independent artifact and selected-checkpoint replay for Roman depth3 seed34 v1.

Writes completion_audit.json only after all four arms on one dataset pass.
The full CUDA replay recomputes both validation and held-out member logits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import roman_depth3_seed34 as study

MAX_LOGIT_DRIFT = 1e-4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text())


def check_trace(path: Path, selected_epoch: int, best_acc: float, best_ce: float):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == study.EPOCHS, f"Wrong trace length: {path}")
    replay_acc, replay_ce, replay_epoch = -1.0, float("inf"), 0
    for i, row in enumerate(rows, 1):
        require(int(row["epoch"]) == i, f"Trace epoch order: {path}")
        acc, ce = float(row["valid_accuracy"]), float(row["valid_ce"])
        require(math.isfinite(acc) and math.isfinite(ce) and 0 <= acc <= 1,
                f"Invalid validation metric: {path}:{i}")
        improved = (acc > replay_acc + 1e-12 or
                    (abs(acc - replay_acc) <= 1e-12 and ce < replay_ce - 1e-12))
        require(int(row["improved"]) == int(improved),
                f"Invalid trace improvement flag: {path}:{i}")
        if improved:
            replay_acc, replay_ce, replay_epoch = acc, ce, i
    require(replay_epoch == selected_epoch, f"Wrong selected epoch: {path}")
    require(abs(replay_acc - best_acc) < 1e-7 and
            abs(replay_ce - best_ce) < 1e-6,
            f"Trace winner differs from selected validation metrics: {path}")
    return rows


def check_predictions(model, bundle, payload, row, run_path, device):
    with np.load(run_path / "selected_predictions.npz", allow_pickle=False) as f:
        saved = {key: f[key].copy() for key in f.files}
    expected_keys = {"valid_member_logits", "valid_indices", "valid_labels",
                     "test_member_logits", "test_indices", "test_labels"}
    require(set(saved) == expected_keys, f"Prediction schema changed: {run_path}")
    require(np.array_equal(saved["valid_indices"], bundle.valid_idx.cpu().numpy()),
            f"Validation IDs changed: {run_path}")
    require(np.array_equal(saved["valid_labels"], bundle.valid_y.cpu().numpy()),
            f"Validation labels changed: {run_path}")
    require(np.array_equal(saved["test_indices"], bundle.test_idx_cpu.numpy()),
            f"Held-out IDs changed: {run_path}")
    require(np.array_equal(saved["test_labels"], bundle.test_y_cpu.numpy()),
            f"Held-out labels changed: {run_path}")
    result = {}
    for part in ("valid", "test"):
        idx = bundle.valid_idx if part == "valid" else bundle.test_idx_cpu.to(device)
        labels = bundle.valid_y if part == "valid" else bundle.test_y_cpu.to(device)
        model.eval()
        with torch.no_grad():
            logits = study.member_logits(model, bundle)[:, idx].detach().cpu().numpy()
        recorded = saved[f"{part}_member_logits"]
        require(logits.shape == recorded.shape,
                f"Selected {part} logit shape changed: {run_path}")
        max_drift = float(np.max(np.abs(logits - recorded)))
        require(max_drift <= MAX_LOGIT_DRIFT,
                f"Selected {part} logits drift {max_drift}: {run_path}")
        fresh = torch.from_numpy(logits).float()
        archived = torch.from_numpy(recorded).float()
        lab_cpu = labels.cpu()
        require(torch.equal(fresh.argmax(-1), archived.argmax(-1)),
                f"Selected {part} member decisions changed: {run_path}")
        fresh_pool = fresh.mean(0)
        old_pool = archived.mean(0)
        require(torch.equal(fresh_pool.argmax(-1), old_pool.argmax(-1)),
                f"Selected {part} pooled decisions changed: {run_path}")
        acc = float((fresh_pool.argmax(-1) == lab_cpu).float().mean().item())
        ce = float(F.cross_entropy(fresh_pool, lab_cpu).item())
        require(abs(acc - float(row[f"{part}_accuracy"])) < 1e-7,
                f"Selected {part} accuracy differs: {run_path}")
        require(abs(ce - float(row[f"{part}_ce"])) < 1e-4,
                f"Selected {part} cross-entropy differs: {run_path}")
        result[f"{part}_max_logit_drift"] = max_drift
        if part == "test":
            member_acc = [float((fresh[m].argmax(-1) == lab_cpu).float().mean().item())
                          for m in range(study.MEMBERS)]
            result["mean_member_test_accuracy"] = float(np.mean(member_acc))
            result["pooling_gain"] = acc - result["mean_member_test_accuracy"]
    return result


def homophily(dataset):
    _, labels, _, edges, _, _, _ = study.load_roman()
    source, target = edges
    nonloop = source != target
    matching = (labels[source[nonloop]] == labels[target[nonloop]])
    return {
        "definition": "fraction of non-self directed entries in the symmetrized training edge_index whose endpoint labels agree",
        "matching_entries": int(matching.sum().item()),
        "nonself_edge_entries": int(nonloop.sum().item()),
        "fraction": float(matching.float().mean().item()),
        "use": "post hoc dataset descriptor; not a model input or selection criterion",
    }


def audit_dataset(dataset: str, device: torch.device):
    root = study.ROOT / "results" / dataset
    manifest_path = root / "source_manifest.json"
    require(manifest_path.is_file(), "Missing source manifest")
    bundle, descriptor = study.load_graph(dataset, device)
    require(read_json(manifest_path) == study.expected_manifest(dataset, descriptor),
            "Manifest does not match current source/data/protocol")
    manifest_sha = study.sha256_file(manifest_path)
    seed_dirs = {p.name for p in root.iterdir() if p.is_dir()}
    require(seed_dirs == {f"seed{s}" for s in study.SEEDS},
            f"Expected exactly two seed directories, found {seed_dirs}")
    records = []
    for seed in study.SEEDS:
        per_seed = root / f"seed{seed}"
        arms = {p.name for p in per_seed.iterdir() if p.is_dir()}
        require(arms == set(study.ARMS),
                f"Expected exactly two complete arms for seed {seed}: {arms}")
        tied_init = read_json(per_seed / "tied" / "initialization.json")
        for arm in study.ARMS:
            run = per_seed / arm
            row = read_json(run / "result.json")
            require(row["protocol"] == "roman_posthoc_ogb300_selfloop_depth3_seed34_v1" and
                    row["dataset"] == dataset and row["published_split"] == 0 and
                    row["optimization_seed"] == seed and row["arm"] == arm and
                    row["source_manifest_sha256"] == manifest_sha and
                    row["epochs_run"] == study.EPOCHS,
                    f"Run identity/protocol mismatch: {run}")
            expected_artifacts = {"checkpoint.pt", "validation_trace.csv",
                                  "initialization.json", "initial_logits.npy",
                                  "selected_predictions.npz"}
            require(set(row["artifact_sha256"]) == expected_artifacts,
                    f"Artifact list differs: {run}")
            for filename, digest in row["artifact_sha256"].items():
                require(study.sha256_file(run / filename) == digest,
                        f"Artifact hash mismatch: {run / filename}")
            init = read_json(run / "initialization.json")
            require(init["initial_logits_sha256"] ==
                    study.sha256_file(run / "initial_logits.npy"),
                    f"Initial logit hash mismatch: {run}")
            for key in ("canonical_tied_state_sha256", "cpu_rng_sha256",
                        "cuda_rng_sha256"):
                require(init[key] == tied_init[key],
                        f"Initial matched-state field differs: {run}:{key}")
            require(float(init["paired_initial_logits_max_abs_diff"]) <=
                    study.INITIAL_LOGIT_TOL,
                    f"Paired initial logit tolerance exceeded: {run}")
            study.seed_all(seed)
            model, canonical = study.make_model(
                arm, bundle.x.size(1), bundle.classes, device)
            require(canonical == init["canonical_tied_state_sha256"],
                    f"Reconstructed initial weights differ: {run}")
            require(study.hash_tensor(torch.get_rng_state()) == init["cpu_rng_sha256"] and
                    study.hash_tensor(torch.cuda.get_rng_state(device)) == init["cuda_rng_sha256"],
                    f"Reconstructed initial RNG differs: {run}")
            require(sum(p.numel() for p in model.parameters()) ==
                    int(row["parameter_count"]) == int(init["parameter_count"]),
                    f"Parameter count differs: {run}")
            model.eval()
            with torch.no_grad():
                initial_logits = study.member_logits(model, bundle).cpu().numpy()
            archived_init = np.load(run / "initial_logits.npy", allow_pickle=False)
            init_drift = float(np.max(np.abs(initial_logits - archived_init)))
            require(init_drift <= study.INITIAL_LOGIT_TOL,
                    f"Reconstructed initial logits differ: {run}: {init_drift}")
            check_trace(run / "validation_trace.csv", int(row["selected_epoch"]),
                        float(row["valid_accuracy"]), float(row["valid_ce"]))
            checkpoint = torch.load(run / "checkpoint.pt", map_location=device,
                                    weights_only=True)
            require(checkpoint["epoch"] == row["selected_epoch"] and
                    checkpoint["seed"] == seed and checkpoint["arm"] == arm,
                    f"Checkpoint metadata differs: {run}")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            replay = check_predictions(model, bundle, None, row, run, device)
            records.append({
                "seed": seed, "arm": arm,
                "selected_epoch": int(row["selected_epoch"]),
                "valid_accuracy": float(row["valid_accuracy"]),
                "test_accuracy": float(row["test_accuracy"]),
                "test_ce": float(row["test_ce"]),
                "parameter_count": int(row["parameter_count"]),
                "train_seconds": float(row["train_seconds"]),
                "checkpoint_sha256": row["artifact_sha256"]["checkpoint.pt"],
                "initial_replay_max_logit_drift": init_drift,
                **replay,
            })
            del model
            torch.cuda.empty_cache()
    keyed = {(r["seed"], r["arm"]): r for r in records}
    paired = {}
    for left, right in (("tied", "untied_propagation"),):
        diffs = [keyed[s, left]["test_accuracy"] -
                 keyed[s, right]["test_accuracy"] for s in study.SEEDS]
        paired[f"{left}_minus_{right}"] = {
            "seeds": list(study.SEEDS),
            "differences": diffs,
            "mean": float(np.mean(diffs)),
            "sample_sd": float(np.std(diffs, ddof=1)),
        }
    result = {
        "status": "COMPLETE_CUDA_REPLAY_PASS",
        "dataset": dataset,
        "unit_of_replication": "optimizer seed on one published fixed split",
        "source_manifest_sha256": manifest_sha,
        "raw_data_redistributed": False,
        "records": records,
        "paired_differences": paired,
        "edge_homophily": homophily(dataset),
        "numeric_tolerance": {
            "initial_member_logits_max_abs": study.INITIAL_LOGIT_TOL,
            "selected_member_logits_max_abs": MAX_LOGIT_DRIFT,
            "exact_member_and_pooled_decisions": True,
        },
    }
    study.write_json(root / "completion_audit.json", result)
    print("COMPLETE_CUDA_REPLAY_PASS", dataset, len(records), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=tuple(study.DATA_FILES))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available(),
            "Full verifier requires CUDA")
    torch.set_num_threads(2)
    audit_dataset(args.dataset, device)


if __name__ == "__main__":
    main()
