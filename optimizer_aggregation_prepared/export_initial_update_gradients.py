"""Postfreeze complete raw-gradient export and independent arithmetic replay.

The original diagnostic did not retain gradients. This script reruns all
12 train-label-only initial states and exports the four member graph-gradient
vectors per state, then checks NumPy-derived statistics against the frozen
diagnostic result. It performs no model selection or test evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import mechanism as mech
import tuning as base


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
SEEDS = (0, 1, 2)
ETA = 0.001
EPS = 1e-8


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_member_graph_gradients(bundle, seed):
    base.seed_all(seed)
    model, canonical = base.make_model("tied", bundle, torch.device("cpu"))
    model.train()
    graph = list(model.residual_modules.parameters())
    require(graph, "Expected trainable graph parameters")
    outputs = tuple(model(bundle.graph, bundle.x, tabm_seed=m) for m in range(4))
    rows = []
    for output in outputs:
        loss = F.cross_entropy(output.index_select(0, bundle.train_idx), bundle.train_y)
        grads = torch.autograd.grad(loss, graph)
        rows.append(torch.cat([grad.detach().reshape(-1).cpu() for grad in grads]).numpy())
    value = np.stack(rows).astype(np.float32, copy=False)
    require(value.shape[0] == 4 and np.isfinite(value).all(), "Invalid raw member gradients")
    return value, canonical


def actual_graph_updates(bundle, seed):
    device = torch.device("cpu")
    base.seed_all(seed)
    tied, tied_canonical = base.make_model("tied", bundle, device)
    initial_rng = mech.rng_snapshot(device)
    tied_graph = list(tied.residual_modules.parameters())
    initial = torch.cat([p.detach().reshape(-1) for p in tied_graph]).clone()
    base.seed_all(seed)
    sync, sync_canonical = mech.make_model("sync", bundle, device)
    require(tied_canonical == sync_canonical and
            mech.rng_equal(initial_rng, mech.rng_snapshot(device)),
            "Actual-update constructors or RNG differ")
    mech.rng_restore(initial_rng, device)
    tied_opt = torch.optim.AdamW(tied.parameters(), lr=ETA, betas=(0.9, 0.999),
                                 eps=EPS, weight_decay=0)
    mech.one_update(tied, "tied", bundle, tied_opt)
    tied_after = mech.rng_snapshot(device)
    tied_delta = torch.cat([p.detach().reshape(-1) for p in tied_graph]) - initial
    mech.rng_restore(initial_rng, device)
    sync_opt = torch.optim.AdamW(sync.parameters(), lr=ETA, betas=(0.9, 0.999),
                                 eps=EPS, weight_decay=0)
    mech.one_update(sync, "sync", bundle, sync_opt)
    sync_after = mech.rng_snapshot(device)
    require(mech.rng_equal(tied_after, sync_after), "Actual-update dropout RNG differs")
    mech.assert_equal_graph_weights(sync)
    sync_delta = torch.cat([p.detach().reshape(-1)
                            for p in mech.graph_stack_parameters(sync)[0]]) - initial
    return tied_delta.numpy().astype(np.float32, copy=False), \
        sync_delta.numpy().astype(np.float32, copy=False), tied_canonical


def replay_metrics(raw: np.ndarray) -> dict:
    mean_g = raw.mean(axis=0, dtype=np.float32)
    tied = (-ETA * mean_g / (np.abs(mean_g) + EPS)).astype(np.float64)
    sync = (-ETA * raw / (np.abs(raw) + EPS)).mean(axis=0,
                                                    dtype=np.float32).astype(np.float64)
    gt = mean_g.astype(np.float64)
    norm_t, norm_s = np.linalg.norm(tied), np.linalg.norm(sync)
    require(norm_t > 0 and norm_s > 0, "Degenerate raw-gradient replay")
    return {
        "update_cosine": float(np.dot(tied, sync) / (norm_t * norm_s)),
        "opposite_sign_fraction_all_graph_coordinates": float(np.mean(tied * sync < 0)),
        "sync_zero_update_fraction_all_graph_coordinates": float(np.mean(sync == 0)),
        "tied_graph_update_l2_norm": float(norm_t),
        "sync_graph_update_l2_norm": float(norm_s),
        "sync_over_tied_l2_norm_ratio": float(norm_s / norm_t),
        "relative_l2_update_difference": float(np.linalg.norm(tied - sync) / norm_t),
        "mean_gradient_dot_tied_update": float(np.dot(gt, tied)),
        "mean_gradient_dot_sync_update": float(np.dot(gt, sync)),
    }


def main():
    require(mech.check_freeze() and base.check_freeze(), "Source/data freeze gate failed")
    original_freeze = ROOT / "INITIAL_UPDATE_DIAGNOSTIC_FREEZE.json"
    original_result = ROOT / "INITIAL_UPDATE_DIAGNOSTIC_RESULTS.json"
    require(original_freeze.is_file() and original_result.is_file(),
            "Frozen diagnostic and result required")
    original = json.loads(original_result.read_text())
    require(original["freeze_sha256"] == sha(original_freeze) and len(original["rows"]) == 12,
            "Original diagnostic result is incomplete or mismatched")
    expected_rows = {(row["dataset"], row["seed"]): row for row in original["rows"]}
    require(set(expected_rows) == {(d, s) for d in DATASETS for s in SEEDS},
            "Original diagnostic has unexpected rows")
    final = ROOT / "initial_update_raw_gradients"
    work = ROOT / "initial_update_raw_gradients.inprogress"
    require(not final.exists() and not work.exists(), "Raw-gradient export already exists")
    work.mkdir()
    rows = []
    for dataset in DATASETS:
        bundle, _ = base.load_graph(dataset, torch.device("cpu"), include_test=False)
        for seed in SEEDS:
            raw, canonical = raw_member_graph_gradients(bundle, seed)
            actual_tied, actual_sync, actual_canonical = actual_graph_updates(bundle, seed)
            original_row = expected_rows[(dataset, seed)]
            require(canonical == actual_canonical == original_row["canonical_initial_state_sha256"] and
                    raw.shape == (4, original_row["graph_parameter_coordinates"]),
                    f"Initial model/dimensions changed: {dataset}/{seed}")
            derived = replay_metrics(raw)
            differences = {name: abs(value - original_row[name])
                           for name, value in derived.items()}
            require(max(differences.values()) <= 1e-5,
                    f"Independent NumPy arithmetic differs: {dataset}/{seed}: {differences}")
            mean_g = raw.mean(axis=0, dtype=np.float32)
            formula_tied = -ETA * mean_g / (np.abs(mean_g) + EPS)
            formula_sync = (-ETA * raw / (np.abs(raw) + EPS)).mean(axis=0,
                                                                   dtype=np.float32)
            actual_tied_error = float(np.max(np.abs(actual_tied - formula_tied)))
            actual_sync_error = float(np.max(np.abs(actual_sync - formula_sync)))
            actual_between = float(np.max(np.abs(actual_tied - actual_sync)))
            require(actual_tied_error <= 1e-5 and actual_sync_error <= 1e-5 and
                    abs(actual_between - original_row[
                        "actual_tied_vs_sync_max_abs_parameter_update_difference"]) <= 1e-5,
                    f"Actual update replay differs: {dataset}/{seed}")
            file = work / f"{dataset}_seed{seed}.npz"
            np.savez_compressed(file, member_graph_gradients=raw,
                                actual_tied_graph_update=actual_tied,
                                actual_sync_graph_update=actual_sync)
            rows.append({"dataset": dataset, "seed": seed,
                         "file": file.name, "sha256": sha(file),
                         "shape": list(raw.shape),
                         "max_abs_aggregate_replay_difference": max(differences.values()),
                         "actual_tied_formula_max_abs_error": actual_tied_error,
                         "actual_sync_formula_max_abs_error": actual_sync_error,
                         "actual_tied_sync_max_abs_difference": actual_between,
                         "numpy_replay": derived})
            print(json.dumps({"exported": [dataset, seed],
                              "max_replay_difference": max(differences.values())}), flush=True)
    require(len(rows) == 12, "Incomplete raw-gradient export")
    manifest = {"protocol": "postfreeze_initial_update_raw_gradient_export_v1",
                "source_script_sha256": sha(Path(__file__)),
                "original_diagnostic_freeze_sha256": sha(original_freeze),
                "original_diagnostic_result_sha256": sha(original_result),
                "data_rule": "full label vector read for integrity; returned bundle exposes train/validation labels, and only train labels enter gradients",
                "rows": rows}
    (work / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    work.rename(final)
    print(json.dumps({"exported_rows": 12,
                      "manifest_sha256": sha(final / "MANIFEST.json"),
                      "max_replay_difference": max(row["max_abs_aggregate_replay_difference"]
                                                   for row in rows)}), flush=True)


if __name__ == "__main__":
    main()
