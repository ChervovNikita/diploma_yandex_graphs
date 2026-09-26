"""Prospectively fixed, train-label-only first AdamW update comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import mechanism as mech
import tuning as base


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
SEEDS = (0, 1, 2)
ETA = 0.001
EPS = 1e-8
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.0
REAL_UPDATE_TOL = 1e-5
SOURCE_FILES = ("initial_update_diagnostic.py", "INITIAL_UPDATE_DIAGNOSTIC_PROTOCOL.md")


def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_freeze() -> dict:
    return {
        "protocol": "initial_adam_update_order_train_only_v1",
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "base_freeze_sha256": base.check_freeze(),
        "mechanism_freeze_sha256": mech.check_freeze(),
        "matrix": {"datasets": DATASETS, "seeds": SEEDS, "device": "cpu",
                   "learning_rate": ETA, "betas": BETAS, "epsilon": EPS,
                   "weight_decay": WEIGHT_DECAY,
                   "real_update_max_abs_error_tolerance": REAL_UPDATE_TOL,
                   "data_rule": "loader hashes full label vector for integrity; returned bundle excludes test indices/labels; diagnostic gradients/statistics use training labels only",
                   "model_rule": "frozen width-128 two-block SAGE, four member paths, train-mode dropout",
                   "metrics": ("update_cosine", "opposite_sign_fraction_all_graph_coordinates",
                               "sync_zero_update_fraction_all_graph_coordinates",
                               "tied_graph_update_l2_norm", "sync_graph_update_l2_norm",
                               "sync_over_tied_l2_norm_ratio",
                               "relative_l2_update_difference", "mean_gradient_dot_tied_update",
                               "mean_gradient_dot_sync_update")},
    }


def check_freeze() -> str:
    path = ROOT / "INITIAL_UPDATE_DIAGNOSTIC_FREEZE.json"
    require(path.is_file(), "Missing prospective initial-update freeze")
    expected = json.loads(json.dumps(expected_freeze(), sort_keys=True, allow_nan=False))
    require(json.loads(path.read_text()) == expected,
            "Diagnostic source, original data, or fixed matrix changed")
    return sha(path)


def flatten(parameters) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in parameters])


def graph_parameters(model) -> list[torch.nn.Parameter]:
    parameters = list(model.residual_modules.parameters())
    require(parameters and all(p.requires_grad for p in parameters),
            "Expected trainable tied graph stack")
    return parameters


def member_gradients(model, bundle) -> torch.Tensor:
    model.train()
    parameters = graph_parameters(model)
    logits = base.member_logits(model, "tied", bundle)
    require(len(logits) == 4, "Expected four member outputs")
    members = []
    for output in logits:
        loss = F.cross_entropy(output.index_select(0, bundle.train_idx), bundle.train_y)
        require(bool(torch.isfinite(loss)), "Nonfinite initial training loss")
        grads = torch.autograd.grad(loss, parameters)
        members.append(torch.cat([gradient.detach().reshape(-1) for gradient in grads]))
    result = torch.stack(members)
    require(bool(torch.isfinite(result).all()), "Nonfinite initial graph gradient")
    return result


def diagnose(dataset: str, seed: int, bundle, freeze_sha: str) -> dict:
    device = torch.device("cpu")
    base.seed_all(seed)
    tied, tied_canonical = base.make_model("tied", bundle, device)
    initial_rng = mech.rng_snapshot(device)
    initial_graph = flatten(graph_parameters(tied)).clone()
    base.seed_all(seed)
    sync, sync_canonical = mech.make_model("sync", bundle, device)
    require(tied_canonical == sync_canonical and
            mech.rng_equal(initial_rng, mech.rng_snapshot(device)) and
            torch.equal(initial_graph, flatten(mech.graph_stack_parameters(sync)[0])),
            f"Initial TIED/SYNC matching failed: {dataset}/{seed}")
    mech.assert_equal_graph_weights(sync)

    mech.rng_restore(initial_rng, device)
    member_g = member_gradients(tied, bundle)
    mean_g = member_g.mean(0)
    tied_formula = -ETA * mean_g / (mean_g.abs() + EPS)
    sync_formula = (-ETA * member_g / (member_g.abs() + EPS)).mean(0)
    require(bool(torch.isfinite(tied_formula).all()) and
            bool(torch.isfinite(sync_formula).all()), "Nonfinite analytic update")

    mech.rng_restore(initial_rng, device)
    tied_optimizer = torch.optim.AdamW(tied.parameters(), lr=ETA, betas=BETAS,
                                       eps=EPS, weight_decay=WEIGHT_DECAY)
    mech.one_update(tied, "tied", bundle, tied_optimizer)
    tied_after_rng = mech.rng_snapshot(device)
    tied_real = flatten(graph_parameters(tied)) - initial_graph

    mech.rng_restore(initial_rng, device)
    sync_optimizer = torch.optim.AdamW(sync.parameters(), lr=ETA, betas=BETAS,
                                       eps=EPS, weight_decay=WEIGHT_DECAY)
    mech.one_update(sync, "sync", bundle, sync_optimizer)
    sync_after_rng = mech.rng_snapshot(device)
    mech.assert_equal_graph_weights(sync)
    sync_real = flatten(mech.graph_stack_parameters(sync)[0]) - initial_graph
    require(mech.rng_equal(tied_after_rng, sync_after_rng),
            f"Dropout RNG differs after matched actual updates: {dataset}/{seed}")
    tied_error = float((tied_real - tied_formula).abs().max().item())
    sync_error = float((sync_real - sync_formula).abs().max().item())
    require(tied_error <= REAL_UPDATE_TOL and sync_error <= REAL_UPDATE_TOL,
            f"Analytic and actual AdamW update differ: {dataset}/{seed}: {tied_error}, {sync_error}")

    a = tied_formula.double()
    b = sync_formula.double()
    mean_g64 = mean_g.double()
    count = a.numel()
    require(count > 0 and float(a.norm()) > 0 and float(b.norm()) > 0,
            "Degenerate initial update")
    row = {
        "dataset": dataset, "seed": seed,
        "freeze_sha256": freeze_sha,
        "canonical_initial_state_sha256": tied_canonical,
        "graph_parameter_coordinates": count,
        "update_cosine": float(torch.dot(a, b) / (a.norm() * b.norm())),
        "opposite_sign_coordinate_count": int(((a * b) < 0).sum().item()),
        "opposite_sign_fraction_all_graph_coordinates": float(((a * b) < 0).double().mean().item()),
        "sync_zero_update_coordinate_count": int((b == 0).sum().item()),
        "sync_zero_update_fraction_all_graph_coordinates": float((b == 0).double().mean().item()),
        "tied_graph_update_l2_norm": float(a.norm()),
        "sync_graph_update_l2_norm": float(b.norm()),
        "sync_over_tied_l2_norm_ratio": float(b.norm() / a.norm()),
        "relative_l2_update_difference": float((a - b).norm() / a.norm()),
        "mean_gradient_dot_tied_update": float(torch.dot(mean_g64, a)),
        "mean_gradient_dot_sync_update": float(torch.dot(mean_g64, b)),
        "analytic_vs_actual_tied_max_abs_error": tied_error,
        "analytic_vs_actual_sync_max_abs_error": sync_error,
        "actual_tied_vs_sync_max_abs_parameter_update_difference":
            float((tied_real - sync_real).abs().max().item()),
        "matched_post_step_rng": True,
    }
    require(all(math.isfinite(v) for k, v in row.items() if isinstance(v, float)),
            "Nonfinite diagnostic statistic")
    return row


def run() -> None:
    freeze_sha = check_freeze()
    result_path = ROOT / "INITIAL_UPDATE_DIAGNOSTIC_RESULTS.json"
    csv_path = ROOT / "initial_update_diagnostic.csv"
    require(not result_path.exists() and not csv_path.exists(),
            "Diagnostic results already exist; refusing overwrite")
    rows = []
    for dataset in DATASETS:
        bundle, _ = base.load_graph(dataset, torch.device("cpu"), include_test=False)
        for seed in SEEDS:
            row = diagnose(dataset, seed, bundle, freeze_sha)
            rows.append(row)
            print(json.dumps({"completed": [dataset, seed],
                              "formula_tied_error": row["analytic_vs_actual_tied_max_abs_error"],
                              "formula_sync_error": row["analytic_vs_actual_sync_max_abs_error"]}),
                  flush=True)
    require(len(rows) == 12, "Incomplete initial-update matrix")
    payload = {"protocol": "initial_adam_update_order_train_only_v1",
               "freeze_sha256": freeze_sha,
               "interpretation": "Initial train-loss update arithmetic only; no generalization prediction",
               "rows": rows}
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "freeze_sha256": freeze_sha,
                      "results_sha256": sha(result_path), "csv_sha256": sha(csv_path)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "check-freeze", "run"))
    args = parser.parse_args()
    if args.command == "freeze":
        path = ROOT / "INITIAL_UPDATE_DIAGNOSTIC_FREEZE.json"
        require(not path.exists(), "Refusing to overwrite prospective diagnostic freeze")
        path.write_text(json.dumps(expected_freeze(), indent=2, sort_keys=True) + "\n")
        print(json.dumps({"initial_update_freeze_sha256": sha(path)}), flush=True)
    elif args.command == "check-freeze":
        print(json.dumps({"initial_update_freeze_sha256": check_freeze()}), flush=True)
    else:
        run()


if __name__ == "__main__":
    main()
