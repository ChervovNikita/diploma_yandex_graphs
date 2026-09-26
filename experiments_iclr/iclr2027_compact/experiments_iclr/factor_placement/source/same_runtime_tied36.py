"""Separate same-runtime TIED grid; original primary train_one is unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import tuning as primary


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "same_runtime_tied36_results"
FREEZE = OUT / "FROZEN_STUDY.json"
PROTOCOL = "same_runtime_tied36_v1"
PRIMARY_LOCK_SHA = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"
DATASETS = ("cora", "wikics")
SEEDS = (0, 1, 2)
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01),
              (0.001, 0.0), (0.001, 0.01),
              (0.003, 0.0), (0.003, 0.01))
DEFAULT = (0.001, 0.0)
SOURCES = ("same_runtime_tied36.py", "verify_same_runtime_tied36.py",
           "SAME_RUNTIME_TIED36_PROTOCOL.md", "SAME_RUNTIME_TIED36_INIT_ANCHOR.json",
           "SAME_RUNTIME_TIED36_DESIGN_ORIGINAL.md")
PRIMARY_SOURCES = ("tuning.py", "models.py", "verify_tuning.py", "FROZEN_STUDY.json")
ANCHOR = ROOT / "SAME_RUNTIME_TIED36_INIT_ANCHOR.json"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def key(dataset: str, lr: float, wd: float, seed: int) -> str:
    return f"{dataset}/tied/{primary.candidate_name(lr, wd)}/seed{seed}"


def cell_dir(dataset: str, lr: float, wd: float, seed: int) -> Path:
    return OUT / "results" / key(dataset, lr, wd, seed)


def exact_primary_lock() -> dict:
    path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    require(path.is_file() and sha(path) == PRIMARY_LOCK_SHA,
            "Exact independently audited 432-cell lock is required")
    lock = json.loads(path.read_text())
    require(lock["protocol"] == "validation_tuning_sensitivity_v1" and
            len(lock["cells"]) == 432 and
            lock["freeze_sha256"] == primary.check_freeze(),
            "Primary lock does not match frozen source/data")
    return lock


def expected_freeze() -> dict:
    original_sha = primary.check_freeze()
    exact_primary_lock()
    anchor = json.loads(ANCHOR.read_text())
    require(anchor["protocol"] == "same_runtime_tied36_initialization_anchor_v1" and
            anchor["mechanism_validation_lock_sha256"] ==
            "712504c4b3c27453d305ba69c5f1b2f07e8ea13a92bcfc5881c9e5bebad8a123" and
            set(anchor["rows"]) == {f"{dataset}/seed{seed}"
                                    for dataset in DATASETS for seed in SEEDS},
            "Independent default TIED initialization anchor differs")
    original = json.loads((ROOT / "FROZEN_STUDY.json").read_text())
    design = ROOT / "SAME_RUNTIME_TIED36_DESIGN_ORIGINAL.md"
    require(sha(design) ==
            "06797e5cee6512806b439490167304a7b9755a71ccb4f3dbbb75cd0990f65df4",
            "Original 03:56 TIED36 design differs")
    return {
        "protocol": PROTOCOL,
        "posthoc_after_primary_outcomes": True,
        "original_frozen_study_sha256": original_sha,
        "original_validation_lock_sha256": PRIMARY_LOCK_SHA,
        "mechanism_default_tied_initialization_anchor_sha256": sha(ANCHOR),
        "prospective_original_design_sha256": sha(design),
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "original_source_sha256": {name: sha(ROOT / name) for name in PRIMARY_SOURCES},
        "graphs": {dataset: original["graphs"][dataset] for dataset in DATASETS},
        "matrix": {"datasets": list(DATASETS), "arm": "tied", "seeds": list(SEEDS),
                   "candidates": [{"lr": lr, "weight_decay": wd}
                                  for lr, wd in CANDIDATES],
                   "default_candidate": list(DEFAULT), "epochs": primary.EPOCHS,
                   "members": primary.MEMBERS, "width": primary.WIDTH,
                   "depth": primary.DEPTH, "dropout": primary.DROP,
                   "training_function": "original tuning.train_one with result root redirected only",
                   "candidate_rule": "highest three-seed mean selected validation accuracy, then lowest mean CE, then lower LR/decay",
                   "test_rule": "after independent 36-cell, all-layer 72-cell, and original 432-cell locks; selected/default only",
                   "complete_study_audit_cutoff_utc": "2026-09-26T06:00:00Z"},
    }


def check_freeze() -> str:
    require(FREEZE.is_file(), "Prospective TIED36 freeze missing")
    require(json.loads(FREEZE.read_text()) == expected_freeze(),
            "TIED36 source, data, anchor, or matrix differs from freeze")
    return sha(FREEZE)


def preflight(device: torch.device) -> None:
    freeze_sha = check_freeze()
    anchor = json.loads(ANCHOR.read_text())
    rows = {}
    for dataset in DATASETS:
        bundle, _ = primary.load_graph(dataset, device, include_test=False)
        for seed in SEEDS:
            primary.seed_all(seed)
            model, canonical = primary.make_model("tied", bundle, device)
            actual = primary.initial_audit(model, "tied", bundle, seed, device, canonical)
            expected = anchor["rows"][f"{dataset}/seed{seed}"]
            require(all(actual[field] == value for field, value in expected.items()
                        if field != "initial_member_logits_sha256"),
                    f"Default TIED initialization differs from same-runtime anchor: {dataset}/{seed}")
            rows[f"{dataset}/seed{seed}"] = actual
    report = {"protocol": PROTOCOL, "status": "PASS", "freeze_sha256": freeze_sha,
              "device": str(device), "anchor_sha256": sha(ANCHOR), "rows": rows}
    path = OUT / f"PREFLIGHT_{device.type.upper()}.json"
    require(not path.exists(), "Refusing to overwrite TIED36 preflight")
    primary.write_json(path, report)
    print(json.dumps({"preflight": "PASS", "device": str(device),
                      "freeze_sha256": freeze_sha, "initializations": len(rows)}), flush=True)


def run_dataset(dataset: str, device: torch.device) -> None:
    freeze_sha = check_freeze()
    require(dataset in DATASETS and device.type == "cuda", "Frozen CUDA graph required")
    preflight_path = OUT / "PREFLIGHT_CUDA.json"
    require(preflight_path.is_file(), "Run same-runtime CUDA preflight first")
    preflight_report = json.loads(preflight_path.read_text())
    require(preflight_report["status"] == "PASS" and
            preflight_report["freeze_sha256"] == freeze_sha and
            preflight_report["anchor_sha256"] == sha(ANCHOR),
            "Same-runtime TIED initialization preflight differs")
    bundle, _ = primary.load_graph(dataset, device, include_test=False)
    for lr, wd in CANDIDATES:
        for seed in SEEDS:
            # This invokes the byte-for-byte original update and selection loop.
            # Only its output root is redirected into this separate study.
            old_root = primary.ROOT
            try:
                primary.ROOT = OUT
                primary.train_one(dataset, bundle, freeze_sha, "tied", lr, wd, seed, device)
            finally:
                primary.ROOT = old_root
            export_validation_companion(dataset, lr, wd, seed, bundle, device)


def export_validation_companion(dataset, lr, wd, seed, bundle, device) -> None:
    """Archive post-training validation logits before independent lock.

    This does not choose or modify the original training checkpoint. It gives
    the independent auditor observable arrays for bounded replay and exact
    hard-decision comparison despite sparse CUDA reduction drift.
    """
    cell = cell_dir(dataset, lr, wd, seed)
    record = json.loads((cell / "result.json").read_text())
    output = cell / "validation_companion.npz"
    manifest = cell / "validation_companion.json"
    if record["failure"] is not None:
        require(not output.exists() and not manifest.exists(),
                "Failed cell cannot have validation companion")
        return
    if output.exists() or manifest.exists():
        require(output.is_file() and manifest.is_file(),
                "Incomplete existing validation companion")
        old = json.loads(manifest.read_text())
        require(old["checkpoint_sha256"] == record["checkpoint_sha256"] and
                old["predictions_sha256"] == sha(output),
                "Existing validation companion changed")
        return
    primary.seed_all(seed)
    model, _ = primary.make_model("tied", bundle, device)
    payload = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    va, vc, pooled, members = primary.evaluate(model, "tied", bundle,
                                                bundle.valid_idx, bundle.valid_y)
    require(abs(va - record["selected_valid_accuracy"]) <= 1e-7 and
            abs(vc - record["selected_valid_ce"]) <= 1e-5,
            "Post-training validation companion metric differs")
    np.savez_compressed(output, valid_indices=bundle.valid_idx.cpu().numpy(),
                        valid_labels=bundle.valid_y.cpu().numpy(),
                        valid_pooled_logits=pooled.numpy(),
                        valid_member_logits=members.numpy())
    primary.write_json(manifest, {
        "protocol": PROTOCOL, "dataset": dataset, "lr": lr,
        "weight_decay": wd, "seed": seed,
        "checkpoint_sha256": record["checkpoint_sha256"],
        "result_sha256": sha(cell / "result.json"),
        "predictions_sha256": sha(output),
        "valid_accuracy": va, "valid_ce": vc,
        "original_selected_pooled_logits_sha256":
            record["selected_valid_pooled_logits_sha256"],
        "companion_pooled_logits_sha256": primary.tensor_sha(pooled),
    })


def score(dataset: str, device: torch.device) -> None:
    freeze_sha = check_freeze()
    require(dataset in DATASETS, "Graph outside fixed TIED36 study")
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    require(lock_path.is_file(), "Independent 36-cell validation lock required")
    lock = json.loads(lock_path.read_text())
    require(lock["protocol"] == PROTOCOL and lock["freeze_sha256"] == freeze_sha and
            lock["original_validation_lock_sha256"] == PRIMARY_LOCK_SHA and
            len(lock["cells"]) == 36 and set(lock["selections"]) == set(DATASETS),
            "TIED36 validation lock or primary binding differs")
    exact_primary_lock()
    all_layer_lock_path = ROOT / "all_layer_factor_results" / "VALIDATION_SELECTION_LOCK.json"
    all_layer_freeze_path = ROOT / "all_layer_factor_results" / "FROZEN_ALL_LAYER_STUDY.json"
    require(all_layer_lock_path.is_file() and all_layer_freeze_path.is_file(),
            "Both TIED36 and all-layer validation locks are required before test scoring")
    all_layer_lock = json.loads(all_layer_lock_path.read_text())
    require(all_layer_lock["protocol"] == "all_layer_factor_placement_posthoc_v1" and
            len(all_layer_lock["cells"]) == 72 and
            all_layer_lock["freeze_sha256"] == sha(all_layer_freeze_path) and
            all_layer_lock["primary_selection_lock_sha256"] == PRIMARY_LOCK_SHA,
            "All-layer validation lock differs from frozen source/primary lock")
    bundle, _ = primary.load_graph(dataset, device, include_test=True)
    selected = tuple(lock["selections"][dataset]["selected_candidate"])
    for lr, wd in dict.fromkeys((selected, DEFAULT)):
        for seed in SEEDS:
            identity = key(dataset, lr, wd, seed)
            frozen_cell = lock["cells"][identity]
            cell = cell_dir(dataset, lr, wd, seed)
            require(sha(cell / "result.json") == frozen_cell["result_sha256"] and
                    sha(cell / "checkpoint.pt") == frozen_cell["checkpoint_sha256"],
                    f"TIED36 selected/default cell changed after lock: {identity}")
            out = OUT / "scores" / identity
            require(not out.exists(), f"Refusing to overwrite score: {identity}")
            primary.seed_all(seed)
            model, _ = primary.make_model("tied", bundle, device)
            payload = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state_dict"], strict=True)
            va, vc, vp, _ = primary.evaluate(model, "tied", bundle,
                                              bundle.valid_idx, bundle.valid_y)
            old = json.loads((cell / "result.json").read_text())
            require(abs(va - old["selected_valid_accuracy"]) <= 1e-7 and
                    abs(vc - old["selected_valid_ce"]) <= 1e-6,
                    f"TIED36 validation replay differs: {identity}")
            ta, tc, tp, tm = primary.evaluate(model, "tied", bundle,
                                              bundle.test_idx, bundle.test_y)
            out.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(out / "predictions.npz",
                                valid_pooled_logits=vp.numpy(),
                                test_pooled_logits=tp.numpy(),
                                test_member_logits=tm.numpy())
            primary.write_json(out / "score.json", {
                "protocol": PROTOCOL, "freeze_sha256": freeze_sha,
                "validation_selection_lock_sha256": sha(lock_path),
                "all_layer_validation_lock_sha256": sha(all_layer_lock_path),
                "original_validation_lock_sha256": PRIMARY_LOCK_SHA,
                "dataset": dataset, "arm": "tied", "lr": lr,
                "weight_decay": wd, "seed": seed,
                "selected_candidate": (lr, wd) == selected,
                "predeclared_default": (lr, wd) == DEFAULT,
                "checkpoint_sha256": frozen_cell["checkpoint_sha256"],
                "valid_accuracy": va, "valid_ce": vc,
                "test_accuracy": ta, "test_ce": tc,
                "test_predictions_sha256": sha(out / "predictions.npz"),
            })
            print(json.dumps({"scored": identity, "test_accuracy": ta}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    commands.add_parser("check-freeze")
    commands.add_parser("preflight").add_argument("--device", default="cuda:0")
    run = commands.add_parser("run")
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--device", default="cuda:0")
    scoring = commands.add_parser("score")
    scoring.add_argument("--dataset", choices=DATASETS, required=True)
    scoring.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not FREEZE.exists(), "Refusing to overwrite prospective TIED36 freeze")
        expected = expected_freeze()
        OUT.mkdir(parents=True, exist_ok=True)
        primary.write_json(FREEZE, expected)
        print(json.dumps({"freeze_sha256": sha(FREEZE)}), flush=True)
    elif args.command == "check-freeze":
        print(json.dumps({"freeze_sha256": check_freeze()}), flush=True)
    elif args.command == "preflight":
        preflight(torch.device(args.device))
    elif args.command == "run":
        run_dataset(args.dataset, torch.device(args.device))
    elif args.command == "score":
        score(args.dataset, torch.device(args.device))


if __name__ == "__main__":
    main()
