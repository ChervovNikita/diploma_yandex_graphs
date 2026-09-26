"""Post hoc parameter-count sensitivity: narrow UNTIED, original train_one."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import tuning as primary


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "narrow_untied72_results"
WIDTH_SCAN = OUT / "NARROW_UNTIED72_WIDTH_SCAN.json"
FREEZE = OUT / "FROZEN_NARROW_UNTIED72_STUDY.json"
PROTOCOL = "narrow_untied72_capacity_sensitivity_v1"
PRIMARY_LOCK_SHA = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"
TIED36_LOCK_SHA = "fb860e2eb9c5c107c11c6c7d71547b79c4e690a95b74da460838dfa93d27be26"
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
SEEDS = (0, 1, 2)
CANDIDATES = primary.CANDIDATES
DEFAULT = primary.DEFAULT
SOURCES = ("narrow_untied72.py", "verify_narrow_untied72.py", "NARROW_UNTIED72_PROTOCOL.md")
PRIMARY_SOURCES = ("tuning.py", "models.py", "verify_tuning.py", "FROZEN_STUDY.json")


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
    return f"{dataset}/untied/{primary.candidate_name(lr, wd)}/seed{seed}"


def cell_dir(dataset: str, lr: float, wd: float, seed: int) -> Path:
    return OUT / "results" / key(dataset, lr, wd, seed)


def original_locks() -> None:
    original = ROOT / "VALIDATION_SELECTION_LOCK.json"
    tied36 = ROOT / "same_runtime_tied36_results/VALIDATION_SELECTION_LOCK.json"
    require(sha(original) == PRIMARY_LOCK_SHA and sha(tied36) == TIED36_LOCK_SHA,
            "Exact primary and same-runtime TIED36 validation locks required")
    p, t = json.loads(original.read_text()), json.loads(tied36.read_text())
    require(len(p["cells"]) == 432 and p["freeze_sha256"] == primary.check_freeze() and
            len(t["cells"]) == 36 and t["original_validation_lock_sha256"] == PRIMARY_LOCK_SHA,
            "Comparator locks differ from frozen studies")


def scan_widths() -> None:
    require(not OUT.exists(), "Refusing to overwrite width scan or study")
    primary.check_freeze()
    original_locks()
    original_preflight = json.loads((ROOT / "preflight.json").read_text())
    require(original_preflight["freeze_sha256"] == primary.check_freeze(),
            "Original parameter-count preflight differs")
    rows = {}
    old_width = primary.WIDTH
    try:
        for dataset in DATASETS:
            bundle, _ = primary.load_graph(dataset, torch.device("cpu"), include_test=False)
            primary.WIDTH = 128
            primary.seed_all(0)
            tied, _ = primary.make_model("tied", bundle, torch.device("cpu"))
            target = sum(p.numel() for p in tied.parameters())
            require(target == original_preflight["graphs"][dataset]["parameter_counts"]["tied"],
                    f"TIED128 target count differs: {dataset}")
            del tied
            curve = []
            for width in range(32, 129):
                primary.WIDTH = width
                primary.seed_all(0)
                model, _ = primary.make_model("untied", bundle, torch.device("cpu"))
                count = sum(p.numel() for p in model.parameters())
                curve.append({"width": width, "untied_parameters": count,
                              "absolute_difference": abs(count - target)})
                del model
            chosen = min(curve, key=lambda r: (r["absolute_difference"], r["width"]))
            rows[dataset] = {"tied128_parameters": target, "width_curve": curve,
                             "selected_width": chosen["width"],
                             "selected_untied_parameters": chosen["untied_parameters"],
                             "selected_absolute_difference": chosen["absolute_difference"]}
    finally:
        primary.WIDTH = old_width
    require(set(rows) == set(DATASETS) and all(len(v["width_curve"]) == 97 for v in rows.values()),
            "Incomplete deterministic width scan")
    OUT.mkdir(parents=True, exist_ok=False)
    primary.write_json(WIDTH_SCAN, {"protocol": PROTOCOL,
                                   "selection_rule": "minimize absolute parameter difference over all integer widths 32..128; lower width breaks ties",
                                   "original_validation_lock_sha256": PRIMARY_LOCK_SHA,
                                   "same_runtime_tied36_validation_lock_sha256": TIED36_LOCK_SHA,
                                   "graphs": rows, "width_selection_uses_test_labels": False,
                                   "loader_reads_full_arrays_for_fingerprints": True})
    print(json.dumps({"width_scan_sha256": sha(WIDTH_SCAN),
                      "selected": {d: (r["selected_width"], r["selected_untied_parameters"],
                                       r["tied128_parameters"]) for d, r in rows.items()}},
                     sort_keys=True), flush=True)


def widths() -> dict:
    report = json.loads(WIDTH_SCAN.read_text())
    require(report["protocol"] == PROTOCOL and set(report["graphs"]) == set(DATASETS) and
            report["original_validation_lock_sha256"] == PRIMARY_LOCK_SHA and
            report["same_runtime_tied36_validation_lock_sha256"] == TIED36_LOCK_SHA,
            "Frozen width scan identity differs")
    selected = {}
    for dataset in DATASETS:
        row = report["graphs"][dataset]
        curve = row["width_curve"]
        require([r["width"] for r in curve] == list(range(32, 129)) and
                all(r["absolute_difference"] == abs(r["untied_parameters"] - row["tied128_parameters"])
                    for r in curve), f"Width scan/count arithmetic differs: {dataset}")
        chosen = min(curve, key=lambda r: (r["absolute_difference"], r["width"]))
        require(row["selected_width"] == chosen["width"] and
                row["selected_untied_parameters"] == chosen["untied_parameters"] and
                row["selected_absolute_difference"] == chosen["absolute_difference"],
                f"Width choice differs: {dataset}")
        selected[dataset] = row["selected_width"]
    return selected


def expected_freeze() -> dict:
    require(primary.WIDTH == 128, "Restore original global width before freeze checks")
    original_locks()
    selected = widths()
    original = json.loads((ROOT / "FROZEN_STUDY.json").read_text())
    return {"protocol": PROTOCOL, "posthoc_after_primary_and_factor_outcomes": True,
            "primary_frozen_study_sha256": primary.check_freeze(),
            "original_validation_lock_sha256": PRIMARY_LOCK_SHA,
            "same_runtime_tied36_validation_lock_sha256": TIED36_LOCK_SHA,
            "width_scan_sha256": sha(WIDTH_SCAN), "selected_widths": selected,
            "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
            "primary_source_sha256": {name: sha(ROOT / name) for name in PRIMARY_SOURCES},
            "graphs": {d: original["graphs"][d] for d in DATASETS},
            "matrix": {"datasets": list(DATASETS), "arm": "untied", "seeds": list(SEEDS),
                       "candidates": [{"lr": lr, "weight_decay": wd} for lr, wd in CANDIDATES],
                       "default_candidate": list(DEFAULT), "epochs": primary.EPOCHS,
                       "members": primary.MEMBERS, "depth": primary.DEPTH,
                       "dropout": primary.DROP,
                       "training_function": "byte-identical original tuning.train_one with output root and frozen graph width redirected",
                       "candidate_rule": "highest three-seed mean validation accuracy, then lowest mean CE, then lower LR/decay",
                       "test_rule": "after independent 72-cell, original 432-cell, and same-runtime TIED36 validation locks; selected/default only",
                       "complete_result_target_utc": "2026-09-26T07:50:00Z"}}


def check_freeze() -> str:
    require(FREEZE.is_file() and json.loads(FREEZE.read_text()) == expected_freeze(),
            "Narrow UNTIED source/data/width matrix differs from freeze")
    return sha(FREEZE)


def preflight(device: torch.device) -> None:
    freeze_sha = check_freeze()
    selected = widths()
    checks = {}
    old_width = primary.WIDTH
    try:
        for dataset in DATASETS:
            bundle, _ = primary.load_graph(dataset, device, include_test=False)
            primary.WIDTH = selected[dataset]
            primary.seed_all(0)
            model, canonical = primary.make_model("untied", bundle, device)
            row = primary.initial_audit(model, "untied", bundle, 0, device, canonical)
            expected_count = json.loads(WIDTH_SCAN.read_text())["graphs"][dataset]["selected_untied_parameters"]
            require(row["parameter_count"] == expected_count and
                    row["paired_initial_logits_max_abs_diff"] <= primary.INITIAL_LOGIT_TOL,
                    f"Narrow UNTIED initialization/count differs: {dataset}")
            checks[dataset] = row
    finally:
        primary.WIDTH = old_width
    output = OUT / f"PREFLIGHT_{device.type.upper()}.json"
    require(not output.exists(), "Refusing preflight overwrite")
    primary.write_json(output, {"protocol": PROTOCOL, "freeze_sha256": freeze_sha,
                                "device": str(device), "graphs": checks, "status": "PASS"})
    print(json.dumps({"preflight": "PASS", "device": str(device),
                      "freeze_sha256": freeze_sha}), flush=True)


def export_validation(dataset: str, lr: float, wd: float, seed: int, bundle, device) -> None:
    cell = cell_dir(dataset, lr, wd, seed)
    record = json.loads((cell / "result.json").read_text())
    predictions = cell / "validation_companion.npz"
    manifest = cell / "validation_companion.json"
    if record["failure"] is not None:
        require(not predictions.exists() and not manifest.exists(), "Failed cell has companion")
        return
    if predictions.exists() or manifest.exists():
        require(predictions.is_file() and manifest.is_file(), "Incomplete companion")
        old = json.loads(manifest.read_text())
        require(old["checkpoint_sha256"] == record["checkpoint_sha256"] and
                old["predictions_sha256"] == sha(predictions), "Existing companion differs")
        return
    primary.seed_all(seed)
    model, _ = primary.make_model("untied", bundle, device)
    payload = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    va, vc, pooled, members = primary.evaluate(model, "untied", bundle,
                                                 bundle.valid_idx, bundle.valid_y)
    require(abs(va - record["selected_valid_accuracy"]) <= 1e-7 and
            abs(vc - record["selected_valid_ce"]) <= 1e-5,
            "Validation companion metric differs")
    np.savez_compressed(predictions, valid_indices=bundle.valid_idx.cpu().numpy(),
                        valid_labels=bundle.valid_y.cpu().numpy(),
                        valid_pooled_logits=pooled.numpy(),
                        valid_member_logits=members.numpy())
    primary.write_json(manifest, {"protocol": PROTOCOL, "dataset": dataset,
                                  "lr": lr, "weight_decay": wd, "seed": seed,
                                  "width": primary.WIDTH,
                                  "checkpoint_sha256": record["checkpoint_sha256"],
                                  "result_sha256": sha(cell / "result.json"),
                                  "predictions_sha256": sha(predictions),
                                  "valid_accuracy": va, "valid_ce": vc,
                                  "selected_pooled_logits_sha256": record["selected_valid_pooled_logits_sha256"]})


def run_dataset(dataset: str, device: torch.device) -> None:
    freeze_sha = check_freeze()
    require(dataset in DATASETS and device.type == "cuda", "Frozen CUDA dataset required")
    preflight_path = OUT / "PREFLIGHT_CUDA.json"
    require(preflight_path.is_file(), "Run CUDA preflight before training")
    preflight_report = json.loads(preflight_path.read_text())
    require(preflight_report["status"] == "PASS" and
            preflight_report["freeze_sha256"] == freeze_sha, "CUDA preflight differs")
    bundle, _ = primary.load_graph(dataset, device, include_test=False)
    old_root, old_width = primary.ROOT, primary.WIDTH
    try:
        primary.WIDTH = widths()[dataset]
        primary.ROOT = OUT
        for lr, wd in CANDIDATES:
            for seed in SEEDS:
                primary.train_one(dataset, bundle, freeze_sha, "untied", lr, wd, seed, device)
                export_validation(dataset, lr, wd, seed, bundle, device)
    finally:
        primary.ROOT, primary.WIDTH = old_root, old_width


def score(dataset: str, device: torch.device) -> None:
    freeze_sha = check_freeze()
    require(dataset in DATASETS, "Graph outside frozen sensitivity")
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    require(lock_path.is_file(), "Complete independent 72-cell validation lock required")
    lock = json.loads(lock_path.read_text())
    require(lock["protocol"] == PROTOCOL and lock["freeze_sha256"] == freeze_sha and
            len(lock["cells"]) == 72 and set(lock["selections"]) == set(DATASETS) and
            lock["original_validation_lock_sha256"] == PRIMARY_LOCK_SHA and
            lock["same_runtime_tied36_validation_lock_sha256"] == TIED36_LOCK_SHA,
            "Narrow UNTIED validation lock incomplete or different")
    original_locks()
    bundle, _ = primary.load_graph(dataset, device, include_test=True)
    selected = tuple(lock["selections"][dataset]["selected_candidate"])
    old_width = primary.WIDTH
    try:
        primary.WIDTH = widths()[dataset]
        for lr, wd in dict.fromkeys((selected, DEFAULT)):
            for seed in SEEDS:
                identity = key(dataset, lr, wd, seed)
                cell = cell_dir(dataset, lr, wd, seed)
                frozen = lock["cells"][identity]
                require(sha(cell / "result.json") == frozen["result_sha256"] and
                        sha(cell / "checkpoint.pt") == frozen["checkpoint_sha256"],
                        f"Locked narrow cell differs: {identity}")
                output = OUT / "scores" / identity
                require(not output.exists(), f"Refusing score overwrite: {identity}")
                primary.seed_all(seed)
                model, _ = primary.make_model("untied", bundle, device)
                payload = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(payload["state_dict"], strict=True)
                va, vc, vp, _ = primary.evaluate(model, "untied", bundle,
                                                  bundle.valid_idx, bundle.valid_y)
                record = json.loads((cell / "result.json").read_text())
                require(abs(va - record["selected_valid_accuracy"]) <= 1e-7 and
                        abs(vc - record["selected_valid_ce"]) <= 1e-5,
                        f"Selected validation replay differs: {identity}")
                ta, tc, tp, tm = primary.evaluate(model, "untied", bundle,
                                                  bundle.test_idx, bundle.test_y)
                output.mkdir(parents=True)
                np.savez_compressed(output / "predictions.npz",
                                    valid_pooled_logits=vp.numpy(),
                                    test_pooled_logits=tp.numpy(),
                                    test_member_logits=tm.numpy())
                primary.write_json(output / "score.json", {
                    "protocol": PROTOCOL, "freeze_sha256": freeze_sha,
                    "validation_selection_lock_sha256": sha(lock_path),
                    "original_validation_lock_sha256": PRIMARY_LOCK_SHA,
                    "same_runtime_tied36_validation_lock_sha256": TIED36_LOCK_SHA,
                    "dataset": dataset, "arm": "untied", "width": primary.WIDTH,
                    "lr": lr, "weight_decay": wd, "seed": seed,
                    "selected_candidate": (lr, wd) == selected,
                    "predeclared_default": (lr, wd) == DEFAULT,
                    "checkpoint_sha256": frozen["checkpoint_sha256"],
                    "valid_accuracy": va, "valid_ce": vc,
                    "test_accuracy": ta, "test_ce": tc,
                    "test_predictions_sha256": sha(output / "predictions.npz")})
                print(json.dumps({"scored": identity, "test_accuracy": ta}), flush=True)
    finally:
        primary.WIDTH = old_width


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scan-widths")
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
    if args.command == "scan-widths":
        scan_widths()
    elif args.command == "freeze":
        require(not FREEZE.exists(), "Refusing to overwrite prospective narrow freeze")
        primary.write_json(FREEZE, expected_freeze())
        print(json.dumps({"freeze_sha256": sha(FREEZE)}), flush=True)
    elif args.command == "check-freeze":
        print(json.dumps({"freeze_sha256": check_freeze()}), flush=True)
    elif args.command == "preflight":
        preflight(torch.device(args.device))
    elif args.command == "run":
        run_dataset(args.dataset, torch.device(args.device))
    else:
        score(args.dataset, torch.device(args.device))


if __name__ == "__main__":
    main()
