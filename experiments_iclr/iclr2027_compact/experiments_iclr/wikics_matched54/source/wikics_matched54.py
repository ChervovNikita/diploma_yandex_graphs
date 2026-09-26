"""Frozen post hoc WikiCS parameter-matched 54-cell comparator study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import tuning as primary


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "wikics_matched54_results"
FREEZE = OUT / "FROZEN_WIKICS_MATCHED54.json"
PROTOCOL = "wikics_matched54_posthoc_v1"
ARMS = ("base", "ens", "private_last")
SEEDS = (0, 1, 2)
CANDIDATES = primary.CANDIDATES
DEFAULT = primary.DEFAULT
WIDTHS = {"base": 198, "ens": 92, "private_last": 128}
PARAMETER_COUNTS = {"base": 456004, "ens": 457464, "private_last": 455552}
TARGET = 455552
SOURCES = ("wikics_matched54.py", "verify_wikics_matched54.py",
           "WIKICS_MATCHED54_PROTOCOL.md", "WIKICS_MATCHED54_WIDTH_SCAN.json")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def key(arm, lr, wd, seed):
    return f"wikics/{arm}/{primary.candidate_name(lr, wd)}/seed{seed}"


def cell_dir(arm, lr, wd, seed):
    return OUT / "results" / key(arm, lr, wd, seed)


def width_scan():
    def base(w):
        return 10 * w * w + 323 * w + 10
    base_rows = sorted((abs(base(w) - TARGET), w, base(w)) for w in range(32, 513))
    ens_rows = sorted((abs(4 * base(w) - TARGET), w, 4 * base(w)) for w in range(32, 129))
    require(base_rows[0] == (452, 198, 456004) and
            ens_rows[0] == (1912, 92, 457464), "Width scan changed")
    return {"protocol": PROTOCOL, "criterion": "minimum absolute stored parameter-count gap, then lower width; no outcomes used",
            "private_last_width": 128, "private_last_parameters": TARGET,
            "base_formula": "10*w*w+323*w+10", "base_range": [32, 512],
            "base_selected_width": 198, "base_parameters": 456004, "base_gap": 452,
            "ens_formula": "4*(10*w*w+323*w+10)", "ens_range": [32, 128],
            "ens_selected_width": 92, "ens_parameters": 457464, "ens_gap": 1912}


def expected_freeze():
    require(primary.WIDTH == 128, "Original width must be restored")
    require(json.loads((ROOT / "WIKICS_MATCHED54_WIDTH_SCAN.json").read_text()) == width_scan(),
            "Parameter-only width scan differs")
    original = json.loads((ROOT / "FROZEN_STUDY.json").read_text())
    original_preflight = json.loads((ROOT / "preflight.json").read_text())
    require(original_preflight["graphs"]["wikics"]["parameter_counts"]["private_last"] == TARGET,
            "Original private-last target differs")
    return {"protocol": PROTOCOL, "posthoc_after_primary_outcomes": True,
            "original_frozen_study_sha256": primary.check_freeze(),
            "original_validation_lock_sha256": sha(ROOT / "VALIDATION_SELECTION_LOCK.json"),
            "primary_source_sha256": {n: sha(ROOT / n) for n in ("tuning.py", "models.py", "verify_tuning.py", "FROZEN_STUDY.json")},
            "source_sha256": {n: sha(ROOT / n) for n in SOURCES},
            "graph": original["graphs"]["wikics"],
            "matrix": {"dataset": "wikics", "arms": list(ARMS), "widths": WIDTHS,
                       "parameter_counts": PARAMETER_COUNTS,
                       "seeds": list(SEEDS), "candidates": [{"lr": lr, "weight_decay": wd} for lr, wd in CANDIDATES],
                       "default_candidate": list(DEFAULT), "epochs": primary.EPOCHS,
                       "members": primary.MEMBERS, "depth": primary.DEPTH, "dropout": primary.DROP,
                       "training_function": "original tuning.train_one; only ROOT and WIDTH redirected",
                       "candidate_rule": "highest mean selected validation accuracy, then lowest mean CE, then lower LR/decay",
                       "test_rule": "after complete independent 54-cell validation lock; selected/default only",
                       "complete_audit_cutoff_utc": "2026-09-26T08:15:00Z"}}


def check_freeze():
    require(FREEZE.is_file() and json.loads(FREEZE.read_text()) == expected_freeze(),
            "Matched54 source/data/matrix differs from freeze")
    return sha(FREEZE)


def with_width(arm):
    require(arm in ARMS, "Arm not frozen")
    primary.WIDTH = WIDTHS[arm]


def preflight(device):
    freeze_sha = check_freeze()
    bundle, _ = primary.load_graph("wikics", device, include_test=False)
    rows = {}
    for arm in ARMS:
        with_width(arm)
        for seed in SEEDS:
            primary.seed_all(seed)
            model, canonical = primary.make_model(arm, bundle, device)
            row = primary.initial_audit(model, arm, bundle, seed, device, canonical)
            require(row["parameter_count"] == PARAMETER_COUNTS[arm],
                    f"Parameter mismatch: {arm}/seed{seed}")
            rows[f"{arm}/seed{seed}"] = row
        primary.WIDTH = 128
    path = OUT / f"PREFLIGHT_{device.type.upper()}.json"
    require(not path.exists(), "Preflight overwrite refused")
    primary.write_json(path, {"protocol": PROTOCOL, "freeze_sha256": freeze_sha,
                              "device": str(device), "rows": rows, "status": "PASS"})
    print(json.dumps({"preflight": "PASS", "freeze_sha256": freeze_sha}), flush=True)


def export_validation(arm, lr, wd, seed, bundle, device):
    cell = cell_dir(arm, lr, wd, seed)
    record = json.loads((cell / "result.json").read_text())
    path, manifest = cell / "validation_companion.npz", cell / "validation_companion.json"
    if record["failure"] is not None:
        require(not path.exists() and not manifest.exists(), "Failed cell has companion")
        return
    if path.exists() or manifest.exists():
        require(path.is_file() and manifest.is_file() and
                json.loads(manifest.read_text())["predictions_sha256"] == sha(path),
                "Existing companion changed")
        return
    primary.seed_all(seed)
    model, _ = primary.make_model(arm, bundle, device)
    checkpoint = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    va, vc, pooled, members = primary.evaluate(model, arm, bundle, bundle.valid_idx, bundle.valid_y)
    require(abs(va - record["selected_valid_accuracy"]) <= 1e-7 and
            abs(vc - record["selected_valid_ce"]) <= 1e-5, "Validation companion metric differs")
    np.savez_compressed(path, valid_indices=bundle.valid_idx.cpu().numpy(),
                        valid_labels=bundle.valid_y.cpu().numpy(),
                        valid_pooled_logits=pooled.numpy(), valid_member_logits=members.numpy())
    primary.write_json(manifest, {"protocol": PROTOCOL, "arm": arm, "width": WIDTHS[arm],
                                  "lr": lr, "weight_decay": wd, "seed": seed,
                                  "checkpoint_sha256": record["checkpoint_sha256"],
                                  "result_sha256": sha(cell / "result.json"),
                                  "predictions_sha256": sha(path), "valid_accuracy": va, "valid_ce": vc})


def run(arm, device, partition):
    freeze_sha = check_freeze()
    require(device.type == "cuda", "CUDA training required")
    preflight_path = OUT / "PREFLIGHT_CUDA.json"
    require(preflight_path.is_file() and
            json.loads(preflight_path.read_text())["freeze_sha256"] == freeze_sha,
            "Frozen CUDA preflight required")
    bundle, _ = primary.load_graph("wikics", device, include_test=False)
    saved_root, saved_width = primary.ROOT, primary.WIDTH
    try:
        primary.ROOT = OUT
        with_width(arm)
        for candidate_index, (lr, wd) in enumerate(CANDIDATES):
            for seed in SEEDS:
                if (candidate_index * len(SEEDS) + seed) % 2 != partition:
                    continue
                primary.train_one("wikics", bundle, freeze_sha, arm, lr, wd, seed, device)
                export_validation(arm, lr, wd, seed, bundle, device)
    finally:
        primary.ROOT, primary.WIDTH = saved_root, saved_width


def score(device):
    freeze_sha = check_freeze()
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    require(lock_path.is_file(), "Complete validation lock required")
    lock = json.loads(lock_path.read_text())
    require(lock["protocol"] == PROTOCOL and lock["freeze_sha256"] == freeze_sha and
            len(lock["cells"]) == 54 and set(lock["selections"]) == set(ARMS),
            "Validation lock differs")
    bundle, _ = primary.load_graph("wikics", device, include_test=True)
    saved_width = primary.WIDTH
    try:
        for arm in ARMS:
            with_width(arm)
            selected = tuple(lock["selections"][arm]["selected_candidate"])
            for lr, wd in dict.fromkeys((selected, DEFAULT)):
                for seed in SEEDS:
                    identity = key(arm, lr, wd, seed)
                    cell = cell_dir(arm, lr, wd, seed)
                    frozen = lock["cells"][identity]
                    require(sha(cell / "checkpoint.pt") == frozen["checkpoint_sha256"] and
                            sha(cell / "result.json") == frozen["result_sha256"],
                            f"Locked cell differs: {identity}")
                    output = OUT / "scores" / identity
                    require(not output.exists(), f"Score overwrite refused: {identity}")
                    primary.seed_all(seed)
                    model, _ = primary.make_model(arm, bundle, device)
                    payload = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
                    model.load_state_dict(payload["state_dict"], strict=True)
                    va, vc, vp, _ = primary.evaluate(model, arm, bundle, bundle.valid_idx, bundle.valid_y)
                    ta, tc, tp, tm = primary.evaluate(model, arm, bundle, bundle.test_idx, bundle.test_y)
                    output.mkdir(parents=True)
                    np.savez_compressed(output / "predictions.npz", valid_pooled_logits=vp.numpy(),
                                        test_pooled_logits=tp.numpy(), test_member_logits=tm.numpy())
                    primary.write_json(output / "score.json", {
                        "protocol": PROTOCOL, "freeze_sha256": freeze_sha,
                        "validation_selection_lock_sha256": sha(lock_path),
                        "dataset": "wikics", "arm": arm, "width": WIDTHS[arm],
                        "lr": lr, "weight_decay": wd, "seed": seed,
                        "selected_candidate": (lr, wd) == selected,
                        "predeclared_default": (lr, wd) == DEFAULT,
                        "checkpoint_sha256": frozen["checkpoint_sha256"],
                        "valid_accuracy": va, "valid_ce": vc,
                        "test_accuracy": ta, "test_ce": tc,
                        "test_predictions_sha256": sha(output / "predictions.npz")})
                    print(json.dumps({"scored": identity, "test_accuracy": ta}), flush=True)
    finally:
        primary.WIDTH = saved_width


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    commands.add_parser("check-freeze")
    commands.add_parser("preflight").add_argument("--device", default="cuda:0")
    run_command = commands.add_parser("run")
    run_command.add_argument("--arm", choices=ARMS, required=True)
    run_command.add_argument("--partition", type=int, choices=(0, 1), required=True)
    run_command.add_argument("--device", default="cuda:0")
    commands.add_parser("score").add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not FREEZE.exists(), "Freeze overwrite refused")
        OUT.mkdir(parents=True, exist_ok=True)
        primary.write_json(FREEZE, expected_freeze())
        print(json.dumps({"freeze_sha256": sha(FREEZE)}), flush=True)
    elif args.command == "check-freeze":
        print(json.dumps({"freeze_sha256": check_freeze()}), flush=True)
    elif args.command == "run":
        run(args.arm, torch.device(args.device), args.partition)
    elif args.command == "preflight":
        preflight(torch.device(args.device))
    else:
        score(torch.device(args.device))


if __name__ == "__main__":
    main()
