"""Exploratory same-runtime width-128 UNTIED control for WikiCS split 0."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import narrow_untied72 as narrow
import tuning as primary


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "wikics_wide_untied18_results"
FREEZE = OUT / "FROZEN_WIKICS_WIDE_UNTIED18_STUDY.json"
PROTOCOL = "wikics_wide_untied18_same_runtime_v1"
NARROW_FREEZE_SHA = "ed987170441ddddd899ab50711e9a977c73b794537186014dd8736163ecfb87a"
DATASET = "wikics"
SEEDS = (0, 1, 2)
CANDIDATES = primary.CANDIDATES
DEFAULT = primary.DEFAULT
SOURCES = ("wikics_wide_untied18.py", "verify_wikics_wide_untied18.py",
           "WIKICS_WIDE_UNTIED18_PROTOCOL.md")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def key(lr, wd, seed) -> str:
    return f"wikics/untied/{primary.candidate_name(lr, wd)}/seed{seed}"


def cell_dir(lr, wd, seed) -> Path:
    return OUT / "results" / key(lr, wd, seed)


def expected_freeze() -> dict:
    require(primary.WIDTH == 128, "Original width128 required")
    narrow.original_locks()
    require(narrow.check_freeze() == NARROW_FREEZE_SHA,
            "Frozen narrow sensitivity differs")
    original = json.loads((ROOT / "FROZEN_STUDY.json").read_text())
    return {"protocol": PROTOCOL, "posthoc_after_primary_and_narrow_design": True,
            "primary_frozen_study_sha256": primary.check_freeze(),
            "original_validation_lock_sha256": narrow.PRIMARY_LOCK_SHA,
            "same_runtime_tied36_validation_lock_sha256": narrow.TIED36_LOCK_SHA,
            "narrow_untied72_freeze_sha256": NARROW_FREEZE_SHA,
            "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
            "primary_source_sha256": {name: sha(ROOT / name) for name in
                                      ("tuning.py", "models.py", "verify_tuning.py", "FROZEN_STUDY.json")},
            "graph": original["graphs"][DATASET],
            "matrix": {"dataset": DATASET, "arm": "untied", "width": 128,
                       "seeds": list(SEEDS),
                       "candidates": [{"lr": lr, "weight_decay": wd} for lr, wd in CANDIDATES],
                       "default_candidate": list(DEFAULT), "epochs": primary.EPOCHS,
                       "members": primary.MEMBERS, "depth": primary.DEPTH, "dropout": primary.DROP,
                       "training_function": "byte-identical original tuning.train_one with result root redirected",
                       "candidate_rule": "highest three-seed mean validation accuracy, then lowest mean CE, then lower LR/decay",
                       "test_rule": "after complete 18-cell wide and 72-cell narrow validation locks; selected/default only",
                       "complete_result_target_utc": "2026-09-26T07:50:00Z"}}


def check_freeze() -> str:
    require(FREEZE.is_file() and json.loads(FREEZE.read_text()) == expected_freeze(),
            "Wide WikiCS source/data/matrix differs from freeze")
    return sha(FREEZE)


def preflight(device: torch.device) -> None:
    freeze_sha = check_freeze()
    bundle, _ = primary.load_graph(DATASET, device, include_test=False)
    rows = {}
    for seed in SEEDS:
        primary.seed_all(seed)
        model, canonical = primary.make_model("untied", bundle, device)
        row = primary.initial_audit(model, "untied", bundle, seed, device, canonical)
        require(row["parameter_count"] == 703232 and
                row["paired_initial_logits_max_abs_diff"] <= primary.INITIAL_LOGIT_TOL,
                f"Wide WikiCS initialization differs: seed{seed}")
        rows[f"seed{seed}"] = row
    output = OUT / f"PREFLIGHT_{device.type.upper()}.json"
    require(not output.exists(), "Refusing preflight overwrite")
    primary.write_json(output, {"protocol": PROTOCOL, "freeze_sha256": freeze_sha,
                                "device": str(device), "rows": rows, "status": "PASS"})
    print(json.dumps({"preflight": "PASS", "freeze_sha256": freeze_sha}), flush=True)


def export_validation(lr, wd, seed, bundle, device) -> None:
    cell = cell_dir(lr, wd, seed)
    result = json.loads((cell / "result.json").read_text())
    predictions = cell / "validation_companion.npz"
    manifest = cell / "validation_companion.json"
    if result["failure"] is not None:
        require(not predictions.exists() and not manifest.exists(), "Failed cell has companion")
        return
    if predictions.exists() or manifest.exists():
        require(predictions.is_file() and manifest.is_file(), "Incomplete companion")
        old = json.loads(manifest.read_text())
        require(old["checkpoint_sha256"] == result["checkpoint_sha256"] and
                old["predictions_sha256"] == sha(predictions), "Existing companion differs")
        return
    primary.seed_all(seed)
    model, _ = primary.make_model("untied", bundle, device)
    checkpoint = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    va, vc, pooled, members = primary.evaluate(model, "untied", bundle,
                                                 bundle.valid_idx, bundle.valid_y)
    require(abs(va - result["selected_valid_accuracy"]) <= 1e-7 and
            abs(vc - result["selected_valid_ce"]) <= 1e-5,
            "Wide WikiCS validation companion metric differs")
    np.savez_compressed(predictions, valid_indices=bundle.valid_idx.cpu().numpy(),
                        valid_labels=bundle.valid_y.cpu().numpy(),
                        valid_pooled_logits=pooled.numpy(),
                        valid_member_logits=members.numpy())
    primary.write_json(manifest, {"protocol": PROTOCOL, "lr": lr, "weight_decay": wd,
                                  "seed": seed, "checkpoint_sha256": result["checkpoint_sha256"],
                                  "result_sha256": sha(cell / "result.json"),
                                  "predictions_sha256": sha(predictions),
                                  "valid_accuracy": va, "valid_ce": vc,
                                  "selected_pooled_logits_sha256": result["selected_valid_pooled_logits_sha256"]})


def run(device: torch.device) -> None:
    freeze_sha = check_freeze()
    require(device.type == "cuda", "Frozen wide WikiCS CUDA run required")
    preflight_path = OUT / "PREFLIGHT_CUDA.json"
    require(preflight_path.is_file() and
            json.loads(preflight_path.read_text())["freeze_sha256"] == freeze_sha,
            "CUDA preflight required")
    bundle, _ = primary.load_graph(DATASET, device, include_test=False)
    old_root = primary.ROOT
    try:
        primary.ROOT = OUT
        for lr, wd in CANDIDATES:
            for seed in SEEDS:
                primary.train_one(DATASET, bundle, freeze_sha, "untied", lr, wd, seed, device)
                export_validation(lr, wd, seed, bundle, device)
    finally:
        primary.ROOT = old_root


def score(device: torch.device) -> None:
    freeze_sha = check_freeze()
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    narrow_lock_path = narrow.OUT / "VALIDATION_SELECTION_LOCK.json"
    require(lock_path.is_file() and narrow_lock_path.is_file(),
            "Both complete wide18 and narrow72 validation locks required")
    lock = json.loads(lock_path.read_text())
    narrow_lock = json.loads(narrow_lock_path.read_text())
    require(lock["protocol"] == PROTOCOL and lock["freeze_sha256"] == freeze_sha and
            len(lock["cells"]) == 18 and lock["narrow_validation_lock_sha256"] == sha(narrow_lock_path) and
            len(narrow_lock["cells"]) == 72 and narrow_lock["freeze_sha256"] == NARROW_FREEZE_SHA,
            "Validation locks incomplete or different")
    narrow.original_locks()
    bundle, _ = primary.load_graph(DATASET, device, include_test=True)
    selected = tuple(lock["selection"]["selected_candidate"])
    for lr, wd in dict.fromkeys((selected, DEFAULT)):
        for seed in SEEDS:
            identity = key(lr, wd, seed)
            cell = cell_dir(lr, wd, seed)
            frozen = lock["cells"][identity]
            require(sha(cell / "result.json") == frozen["result_sha256"] and
                    sha(cell / "checkpoint.pt") == frozen["checkpoint_sha256"],
                    f"Wide WikiCS locked cell differs: {identity}")
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
                    f"Wide validation replay differs: {identity}")
            ta, tc, tp, tm = primary.evaluate(model, "untied", bundle,
                                              bundle.test_idx, bundle.test_y)
            output.mkdir(parents=True)
            np.savez_compressed(output / "predictions.npz",
                                valid_pooled_logits=vp.numpy(), test_pooled_logits=tp.numpy(),
                                test_member_logits=tm.numpy())
            primary.write_json(output / "score.json", {
                "protocol": PROTOCOL, "freeze_sha256": freeze_sha,
                "validation_selection_lock_sha256": sha(lock_path),
                "narrow_validation_lock_sha256": sha(narrow_lock_path),
                "original_validation_lock_sha256": narrow.PRIMARY_LOCK_SHA,
                "same_runtime_tied36_validation_lock_sha256": narrow.TIED36_LOCK_SHA,
                "dataset": DATASET, "arm": "untied", "width": 128,
                "lr": lr, "weight_decay": wd, "seed": seed,
                "selected_candidate": (lr, wd) == selected,
                "predeclared_default": (lr, wd) == DEFAULT,
                "checkpoint_sha256": frozen["checkpoint_sha256"],
                "valid_accuracy": va, "valid_ce": vc,
                "test_accuracy": ta, "test_ce": tc,
                "test_predictions_sha256": sha(output / "predictions.npz")})
            print(json.dumps({"scored": identity, "test_accuracy": ta}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    commands.add_parser("check-freeze")
    commands.add_parser("preflight").add_argument("--device", default="cuda:0")
    commands.add_parser("run").add_argument("--device", default="cuda:0")
    commands.add_parser("score").add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not FREEZE.exists(), "Refusing wide WikiCS freeze overwrite")
        OUT.mkdir(parents=True, exist_ok=False)
        primary.write_json(FREEZE, expected_freeze())
        print(json.dumps({"freeze_sha256": sha(FREEZE)}), flush=True)
    elif args.command == "check-freeze":
        print(json.dumps({"freeze_sha256": check_freeze()}), flush=True)
    elif args.command == "preflight":
        preflight(torch.device(args.device))
    elif args.command == "run":
        run(torch.device(args.device))
    else:
        score(torch.device(args.device))


if __name__ == "__main__":
    main()
