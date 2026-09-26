"""Independent fail-closed audit of the frozen OGBL-Collab extension.

Place next to ogbl_collab_frozen.py under experiments_iclr. This verifier never
chooses a test result. It checks all required seed/arm artifacts and independently
recomputes validation selection and OGB Hits@50 from saved raw member scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import ogbl_collab_frozen as study

REQUIRED = ("tied", "untied", "ens")
SEEDS = (0, 1, 2)
EXPECTED_STEPS = list(range(20, 401, 20))
ATOL = 2e-4


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def independent_metrics(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    require(positive.ndim == 1 and negative.ndim == 1, "Scores must be vectors")
    require(len(positive) > 0 and len(negative) >= 50, "Empty or short score vectors")
    require(np.isfinite(positive).all() and np.isfinite(negative).all(), "Nonfinite score")
    threshold = np.sort(negative)[-50]
    hits = float(np.count_nonzero(positive > threshold) / len(positive))
    bce = float((np.logaddexp(0, -positive).mean() +
                 np.logaddexp(0, negative).mean()) / 2)
    return hits, bce


def close(got: float, expected: float, name: str, tol: float = 1e-7) -> None:
    require(math.isfinite(got) and math.isfinite(expected), f"Nonfinite {name}")
    require(abs(got - expected) <= tol, f"{name}: got {got}, expected {expected}")


def independently_selected(history: list[dict[str, str]]) -> tuple[int, float, float]:
    best_step, best_hits, best_bce = 0, -1.0, float("inf")
    for row in history:
        step = int(row["step"])
        hits = float(row["valid_pooled_hits50"])
        bce = float(row["valid_pooled_bce"])
        improved = hits > best_hits + 1e-12 or (
            abs(hits - best_hits) <= 1e-12 and bce < best_bce - 1e-12)
        require(int(row["selected_so_far"]) == int(improved),
                f"Wrong selected_so_far at step {step}")
        if improved:
            best_step, best_hits, best_bce = step, hits, bce
    return best_step, best_hits, best_bce


def audit_run(result_root: Path, variant: str, seed: int,
              actual_data_hashes: dict[str, str] | None,
              bundle: study.DataBundle | None, replay_cpu: bool) -> dict[str, object]:
    run_dir = result_root / variant / f"seed_{seed}"
    require(run_dir.is_dir(), f"Missing required run {run_dir}")
    paths = {name: run_dir / name for name in (
        "run_config.json", "history.csv", "selected_checkpoint.pt",
        "selected_predictions.npz", "selected.json")}
    for name, path in paths.items():
        require(path.is_file() and path.stat().st_size > 0,
                f"Missing or empty {variant}/{seed}/{name}")
    config = json.loads(paths["run_config.json"].read_text())
    selected = json.loads(paths["selected.json"].read_text())
    require(config["protocol"] == study.PROTOCOL == selected["protocol"], "Protocol mismatch")
    require(config["variant"] == selected["variant"] == variant, "Variant mismatch")
    require(config["seed"] == selected["seed"] == seed, "Seed mismatch")
    for key, expected in (("steps", 400), ("eval_every", 20), ("pairs_per_step", 65536),
                          ("width", 128), ("depth", 2), ("dropout", 0.2),
                          ("lr", 0.001), ("weight_decay", 0.0)):
        require(config[key] == expected, f"Frozen config {key} changed")
    require(config["message_graph"] == "official train edges only for all stages",
            "Message graph rule changed")
    require(config["source"]["runner_sha256"] == file_sha256(Path(study.__file__)),
            "Runner source changed since this run")
    require(config["source"]["models_sha256"] == file_sha256(study.ROOT / "models.py"),
            "Model source changed since this run")
    if actual_data_hashes is not None:
        require(config["data_fingerprints"] == actual_data_hashes, "Data split hash changed")
    require(selected["steps_run"] == 400, "Run did not finish 400 steps")
    for name, key in (("history.csv", "history_sha256"),
                      ("selected_checkpoint.pt", "checkpoint_sha256"),
                      ("selected_predictions.npz", "predictions_sha256")):
        require(selected[key] == file_sha256(paths[name]), f"Corrupt {name}")
    with paths["history.csv"].open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == study.HISTORY_COLUMNS, "History columns changed")
        history = list(reader)
    require([int(row["step"]) for row in history] == EXPECTED_STEPS,
            "History must have 20 contiguous scheduled validation points")
    for row in history:
        for name in study.HISTORY_COLUMNS:
            require(math.isfinite(float(row[name])), f"Nonfinite history {name}")
    chosen_step, chosen_hits, chosen_bce = independently_selected(history)
    require(selected["selected_step"] == chosen_step, "Checkpoint choice does not replay")
    close(selected["valid_hits50"], chosen_hits, "selected validation Hits@50")
    close(selected["valid_bce"], chosen_bce, "selected validation BCE", 1e-6)
    checkpoint = torch.load(paths["selected_checkpoint.pt"], map_location="cpu", weights_only=False)
    require(checkpoint["protocol"] == study.PROTOCOL, "Checkpoint protocol mismatch")
    require(checkpoint["step"] == chosen_step and checkpoint["variant"] == variant and
            checkpoint["seed"] == seed, "Checkpoint identity mismatch")
    model = study.LinkSystem(variant, torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    require(sum(p.numel() for p in model.parameters()) == config["parameter_count"] ==
            selected["parameter_count"], "Parameter count changed")
    model.eval()
    with np.load(paths["selected_predictions.npz"], allow_pickle=False) as archive:
        expected_keys = {
            "valid_pos_edges", "valid_neg_edges", "test_pos_edges", "test_neg_edges",
            "valid_pos_member_logits", "valid_neg_member_logits",
            "test_pos_member_logits", "test_neg_member_logits",
        }
        require(set(archive.files) == expected_keys, "Prediction archive schema changed")
        saved = {key: archive[key] for key in archive.files}
    member_count = 1 if variant == "base" else 4
    per_stage = {}
    for stage in ("valid", "test"):
        pos_edges = saved[f"{stage}_pos_edges"]
        neg_edges = saved[f"{stage}_neg_edges"]
        pos = saved[f"{stage}_pos_member_logits"]
        neg = saved[f"{stage}_neg_member_logits"]
        require(pos_edges.ndim == 2 and neg_edges.ndim == 2 and
                pos_edges.shape[1] == neg_edges.shape[1] == 2,
                f"Bad {stage} pair array shapes")
        require(pos.shape == (member_count, len(pos_edges)) and
                neg.shape == (member_count, len(neg_edges)),
                f"Bad {stage} member score shapes")
        require(np.isfinite(pos).all() and np.isfinite(neg).all(),
                f"Nonfinite {stage} member score")
        if bundle is not None:
            for kind in ("pos", "neg"):
                split_key = "edge" if kind == "pos" else "edge_neg"
                require(np.array_equal(saved[f"{stage}_{kind}_edges"],
                                       bundle.split[stage][split_key].numpy()),
                        f"{stage} {kind} edges differ from official split")
        pooled_pos = np.mean(pos, axis=0)
        pooled_neg = np.mean(neg, axis=0)
        pooled_hits, pooled_bce = independent_metrics(pooled_pos, pooled_neg)
        close(selected[f"{stage}_hits50"], pooled_hits, f"{stage} Hits@50")
        close(selected[f"{stage}_bce"], pooled_bce, f"{stage} BCE", 1e-6)
        for member in range(member_count):
            member_hits, _ = independent_metrics(pos[member], neg[member])
            close(selected[f"{stage}_member_hits50"][member], member_hits,
                  f"{stage} member {member} Hits@50")
        per_stage[stage] = {"hits50": pooled_hits, "bce": pooled_bce}
    if replay_cpu:
        require(bundle is not None, "CPU replay requires the OGB dataset")
        torch.set_num_threads(min(torch.get_num_threads(), 8))
        graph = bundle.graph
        x = bundle.x
        with torch.no_grad():
            for member in range(member_count):
                embeddings = model.node_embeddings(graph, x, member)
                for kind in ("pos", "neg"):
                    pair_key = f"valid_{kind}_edges"
                    saved_key = f"valid_{kind}_member_logits"
                    pairs = torch.from_numpy(saved[pair_key][:64].astype(np.int64))
                    replayed = model.scores(embeddings, pairs, member).cpu().numpy()
                    stored = saved[saved_key][member, :64]
                    require(np.allclose(replayed, stored, rtol=ATOL, atol=ATOL),
                            f"CPU replay mismatch for {variant}/{seed}/{kind}/member{member}")
    return {"variant": variant, "seed": seed, "selected_step": chosen_step,
            "valid": per_stage["valid"], "test": per_stage["test"],
            "parameter_count": config["parameter_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen OGBL-Collab artifacts")
    parser.add_argument("--result-root", default="experiments_iclr/ogbl_collab_results")
    parser.add_argument("--data-root", default="data/ogb")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--replay-cpu", action="store_true")
    parser.add_argument("--audit-output", default="experiments_iclr/ogbl_collab_results/audit.json")
    args = parser.parse_args()
    result_root = study.repo_path(args.result_root)
    require(result_root.is_dir(), "No result root")
    variant_names = REQUIRED + (("base",) if args.include_base else ())
    bundle = study.DataBundle(study.repo_path(args.data_root), torch.device("cpu"))
    data_hashes = bundle.fingerprints
    rows = [audit_run(result_root, variant, seed, data_hashes, bundle, args.replay_cpu)
            for variant in variant_names for seed in SEEDS]
    require(len(rows) == 3 * len(variant_names), "Wrong number of audited arms")
    summary = {"protocol": study.PROTOCOL, "required_pairs": len(rows), "runs": rows,
               "paired_tied_minus_untied_test_hits50": [
                   next(r for r in rows if r["variant"] == "tied" and r["seed"] == seed)["test"]["hits50"] -
                   next(r for r in rows if r["variant"] == "untied" and r["seed"] == seed)["test"]["hits50"]
                   for seed in SEEDS],
               "cpu_replay": args.replay_cpu, "data_fingerprints": data_hashes}
    audit_output = study.repo_path(args.audit_output)
    require(not audit_output.exists(), f"Audit output already exists: {audit_output}")
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = audit_output.with_suffix(audit_output.suffix + ".tmp")
    temp_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temp_output.replace(audit_output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
