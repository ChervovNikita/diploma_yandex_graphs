"""Independent NumPy-only verification for anonymous ogbn-arxiv decisions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ("tied", "untied_propagation", "private_first", "private_last")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    h.update(str(value.dtype).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def require(ok, message: str):
    if not ok:
        raise RuntimeError(message)


def check_trace(path: Path, row: dict):
    with path.open(newline="") as stream:
        lines = list(csv.DictReader(stream))
    require(len(lines) == 300, f"Trace length: {path}")
    best_acc, best_ce, best_epoch = -1.0, float("inf"), 0
    for i, line in enumerate(lines, 1):
        require(int(line["epoch"]) == i, f"Trace order: {path}")
        acc, ce = float(line["valid_accuracy"]), float(line["valid_ce"])
        better = (acc > best_acc + 1e-12 or
                  (abs(acc - best_acc) <= 1e-12 and ce < best_ce - 1e-12))
        require(int(line["improved"]) == int(better), f"Trace flag: {path}")
        if better:
            best_acc, best_ce, best_epoch = acc, ce, i
    require(best_epoch == row["selected_epoch"] and
            abs(best_acc - row["valid_accuracy"]) < 1e-6 and
            abs(best_ce - row["valid_ce"]) < 1e-5,
            f"Selected validation epoch/score: {path}")


def main():
    base = ROOT / "results/ogbn_arxiv"
    manifest_path = base / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((base / "completion_audit.json").read_text())
    derivations = json.loads((ROOT / "decision_derivation_manifest.json").read_text())
    require(audit["status"] == "COMPLETE_CUDA_REPLAY_PASS" and
            len(audit["records"]) == 12, "Full CUDA audit")
    require(audit["source_manifest_sha256"] == sha(manifest_path), "Source manifest")
    require(manifest["protocol"] == "ogbn_arxiv_sharing_boundary_official_time_v1",
            "Protocol identity")
    require(manifest["data_descriptor"] == json.loads((ROOT /
            "experiments_iclr/ogbn_arxiv_reference_dataset_manifest.json").read_text()),
            "Official OGB data descriptor")
    for relative, digest in manifest["source_sha256"].items():
        if relative == "data/ogb/ogbn_arxiv_official.zip":
            require(len(digest) == 64, "Public archive hash syntax")
        else:
            require(sha(ROOT / relative) == digest, f"Source hash: {relative}")
    require({p.name for p in base.iterdir() if p.is_dir()} ==
            {"seed0", "seed1", "seed2"}, "Seed directory set")
    descriptor = manifest["data_descriptor"]
    scores = {}
    for seed in range(3):
        seed_dir = base / f"seed{seed}"
        require({p.name for p in seed_dir.iterdir() if p.is_dir()} == set(ARMS),
                f"Arm directory set: {seed_dir}")
        tied = json.loads((seed_dir / "tied/initialization.json").read_text())
        for arm in ARMS:
            run = seed_dir / arm
            relative = str(run.relative_to(ROOT))
            row = json.loads((run / "result.json").read_text())
            require(row["dataset"] == "ogbn_arxiv" and
                    row["optimization_seed"] == seed and row["arm"] == arm and
                    row["epochs_run"] == 300 and
                    row["source_manifest_sha256"] == sha(manifest_path),
                    f"Run identity: {relative}")
            for filename in ("initialization.json", "validation_trace.csv"):
                require(sha(run / filename) == row["artifact_sha256"][filename],
                        f"Retained artifact hash: {relative}/{filename}")
            init = json.loads((run / "initialization.json").read_text())
            for key in ("canonical_tied_state_sha256", "cpu_rng_sha256",
                        "cuda_rng_sha256"):
                require(init[key] == tied[key], f"Matched initialization: {relative}")
            require(init["paired_initial_logits_max_abs_diff"] <= 1e-5,
                    f"Initial member equality: {relative}")
            check_trace(run / "validation_trace.csv", row)
            provenance = derivations[relative]
            require(provenance["source_selected_logits_sha256"] ==
                    row["artifact_sha256"]["selected_predictions.npz"],
                    f"Source selected-logit hash: {relative}")
            require(sha(run / "selected_decisions.npz") ==
                    provenance["selected_decisions_sha256"],
                    f"Selected-decision hash: {relative}")
            with np.load(run / "selected_decisions.npz", allow_pickle=False) as f:
                decisions = {name: f[name].copy() for name in f.files}
            expected_keys = {f"{part}_{field}" for part in ("valid", "test")
                             for field in ("member_pred", "pooled_pred", "labels", "indices")}
            require(set(decisions) == expected_keys, f"Decision schema: {relative}")
            full = next(r for r in audit["records"] if r["seed"] == seed and
                        r["arm"] == arm)
            for part in ("valid", "test"):
                count = descriptor["split_sizes"][part]
                members = decisions[f"{part}_member_pred"]
                pooled = decisions[f"{part}_pooled_pred"]
                labels = decisions[f"{part}_labels"]
                indices = decisions[f"{part}_indices"]
                require(members.shape == (4, count) and pooled.shape == (count,) and
                        labels.shape == (count,) and indices.shape == (count,),
                        f"Decision shapes: {relative}:{part}")
                require(all(int(v.min()) >= 0 and int(v.max()) < descriptor["num_classes"]
                            for v in (members, pooled, labels)),
                        f"Decision class range: {relative}:{part}")
                require(tensor_sha(indices.astype(np.int64)) ==
                        descriptor["fingerprints_sha256"][f"{part}_index"],
                        f"Official node IDs: {relative}:{part}")
                acc = float(np.mean(pooled == labels))
                require(abs(acc - row[f"{part}_accuracy"]) < 1e-6 and
                        abs(acc - full[f"{part}_accuracy"]) < 1e-6,
                        f"Pooled accuracy: {relative}:{part}")
                if part == "test":
                    member_acc = float(np.mean(members == labels[None, :]))
                    require(abs(member_acc - full["mean_member_test_accuracy"]) < 1e-6,
                            f"Mean member accuracy: {relative}")
                    scores[seed, arm] = (acc, row["parameter_count"])
    require(len(derivations) == 12 and len(scores) == 12,
            "Expected exactly 12 complete arms")
    paired = []
    for seed in range(3):
        first, p_first = scores[seed, "private_first"]
        last, p_last = scores[seed, "private_last"]
        require(p_first == p_last, f"Equal partial-arm parameter count: seed{seed}")
        paired.append(first - last)
    expected = audit["paired_differences"]["private_first_minus_private_last"]["differences"]
    require(float(np.max(np.abs(np.array(paired) - expected))) < 1e-6,
            "Paired private-first versus private-last contrasts")
    file_hashes = json.loads((ROOT / "evidence_manifest.json").read_text())["sha256"]
    for relative, digest in file_hashes.items():
        require(sha(ROOT / relative) == digest, f"Stage file hash: {relative}")
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
              if p.is_file() and p.name != "evidence_manifest.json"}
    require(actual == set(file_hashes), "Stage file set")
    print(json.dumps({
        "status": "OGB_LITE_DECISION_AUDIT_PASS",
        "runs": 12,
        "private_first_minus_private_last_percentage_points": [100 * v for v in paired],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
