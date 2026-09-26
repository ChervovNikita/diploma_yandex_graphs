"""NumPy-only audit of the complete 24-arm filtered-Chameleon decision stage."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GRAPH = "chameleon_filtered"
DEPTH_ARMS = ("tied", "untied_propagation")
POSITION_ARMS = ("tied", "untied_propagation", "private_first", "private_last")


def require(ok, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read(path: Path):
    return json.loads(path.read_text())


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


def trace_check(path: Path, result: dict) -> None:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 300, f"Validation length: {path}")
    best_acc, best_ce, best_epoch = -1.0, float("inf"), 0
    for epoch, row in enumerate(rows, 1):
        require(int(row["epoch"]) == epoch, f"Validation order: {path}")
        acc, ce = float(row["valid_accuracy"]), float(row["valid_ce"])
        require(np.isfinite(acc) and np.isfinite(ce) and 0 <= acc <= 1,
                f"Invalid validation metric: {path}:{epoch}")
        improved = (acc > best_acc + 1e-12 or
                    (abs(acc - best_acc) <= 1e-12 and ce < best_ce - 1e-12))
        require(int(row["improved"]) == int(improved),
                f"Validation improvement flag: {path}:{epoch}")
        if improved:
            best_acc, best_ce, best_epoch = acc, ce, epoch
    require(best_epoch == result["selected_epoch"] and
            abs(best_acc - result["valid_accuracy"]) < 1e-6 and
            abs(best_ce - result["valid_ce"]) < 1e-5,
            f"Validation-selected checkpoint: {path}")


def run_check(relative: Path, graph: str, arm: str, seed: int, manifest: dict,
              audit_record: dict, derivations: dict) -> tuple[float, int]:
    run = ROOT / relative
    row = read(run / "result.json")
    require(row["dataset"] == graph and row["arm"] == arm and
            row["optimization_seed"] == seed and row["epochs_run"] == 300 and
            row["protocol"] == manifest["protocol"] and
            row["published_split"] == manifest["split"] and
            row["source_manifest_sha256"] == sha(ROOT / relative.parents[1] /
                                                   "source_manifest.json"),
            f"Run identity: {relative}")
    require(row["selected_epoch"] == audit_record["selected_epoch"] and
            row["parameter_count"] == audit_record["parameter_count"] and
            row["artifact_sha256"]["checkpoint.pt"] ==
            audit_record["checkpoint_sha256"],
            f"Checkpoint audit link: {relative}")
    if "depth" in row:
        require(row["depth"] == manifest["configuration"]["layers"],
                f"Run depth: {relative}")
    for filename in ("initialization.json", "validation_trace.csv"):
        require(sha(run / filename) == row["artifact_sha256"][filename],
                f"Retained run hash: {relative}/{filename}")
    trace_check(run / "validation_trace.csv", row)
    init = read(run / "initialization.json")
    require(init["paired_initial_logits_max_abs_diff"] <= 1e-5,
            f"Initial logits: {relative}")
    provenance = derivations[str(relative)]
    require(provenance["source_selected_logits_sha256"] ==
            row["artifact_sha256"]["selected_predictions.npz"] and
            sha(run / "selected_decisions.npz") ==
            provenance["selected_decisions_sha256"],
            f"Decision derivation: {relative}")
    with np.load(run / "selected_decisions.npz", allow_pickle=False) as f:
        decisions = {key: f[key].copy() for key in f.files}
    keys = {f"{part}_{field}" for part in ("valid", "test")
            for field in ("member_pred", "pooled_pred", "labels", "indices")}
    require(set(decisions) == keys, f"Decision schema: {relative}")
    descriptor = manifest["data_descriptor"]
    for part in ("valid", "test"):
        count = descriptor["split_sizes"][part]
        member = decisions[f"{part}_member_pred"]
        pool = decisions[f"{part}_pooled_pred"]
        labels = decisions[f"{part}_labels"]
        indices = decisions[f"{part}_indices"]
        require(member.shape == (4, count) and pool.shape == (count,) and
                labels.shape == (count,) and indices.shape == (count,),
                f"Decision shapes: {relative}:{part}")
        require(all(0 <= int(v.min()) and int(v.max()) < descriptor["num_classes"]
                    for v in (member, pool, labels)),
                f"Decision class range: {relative}:{part}")
        require(tensor_sha(indices.astype(np.int64)) ==
                descriptor["fingerprints_sha256"][f"{part}_indices"],
                f"Official indices: {relative}:{part}")
        accuracy = float(np.mean(pool == labels))
        require(abs(accuracy - row[f"{part}_accuracy"]) < 1e-6 and
                abs(accuracy - audit_record[f"{part}_accuracy"]) < 1e-6,
                f"Pooled accuracy: {relative}:{part}")
        if part == "test":
            member_accuracy = float(np.mean(member == labels[None, :]))
            require(abs(member_accuracy -
                        audit_record["mean_member_test_accuracy"]) < 1e-6,
                    f"Member accuracy: {relative}")
            return accuracy, row["parameter_count"]
    raise AssertionError("unreachable")


def group_check(base: Path, graph: str, arms: tuple[str, ...], seed_count: int,
                expected_protocol: str, expected_manifest_sha: str,
                derivations: dict):
    manifest_path = ROOT / base / "source_manifest.json"
    manifest = read(manifest_path)
    audit = read(ROOT / base / "completion_audit.json")
    require(sha(manifest_path) == expected_manifest_sha and
            audit["source_manifest_sha256"] == expected_manifest_sha,
            f"Frozen source manifest: {base}")
    require(manifest["protocol"] == expected_protocol and
            manifest["dataset"] == graph and manifest["arms"] == list(arms) and
            manifest["optimization_seeds"] == [0, 1, 2],
            f"Group protocol: {base}")
    require(audit["status"] == "COMPLETE_CUDA_REPLAY_PASS" and
            len(audit["records"]) == 3 * len(arms) and
            audit["numeric_tolerance"]["exact_member_and_pooled_decisions"],
            f"CUDA replay audit: {base}")
    require({(r["seed"], r["arm"]) for r in audit["records"]} ==
            {(seed, arm) for seed in range(3) for arm in arms},
            f"CUDA replay record set: {base}")
    require({p.name for p in (ROOT / base).iterdir() if p.is_dir()} ==
            {"seed0", "seed1", "seed2"}, f"Seed set: {base}")
    for relative, digest in manifest["source_sha256"].items():
        if relative.startswith("data/"):
            require(relative == "data/chameleon_filtered.npz" and
                    digest == read(ROOT / "PRETRAIN_FREEZE.json")["official_raw_npz_sha256"],
                    f"Public raw-data hash: {relative}")
        else:
            require(sha(ROOT / relative) == digest,
                    f"Frozen source hash: {relative}")
    scores = {}
    for seed in range(seed_count):
        seed_dir = ROOT / base / f"seed{seed}"
        require({p.name for p in seed_dir.iterdir() if p.is_dir()} == set(arms),
                f"Complete arm set: {seed_dir}")
        tied_init = read(seed_dir / "tied" / "initialization.json")
        for arm in arms:
            init = read(seed_dir / arm / "initialization.json")
            for key in ("canonical_tied_state_sha256", "cpu_rng_sha256",
                        "cuda_rng_sha256"):
                require(init[key] == tied_init[key],
                        f"Paired initialization: {seed_dir}/{arm}:{key}")
            record = next(r for r in audit["records"] if r["seed"] == seed and
                          r["arm"] == arm)
            score = run_check(base / f"seed{seed}" / arm, graph, arm, seed,
                              manifest, record, derivations)
            scores[seed, arm] = score
    if arms == DEPTH_ARMS:
        left, right = "tied", "untied_propagation"
    else:
        left, right = "private_first", "private_last"
        for seed in range(3):
            require(scores[seed, left][1] == scores[seed, right][1],
                    f"Unequal private parameter count: {base}/seed{seed}")
    contrasts = [scores[seed, left][0] - scores[seed, right][0]
                 for seed in range(3)]
    archived = audit["paired_differences"][f"{left}_minus_{right}"]["differences"]
    require(float(np.max(np.abs(np.array(contrasts) - archived))) < 1e-6,
            f"Paired scores: {base}")
    return [100 * difference for difference in contrasts], manifest["data_descriptor"]


def main() -> None:
    derivations = read(ROOT / "decision_derivation_manifest.json")
    require(len(derivations) == 24, "Expected 24 selected decision records")
    depth_freeze = read(ROOT / "PRETRAIN_FREEZE.json")
    position_freeze = read(ROOT / "POSITION_PRETRAIN_FREEZE.json")
    require(depth_freeze["status"] == "FILTERED_DEPTH_FROZEN_BEFORE_TRAINING" and
            position_freeze["status"] == "FILTERED_POSITION_FROZEN_BEFORE_TRAINING" and
            depth_freeze["outcome_files_seen"] ==
            position_freeze["outcome_files_seen"] == 0 and
            position_freeze["depth_pretrain_freeze_sha256"] ==
            sha(ROOT / "PRETRAIN_FREEZE.json") and
            position_freeze["position_prediction_sha256"] ==
            depth_freeze["position_prediction_sha256"] ==
            sha(ROOT / "POSITION_PREDICTION_FREEZE.md") and
            depth_freeze["data_descriptor"] == position_freeze["data_descriptor"] and
            depth_freeze["official_raw_npz_sha256"] ==
            "bf46f07e1fb5249280447e5fe3100f3e82fc4b93ad1e13ffcfdec924b6ac0bb5" and
            {(r["depth"], r["seed"], r["arm"]) for r in depth_freeze["planned_cells"]} ==
            {(d, s, a) for d in (2, 5) for s in range(3) for a in DEPTH_ARMS} and
            {(r["seed"], r["arm"]) for r in position_freeze["planned_cells"]} ==
            {(s, a) for s in range(3) for a in POSITION_ARMS},
            "Pretraining freeze chain")
    result = {}
    for depth in (2, 5):
        base = Path("results") / GRAPH / f"depth{depth}"
        expected = depth_freeze["source_manifests"][f"depth{depth}"]["sha256"]
        paired, descriptor = group_check(base, GRAPH, DEPTH_ARMS, 3,
                                         "filtered_chameleon_sage_depth_v1", expected,
                                         derivations)
        require(descriptor == depth_freeze["data_descriptor"],
                f"Frozen depth data: {depth}")
        result[f"depth{depth}_tied_minus_untied_pp"] = paired
    base = Path("position_results") / GRAPH
    expected = position_freeze["source_manifest_sha256"]
    paired, descriptor = group_check(base, GRAPH, POSITION_ARMS, 3,
                                     "filtered_chameleon_sharing_position_v1", expected,
                                     derivations)
    require(descriptor == position_freeze["data_descriptor"],
            "Frozen position data")
    result["private_first_minus_private_last_pp"] = paired
    hashes = read(ROOT / "evidence_manifest.json")["sha256"]
    for relative, digest in hashes.items():
        require(sha(ROOT / relative) == digest, f"Stage hash: {relative}")
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()
              and p.name != "evidence_manifest.json"}
    require(actual == set(hashes), "Stage file set differs")
    print(json.dumps({"status": "FILTERED_CHAMELEON_24_ARM_DECISION_AUDIT_PASS",
                      "runs": 24, "paired_percentage_points": result}, sort_keys=True))


if __name__ == "__main__":
    main()
