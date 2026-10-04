#!/usr/bin/env python3
"""Available-input authentication only; root review precedes execution on peptide."""
import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path
import socket

PHASE = Path("/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930")
AVAILABLE = PHASE / "pubmed_heart_official_acquisition_server_20261004_v2/available/pubmed"
EXPECTED = {
    "train_pos.txt": "c6de89d86371909f738d620846540168d4b6256fed88dc9d8ab3609cb5357fb4",
    "valid_pos.txt": "31f55d457ce8ea1a75b0b501e0707814e7968be1e9e0b01f0bbff2b8ef9cdc0b",
    "heart_valid_samples.npy": "3a2f9ea0c11dd5221d9a36771bcd0e2d273a4d059aaa2dd3dff03fc46e9e66e4",
    "gnn_feature": "c895f9e8e2d96eae8d610e7be8740f77fe1cb6b40e02a73046b2f02aa63a8dd5",
}
NODES = 19717


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def server_guard():
    phase = PHASE.resolve(strict=True)
    if platform.system() != "Linux" or socket.gethostname() != "peptide" or not Path(__file__).resolve().is_relative_to(phase):
        raise RuntimeError("Inspector/adapter must run inside the pinned peptide phase")
    if not AVAILABLE.resolve(strict=True).is_relative_to(phase):
        raise RuntimeError("Available inputs leave the pinned phase")
    return phase


def read_rows(name, expected_rows):
    rows = []
    with (AVAILABLE / name).open("rt", encoding="utf-8") as stream:
        for line in stream:
            values = line.strip().split("\t")
            if len(values) != 2:
                raise ValueError("Invalid positive row")
            u, v = map(int, values)
            if not (0 <= u < NODES and 0 <= v < NODES) or u == v:
                raise ValueError("Positive endpoint range/self-loop mismatch")
            rows.append((u, v))
    if len(rows) != expected_rows or len({tuple(sorted(p)) for p in rows}) != len(rows):
        raise ValueError("Positive row count or duplicate mismatch")
    return rows


def load_available():
    """Read exactly four pinned files; return native values without transformations."""
    server_guard()
    import numpy as np
    import torch

    identities = {}
    for name, expected in EXPECTED.items():
        path = AVAILABLE / name
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError("Available file identity mismatch: " + name)
        identities[name] = {"sha256": expected, "bytes": path.stat().st_size}
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError("A Torch version supporting weights_only=True is required; no pickle fallback")
    supplied = torch.load(AVAILABLE / "gnn_feature", map_location="cpu", weights_only=True)
    if not isinstance(supplied, dict) or "entity_embedding" not in supplied:
        raise ValueError("Supplied feature key is absent")
    x = supplied["entity_embedding"]
    if not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.ndim != 2 or x.shape[0] != NODES or x.shape[1] < 1:
        raise ValueError("Feature tensor population/layout mismatch")
    if x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
        raise ValueError("Native float32 finite feature requirement failed; no automatic cast")
    train_rows = read_rows("train_pos.txt", 37676)
    valid_rows = read_rows("valid_pos.txt", 2216)
    train_pairs = {tuple(sorted(p)) for p in train_rows}
    if train_pairs & {tuple(sorted(p)) for p in valid_rows}:
        raise ValueError("TRAIN/VALID overlap")
    pool = np.load(AVAILABLE / "heart_valid_samples.npy", mmap_mode="r", allow_pickle=False)
    if pool.shape != (2216, 500, 2) or pool.dtype != np.dtype("<i8") or not pool.flags.c_contiguous:
        raise ValueError("Complete VALID pool schema mismatch")
    return x, train_rows, valid_rows, pool, identities


def inspect_available():
    server_guard()
    import numpy as np
    import torch

    x, train, valid, pool, identities = load_available()
    positive = np.asarray(valid, dtype=np.int64)
    first = pool[:, :250]
    second = pool[:, 250:]
    source_anchor_errors = int(np.count_nonzero(first[:, :, 0] != positive[:, 0, None]))
    target_anchor_errors = int(np.count_nonzero(second[:, :, 1] != positive[:, 1, None]))
    range_errors = int(np.count_nonzero((pool < 0) | (pool >= NODES)))
    self_loops = int(np.count_nonzero(pool[:, :, 0] == pool[:, :, 1]))
    lower = np.minimum(pool[:, :, 0], pool[:, :, 1])
    upper = np.maximum(pool[:, :, 0], pool[:, :, 1])
    codes = lower * NODES + upper
    own_codes = np.minimum(positive[:, 0], positive[:, 1]) * NODES + np.maximum(positive[:, 0], positive[:, 1])
    own_positive_errors = int(np.count_nonzero(codes == own_codes[:, None]))
    any_valid_positive_candidates = int(np.count_nonzero(np.isin(codes, own_codes, kind="sort")))
    train_codes = np.asarray([min(u, v) * NODES + max(u, v) for u, v in train], dtype=np.int64)
    train_positive_errors = int(np.count_nonzero(np.isin(codes, train_codes, kind="sort")))
    duplicates = sum(500 - np.unique(row).size for row in codes)
    checks = {"out_of_range_endpoint_values": range_errors, "first250_source_anchor_mismatches": source_anchor_errors,
              "last250_target_anchor_mismatches": target_anchor_errors, "self_loop_candidates": self_loops,
              "own_positive_candidates": own_positive_errors, "TRAIN_positive_candidates": train_positive_errors}
    return {"schema": "pubmed_available_input_inspection_v1", "files": identities,
            "TRAIN_rows": len(train), "VALID_rows": len(valid), "node_population": NODES,
            "feature": {"key": "entity_embedding", "shape": list(x.shape), "dtype": str(x.dtype), "layout": str(x.layout), "finite": True,
                        "load_policy": "torch.load(weights_only=True,map_location=cpu)", "donor_origin": "Unresolved from currently identified pinned source"},
            "VALID_pool": {"shape": list(pool.shape), "dtype": str(pool.dtype), "all_rows_checked": True, "positive_row_order_preserved": True,
                           "anchor_order_source": "pinned rank_and_merge_edges: first250=(source,node), last250=(node,target)", "checks": checks,
                           "any_VALID_positive_candidates": any_valid_positive_candidates,
                           "other_VALID_positive_candidates": any_valid_positive_candidates - own_positive_errors,
                           "other_VALID_positive_collisions_are_nonblocking": True,
                           "VALID_collision_policy": "Author VALID generator filters TRAIN and the current query; other VALID positives are reported without altering the released pool",
                           "within_row_duplicate_candidates": int(duplicates), "duplicates_are_nonblocking": True,
                           "duplicate_reason": "Native random fallback np.random.choice defaults to replacement; preserve the released pool", "unknown_nonlink_semantics": True},
            "available_geometry_checks_pass": all(value == 0 for value in checks.values()),
            "feature_donor_authority_complete": False, "negative_pool_authority_complete": False, "training_admission": False,
            "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    phase = server_guard()
    receipt = args.receipt.resolve()
    if receipt.exists() or not receipt.is_relative_to(phase) or not receipt.parent.is_dir():
        raise RuntimeError("Use a fresh scalar receipt inside the pinned phase")
    result = inspect_available()
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["available_geometry_checks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
