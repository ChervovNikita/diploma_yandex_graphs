# SPDX-License-Identifier: Apache-2.0
"""Inactive exporter from a local official raw archive; never downloads data."""
import argparse
import json
import time
from pathlib import Path

from common import (PACKET, array_hash, failure_record, require_release,
                    sha256_file, validate_roles, write_json)

RAW_KEYS = {"node_features", "node_labels", "edges", "train_masks", "val_masks", "test_masks"}
OFFICIAL_URL = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/tolokers.npz"


def export(args, release, output):
    import numpy as np
    started = time.perf_counter()
    with np.load(args.raw_npz, allow_pickle=False) as raw:
        if set(raw.files) != RAW_KEYS:
            raise ValueError("Raw archive must have the six official release keys")
        x, edges = raw["node_features"], raw["edges"]
        if x.dtype != np.float32 or x.shape != (11758, 10) or not np.isfinite(x).all():
            raise ValueError("Expected untransformed official float32 Tolokers features")
        n = x.shape[0]
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu" or not edges.size:
            raise ValueError("Expected raw numeric edges [E_raw,2]")
        edges = edges.astype(np.int64, copy=False)
        if edges.min() < 0 or edges.max() >= n or np.any(edges[:, 0] == edges[:, 1]):
            raise ValueError("Raw support cannot be used by paired native NSD")
        train_masks, valid_masks = raw["train_masks"], raw["val_masks"]
        if (train_masks.shape != (10, n) or valid_masks.shape != (10, n)
                or train_masks.dtype != np.bool_ or valid_masks.dtype != np.bool_):
            raise ValueError("Expected official bool masks [10,N]")
        # The raw label vector is decoded inside this root-authorized exporter.
        # Only TRAIN/VALID entries are selected/emitted; TEST masks are not read.
        # The runner cannot accept the raw label vector or any TEST truth key.
        train_index = np.flatnonzero(train_masks[0]).astype(np.int64)
        valid_index = np.flatnonzero(valid_masks[0]).astype(np.int64)
        labels = raw["node_labels"]
        if labels.shape != (n,) or labels.dtype.kind not in "iu":
            raise ValueError("Invalid raw label-vector schema")
        train_y = labels[train_index].astype(np.int64)
        valid_y = labels[valid_index].astype(np.int64)
        del labels
        # PyG2.6.1 to_undirected adds the reverse then row-sorts/coalesces.
        # No guessed edge count; no feature normalization or native math rewrite.
        raw_keys = edges[:, 0] * n + edges[:, 1]
        expanded = np.concatenate((raw_keys, edges[:, 1] * n + edges[:, 0]))
        canonical = np.unique(expanded)
        edge_index = np.ascontiguousarray(np.stack((canonical // n, canonical % n)), dtype=np.int64)
        arrays = {"x": np.ascontiguousarray(x), "edge_index": edge_index,
                  "train_index": train_index, "train_y": train_y,
                  "valid_index": valid_index, "valid_y": valid_y}
        metadata = {"schema": "nsd_train_valid_roles_v1", "dataset": "tolokers",
                    "exposure_classification": "original_paper_benchmark_exploratory",
                    "official_split": 0, "official_split_count": 10,
                    "official_source_url": OFFICIAL_URL,
                    "raw_archive_sha256": sha256_file(args.raw_npz),
                    "root_release_sha256": sha256_file(args.release),
                    "source_seal_sha256": release["source_seal_sha256"],
                    "support_convention": "pyg_2.6.1_to_undirected_row_coalesce",
                    "support_counts": {"raw_entries": int(len(raw_keys)),
                                       "raw_unique_directed": int(np.unique(raw_keys).size),
                                       "expanded_entries": int(expanded.size),
                                       "canonical_directed": int(canonical.size),
                                       "canonical_undirected": int(canonical.size // 2)},
                    "role_counts": {"train": len(train_index), "valid": len(valid_index)},
                    "role_class_counts": {role: np.bincount(arrays[role + "_y"], minlength=2).tolist()
                                          for role in ("train", "valid")},
                    "features": {"shape": list(x.shape), "dtype": str(x.dtype), "transform": "none"},
                    "array_sha256": {name: array_hash(value) for name, value in arrays.items()},
                    "test_truth_exported": False, "test_truth_keys_accepted_by_runner": False,
                    "raw_label_vector_decoded_inside_exporter": True,
                    "test_labels_selected_or_emitted": False,
                    "canonical_count_is_actual_not_expected_constant": True,
                    "exporter_numpy": np.__version__}
        validate_roles(np, arrays, metadata)
        archive = output / "roles.npz"
        np.savez(archive, **arrays)
        metadata["archive_sha256"] = sha256_file(archive)
        metadata["export_seconds"] = time.perf_counter() - started
        write_json(output / "ROLE.json", metadata)
    return {"status": "exported_train_valid_only", "archive": str(archive),
            "archive_sha256": metadata["archive_sha256"],
            "actual_support_counts": metadata["support_counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release")
    parser.add_argument("--raw-npz")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "inactive", "action": "export", "downloads": False,
                          "input": "local official raw Tolokers archive after root audit/release",
                          "output_keys": sorted(["x", "edge_index", "train_index", "train_y", "valid_index", "valid_y"])}))
        return
    if not args.raw_npz or not args.output:
        parser.error("--raw-npz and --output are required for execution")
    release, output = require_release(args, "export", args.raw_npz)
    if release.get("official_source_url") != OFFICIAL_URL:
        write_json(output / "EXPORT_FAILURE.json", {"status": "failed", "message": "Official raw provenance URL binding is required"})
        raise RuntimeError("Release is not bound to the official raw source URL")
    try:
        result = export(args, release, output)
        write_json(output / "EXPORT_RESULT.json", result)
        print(json.dumps(result))
    except Exception as error:
        write_json(output / "EXPORT_FAILURE.json", failure_record(error, "official_role_export"))
        raise


if __name__ == "__main__":
    main()
