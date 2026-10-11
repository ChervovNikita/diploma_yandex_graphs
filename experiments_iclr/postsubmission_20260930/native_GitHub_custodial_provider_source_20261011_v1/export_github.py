"""Source-only custodial export of the canonical PyG GitHub NPZ release.

Root must separately admit existing-byte preparation. This program has no
downloader or model. It parses all targets as a custodian, including CONFIRM,
and exports only TRAIN truth to the model graph. It computes no scores.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time


NODES = 37700
FEATURES = 128
CLASSES = 2
RAW_EDGES = 578006
ROLE_SEED = 11709
ROLE_COUNTS = {"train": 22620, "select": 3770, "cal": 3770, "confirm": 7540}
PYG_COMMIT = "76ff9c2ce18c8cebf52122b57e2aeadce9793d10"
PROVIDER_SOURCE_SHA256 = "421aa192f8ba799f1517f1ea3f5acbda2613a196af14db1087b27559d78cf5d0"
DOWNLOAD_URL = "https://graphmining.ai/datasets/ptg/github.npz"
HOSTNAME = "anogena-2-0"
GPU_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def describe(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def verify(item):
    require(set(item) >= {"path", "bytes", "sha256"}, "Complete file receipt required")
    require(isinstance(item["bytes"], int) and item["bytes"] > 0,
            "Root must bind actual nonempty bytes")
    require(isinstance(item["sha256"], str) and len(item["sha256"]) == 64,
            "Root must bind exact SHA-256")
    path = Path(item["path"])
    require(path.is_file() and path.stat().st_size == item["bytes"]
            and digest(path) == item["sha256"], "Bound file receipt differs")
    return path


def route_identity():
    # Normal authorized host execution; no project reads before route checks.
    require(socket.gethostname() == HOSTNAME, "Excluded route: wrong hostname")
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True, timeout=15).splitlines()
    require(uuids == [GPU_UUID], "Excluded route: wrong GPU allocation")
    return {"hostname": HOSTNAME, "gpu_uuid": GPU_UUID}


def array_digest(np, array, dtype):
    value = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def prepare(config, config_path, route):
    start = time.monotonic()
    admission = config.get("admission", {})
    require(admission.get("custodial_export") is True, "Custodial export is not admitted")
    for name in ("data_download", "resource_qualification", "scientific_fit", "TEST_scoring"):
        require(admission.get(name) is False, "This export does not admit " + name)
    require(config.get("dataset") == "GitHub"
            and config.get("provider_PyG_commit") == PYG_COMMIT
            and config.get("canonical_download_url") == DOWNLOAD_URL,
            "Exact pinned GitHub provider required")
    require(config.get("research_role") == "prospective_confirmation_scoped_history_no_prior_evaluation_found",
            "Root must preserve the limited history claim")
    require(config.get("global_unused_certificate") is False,
            "The scoped history audit is not a global unused certificate")
    require(config.get("split_protocol") == "RandomState11709_full_ID_60_20_20_VALID_first_half_CAL",
            "Exact root prospective split required")
    source_receipts = [describe(verify(item)) for item in config["source_receipts"]]
    require(any(item["sha256"] == PROVIDER_SOURCE_SHA256 for item in source_receipts),
            "Exact public PyG GitHub provider source required")
    runtime_receipts = [describe(verify(item)) for item in config["runtime_source_receipts"]]
    require(runtime_receipts, "Root must bind actual runtime sources")
    input_admission = describe(verify(config["input_preparation_admission_receipt"]))
    eligibility_decision = describe(verify(config["root_history_and_access_decision_receipt"]))
    acquisition = describe(verify(config["byte_acquisition_receipt"]))
    raw_path = verify(config["input_npz"])
    raw_receipt = describe(raw_path)
    output = Path(config["output_directory"])
    require(not output.exists(), "Use a new output directory; never replace an export")

    # Custodial numerical imports only after route, admission and byte bindings.
    import numpy as np
    import torch
    import torch_geometric
    import torch_sparse
    from torch_geometric.data import Data
    from torch_geometric.transforms import ToSparseTensor
    from torch_sparse import SparseTensor

    actual_versions = {"numpy": np.__version__, "torch": torch.__version__,
                       "torch_geometric": torch_geometric.__version__,
                       "torch_sparse": torch_sparse.__version__}
    require(actual_versions == config["expected_runtime_versions"],
            "Actual runtime versions differ from root binding")
    runtime_modules = {}
    for name in ("numpy", "torch", "torch_geometric",
                 "torch_geometric.transforms.to_sparse_tensor", "torch_sparse", "torch_sparse.tensor"):
        runtime_modules[name] = describe(importlib.import_module(name).__file__)
    bound_runtime = {(item["path"], item["bytes"], item["sha256"]) for item in runtime_receipts}
    require(all((item["path"], item["bytes"], item["sha256"]) in bound_runtime
                for item in runtime_modules.values()),
            "Imported runtime paths/bytes/hashes differ from root binding")

    # Provider loads numeric features/target/edges; pickle is unnecessary here.
    with np.load(raw_path, allow_pickle=False) as stored:
        require(len(stored.files) == 3 and set(stored.files) == {"features", "target", "edges"},
                "Exact numeric provider NPZ field roster required")
        features = stored["features"]
        require(features.shape == (NODES, FEATURES) and features.dtype.kind in "fi",
                "Full original provider feature matrix required")
        x = np.ascontiguousarray(features, dtype=np.float32)
        require(np.isfinite(x).all(), "Provider features must be finite")
        raw_edges = stored["edges"]
        require(raw_edges.shape == (RAW_EDGES, 2) and raw_edges.dtype.kind in "iu",
                "Full stored directed provider edge roster required; fail before target access")
        raw_edges = np.ascontiguousarray(raw_edges.T, dtype=np.int64)
        require(raw_edges.min() >= 0 and raw_edges.max() < NODES,
                "Provider edge IDs outside fixed complete graph")

        # All roles are assigned from complete node IDs before target access.
        permutation = np.random.RandomState(ROLE_SEED).permutation(NODES)
        train_end = 3 * NODES // 5
        valid_end = 4 * NODES // 5
        valid = permutation[train_end:valid_end]
        midpoint = len(valid) // 2
        roles = {"train": np.sort(permutation[:train_end]),
                 "cal": np.sort(valid[:midpoint]), "select": np.sort(valid[midpoint:]),
                 "confirm": np.sort(permutation[valid_end:])}
        require({key: len(value) for key, value in roles.items()} == ROLE_COUNTS,
                "Fixed prospective role counts differ")
        require(np.array_equal(np.sort(np.concatenate(list(roles.values()))), np.arange(NODES)),
                "Prospective roles must partition every node exactly once")
        targets = stored["target"]
        require(targets.shape == (NODES,) and targets.dtype.kind in "iu",
                "Complete integer binary target vector required")
        targets = np.ascontiguousarray(targets, dtype=np.int64)
        require(targets.min() >= 0 and targets.max() < CLASSES,
                "Targets outside fixed binary domain")

    # Explicit prospective support preparation, distinct from provider's raw load.
    public_data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(raw_edges), num_nodes=NODES)
    public_data = ToSparseTensor()(public_data)
    require(isinstance(public_data.adj_t, SparseTensor), "Bound torch_sparse path required")
    adjacency = public_data.adj_t.to_symmetric()
    require(adjacency.sparse_sizes() == (NODES, NODES), "Complete sparse graph required")
    row, col, values = adjacency.coo()
    require(values is None and 0 < row.numel() <= 2 * RAW_EDGES,
            "Unweighted symmetric/coalesced support required")
    edge_index = torch.stack((col, row)).to(torch.int64).contiguous()
    raw_loop_ids = np.unique(raw_edges[0, raw_edges[0] == raw_edges[1]])
    loop_ids = edge_index[0, edge_index[0] == edge_index[1]].numpy()
    require(np.array_equal(np.sort(loop_ids), raw_loop_ids),
            "Support must preserve original diagonals and add no self loops")
    edges = edge_index.numpy()
    graph_support_sha = array_digest(np, edges, "<i8")
    feature_sha = array_digest(np, x, "<f4")
    train_y = targets[roles["train"]].copy()
    safe_y = torch.zeros(NODES, dtype=torch.int64)
    train_mask = torch.zeros(NODES, dtype=torch.bool)
    train_ids = torch.from_numpy(roles["train"])
    safe_y[train_ids] = torch.from_numpy(train_y)
    train_mask[train_ids] = True
    require(bool((safe_y[~train_mask] == 0).all()), "Non-TRAIN model truth must be zero")

    output.mkdir(parents=True, exist_ok=False)
    (output / "scorer_only").mkdir()
    torch.save({"x": torch.from_numpy(x), "edge_index": edge_index,
                "y": safe_y, "train_mask": train_mask}, output / "graph.pt")
    np.savez(output / "train.npz", x=x, edge_index=edges, ids=roles["train"], y=train_y)
    np.savez(output / "select.npz", ids=roles["select"], y=targets[roles["select"]])
    np.savez(output / "cal_targets.npz", ids=roles["cal"], y=targets[roles["cal"]])
    confirm_target_path = output / "scorer_only/confirm_targets.npz"
    np.savez(confirm_target_path, ids=roles["confirm"], y=targets[roles["confirm"]])
    os.chmod(confirm_target_path, 0o600)
    os.chmod(output / "cal_targets.npz", 0o600)
    for role in ("train", "select", "cal", "confirm"):
        np.save(output / (role + "_ids.npy"), roles[role], allow_pickle=False)
    role_identity = {"ordered_ids": {key: array_digest(np, value, "<i8")
                                    for key, value in roles.items()},
                     "role_seed": ROLE_SEED, "RNG": "numpy.random.RandomState MT19937",
                     "counts": ROLE_COUNTS, "official_split": False,
                     "protocol": config["split_protocol"],
                     "VALID_partition_order": "first half CAL, second half SELECT of original permutation slice",
                     "output_ID_order": "ascending within each role",
                     "targets_parsed_by_custodian_including_CONFIRM": True,
                     "role_assignment_uses_labels_or_predictions": False}
    provider_identity = {"schema": "GitHub_custodial_provider_identity_v1",
                         "dataset": "GitHub", "provider_PyG_commit": PYG_COMMIT,
                         "canonical_download_url": DOWNLOAD_URL, "raw_npz": raw_receipt,
                         "raw_release_version": "URL unversioned; exact acquired bytes bound by SHA256",
                         "source_receipts": source_receipts, "runtime_sources": runtime_receipts,
                         "nodes": NODES, "features": FEATURES, "classes": CLASSES,
                         "raw_stored_edges": RAW_EDGES, "edges": edge_index.shape[1],
                         "self_edges": len(loop_ids), "new_self_loops": 0,
                         "feature_sha256": feature_sha, "graph_support_sha256": graph_support_sha,
                         "feature_transform": "release features castfloat32 only; no reconstruction",
                         "support_transform": "ToSparseTensor then to_symmetric, coalesced; no new loops",
                         "provider_raw_process_adds_symmetry_or_loops": False,
                         "edge_order": "COO source=SparseTensor column,target=row",
                         "roles": role_identity, "global_unused_certificate": False}
    identity_path = output / "PROVIDER_IDENTITY.json"
    identity_path.write_text(json.dumps(provider_identity, sort_keys=True, indent=2) + "\n")
    profile = {"dataset": "GitHub", "nodes": NODES, "input_dim": FEATURES, "classes": CLASSES,
               "train_rows": ROLE_COUNTS["train"], "select_rows": ROLE_COUNTS["select"],
               "edges": edge_index.shape[1], "self_edges": len(loop_ids),
               "provider_identity_sha256": digest(identity_path), "graph_support_sha256": graph_support_sha}
    files = {str(path.relative_to(output)): describe(path)
             for path in sorted(output.rglob("*")) if path.is_file()}
    result = {"schema": "GitHub_custodial_export_receipt_v1", "complete": True,
              "UTC": datetime.now(timezone.utc).isoformat(), "research_role": config["research_role"],
              "dataset": "GitHub", "profile": profile, "roles": role_identity, "files": files,
              "raw_npz": raw_receipt, "config": describe(config_path),
              "exporter_source": describe(Path(__file__)), "source_receipts": source_receipts,
              "input_preparation_admission_receipt": input_admission,
              "root_history_and_access_decision_receipt": eligibility_decision,
              "byte_acquisition_receipt": acquisition,
              "runtime": dict(route, python=sys.version, executable=sys.executable,
                              platform=platform.platform(), versions=actual_versions,
                              module_sources=runtime_modules),
              "seconds": time.monotonic() - start,
              "custodian_CONFIRM_targets_parsed": True, "CONFIRM_targets_in_model_artifacts": False,
              "SELECT_CAL_targets_in_model_graph": False, "native_models_or_fits": 0,
              "predictions_or_predictive_scores": 0, "TEST_scoring": False,
              "global_unused_certificate": False, "resource_qualification_admitted": False,
              "scientific_fit_admitted": False, "TEST_release_admitted": False}
    receipt_path = output / "CUSTODIAL_EXPORT_RECEIPT.json"
    receipt_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": True, "receipt": describe(receipt_path), "profile": profile,
                      "custodian_CONFIRM_targets_parsed": True, "predictive_scores": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    route = route_identity()
    config = json.loads(args.config.read_text())
    prepare(config, args.config, route)


if __name__ == "__main__":
    main()
