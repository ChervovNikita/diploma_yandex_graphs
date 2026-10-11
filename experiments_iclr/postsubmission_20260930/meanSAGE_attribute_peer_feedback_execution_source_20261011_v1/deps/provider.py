"""Inactive custodial Coauthor Physics exporter; root admission required.

Parses all author targets for ten class-balanced prospective masks, including
CONFIRM targets. No dataset downloader, model, prediction or score exists here.
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

NODES = 34493
FEATURES = 8415
CLASSES = 5
PROVIDER_EDGES = 495924
EDGES = 530417
BLOCK_SEEDS = (8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623)
ROLE_SEED_OFFSET = 5000011
ROLE_COUNTS = {"train": 100, "select": 150, "cal": 150, "confirm": 34093}
PYG_COMMIT = "76ff9c2ce18c8cebf52122b57e2aeadce9793d10"
AUTHOR_COMMIT = "1e72912a0810cdf27ae54fd589a3b43358a2b161"
AUTHOR_BLOB = "8d8bc12e29aa67f6753a712080eae81079bd7d01"
RAW_BYTES = 16374005
DOWNLOAD_URL = "https://raw.githubusercontent.com/shchur/gnn-benchmark/" + AUTHOR_COMMIT + "/data/npz/ms_academic_phy.npz"
PROVIDER_SOURCE_HASHES = {"3666bd68b6a90c4d52ee5082e3459acd065b049f27310577249651722f85dd74",
                          "cf84847e11ef7ee003b6b65b374b29ed545c804675efc63dc63d17b0aee6ac26"}
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


def git_blob_digest(path):
    path = Path(path)
    h = hashlib.sha1(b"blob " + str(path.stat().st_size).encode("ascii") + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def prepare(config, config_path, route):
    start = time.monotonic()
    admission = config.get("admission", {})
    require(admission.get("custodial_export") is True, "Custodial export is not admitted")
    for name in ("data_download", "resource_qualification", "scientific_fit", "TEST_scoring"):
        require(admission.get(name) is False, "This export does not admit " + name)
    require(config.get("dataset") == "CoauthorPhysics"
            and config.get("provider_PyG_commit") == PYG_COMMIT
            and config.get("author_commit") == AUTHOR_COMMIT
            and config.get("author_git_blob") == AUTHOR_BLOB
            and config.get("canonical_download_url") == DOWNLOAD_URL,
            "Exact pinned author/PyG Physics source required")
    require(config.get("research_role") == "prospective_confirmation_scoped_history_no_prior_evaluation_found"
            and config.get("global_unused_certificate") is False,
            "Preserve the bounded historical claim")
    require(config.get("split_protocol") == "ten_blocks_per_class_20TRAIN_30SELECT_30CAL_restCONFIRM"
            and tuple(config.get("block_seeds", [])) == BLOCK_SEEDS
            and config.get("role_seed_offset") == ROLE_SEED_OFFSET,
            "Exact root-frozen mask protocol required")
    require(config.get("model_support_protocol") == "provider_no_loops_plus_one_loop_per_node_coalesced"
            and config.get("normalize_features") is False,
            "Root-frozen common model support and raw binary features required")
    source_receipts = [describe(verify(item)) for item in config["source_receipts"]]
    require(PROVIDER_SOURCE_HASHES <= {item["sha256"] for item in source_receipts},
            "Both pinned provider and NPZ reader required")
    runtime_receipts = [describe(verify(item)) for item in config["runtime_source_receipts"]]
    require(runtime_receipts, "Root must bind actual runtime sources")
    input_admission = describe(verify(config["input_preparation_admission_receipt"]))
    eligibility_decision = describe(verify(config["root_history_and_access_decision_receipt"]))
    acquisition = describe(verify(config["byte_acquisition_receipt"]))
    raw_path = verify(config["input_npz"])
    require(raw_path.stat().st_size == RAW_BYTES and git_blob_digest(raw_path) == AUTHOR_BLOB,
            "Exact pinned author NPZ bytes/git blob required")
    raw_receipt = dict(describe(raw_path), git_blob_sha1=AUTHOR_BLOB)
    output = Path(config["output_directory"])
    require(not output.exists(), "Use a new output directory; never replace an export")

    # Numerical imports only after route, admission, source and byte bindings.
    import numpy as np
    import scipy
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    import torch
    import torch_geometric
    from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected

    actual_versions = {"numpy": np.__version__, "scipy": scipy.__version__,
                       "torch": torch.__version__, "torch_geometric": torch_geometric.__version__}
    require(actual_versions == config["expected_runtime_versions"], "Runtime versions differ")
    runtime_modules = {}
    for name in ("numpy", "scipy", "scipy.sparse", "scipy.sparse.csgraph", "torch", "torch_geometric",
                 "torch_geometric.utils.loop", "torch_geometric.utils.undirected"):
        runtime_modules[name] = describe(importlib.import_module(name).__file__)
    bound_runtime = {(item["path"], item["bytes"], item["sha256"]) for item in runtime_receipts}
    require(all((item["path"], item["bytes"], item["sha256"]) in bound_runtime
                for item in runtime_modules.values()), "Imported runtime source bytes differ")

    with np.load(raw_path, allow_pickle=False) as stored:
        required = {"attr_data", "attr_indices", "attr_indptr", "attr_shape",
                    "adj_data", "adj_indices", "adj_indptr", "adj_shape", "labels"}
        require(required <= set(stored.files), "Complete author sparse NPZ fields required")
        require(tuple(stored["attr_shape"].tolist()) == (NODES, FEATURES)
                and tuple(stored["adj_shape"].tolist()) == (NODES, NODES),
                "Full author graph/feature dimensions differ; fail before label access")
        # Literal pinned PyG reader semantics: dense FP32, binarize positive entries.
        attributes = sp.csr_matrix((stored["attr_data"], stored["attr_indices"], stored["attr_indptr"]),
                                   (NODES, FEATURES))
        x_tensor = torch.from_numpy(attributes.todense()).to(torch.float)
        x_tensor[x_tensor > 0] = 1
        require(bool(torch.isfinite(x_tensor).all()), "Finite full provider features required")
        x = x_tensor.numpy()
        adjacency = sp.csr_matrix((stored["adj_data"], stored["adj_indices"], stored["adj_indptr"]),
                                  (NODES, NODES)).tocoo()
        raw_row = torch.from_numpy(adjacency.row).to(torch.long)
        raw_col = torch.from_numpy(adjacency.col).to(torch.long)
        raw_edge_index = torch.stack((raw_row, raw_col), dim=0)
        edge_index, _ = remove_self_loops(raw_edge_index)
        edge_index = to_undirected(edge_index, num_nodes=NODES).contiguous()
        require(edge_index.shape == (2, PROVIDER_EDGES) and edge_index.dtype == torch.int64,
                "Exact whole-provider coalesced directed support count required")
        require(bool((edge_index[0] != edge_index[1]).all()) and int(edge_index.min()) >= 0
                and int(edge_index.max()) < NODES, "Loop-free valid provider support required")
        provider_edges = edge_index.numpy()
        component_graph = sp.csr_matrix((np.ones(PROVIDER_EDGES, dtype=np.int8),
                                        (provider_edges[0], provider_edges[1])),
                                       shape=(NODES, NODES))
        components = connected_components(component_graph, directed=False, return_labels=False)
        require(components == 1,
                "Full archive must already be the complete published component; no subset selection")
        targets = stored["labels"]
        require(targets.shape == (NODES,) and targets.dtype.kind in "iu",
                "Full author integer label vector required")
        targets = np.ascontiguousarray(targets, dtype=np.int64)
        require(np.array_equal(np.unique(targets), np.arange(CLASSES)), "Fixed five-class label domain required")

    provider_graph_support_sha = array_digest(np, provider_edges, "<i8")
    # Separate root-adopted model preprocessing: one diagonal entry per node.
    # It is not part of the raw PyG provider transform and uses no labels.
    edge_index, _ = add_self_loops(edge_index, num_nodes=NODES)
    edge_index = to_undirected(edge_index, num_nodes=NODES).contiguous()
    require(edge_index.shape == (2, EDGES), "Fixed common model support count differs")
    loops = edge_index[0, edge_index[0] == edge_index[1]].numpy()
    require(np.array_equal(np.sort(loops), np.arange(NODES)),
            "Root model support must have exactly one self loop per node")
    edges = edge_index.numpy()
    graph_support_sha = array_digest(np, edges, "<i8")
    feature_sha = array_digest(np, x, "<f4")
    blocks = {}
    for block_seed in BLOCK_SEEDS:
        rng = np.random.RandomState(block_seed + ROLE_SEED_OFFSET)
        parts = {key: [] for key in ROLE_COUNTS}
        for class_id in range(CLASSES):
            class_ids = np.sort(np.flatnonzero(targets == class_id)).astype(np.int64)
            require(len(class_ids) > 80, "Every class must leave a confirmation population")
            shuffled = rng.permutation(class_ids)
            parts["train"].append(shuffled[:20])
            parts["select"].append(shuffled[20:50])
            parts["cal"].append(shuffled[50:80])
            parts["confirm"].append(shuffled[80:])
        roles = {key: np.sort(np.concatenate(value)) for key, value in parts.items()}
        require({key: len(value) for key, value in roles.items()} == ROLE_COUNTS,
                "Fixed per-block role counts differ")
        require(np.array_equal(np.sort(np.concatenate(list(roles.values()))), np.arange(NODES)),
                "Every block must partition the full node population exactly once")
        blocks[block_seed] = roles

    output.mkdir(parents=True, exist_ok=False)
    (output / "scorer_only").mkdir()
    (output / "calibration_only").mkdir()
    np.savez(output / "provider_support.npz", edge_index=provider_edges)
    role_identity = {}
    for block_seed, roles in blocks.items():
        directory = output / ("block_" + str(block_seed))
        directory.mkdir()
        # Compression changes only physical serialization; exact consumer fields remain.
        np.savez_compressed(directory / "train.npz", x=x, edge_index=edges,
                            ids=roles["train"], y=targets[roles["train"]])
        np.savez(directory / "select.npz", ids=roles["select"], y=targets[roles["select"]])
        cal_path = output / "calibration_only" / ("block_" + str(block_seed) + "_cal_targets.npz")
        np.savez(cal_path, ids=roles["cal"], y=targets[roles["cal"]])
        confirm_path = output / "scorer_only" / ("block_" + str(block_seed) + "_confirm_targets.npz")
        np.savez(confirm_path, ids=roles["confirm"], y=targets[roles["confirm"]])
        os.chmod(cal_path, 0o600)
        os.chmod(confirm_path, 0o600)
        for role in ROLE_COUNTS:
            np.save(directory / (role + "_ids.npy"), roles[role], allow_pickle=False)
        role_identity[str(block_seed)] = {
            "ordered_id_sha256": {key: array_digest(np, value, "<i8") for key, value in roles.items()},
            "counts": ROLE_COUNTS, "RNG": "numpy.random.RandomState MT19937",
            "role_seed": block_seed + ROLE_SEED_OFFSET,
            "algorithm": "classes0..4 in order; ascending class IDs; one RNG per block; per-class permutation; first20TRAIN,next30SELECT,next30CAL,restCONFIRM; sort each final role",
            "role_assignment_uses_targets_for_class_stratification": True,
            "role_assignment_uses_predictions_or_outcomes": False,
            "published_train_select_per_class_counts_preserved": True,
            "official_mask": False}
    provider_identity = {
        "schema": "CoauthorPhysics_custodial_provider_identity_v1", "dataset": "CoauthorPhysics",
        "provider_PyG_commit": PYG_COMMIT, "author_commit": AUTHOR_COMMIT, "author_git_blob": AUTHOR_BLOB,
        "canonical_download_url": DOWNLOAD_URL, "raw_npz": raw_receipt,
        "source_receipts": source_receipts, "runtime_sources": runtime_receipts,
        "nodes": NODES, "features": FEATURES, "classes": CLASSES,
        "provider_edges": PROVIDER_EDGES, "provider_self_edges": 0,
        "edges": EDGES, "self_edges": NODES,
        "feature_sha256": feature_sha, "graph_support_sha256": graph_support_sha,
        "provider_graph_support_sha256": provider_graph_support_sha,
        "feature_transform": "pinned PyG sparse CSR todense, castfloat32, positive entries to1",
        "support_transform": "pinned PyG CSR adjacencyCOO, remove_self_loops, to_undirected coalesce",
        "model_support_transform": "separate root freeze: add one self loop per node then coalesce, no feature normalization",
        "component_count": 1, "node_mapping": "all original NPZ rows preserved without subset/reordering",
        "provider_new_self_loops": 0, "model_new_self_loops": NODES,
        "edge_order": "pinned PyG to_undirected source-major ordered COO",
        "blocks": role_identity, "custom_mask_protocol": config["split_protocol"],
        "targets_parsed_by_custodian_including_CONFIRM": True, "global_unused_certificate": False,
        "masks_are_not_independent_graphs": True}
    identity_path = output / "PROVIDER_IDENTITY.json"
    identity_path.write_text(json.dumps(provider_identity, sort_keys=True, indent=2) + "\n")
    provider_identity_sha = digest(identity_path)
    profiles = {}
    for block_seed in BLOCK_SEEDS:
        profile = {"dataset": "CoauthorPhysics", "nodes": NODES, "input_dim": FEATURES, "classes": CLASSES,
                   "train_rows": 100, "select_rows": 150, "edges": EDGES, "self_edges": NODES,
                   "provider_identity_sha256": provider_identity_sha, "graph_support_sha256": graph_support_sha}
        profiles[str(block_seed)] = profile
        (output / ("block_" + str(block_seed)) / "PROVIDER_PROFILE.json").write_text(
            json.dumps(profile, indent=2) + "\n")
    files = {str(path.relative_to(output)): describe(path)
             for path in sorted(output.rglob("*")) if path.is_file()}
    result = {"schema": "CoauthorPhysics_custodial_export_receipt_v1", "complete": True,
              "UTC": datetime.now(timezone.utc).isoformat(), "dataset": "CoauthorPhysics",
              "research_role": config["research_role"], "profiles": profiles, "blocks": role_identity,
              "files": files, "raw_npz": raw_receipt, "config": describe(config_path),
              "exporter_source": describe(Path(__file__)), "source_receipts": source_receipts,
              "input_preparation_admission_receipt": input_admission,
              "root_history_and_access_decision_receipt": eligibility_decision,
              "byte_acquisition_receipt": acquisition,
              "runtime": dict(route, python=sys.version, executable=sys.executable,
                              platform=platform.platform(), versions=actual_versions, module_sources=runtime_modules),
              "seconds": time.monotonic() - start, "custodian_CONFIRM_targets_parsed": True,
              "CONFIRM_CAL_SELECT_targets_in_model_train_artifacts": False,
              "native_models_or_fits": 0, "predictions_or_predictive_scores": 0,
              "TEST_scoring": False, "global_unused_certificate": False,
              "resource_qualification_admitted": False, "scientific_fit_admitted": False,
              "TEST_release_admitted": False}
    receipt_path = output / "CUSTODIAL_EXPORT_RECEIPT.json"
    receipt_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": True, "receipt": describe(receipt_path), "profiles": profiles,
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
