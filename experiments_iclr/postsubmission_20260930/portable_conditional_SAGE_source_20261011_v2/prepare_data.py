"""Canonical Physics preparation; explicit local custodian, no fitting."""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import time
from common import checked, receipt, require, runtime, sha, write

RAW_SHA256 = "e4d68468eba5fb8f6b280ffd406bd53e6e4295ba1dd2967d6fe99b978e4b3375"
DOWNLOAD_URL = "https://raw.githubusercontent.com/shchur/gnn-benchmark/1e72912a0810cdf27ae54fd589a3b43358a2b161/data/npz/ms_academic_phy.npz"

NODES = 34493

FEATURES = 8415

CLASSES = 5

PROVIDER_EDGES = 495924

EDGES = 530417

BLOCK_SEEDS = (8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623)

ROLE_SEED_OFFSET = 5000011

ROLE_COUNTS = {"train": 100, "select": 150, "cal": 150, "confirm": 34093}

AUTHOR_COMMIT = "1e72912a0810cdf27ae54fd589a3b43358a2b161"

AUTHOR_BLOB = "8d8bc12e29aa67f6753a712080eae81079bd7d01"

RAW_BYTES = 16374005

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

def prepare(raw_path, output):
    start = time.monotonic()
    raw_path, output = Path(raw_path).resolve(), Path(output).resolve()
    require(raw_path.stat().st_size == RAW_BYTES and sha(raw_path) == RAW_SHA256
            and git_blob_digest(raw_path) == AUTHOR_BLOB, "Exact pinned canonical author NPZ required")
    require(not output.exists(), "Fresh prepared output only")
    versions = runtime()
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    import torch
    from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected
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
    versions['torch_runtime'] = {'version': torch.__version__, 'cuda_build': torch.version.cuda,
        'default_dtype': str(torch.get_default_dtype()), 'cuda_available': torch.cuda.is_available(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision()}
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        versions['torch_runtime']['cuda0'] = {'name': device.name, 'total_memory_bytes': device.total_memory}
    identity = {"dataset": "CoauthorPhysics", "raw": receipt(raw_path),
        "author_commit": AUTHOR_COMMIT, "author_git_blob": AUTHOR_BLOB,
        "nodes": NODES, "input_dim": FEATURES, "classes": CLASSES,
        "provider_edges": PROVIDER_EDGES, "model_edges": EDGES, "self_edges": NODES,
        "feature_sha256": feature_sha, "graph_support_sha256": graph_support_sha,
        "provider_graph_support_sha256": provider_graph_support_sha,
        "normalize_features": False, "custodian_CONFIRM_targets_parsed": True,
        "model_consumers_access_only_TRAIN_SELECT_labels": True,
        "custom_masks_not_official_masks": True, "masks_are_not_independent_graphs": True,
        "blocks": role_identity, "runtime": versions}
    write(output / "PROVIDER_IDENTITY.json", identity)
    profile = {"dataset": "CoauthorPhysics", "nodes": NODES, "input_dim": FEATURES,
        "classes": CLASSES, "edges": EDGES, "self_edges": NODES, "train_rows": 100,
        "select_rows": 150, "provider_identity_sha256": sha(output / "PROVIDER_IDENTITY.json"),
        "graph_support_sha256": graph_support_sha}
    write(output / "PROVIDER_PROFILE.json", profile)
    files = {str(path.relative_to(output)): receipt(path) for path in sorted(output.rglob("*")) if path.is_file()}
    result = {"complete": True, "dataset": "CoauthorPhysics", "profile": profile,
        "block_seeds": list(BLOCK_SEEDS), "role_counts": ROLE_COUNTS, "files": files,
        "raw": receipt(raw_path), "runtime": versions, "seconds": time.monotonic() - start,
        "custodian_CONFIRM_targets_parsed": True, "predictions_or_scores": 0,
        "scientific_fits": 0, "resource_or_numerical_qualification": False}
    write(output / "CUSTODIAL_EXPORT_RECEIPT.json", result)
    return result
