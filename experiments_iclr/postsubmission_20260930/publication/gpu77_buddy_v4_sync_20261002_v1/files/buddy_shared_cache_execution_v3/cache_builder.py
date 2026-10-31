"""Build one offline, training-topology BUDDY cache. No fitting here."""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import sys
import time
from guards import file_sha, implementation_hashes, read_json, verify_family_metadata

HERE = Path(__file__).resolve().parent


def tensor_sha(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.shape, array.dtype)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def offline_dataset(root):
    from ogb.linkproppred import PygLinkPropPredDataset
    def forbid_download(self):
        raise RuntimeError("Dataset absent/incomplete. Acquire official ogbl-collab separately; this builder never downloads.")
    PygLinkPropPredDataset.download = forbid_download
    return PygLinkPropPredDataset(name="ogbl-collab", root=str(root))


def checked_split_loader_source(dataset, expected_sha=None):
    source = inspect.getsource(type(dataset).get_edge_split)
    if not all(name in source for name in ("train.pt", "valid.pt", "test.pt")):
        raise RuntimeError("Installed OGB split loader differs from the per-file convention; inspect and qualify it before building.")
    if expected_sha is not None and hashlib.sha256(source.encode()).hexdigest() != expected_sha:
        raise RuntimeError("Official OGB split loader changed since training preparation.")
    return source


def official_split_file(dataset, split, expected_loader_sha=None):
    """Read one official per-split file; never invoke get_edge_split().

    The retained OGB convention is root/split/<dataset.split>/{train,valid,test}.pt
    with replace_numpy_with_torchtensor conversion. Check the installed loader
    advertises that convention and record its exact source at cache construction.
    An aggregate-only split_dict.pt cannot satisfy closed test access here.
    """
    implementation_hashes()
    if split not in {"train", "valid", "test"}:
        raise ValueError(split)
    if split == "test" and expected_loader_sha is None:
        raise RuntimeError("Final test read requires the qualified loader source digest.")
    source = checked_split_loader_source(dataset, expected_loader_sha)
    split_type = getattr(dataset, "split", None)
    if not isinstance(split_type, str) or Path(split_type).name != split_type:
        raise RuntimeError("Unexpected official OGB split type.")
    path = Path(dataset.root) / "split" / split_type / f"{split}.pt"
    if not path.is_file():
        raise RuntimeError(f"Official {split}.pt absent. Stage the authentic per-split release files; do not open an aggregate split_dict.pt before family lock.")
    import torch
    from ogb.utils.torch_util import replace_numpy_with_torchtensor
    return replace_numpy_with_torchtensor(torch.load(path, map_location="cpu")), path, source


def extract_feature_function():
    from models import verified_source
    import torch
    import torch_sparse
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    tree = ast.parse(verified_source("datasets_elph.py.txt"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HashDataset")
    func = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_generate_sign_features")
    namespace = {"torch": torch, "torch_sparse": torch_sparse, "gcn_norm": gcn_norm}
    module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
    exec(compile(module, "pinned_author_feature_preprocessing", "exec"), namespace)
    return namespace[func.name]


def load_registered_module(name, path):
    """Register before execution so PyG Inspector can resolve class globals."""
    path = Path(path).resolve()
    identity = (str(path), file_sha(path))
    if name in sys.modules:
        existing = sys.modules[name]
        if (getattr(existing, "_buddy_source_identity", None) != identity
                or getattr(existing, "__file__", None) != str(path)
                or getattr(getattr(existing, "__spec__", None), "origin", None) != str(path)):
            raise RuntimeError(f"Registered module identity collision: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No loader for pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        if sys.modules.get(name) is not module:
            raise RuntimeError(f"Module registration changed during loading: {name}")
        module._buddy_source_identity = identity
    except BaseException:
        # The name was absent before this call. Restore that exact prior state
        # after any failure, including a loader that replaced its own entry.
        sys.modules.pop(name, None)
        raise
    return module


def native_hashes(config):
    from models import verified_source
    from types import SimpleNamespace
    verified_source("hashing.py")
    module = load_registered_module("pinned_buddy_hashing", HERE / "vendor" / "hashing.py")
    args = SimpleNamespace(max_hash_hops=2, floor_sf=False, minhash_num_perm=128, hll_p=8, use_zero_one=False)
    return module.ElphHashes(args)


def load_manifest(cache):
    sources = implementation_hashes()
    manifest = read_json(Path(cache) / "manifest.json")
    if manifest.get("implementation_hashes") != sources:
        raise RuntimeError("Cache executable source identity changed.")
    for name, checksum in manifest["files"].items():
        if file_sha(Path(cache) / name) != checksum:
            raise RuntimeError(f"Cache identity mismatch: {name}")
    return manifest


def verify_family_lock(lock, cache_manifest_sha, config_sha):
    value, checkpoints = verify_family_metadata(lock, cache_manifest_sha, config_sha)
    from checkpoint_io import validate_selected_checkpoint
    for checkpoint, metadata in checkpoints:
        validate_selected_checkpoint(checkpoint, metadata)
    return value


def build(args):
    sources = implementation_hashes()
    import numpy as np
    import scipy.sparse as ssp
    import torch
    from torch_geometric.utils import add_self_loops, negative_sampling
    from torch_sparse import coalesce
    config = read_json(HERE / "CONFIG.json")
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise RuntimeError("Cache already exists; use its verified identity, never overwrite it.")
    started = time.monotonic()
    dataset = offline_dataset(Path(args.dataset_root).resolve())
    graph = dataset[0]
    train_split, train_path, split_loader_source = official_split_file(dataset, "train")
    valid_split, valid_path, valid_loader_source = official_split_file(dataset, "valid")
    if split_loader_source != valid_loader_source:
        raise RuntimeError("OGB split loader changed during preparation.")
    (destination / "official_split_loader.py.txt").write_text(split_loader_source)
    if graph.num_nodes != 235868 or graph.x.shape[1] != 128:
        raise RuntimeError("Unexpected official collab graph dimensions.")
    edge_index, edge_weight = coalesce(graph.edge_index, graph.edge_weight.reshape(-1), graph.num_nodes, graph.num_nodes)
    # Exactly the native collab weighted coalescing and degree convention.
    adjacency = ssp.csr_matrix((edge_weight.numpy(), (edge_index[0].numpy(), edge_index[1].numpy())), shape=(graph.num_nodes, graph.num_nodes))
    degrees = torch.tensor(adjacency.sum(axis=0, dtype=float), dtype=torch.float).flatten()
    features = extract_feature_function()(None, graph, edge_index, edge_weight, 0)
    hasher = native_hashes(config)
    hashes, cards = hasher.build_hash_tables(graph.num_nodes, edge_index)
    torch.save({"x": features, "degrees": degrees, "hashes": hashes, "cards": cards}, destination / "common.pt")
    negative_seed = config["negative_seed"]
    random.seed(negative_seed); np.random.seed(negative_seed); torch.manual_seed(negative_seed)
    forbidden, _ = add_self_loops(edge_index, num_nodes=graph.num_nodes)
    train_positive = train_split["edge"]
    train_negative = negative_sampling(forbidden, num_nodes=graph.num_nodes, num_neg_samples=len(train_positive), method="sparse").t()
    if len(train_negative) != len(train_positive):
        raise RuntimeError("Negative sampler did not return one row per positive.")
    split_identity = {}
    for split, positive, negative in (
        ("train", train_positive, train_negative),
        ("valid", valid_split["edge"], valid_split["edge_neg"]),
    ):
        if split == "valid" and len(negative) != 100000:
            raise RuntimeError("Unexpected official collab validation-negative population.")
        links = torch.cat([positive, negative])
        sf = hasher.get_subgraph_features(links, hashes, cards, config["subgraph_feature_batch_size"])
        sf[:, [4, 5]] = 0  # Native use_zero_one=False; do not floor estimates.
        torch.save({"links": links, "sf": sf, "n_positive": len(positive)}, destination / f"{split}.pt")
        split_identity[split] = {"positive_sha256": tensor_sha(positive), "negative_sha256": tensor_sha(negative), "pair_order_sha256": tensor_sha(links), "positive_rows": len(positive), "negative_rows": len(negative)}
    from importlib.metadata import version
    versions = {name: version(name) for name in ["torch", "torch-geometric", "torch-sparse", "numpy", "scipy", "pandas", "datasketch", "ogb"]}
    manifest = {"schema": "buddy-shared-cache-v2", "implementation_hashes": sources, "dataset": "ogbl-collab", "graph_policy": "training_only_all_splits", "node_count": graph.num_nodes, "coalesced_directed_entries": edge_index.shape[1], "edge_index_sha256": tensor_sha(edge_index), "edge_weight_sha256": tensor_sha(edge_weight), "raw_features_sha256": tensor_sha(graph.x), "negative_seed": negative_seed, "config_sha256": file_sha(HERE / "CONFIG.json"), "source_pin_sha256": file_sha(HERE / "SOURCE_PINS.json"), "builder_sha256": file_sha(HERE / "cache_builder.py"), "dependency_versions": versions, "official_split_files": {"train": {"path": str(train_path), "sha256": file_sha(train_path)}, "valid": {"path": str(valid_path), "sha256": file_sha(valid_path)}}, "test_split_opened": False, "split_identity": split_identity, "build_seconds": time.monotonic() - started, "files": {name: file_sha(destination / name) for name in ("common.pt", "train.pt", "valid.pt", "official_split_loader.py.txt")}}
    write_json(destination / "manifest.json", manifest)
    print(json.dumps({"status": "cache_ready", "path": str(destination), "manifest_sha256": file_sha(destination / "manifest.json")}, indent=2))


def finalize(args):
    import torch
    cache = Path(args.output).resolve()
    manifest = load_manifest(cache)
    verify_family_lock(args.family_lock, file_sha(cache / "manifest.json"), file_sha(HERE / "CONFIG.json"))
    if (cache / "test_manifest.json").exists():
        raise RuntimeError("Final cache already exists; verify and reuse it.")
    dataset = offline_dataset(Path(args.dataset_root).resolve())
    graph = dataset[0]
    # The final stage verifies the raw training graph again, and never augments
    # it with validation edges. Only now hydrate official test candidate pairs.
    from torch_sparse import coalesce
    edges, weights = coalesce(graph.edge_index, graph.edge_weight.reshape(-1), graph.num_nodes, graph.num_nodes)
    if tensor_sha(edges) != manifest["edge_index_sha256"] or tensor_sha(weights) != manifest["edge_weight_sha256"] or tensor_sha(graph.x) != manifest["raw_features_sha256"]:
        raise RuntimeError("Official source graph/features changed since cache build.")
    pairs, test_path, source = official_split_file(dataset, "test", expected_loader_sha=manifest["files"]["official_split_loader.py.txt"])
    positive, negative = pairs["edge"], pairs["edge_neg"]
    if len(negative) != 100000:
        raise RuntimeError("Unexpected official collab test-negative population.")
    links = torch.cat([positive, negative])
    common = torch.load(cache / "common.pt", map_location="cpu")
    config = json.loads((HERE / "CONFIG.json").read_text())
    sf = native_hashes(config).get_subgraph_features(links, common["hashes"], common["cards"], config["subgraph_feature_batch_size"])
    sf[:, [4, 5]] = 0
    torch.save({"links": links, "sf": sf, "n_positive": len(positive)}, cache / "test.pt")
    write_json(cache / "test_manifest.json", {"cache_manifest_sha256": file_sha(cache / "manifest.json"), "test_sha256": file_sha(cache / "test.pt"), "official_test_file_sha256": file_sha(test_path), "positive_sha256": tensor_sha(positive), "negative_sha256": tensor_sha(negative), "pair_order_sha256": tensor_sha(links), "graph_policy": "training_only_all_splits", "family_lock_sha256": file_sha(args.family_lock)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["build", "finalize"])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family-lock")
    args = parser.parse_args()
    if args.stage == "finalize" and not args.family_lock:
        parser.error("finalize requires --family-lock")
    build(args) if args.stage == "build" else finalize(args)
