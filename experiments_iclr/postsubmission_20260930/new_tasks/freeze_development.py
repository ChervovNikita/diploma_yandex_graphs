#!/usr/bin/env python3
"""Download, load, and fingerprint development data. Never fit or score a model.

Requires the existing project environment: torch, torch_geometric, numpy, ogb.
All generated files must stay under this script's research directory.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
URLS = {
    "molhiv": {"data": "https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip"},
    "ppi": {"data": "https://data.dgl.ai/dataset/ppi.zip"},
    "mnist_superpixels": {"data": "https://data.pyg.org/datasets/benchmarking-gnns/MNIST_v2.zip"},
    "malnet_tiny": {
        "data": "http://malnet.cc.gatech.edu/graph-data/malnet-graphs-tiny.tar.gz",
        "split": "http://malnet.cc.gatech.edu/split-info/split_info_tiny.zip",
    },
}
VERIFIED_ARCHIVES = {
    "molhiv": "47d747664b9e1653de5aac99bf26c015d88a3daa474f5a32f3fe5c014111375b",
    "ppi": "1f5b2b09ac0f897fa6aa1338c64ab75a5473674cbba89380120bede8cddb2a6a",
}


def inside(path):
    path = Path(path).resolve()
    if path != HERE and HERE not in path.parents:
        raise ValueError(f"Generated/input data path must be inside {HERE}: {path}")
    return path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def acquire(url, path, existing=None, expected=None):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing is not None:
        existing = inside(existing)
        if existing != path:
            if path.exists() and sha(path) != sha(existing):
                raise ValueError(f"Existing archive differs: {path}")
            if not path.exists():
                shutil.copyfile(existing, path)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "GNNM-postsubmission-data-freezer"})
        try:
            with urllib.request.urlopen(request, timeout=45) as source, open(partial, "wb") as dest:
                shutil.copyfileobj(source, dest)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Source archive changed: {path}; expected {expected}, got {digest}")
    return {"source_url": url, "path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def extract(archive, dest):
    dest = inside(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target = (dest / entry.filename).resolve()
                if dest != target and dest not in target.parents:
                    raise ValueError("Unsafe archive path")
                if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("Archive symlink rejected")
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            for entry in t.getmembers():
                target = (dest / entry.name).resolve()
                if dest != target and dest not in target.parents:
                    raise ValueError("Unsafe archive path")
                if entry.issym() or entry.islnk() or entry.isdev():
                    raise ValueError("Archive link/device rejected")
            t.extractall(dest)


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    metadata = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    h.update(tensor.numpy().tobytes(order="C"))
    return metadata | {"sha256": h.hexdigest()}


def summarize(dataset, split, indices=None):
    import torch
    fingerprints = []
    positive = negative = labeled = None
    nodes = edges = max_nodes = max_edges = 0
    class_counts = {}
    for i, data in enumerate(dataset):
        graph = {"split_index": i, "num_nodes": int(data.num_nodes)}
        if indices is not None:
            graph["official_index"] = int(indices[i])
        for key in ["x", "pos", "edge_index", "edge_attr", "y"]:
            value = getattr(data, key, None)
            if value is not None:
                graph[key] = tensor_hash(value)
        n, e = int(data.num_nodes), int(data.edge_index.size(1))
        nodes += n
        edges += e
        max_nodes, max_edges = max(max_nodes, n), max(max_edges, e)
        fingerprints.append(graph)
        # Test labels enter a fingerprint only; no label frequencies or scores.
        if split != "test":
            y = data.y.detach().cpu()
            if y.numel() == 1 and y.dtype == torch.long:
                key = str(int(y.item()))
                class_counts[key] = class_counts.get(key, 0) + 1
            else:
                if y.ndim == 1:
                    y = y.reshape(1, -1)
                mask = torch.isfinite(y)
                p, q, r = ((y == 1) & mask).sum(0), ((y == 0) & mask).sum(0), mask.sum(0)
                if positive is None:
                    positive, negative, labeled = p, q, r
                else:
                    positive += p
                    negative += q
                    labeled += r
    canonical = json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "graphs": len(fingerprints), "nodes": nodes, "directed_edge_entries": edges,
        "max_nodes_per_graph": max_nodes, "max_edge_entries_per_graph": max_edges,
        "canonical_split_sha256": hashlib.sha256(canonical).hexdigest(),
        "graph_tensor_fingerprints": fingerprints,
    }
    if split != "test":
        if class_counts:
            result["class_counts"] = class_counts
        if positive is not None:
            result["label_summary"] = {
                "positive": positive.tolist(), "negative": negative.tolist(),
                "finite_labels": labeled.tolist(),
            }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=URLS, required=True)
    ap.add_argument("--cache", type=Path, default=HERE / "data")
    ap.add_argument("--data-archive", type=Path)
    ap.add_argument("--split-archive", type=Path)
    args = ap.parse_args()
    cache = inside(args.cache) / args.dataset
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "frozen_manifest.json"
    if manifest_path.exists():
        raise SystemExit("A frozen manifest already exists; choose a new cache directory.")
    os.environ.setdefault("TORCH_HOME", str(cache / "torch_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg_cache"))
    archives = []
    ext = ".tar.gz" if args.dataset == "malnet_tiny" else ".zip"
    archive = cache / "archives" / ("data" + ext)
    archives.append(acquire(URLS[args.dataset]["data"], archive, args.data_archive,
                            VERIFIED_ARCHIVES.get(args.dataset)))
    from torch_geometric.datasets import GNNBenchmarkDataset, MalNetTiny, PPI
    from torch_geometric.transforms import LocalDegreeProfile
    providers = []
    official_indices = None
    if args.dataset == "molhiv":
        target = cache / "provider" / "ogbg_molhiv"
        if not target.exists():
            extraction = cache / "archive_extract"
            extract(archive, extraction)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extraction / "hiv"), target)
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(target.parent))
        indices = dataset.get_idx_split()
        official_indices = {"train": indices["train"], "val": indices["valid"], "test": indices["test"]}
        splits = {key: dataset[value] for key, value in official_indices.items()}
        providers.append(PygGraphPropPredDataset)
        feature_rule = "Official OGB categorical atom(9)/bond(3) features unchanged; add_inverse_edge=True. No RDKit conversion."
    elif args.dataset == "ppi":
        root = cache / "provider"
        extract(archive, root / "raw")
        splits = {key: PPI(root=str(root), split=key) for key in ["train", "val", "test"]}
        providers.append(PPI)
        feature_rule = "Official float32 x(50), binary y(121); provider removes self loops; preserve directed edge entries."
    elif args.dataset == "mnist_superpixels":
        root = cache / "provider"
        extract(archive, root / "MNIST" / "raw")
        def add_positions(data):
            import torch
            if data.x.size(-1) != 1 or data.pos.size(-1) != 2:
                raise ValueError("Unexpected MNIST raw feature dimensions; inspect provider before adapting.")
            data.x = torch.cat([data.x, data.pos], dim=-1)
            return data
        splits = {key: GNNBenchmarkDataset(root=str(root), name="MNIST", split=key,
                                          transform=add_positions)
                  for key in ["train", "val", "test"]}
        providers.append(GNNBenchmarkDataset)
        feature_rule = "Concatenate released x(1 intensity) and pos(2 coordinates); preserve all original edges/edge_attr; use x(3) in sparse-mean SAGE."
    else:
        root = cache / "provider"
        extract(archive, root / "raw")
        split_archive = cache / "archives" / "split.zip"
        archives.append(acquire(URLS[args.dataset]["split"], split_archive, args.split_archive))
        extract(split_archive, root / "raw")
        transform = LocalDegreeProfile()
        splits = {key: MalNetTiny(root=str(root), split=key, transform=transform)
                  for key in ["train", "val", "test"]}
        providers.extend([MalNetTiny, LocalDegreeProfile])
        feature_rule = "Keep released directed edges; compute PyG LocalDegreeProfile(5) from each graph, independently; no labels enter features."
    summaries = {key: summarize(value, key, None if official_indices is None else official_indices[key])
                 for key, value in splits.items()}
    versions = {}
    for package in ["torch", "torch-geometric", "numpy", "ogb", "networkx", "rdkit"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    provider_sources = []
    for provider in providers:
        source = inspect.getsource(provider).encode()
        p = cache / (provider.__name__ + "_provider_source.txt")
        p.write_bytes(source)
        provider_sources.append({"class": provider.__name__, "path": str(p), "sha256": sha(p)})
    files = []
    for p in sorted((cache / "provider").rglob("*")):
        if p.is_file():
            files.append({"relative_path": str(p.relative_to(cache)), "bytes": p.stat().st_size,
                          "sha256": sha(p)})
    manifest = {
        "schema": "gnnm-new-development-data-v1", "dataset": args.dataset,
        "cache": str(cache), "archives": archives, "feature_rule": feature_rule,
        "source_sha256": sha(Path(__file__)), "versions": versions,
        "provider_sources": provider_sources, "raw_and_processed_files": files,
        "split_summaries": summaries, "scoring_performed": False,
        "confirmation_data_accessed": False,
        "tensor_hash_spec": "SHA256(json sorted dtype/shape + contiguous native NumPy C-order bytes); split hash sorted compact JSON of per-graph fingerprints and official indices.",
    }
    manifest_temp = manifest_path.with_suffix(".json.partial")
    manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # A interrupted download/load leaves successful archives/provider data in place.
    # A complete freeze is immutable and never overwritten on a retry.
    if manifest_path.exists():
        manifest_temp.unlink()
        raise ValueError("Another process completed this freeze; refusing overwrite.")
    manifest_temp.replace(manifest_path)
    # Console output omits per-task label counts and individual test fingerprints.
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha(manifest_path),
                      "splits": {key: {x: value[x] for x in ["graphs", "nodes", "directed_edge_entries", "canonical_split_sha256"]}
                                 for key, value in summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
