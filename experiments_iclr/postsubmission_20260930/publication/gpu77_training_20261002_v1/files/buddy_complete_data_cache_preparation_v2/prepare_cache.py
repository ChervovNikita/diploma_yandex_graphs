"""Stage closed-test official train/public data and qualify before sealed BUDDY build."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify_manifest(root, expected_sha=None):
    root = Path(root)
    manifest_path = root / "SOURCE_MANIFEST.json"
    checksum = file_sha(manifest_path)
    seal = json.loads((root / "SEAL.json").read_text())
    if checksum != seal["source_manifest_sha256"] or (expected_sha is not None and checksum != expected_sha):
        raise RuntimeError("Source manifest seal mismatch.")
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        path = root / item["path"]
        if path.is_symlink() or file_sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Sealed source changed: {path}")
    return checksum


def confined_path(path, repo):
    path = Path(path).absolute()
    resolved = path.resolve()
    if path != resolved or not resolved.is_relative_to(repo) or resolved == repo:
        raise RuntimeError(f"Path must be inside the repository without symlink traversal: {path}")
    return resolved


def member_metadata(member):
    return {"name": member.filename, "bytes": member.file_size,
            "compressed_bytes": member.compress_size, "CRC": member.CRC}


def check_archive_members(members, expected):
    names = set()
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        mode = member.external_attr >> 16
        if (not name or "\\" in name or "\x00" in name or path.is_absolute()
                or any(part in (".", "..") for part in name.rstrip("/").split("/"))
                or str(path) != name.rstrip("/") or name in names
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
            raise RuntimeError(f"Unsafe/duplicate/nonregular archive member: {name}")
        names.add(name)
    actual = sorted((member_metadata(member) for member in members), key=lambda item: item["name"])
    if actual != sorted(expected, key=lambda item: item["name"]):
        raise RuntimeError("Archive directory metadata differs from the acquired official release.")


def stage_archive(archive, dataset_root, contract):
    archive, dataset_root = Path(archive), Path(dataset_root)
    if dataset_root.exists():
        raise RuntimeError("Dataset root must be fresh; no overwrite or processed-cache reuse.")
    if archive.stat().st_size != contract["archive_bytes"] or file_sha(archive) != contract["archive_sha256"]:
        raise RuntimeError("Already acquired archive bytes changed; do not redownload or substitute.")
    allow = contract["stage_members"]
    if len(set(allow)) != len(allow) or any("test.pt" in name or "split_dict.pt" in name for name in allow):
        raise RuntimeError("Invalid closed-test extraction allowlist.")
    evidence = []
    with zipfile.ZipFile(archive) as packed:
        members = packed.infolist()  # metadata only for closed members
        check_archive_members(members, contract["archive_metadata"])
        by_name = {member.filename: member for member in members}
        if any(name not in by_name or by_name[name].is_dir() for name in allow):
            raise RuntimeError("Required official public/train/valid member is absent.")
        dataset_root.mkdir(parents=True, exist_ok=False)
        for name in allow:
            relative = PurePosixPath(name).relative_to("collab")
            destination = dataset_root / "ogbl_collab" / str(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            checksum = hashlib.sha256()
            count = 0
            with packed.open(by_name[name], "r") as source, destination.open("xb") as target:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)
                    checksum.update(block)
                    count += len(block)
            if count != by_name[name].file_size:
                raise RuntimeError("Truncated official archive member.")
            item = {"archive_member": name, "staged_path": str(destination),
                    "bytes": count, "sha256": checksum.hexdigest()}
            if destination.suffix == ".gz":
                # Preserve both exact ZIP-expanded .csv.gz bytes and exact
                # gzip-expanded public CSV bytes; no test gzip/file is opened.
                csv_checksum = hashlib.sha256()
                csv_bytes = 0
                with gzip.open(destination, "rb") as public_csv:
                    for block in iter(lambda: public_csv.read(8 * 1024 * 1024), b""):
                        csv_checksum.update(block)
                        csv_bytes += len(block)
                item.update(csv_sha256=csv_checksum.hexdigest(), csv_bytes=csv_bytes)
            evidence.append(item)
    return evidence


def verify_installed_ogb(contract):
    distribution = importlib.metadata.distribution("ogb")
    if distribution.version != contract["ogb_version"]:
        raise RuntimeError("OGB version differs from inspected installed sources.")
    sources = {}
    for name, expected in contract["installed_ogb_source_hashes"].items():
        path = Path(distribution.locate_file(f"ogb/{name}")).resolve()
        if file_sha(path) != expected:
            raise RuntimeError(f"Installed OGB source differs from inspected evidence: {name}")
        sources[name] = str(path)
    with Path(sources["linkproppred/master.csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata = {next(iter(row.values())): row["ogbl-collab"] for row in rows}
    expected = {"download_name": "collab", "version": "1", "add_inverse_edge": "True",
                "has_node_attr": "True", "has_edge_attr": "False", "split": "time",
                "additional node files": "None", "additional edge files": "edge_weight,edge_year",
                "is hetero": "False", "binary": "False"}
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise RuntimeError("Installed collab raw/split metadata differs from inspected convention.")
    return {"version": distribution.version, "files": sources, "hashes": contract["installed_ogb_source_hashes"], "collab_metadata": metadata}


def qualify_public_graph(dataset, graph, dataset_root, contract, builder):
    import numpy as np
    import pandas as pd
    import torch
    raw = Path(dataset_root) / "ogbl_collab" / "raw"
    def array(name):
        return pd.read_csv(raw / f"{name}.csv.gz", compression="gzip", header=None).values
    nodes = array("num-node-list")
    edges_count = array("num-edge-list")
    if (nodes.shape != (1, 1) or nodes[0, 0] != contract["node_count"] or edges_count.shape != (1, 1)
            or not np.isfinite(edges_count).all() or not np.equal(edges_count, np.floor(edges_count)).all()):
        raise RuntimeError("Official raw graph count/dimensions differ.")
    pairs = array("edge")
    weights = array("edge_weight")
    years = array("edge_year")
    features = array("node-feat").astype(np.float32)
    count = int(edges_count[0, 0])
    if count <= 0 or pairs.shape != (count, 2) or weights.shape != (count, 1) or years.shape != (count, 1):
        raise RuntimeError("Raw edge/weight/year shape or count mismatch.")
    for name, values in (("edge", pairs), ("weight", weights), ("year", years)):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise RuntimeError(f"Raw {name} must contain finite integers.")
    if (pairs < 0).any() or (pairs >= contract["node_count"]).any() or (pairs[:, 0] == pairs[:, 1]).any():
        raise RuntimeError("Raw graph endpoint range/self-pair mismatch.")
    if (weights <= 0).any() or (years <= 0).any() or (years > contract["train_max_year"]).any():
        raise RuntimeError("Raw weights/year violate complete training-only topology.")
    if features.shape != (contract["node_count"], contract["feature_columns"]) or not np.isfinite(features).all():
        raise RuntimeError("Raw feature dimensions/finiteness differ.")
    expected_edges = np.repeat(pairs.astype(np.int64).T, 2, axis=1)
    expected_edges[:, 1::2] = pairs.astype(np.int64)[:, ::-1].T
    if (graph.num_nodes != contract["node_count"] or graph.edge_index.dtype != torch.int64
            or not torch.equal(graph.edge_index, torch.from_numpy(expected_edges))
            or not torch.equal(graph.x, torch.from_numpy(features)) or graph.x.dtype != torch.float32):
        raise RuntimeError("Processed graph differs from full raw dimensions/endpoint order/features.")
    for name, values in (("edge_weight", weights), ("edge_year", years)):
        actual = getattr(graph, name)
        expected = torch.from_numpy(np.repeat(values, 2, axis=0))
        if actual.shape != expected.shape or not torch.equal(actual.to(torch.float64), expected.to(torch.float64)):
            raise RuntimeError(f"Processed {name} values/order differ from raw reciprocal expansion.")
    if dataset.transform is not None or dataset.pre_transform is not None or dataset.root != str(raw.parent):
        raise RuntimeError("Unexpected dataset transformation/root.")
    return {"pairs": torch.from_numpy(pairs.astype(np.int64)), "weights": torch.from_numpy(weights.reshape(-1)),
            "years": torch.from_numpy(years.reshape(-1)), "node_count": graph.num_nodes,
            "raw_edge_rows": count, "directed_entries": graph.edge_index.shape[1],
            "raw_features_tensor_sha256": builder.tensor_sha(graph.x),
            "ordered_graph_edge_sha256": builder.tensor_sha(graph.edge_index),
            "ordered_graph_weight_sha256": builder.tensor_sha(graph.edge_weight),
            "ordered_graph_year_sha256": builder.tensor_sha(graph.edge_year)}


def qualify_train_split(train, graph, raw, contract, builder):
    import numpy as np
    import torch
    from torch_sparse import coalesce
    for name in ("edge", "weight", "year"):
        if name not in train or not isinstance(train[name], torch.Tensor) or not torch.isfinite(train[name]).all():
            raise RuntimeError(f"Official train split requires finite {name} tensor.")
    pairs, weights, years = train["edge"], train["weight"].reshape(-1), train["year"].reshape(-1)
    count = raw["raw_edge_rows"]
    if pairs.shape != (count, 2) or weights.shape != (count,) or years.shape != (count,) or pairs.dtype != torch.int64:
        raise RuntimeError("Official train rows/dtypes differ from complete raw population.")
    if ((pairs < 0).any() or (pairs >= contract["node_count"]).any() or (pairs[:, 0] == pairs[:, 1]).any()
            or (weights <= 0).any() or (years <= 0).any() or (years > contract["train_max_year"]).any()
            or not torch.equal(weights.double(), weights.double().floor()) or not torch.equal(years.double(), years.double().floor())):
        raise RuntimeError("Official train values/range/year are invalid.")
    def sorted_records(edge, weight, year):
        canonical = np.sort(edge.numpy(), axis=1)
        result = np.column_stack((canonical, weight.numpy(), year.numpy())).astype(np.int64)
        return result[np.lexsort((result[:, 3], result[:, 2], result[:, 1], result[:, 0]))]
    if not np.array_equal(sorted_records(pairs, weights, years), sorted_records(raw["pairs"], raw["weights"], raw["years"])):
        raise RuntimeError("Raw edge/year/weight multiset is not exactly the official train split.")
    graph_edges, graph_weights = coalesce(graph.edge_index, graph.edge_weight.reshape(-1), graph.num_nodes, graph.num_nodes)
    train_edges = torch.cat((pairs.T, pairs[:, [1, 0]].T), dim=1)
    train_edges, train_weights = coalesce(train_edges, torch.cat((weights, weights)).to(graph_weights.dtype), graph.num_nodes, graph.num_nodes)
    if not torch.equal(graph_edges, train_edges) or not torch.equal(graph_weights, train_weights):
        raise RuntimeError("Coalesced weighted graph is not exactly official training-only topology.")
    degrees = torch.zeros(graph.num_nodes, dtype=torch.float64)
    degrees.index_add_(0, graph_edges[1], graph_weights.double())
    return {"status": "complete_train_topology_pass", "train_positive_rows": count,
            "train_positive_sha256": builder.tensor_sha(pairs), "train_weight_sha256": builder.tensor_sha(weights),
            "train_year_sha256": builder.tensor_sha(years), "train_max_year": int(years.max()),
            "coalesced_edge_index_sha256": builder.tensor_sha(graph_edges),
            "coalesced_edge_weight_sha256": builder.tensor_sha(graph_weights),
            "expected_degree_sha256": builder.tensor_sha(degrees.float()),
            "coalesced_directed_entries": graph_edges.shape[1], "raw_train_record_multiset_equal": True}


def qualify_validation(valid, contract, builder):
    import torch
    for name in ("edge", "edge_neg"):
        value = valid.get(name)
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.ndim != 2
                or value.shape[1] != 2 or not len(value) or (value < 0).any() or (value >= contract["node_count"]).any()):
            raise RuntimeError(f"Official validation {name} dimensions/range mismatch.")
    if len(valid["edge_neg"]) != contract["evaluation_negative_rows"]:
        raise RuntimeError("Official validation negative pool must remain exactly 100000 rows.")
    years = valid.get("year")
    if years is not None and (years.numel() != len(valid["edge"]) or not torch.isfinite(years).all() or not (years == contract["valid_year"]).all()):
        raise RuntimeError("Validation year differs from official temporal convention.")
    return {"positive_rows": len(valid["edge"]), "negative_rows": len(valid["edge_neg"]),
            "positive_sha256": builder.tensor_sha(valid["edge"]), "negative_sha256": builder.tensor_sha(valid["edge_neg"])}


def qualify_cache_contents(cache, train, valid, graph, topology, contract, builder):
    import torch
    common = torch.load(Path(cache) / "common.pt", map_location="cpu")
    if (common["x"].shape != (contract["node_count"], contract["feature_columns"])
            or common["x"].dtype != torch.float32 or not torch.isfinite(common["x"]).all()
            or common["degrees"].shape != (contract["node_count"],)
            or not torch.isfinite(common["degrees"]).all()
            or builder.tensor_sha(common["degrees"]) != topology["expected_degree_sha256"]):
        raise RuntimeError("Complete common feature/degree cache is invalid.")
    def finite_tensors(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tensors(item) for item in value.values())
        raise RuntimeError("Unexpected native hash/card cache value.")
    if not finite_tensors(common["hashes"]) or not finite_tensors(common["cards"]):
        raise RuntimeError("Nonfinite native hash/card bank.")
    if set(common["hashes"]) != {0, 1, 2} or common["cards"].shape != (contract["node_count"], 2):
        raise RuntimeError("Incomplete native hash/card hop population.")
    for hop in range(3):
        if (common["hashes"][hop]["minhash"].shape != (contract["node_count"], 128)
                or common["hashes"][hop]["minhash"].dtype != torch.int64
                or common["hashes"][hop]["hll"].shape != (contract["node_count"], 256)
                or common["hashes"][hop]["hll"].dtype != torch.int8):
            raise RuntimeError("Incomplete native hash-bank dimensions.")
    for split, official in (("train", train), ("valid", valid)):
        data = torch.load(Path(cache) / f"{split}.pt", map_location="cpu")
        links, sf = data["links"], data["sf"]
        positive_count = len(official["edge"])
        if (data["n_positive"] != positive_count or links.dtype != torch.int64
                or links.ndim != 2 or links.shape[1] != 2 or (links < 0).any() or (links >= contract["node_count"]).any()
                or not torch.equal(links[:positive_count], official["edge"])
                or sf.shape != (len(links), 8) or sf.dtype != torch.float32
                or not torch.isfinite(sf).all() or (sf[:, [4, 5]] != 0).any()):
            raise RuntimeError(f"Complete {split} candidate/structural cache is invalid.")
        negative = links[positive_count:]
        if split == "valid":
            if not torch.equal(negative, official["edge_neg"]):
                raise RuntimeError("Official validation negatives/order changed.")
        else:
            if len(negative) != positive_count or (negative[:, 0] == negative[:, 1]).any():
                raise RuntimeError("Fixed native training-negative population is invalid.")
            codes = negative[:, 0] * graph.num_nodes + negative[:, 1]
            if torch.unique(codes).numel() != len(codes):
                raise RuntimeError("Duplicate training negative rows.")
            train_codes = torch.unique(graph.edge_index[0] * graph.num_nodes + graph.edge_index[1])
            if torch.isin(codes, train_codes, assume_unique=True).any():
                raise RuntimeError("Training negative overlaps training topology.")
    return {"all_nodes_retained": True, "all_structural_rows_finite": True,
            "native_zero_columns_4_5_exact": True, "native_hash_bank_complete_finite": True,
            "degrees_equal_full_weighted_train_oracle": True, "candidate_positive_and_validation_negative_order_exact": True,
            "training_negatives_unique_train_disjoint_and_nonself": True,
            "future_positives_never_opened_or_filtered": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-output", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--source-dir", default=str(HERE.parent / "buddy_shared_cache_execution_v4"))
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists() or args.cpu_threads < 1:
        raise RuntimeError("Actual authorized repository and positive CPU thread count required.")
    paths = {name: confined_path(getattr(args, name), repo) for name in
             ("archive", "dataset_root", "cache_output", "receipt", "qualification", "source_dir")}
    if not HERE.is_relative_to(repo) or any(paths[name].exists() for name in ("dataset_root", "cache_output", "receipt")):
        raise RuntimeError("Use fresh repository-confined outputs.")
    wrapper_manifest = verify_manifest(HERE)
    contract = json.loads((HERE / "CONTRACT.json").read_text())
    source_manifest = verify_manifest(paths["source_dir"], contract["builder_source_manifest_sha256"])
    installed = verify_installed_ogb(contract)
    # This wrapper builds/qualifies data on CPU only, including on a GPU host.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Pinned OGB1.3.6 omits weights_only for official PyG/NumPy data. Torch2.6+
    # changed that default. Use Torch's documented process-local compatibility
    # switch after source qualification; only the SHA-bound archive is staged.
    # Explicit weights_only=True checkpoint guards remain enforced by Torch.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(paths["source_dir"]))
    import torch
    import cache_builder as builder
    from run import qualification
    qualification(paths["qualification"])
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    receipt = {"schema": "buddy-complete-data-cache-qualification-v2", "UTC": datetime.now(timezone.utc).isoformat(),
               "wrapper_manifest_sha256": wrapper_manifest, "builder_manifest_sha256": source_manifest,
               "CPU_qualification_sha256": file_sha(paths["qualification"]),
               "Torch_version": str(torch.__version__), "OGB_legacy_data_load_environment": "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
               "archive_sha256": contract["archive_sha256"], "installed_ogb": installed,
               "cpu_threads": args.cpu_threads, "GPU_compute": False, "test_members_opened": False,
               "combined_split_accessor_called": False, "dataset_root": str(paths["dataset_root"]),
               "cache_output": str(paths["cache_output"]), "status": "in_progress"}
    try:
        receipt["staged_members"] = stage_archive(paths["archive"], paths["dataset_root"], contract)
        # Direct calls to frozen v4 only: no metadata shim or runtime patch of
        # builder/OGB/Torch functions. Test/aggregate files are never staged.
        dataset = builder.offline_dataset(paths["dataset_root"])
        graph = dataset[0]
        raw = qualify_public_graph(dataset, graph, paths["dataset_root"], contract, builder)
        receipt["public_graph"] = {key: value for key, value in raw.items() if key not in {"pairs", "weights", "years"}}
        train, train_path, loader_source = builder.official_split_file(dataset, "train")
        valid, valid_path, valid_source = builder.official_split_file(dataset, "valid")
        if loader_source != valid_source:
            raise RuntimeError("Installed loader changed during complete-data qualification.")
        receipt["train_topology"] = qualify_train_split(train, graph, raw, contract, builder)
        receipt["validation"] = qualify_validation(valid, contract, builder)
        receipt["qualified_official_split_files"] = {"train": file_sha(train_path), "valid": file_sha(valid_path)}
        receipt["qualification_before_hashing"] = True
        del raw  # release independent raw arrays before full native cache build
        for member in receipt["staged_members"]:
            if file_sha(member["staged_path"]) != member["sha256"]:
                raise RuntimeError("Staged official file changed during qualification.")
        builder.build(argparse.Namespace(dataset_root=str(paths["dataset_root"]), output=str(paths["cache_output"])))
        manifest = builder.load_manifest(paths["cache_output"])
        topology = receipt["train_topology"]
        if (manifest["edge_index_sha256"] != topology["coalesced_edge_index_sha256"]
                or manifest["edge_weight_sha256"] != topology["coalesced_edge_weight_sha256"]
                or manifest["raw_features_sha256"] != receipt["public_graph"]["raw_features_tensor_sha256"]
                or manifest["test_split_opened"] is not False
                or manifest["split_identity"]["train"]["positive_sha256"] != topology["train_positive_sha256"]
                or manifest["split_identity"]["valid"]["positive_sha256"] != receipt["validation"]["positive_sha256"]
                or manifest["split_identity"]["valid"]["negative_sha256"] != receipt["validation"]["negative_sha256"]
                or any(manifest["official_split_files"][name]["sha256"] != value for name, value in receipt["qualified_official_split_files"].items())):
            raise RuntimeError("Cache graph/features identity differs from qualified complete data.")
        receipt["cache_contents"] = qualify_cache_contents(paths["cache_output"], train, valid, graph, topology, contract, builder)
        receipt.update(status="complete_official_train_valid_cache_qualified",
                       cache_manifest_sha256=builder.file_sha(paths["cache_output"] / "manifest.json"))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["seconds_including_staging_qualification_and_build"] = time.monotonic() - started
        paths["receipt"].parent.mkdir(parents=True, exist_ok=True)
        write_json(paths["receipt"], receipt)


if __name__ == "__main__":
    main()
