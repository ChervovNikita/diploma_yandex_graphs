"""Download and verify the three OGB datasets used by the rebuttal extension."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import time
import zipfile

import ogb
import pandas as pd
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset

DATASETS = {
    "ogbg-molbace": {"family": "graphproppred", "graphs": 1513, "archive_sha256": "583b20e86641ca83e7f94b44bfd1cb43e42d50e391b868c77fce77eb504fd5b1"},
    "ogbg-molhiv": {"family": "graphproppred", "graphs": 41127, "archive_sha256": "47d747664b9e1653de5aac99bf26c015d88a3daa474f5a32f3fe5c014111375b"},
    "ogbl-collab": {"family": "linkproppred", "nodes": 235868, "archive_sha256": "c5563198e041c338f0a78e11322bb2eb2de76b68f0e9ae3e3b6d6af2d8ca64cc"},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_archive(name, meta, root):
    downloads = root / ".ogb_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / (meta["download_name"] + ".zip")
    url = meta["url"].replace("http://", "https://", 1)
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        request = urllib.request.Request(url, headers={"User-Agent": "GNNM-rebuttal-data/1"})
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as stream:
                    expected = response.headers.get("Content-Length")
                    size = 0
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                        size += len(chunk)
                    if expected is not None and size != int(expected):
                        raise ValueError("Incomplete dataset download " + name)
                partial.replace(archive)
                break
            except (OSError, ValueError) as error:
                print(f"Download attempt {attempt} failed for {name}: {error}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(2 * attempt)
    if sha256(archive) != DATASETS[name]["archive_sha256"]:
        raise ValueError("Downloaded archive differs from the verified OGB release " + name)
    target = root / name.replace("-", "_")
    release = target / ("RELEASE_v" + str(meta["version"]) + ".txt")
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix=name + "-", dir=downloads) as directory:
            staging = Path(directory).resolve()
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    destination = (staging / member.filename).resolve()
                    if not destination.is_relative_to(staging):
                        raise ValueError("Archive member escapes dataset directory")
                bad = bundle.testzip()
                if bad is not None:
                    raise ValueError("Dataset archive CRC failed " + bad)
                bundle.extractall(staging)
            extracted = staging / meta["download_name"]
            marker = extracted / ("RELEASE_v" + str(meta["version"]) + ".txt")
            if not marker.is_file():
                raise ValueError("Official OGB version marker missing")
            extracted.rename(target)
    if not release.is_file():
        raise ValueError("Existing dataset directory has a different or incomplete release")
    return {
        "dataset": name, "official_metadata_url": meta["url"], "download_url": url,
        "archive": str(archive.relative_to(root)), "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size, "release": str(meta["version"]),
    }


def verify_dataset(name, root):
    specification = DATASETS[name]
    if specification["family"] == "graphproppred":
        dataset = PygGraphPropPredDataset(name=name, root=str(root))
        if len(dataset) != specification["graphs"]:
            raise ValueError("Unexpected official graph count")
        splits = dataset.get_idx_split()
        joined = torch.cat([splits[part].cpu() for part in ("train", "valid", "test")])
        if joined.unique().numel() != len(dataset) or joined.numel() != len(dataset):
            raise ValueError("Graph split coverage or disjointness failed")
        first = dataset[0]
        if first.x.shape[1] != 9 or first.edge_attr.shape[1] != 3:
            raise ValueError("Unexpected OGB atom/bond feature schema")
        return {"graphs": len(dataset), "split": "scaffold", "metric": dataset.eval_metric,
                "split_counts": {part: int(len(splits[part])) for part in splits},
                "atom_feature_columns": 9, "bond_feature_columns": 3}
    dataset = PygLinkPropPredDataset(name=name, root=str(root))
    data = dataset[0]
    if data.num_nodes != specification["nodes"]:
        raise ValueError("Unexpected official Collab node count")
    splits = dataset.get_edge_split()
    counts = {}
    for part in ("train", "valid", "test"):
        edges = splits[part]["edge"]
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.min() < 0 or edges.max() >= data.num_nodes:
            raise ValueError("Invalid official edge endpoints")
        counts[part + "_positive"] = len(edges)
        if "edge_neg" in splits[part]:
            negative = splits[part]["edge_neg"]
            if negative.ndim != 2 or negative.shape[1] != 2:
                raise ValueError("Invalid official negative edge shape")
            counts[part + "_negative"] = len(negative)
    return {"nodes": data.num_nodes, "message_edges": data.edge_index.shape[1],
            "node_features": data.x.shape[1], "split": "time", "metric": dataset.eval_metric,
            "split_counts": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "ogb_transfer_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema": "gnnm-ogb-transfer-data-v1", "ogb_version": ogb.__version__, "datasets": {}}
    package_root = Path(ogb.__file__).resolve().parent
    downloads = root / ".ogb_downloads"
    downloads.mkdir(exist_ok=True)
    for name in args.datasets:
        with (downloads / (name + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            family = DATASETS[name]["family"]
            metadata = pd.read_csv(package_root / family / "master.csv",
                                   index_col=0, keep_default_na=False)[name].to_dict()
            print("Preparing " + name, flush=True)
            receipt = prepare_archive(name, metadata, root)
            receipt.update(verify_dataset(name, root))
            dataset_path = root / name.replace("-", "_")
            receipt["raw_and_split_files"] = {
                str(path.relative_to(dataset_path)): sha256(path)
                for folder in ("raw", "split")
                for path in sorted((dataset_path / folder).rglob("*")) if path.is_file()
            }
            manifest["datasets"][name] = receipt
            write_json(manifest_path, manifest)
            print(json.dumps({key: value for key, value in receipt.items()
                              if key != "raw_and_split_files"}), flush=True)


if __name__ == "__main__":
    main()
