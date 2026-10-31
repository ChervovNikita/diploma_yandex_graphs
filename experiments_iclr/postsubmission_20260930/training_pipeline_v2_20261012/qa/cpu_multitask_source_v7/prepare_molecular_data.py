"""Custodial OGB-MolHIV/MolBACE graph-pack preparation, never a fit or score.

The official scaffold TRAIN is retained. Official VALID is prospectively split
into disjoint SELECT/CAL; official TEST becomes the protected evaluate role.
This four-role standardized protocol is not published author replication.
Numerical providers load only when prepare() is explicitly invoked.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform
import sys

DATASETS = {"ogbg-molhiv", "ogbg-molbace"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(path, project_root):
    path, root = Path(path).resolve(), Path(project_root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset/program custody file lies outside the project")
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": digest_file(path)}


def atomic_json(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def fraction(value):
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("SELECT fraction must be a finite rational between zero and one") from error
    if not 0 < result < 1:
        raise ValueError("SELECT fraction must be strictly between zero and one")
    return result


def stratified_select_cal(ids, labels, *, seed, select_fraction="1/2"):
    """Stable public-ID hash ranking within binary classes, with no scoring."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Declare a nonnegative prospective split seed")
    portion = fraction(select_fraction)
    if len(ids) != len(labels) or not ids or len(set(ids)) != len(ids):
        raise ValueError("Unique official VALID identities and aligned labels required")
    classes = {0: [], 1: []}
    for identifier, label in zip(ids, labels):
        if type(identifier) is not int or identifier < 0 or not math.isfinite(float(label)) or label not in (0, 1):
            raise ValueError("Only complete binary official VALID labels are admitted")
        classes[int(label)].append(identifier)
    selected, calibration = [], []
    for label in (0, 1):
        ranked = sorted(classes[label], key=lambda identifier:
            (hashlib.sha256(f"{seed}:official-valid:{identifier}".encode("ascii")).digest(), identifier))
        count = len(ranked)*portion.numerator//portion.denominator
        if not 1 <= count < len(ranked):
            raise ValueError("Each official VALID class must populate both SELECT and CAL")
        selected.extend(ranked[:count])
        calibration.extend(ranked[count:])
    return sorted(selected), sorted(calibration)


def validate_roles(roles, dataset_size):
    if set(roles) != {"train", "select", "cal", "evaluate"}:
        raise ValueError("Declare exactly four molecular roles")
    seen = set()
    for role, ids in roles.items():
        if not ids or any(type(i) is not int or not 0 <= i < dataset_size for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid or duplicate molecular role identities: " + role)
        if seen.intersection(ids):
            raise ValueError("Molecular role identities overlap")
        seen.update(ids)
    if seen != set(range(dataset_size)):
        raise ValueError("Official split must cover every provider graph exactly once")


@contextmanager
def forbid_provider_prompts():
    previous = builtins.input
    def forbidden(prompt=""):
        raise RuntimeError("Provider requested an interactive cache/version decision; inspect the cache explicitly")
    builtins.input = forbidden
    try:
        yield
    finally:
        builtins.input = previous


def provider_dataset(name, cache_root, *, allow_download):
    """Official PyG OGB loader, with explicit headless download consent only."""
    from ogb.graphproppred import PygGraphPropPredDataset
    download_events = []
    class CustodialDataset(PygGraphPropPredDataset):
        def download(self):
            if not allow_download:
                raise FileNotFoundError("Official OGB cache is missing; explicit --allow-download is required")
            # The original OGB downloader asks for consent through this helper.
            # Consent here comes from the explicit CLI flag, not a simulated UI.
            globals_ = PygGraphPropPredDataset.download.__globals__
            original = globals_.get("decide_download")
            if not callable(original):
                raise RuntimeError("Unknown OGB downloader contract; inspect this provider version")
            globals_["decide_download"] = lambda url: True
            download_events.append({"started": utc(), "official_url": self.url})
            try:
                return super().download()
            finally:
                globals_["decide_download"] = original
                download_events[-1]["ended"] = utc()
    with forbid_provider_prompts():
        dataset = CustodialDataset(name=name, root=str(cache_root))
    return dataset, PygGraphPropPredDataset, download_events


def pack_role(torch, dataset, ids):
    """Concatenate original fields; preserve graph-local edges and graph IDs."""
    atoms, edges, bonds, labels = [], [], [], []
    node_ptr, edge_ptr = [0], [0]
    for identifier in ids:
        graph = dataset[identifier]
        x, edge, attr = graph.x.detach().cpu(), graph.edge_index.detach().cpu(), graph.edge_attr.detach().cpu()
        if x.dtype != torch.long or x.ndim != 2 or x.shape[1] != 9 or not len(x):
            raise ValueError("Provider changed the original categorical atom interface")
        if edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2 or attr.dtype != torch.long or attr.shape != (edge.shape[1], 3):
            raise ValueError("Provider changed the original categorical bond/edge interface")
        if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= len(x)):
            raise ValueError("Provider graph has nonlocal edge indices")
        target = graph.y.detach().cpu().reshape(-1)
        if target.shape != (1,) or not bool(torch.isfinite(target).all()) or bool(((target != 0) & (target != 1)).any()):
            raise ValueError("MolHIV/MolBACE require one complete binary target per graph")
        atoms.append(x)
        edges.append(edge)
        bonds.append(attr)
        labels.append(target.to(torch.float32))
        node_ptr.append(node_ptr[-1]+len(x))
        edge_ptr.append(edge_ptr[-1]+edge.shape[1])
    return {"x": torch.cat(atoms, dim=0), "edge_index": torch.cat(edges, dim=1),
            "edge_attr": torch.cat(bonds, dim=0), "node_ptr": torch.tensor(node_ptr, dtype=torch.long),
            "edge_ptr": torch.tensor(edge_ptr, dtype=torch.long), "ids": torch.tensor(ids, dtype=torch.long),
            "y": torch.stack(labels)}


def provider_custody(provider, dataset):
    programs = []
    for value in (provider, type(dataset).__mro__[1]):
        path = Path(inspect.getfile(value)).resolve()
        if str(path) not in {r["path"] for r in programs}:
            programs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path)})
    versions = {}
    for name in ("torch", "torch-geometric", "ogb", "numpy", "pandas"):
        versions[name] = importlib.metadata.version(name)
    return {"versions": versions, "programs": programs, "python": sys.version,
            "platform": platform.platform(), "dataset_name": dataset.name,
            "split_type": str(dataset.meta_info["split"]), "task_type": str(dataset.task_type),
            "num_tasks": int(dataset.num_tasks), "eval_metric": str(dataset.eval_metric),
            "provider_meta_info": {str(k): str(v) for k, v in dataset.meta_info.items()},
            "pre_transform": None, "transform": None}


def prepare(*, project_root, dataset_name, dataset_root, output, split_seed,
            select_fraction="1/2", allow_download=False):
    """Create fresh data artifacts only; caller owns finite resource supervision."""
    root = Path(project_root).resolve()
    cache, destination = Path(dataset_root).resolve(), Path(output).resolve()
    if dataset_name not in DATASETS:
        raise ValueError("Only official ogbg-molhiv and ogbg-molbace are supported")
    if not cache.is_relative_to(root) or not destination.is_relative_to(root):
        raise ValueError("Molecular cache/output must stay inside the project")
    if destination == cache or destination.is_relative_to(cache) or cache.is_relative_to(destination):
        raise ValueError("Keep fresh output separate from the official provider cache")
    if destination.exists():
        raise FileExistsError("Retain previous preparation; use a fresh output directory")
    portion = fraction(select_fraction)
    if type(split_seed) is not int or not 0 <= split_seed < 2**63 or type(allow_download) is not bool:
        raise ValueError("Explicit split seed and download-consent boolean required")
    destination.mkdir(parents=True)
    phase = "provider_load"
    status = {"schema_version": 1, "dataset": dataset_name, "started": utc(), "status": "preparing",
              "model_constructed": False, "optimizer_created": False, "training_launched": False,
              "scoring_performed": False, "author_protocol_replication": False}
    atomic_json(destination / "STATUS.json", status)
    try:
        import torch
        from adapters.molecular import validate_pack
        dataset, provider, downloads = provider_dataset(dataset_name, cache, allow_download=allow_download)
        custody = provider_custody(provider, dataset)
        if custody["split_type"] != "scaffold" or custody["num_tasks"] != 1:
            raise ValueError("Unexpected official scaffold/binary provider recipe")
        phase = "split_admission"
        official = dataset.get_idx_split()
        if set(official) != {"train", "valid", "test"}:
            raise ValueError("Unexpected official OGB split authority")
        official = {key: [int(i) for i in value.tolist()] for key, value in official.items()}
        if any(not ids or len(set(ids)) != len(ids) for ids in official.values()):
            raise ValueError("Invalid official split identities")
        all_official = [i for ids in official.values() for i in ids]
        if len(set(all_official)) != len(all_official) or set(all_official) != set(range(len(dataset))):
            raise ValueError("Official scaffold split is overlapping or incomplete")
        valid_labels = [float(dataset[i].y.reshape(-1)[0].item()) for i in official["valid"]]
        selected, calibration = stratified_select_cal(official["valid"], valid_labels,
            seed=split_seed, select_fraction=str(portion))
        roles = {"train": official["train"], "select": selected, "cal": calibration, "evaluate": official["test"]}
        validate_roles(roles, len(dataset))
        phase = "provider_cache_custody"
        dataset_directory = Path(dataset.root).resolve()
        if not dataset_directory.is_relative_to(cache):
            raise ValueError("Provider cache moved outside its declared dataset root")
        cache_files = [receipt(path, root) for path in sorted(dataset_directory.rglob("*")) if path.is_file()]
        atomic_json(destination / "PROVIDER_CUSTODY.json", {**custody,
            "cache_root": str(cache.relative_to(root)), "dataset_directory": str(dataset_directory.relative_to(root)),
            "cache_files": cache_files, "download_consent": allow_download, "download_events": downloads})
        split = {"schema_version": 1, "dataset": dataset_name,
            "protocol": "official_scaffold_TRAIN; hash_class_stratified_official_VALID_to_SELECT_CAL; official_TEST_to_evaluate",
            "official_indices": official, "roles": roles, "split_seed": split_seed,
            "select_fraction": str(portion), "rank_key": "SHA256(ASCII(seed:official-valid:graph_id)); per-class floor; sorted output IDs",
            "labels_used_for_role_stratification": True, "predictions_or_scores_used": False,
            "graph_structure_used_for_role_assignment": False, "author_protocol_replication": False,
            "author_protocol_difference": "Published official VALID is divided into SELECT and CAL; this is a separately declared standardized four-role recipe"}
        atomic_json(destination / "SPLITS.json", split)
        packs = {}
        for role, ids in roles.items():
            phase = "pack_"+role
            pack = validate_pack(torch, pack_role(torch, dataset, ids), 1, "complete_binary")
            path = destination / (role.upper()+".pt")
            temporary = path.with_name(path.name+".tmp")
            torch.save(pack, temporary)
            temporary.replace(path)
            packs[role] = {**receipt(path, root), "graphs": len(ids), "nodes": len(pack["x"]),
                           "directed_edges": pack["edge_index"].shape[1], "tasks": 1,
                           "binary_class_counts": {str(label): int((pack["y"] == label).sum().item()) for label in (0, 1)}}
            del pack
        phase = "closure"
        split_receipt = receipt(destination / "SPLITS.json", root)
        provider_receipt = receipt(destination / "PROVIDER_CUSTODY.json", root)
        manifest = {"schema_version": 1, "dataset": dataset_name, "dataset_graphs": len(dataset),
            "task": "graph_binary_classification", "target_policy": "complete_binary",
            "roles": {role: {key: value for key, value in item.items() if key in {"path", "bytes", "sha256"}}
                      for role, item in packs.items()}, "pack_descriptives": packs,
            "split_receipt": split_receipt, "provider_custody_receipt": provider_receipt,
            "preparation_program": receipt(Path(__file__), root),
            "role_validation": {"all_four_disjoint": True, "complete_provider_universe": True,
                                "original_atom_bond_fields": True, "graph_local_edges": True},
            "author_protocol_replication": False, "model_constructed": False,
            "optimizer_created": False, "training_launched": False, "scoring_performed": False,
            "scientific_launch_allowed": False, "complete": True, "closed": utc()}
        atomic_json(destination / "MANIFEST.json", manifest)
        status.update(status="complete_dataprep", closed=utc(), manifest=receipt(destination / "MANIFEST.json", root))
        atomic_json(destination / "STATUS.json", status)
        return manifest
    except BaseException as error:
        status.update(status="failed_dataprep", failed_phase=phase, closed=utc(),
                      error_type=type(error).__name__, error=str(error))
        atomic_json(destination / "STATUS.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset-root", required=True, help="Official provider cache inside project root")
    parser.add_argument("--output", required=True, help="Fresh project directory for four role packs")
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--select-fraction", default="1/2")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly authorize official OGB dataset download")
    args = parser.parse_args()
    manifest = prepare(project_root=args.project_root, dataset_name=args.dataset,
        dataset_root=args.dataset_root, output=args.output, split_seed=args.split_seed,
        select_fraction=args.select_fraction, allow_download=args.allow_download)
    print(json.dumps({"dataset": manifest["dataset"], "complete_dataprep": True,
                      "role_counts": {role: value["graphs"] for role, value in manifest["pack_descriptives"].items()},
                      "manifest": str(Path(args.output).resolve() / "MANIFEST.json"),
                      "training_launched": False, "scoring_performed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
