#!/usr/bin/env python3
"""Authenticate cached official collab TRAIN/raw/VALID and write fit-only packs.

No OGB dataset or split accessor, download, model, optimizer, score, or TEST read.
The cache root is explicit; an archived authority's old dataset_root is ignored.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from adapters.link import (AUTHOR_COMMIT, AUTHOR_REPOSITORY, AUTHOR_SOURCE,
                           bound_file, file_sha, sampler_custody, validate_inputs)


SOURCE_FILES = {
    "split/time/train.pt": {"bytes": 41130429, "sha256": "eb1a1deb3a40fa870fcef21c1f19809ab1806c6395061ef4fbc899fc320e16a1"},
    "split/time/valid.pt": {"bytes": 3885091, "sha256": "aaf6fdbd3a3783c4a0e2b337de8f0cff8e42351a9216fecff2a6b8b056cc01ac"},
    "raw/node-feat.csv.gz": {"bytes": 105975881, "sha256": "1e8e52b4ef3558bd823f64db48de0803fcc5d449eccd24ddd8a0187ab799acf8"},
    "raw/edge.csv.gz": {"bytes": 4933748, "sha256": "20d2281149aa0503453d5a13a99e22fc129964155f30b1caf2057c7886e7b82e"},
}
ARRAY_FIELDS = {"train": {"edge", "weight", "year"}, "valid": {"edge", "weight", "year", "edge_neg"}}
SOURCE_DIRECTORY = "graph_link_competing_family_source_20261003_v1/source"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_authority(authority):
    require(isinstance(authority, dict) and authority.get("schema") == "ncnc-collab-TRAIN-VALID-data-authority-v1", "Expected frozen official collab TRAIN/VALID authority")
    files = authority.get("files")
    require(isinstance(files, dict) and set(files) == set(SOURCE_FILES), "Authority must bind exactly the four allowed TRAIN/raw/VALID files")
    for name, pin in SOURCE_FILES.items():
        require(isinstance(files[name], dict) and all(files[name].get(k) == v for k, v in pin.items()), "Official collab file pin changed: " + name)
    arrays = authority.get("expected_arrays")
    require(isinstance(arrays, dict) and set(arrays) == set(ARRAY_FIELDS), "Expected TRAIN and VALID array custody only")
    for role, fields in ARRAY_FIELDS.items():
        require(isinstance(arrays[role], dict) and set(arrays[role]) == fields, "Official split array fields changed: " + role)
        for spec in arrays[role].values():
            require(isinstance(spec, dict) and set(spec) == {"shape", "dtype", "sha256"}, "Malformed source array custody")
    require(authority.get("full_node_count") == 235868, "Full official node count changed")
    require(isinstance(authority.get("train_raw_tensor_digests"), dict) and set(authority["train_raw_tensor_digests"]) == {"train_records", "raw_features", "ordered_raw_graph"}, "Bind exact TRAIN/raw tensor digests")
    return authority


def authenticate_cache(cache_root, authority):
    """Resolve and hash only the explicit allowlist before any pickle read."""
    cache = Path(cache_root).resolve()
    paths = {}
    for name in SOURCE_FILES:
        pin = authority["files"][name]
        paths[name] = bound_file(cache, {"path": name, "bytes": pin["bytes"], "sha256": pin["sha256"]})
    return paths


def tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.shape, array.dtype)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def load_official_arrays(torch, numpy, pandas, paths, authority):
    # These old official files pickle NumPy arrays. Their exact known file pins
    # were authenticated above, before this trusted legacy deserialization.
    result = {}
    for role in ("train", "valid"):
        stored = torch.load(paths["split/time/"+role+".pt"], map_location="cpu", weights_only=False)
        require(type(stored) is dict and set(stored) == ARRAY_FIELDS[role], "Official source split fields differ")
        result[role] = {}
        for field, value in stored.items():
            spec = authority["expected_arrays"][role][field]
            require(type(value) is numpy.ndarray and str(value.dtype) == spec["dtype"] and list(value.shape) == spec["shape"], "Official source array shape/dtype differs")
            tensor = torch.from_numpy(value)
            require(tensor_sha(tensor) == spec["sha256"], "Official source array digest differs")
            result[role][field] = tensor
    train, valid = result["train"], result["valid"]
    require(tuple(train["edge"].shape) == (1179052, 2) and tuple(valid["edge"].shape) == (60084, 2) and tuple(valid["edge_neg"].shape) == (100000, 2), "Complete collab role records differ")
    require(int(train["year"].min()) == 1963 and int(train["year"].max()) == 2017 and bool((valid["year"] == 2018).all()), "Official collab temporal split differs")
    x = torch.from_numpy(pandas.read_csv(paths["raw/node-feat.csv.gz"], compression="gzip", header=None).values.astype(numpy.float32))
    raw = pandas.read_csv(paths["raw/edge.csv.gz"], compression="gzip", header=None).values.T.astype(numpy.int64)
    # Literal OGB reverse-edge insertion order, authenticated by tensor digest.
    expanded = numpy.repeat(raw, 2, axis=1)
    expanded[0, 1::2] = expanded[1, 0::2]
    expanded[1, 1::2] = expanded[0, 0::2]
    raw_edges = torch.from_numpy(expanded)
    require(tuple(x.shape) == (235868, 128) and tuple(raw_edges.shape) == (2, 2358104), "Complete collab feature/raw graph shapes differ")
    digests = {"train_records": tensor_sha(train["edge"]), "raw_features": tensor_sha(x), "ordered_raw_graph": tensor_sha(raw_edges)}
    require(digests == authority["train_raw_tensor_digests"], "Authenticated TRAIN/raw tensor identities differ")
    return {"x": x, "raw_edge_index": raw_edges}, {"positive": train["edge"]}, {"positive": valid["edge"], "negative": valid["edge_neg"]}, digests


def receipt(project_root, path):
    root, item = Path(project_root).resolve(), Path(path).resolve()
    require(item.is_relative_to(root), "Output/source must stay within the project root")
    return {"path": str(item.relative_to(root)), "bytes": item.stat().st_size, "sha256": file_sha(item)}


def source_bindings(project_root):
    return {name: {"path": SOURCE_DIRECTORY+"/ncn__"+name+".py", **pin} for name, pin in AUTHOR_SOURCE.items()}


def prepare(*, project_root, cache_root, authority_path, output, duplicate_policy="native_record_multiset", admission_path=None):
    project, destination = Path(project_root).resolve(), Path(output).resolve()
    require(destination.is_relative_to(project) and destination != project, "Use a fresh preparation directory within the project")
    require(not destination.exists(), "Never overwrite a prior data packet; use a fresh output directory")
    require(duplicate_policy in {"native_record_multiset", "require_unique_pairs"}, "Declare native record multiset or unique-pair policy")
    authority_file = Path(authority_path).resolve()
    authority = validate_authority(json.loads(authority_file.read_text()))
    paths = authenticate_cache(cache_root, authority)
    admission = None
    if admission_path is not None:
        admission_file = Path(admission_path).resolve()
        require(admission_file.is_relative_to(project), "Archive admission receipt must stay inside project custody")
        admission = json.loads(admission_file.read_text())
        require(admission.get("verified_server_custody") is True and admission.get("official_files") == SOURCE_FILES
                and admission.get("cache_root") == str(Path(cache_root).resolve()), "Archive admission differs from this exact cache")
        require(all(admission.get(k) is False for k in ("test_file_opened", "test_member_extracted", "all_split_accessor_called", "predictive_metrics_computed")), "Archive admission opened a forbidden role or scored data")
    sources = source_bindings(project)
    for item in sources.values():
        bound_file(project, item)
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.utils import negative_sampling
    from ogb.linkproppred import Evaluator
    graph, train, select, digests = load_official_arrays(torch, np, pd, paths, authority)
    audit = validate_inputs(torch, graph, train, select, {"support": {"record_duplicate_policy": duplicate_policy,
        "select_negative_self_policy": "preserve_official_records"}, "metric": {"hits_k": 50}})
    sampler = sampler_custody(negative_sampling)
    evaluator = Path(inspect.getfile(Evaluator)).resolve()
    # Every validation above runs before creating output. No role filtering.
    destination.mkdir(parents=True)
    bindings = {}
    for role, pack in (("graph", graph), ("train", train), ("select", select)):
        path = destination / (role.upper()+".pt")
        torch.save(pack, path)
        bindings[role] = receipt(project, path)
    manifest = {
        "schema_version": 1, "schema": "gnnm-link-collab-fit-data-v1", "UTC": datetime.now(timezone.utc).isoformat(),
        "dataset": "ogbl-collab", "cache_root": str(Path(cache_root).resolve()),
        "source_authority": {"path": str(authority_file), "bytes": authority_file.stat().st_size, "sha256": file_sha(authority_file)},
        "preparation_program": receipt(project, Path(__file__).resolve()),
        "official_source_files": SOURCE_FILES, "link_inputs": bindings, "link_source": sources,
        "sampler": {k: sampler[k] for k in ("name", "source_sha256", "function_sha256")},
        "negative_sampler_custody": sampler,
        "metric": {"hits_k": 50, "evaluator_source_sha256": file_sha(evaluator)},
        "evaluator_custody": {"path": str(evaluator), "bytes": evaluator.stat().st_size, "sha256": file_sha(evaluator)},
        "support": {"graph": "train_only", "maskinput": True, "record_duplicate_policy": duplicate_policy, "select_overlap": "reject",
            "select_negative_self_policy": "preserve_official_records"},
        "input_audit": audit, "tensor_digests": {**digests, "select_positive": tensor_sha(select["positive"]), "select_negative": tensor_sha(select["negative"])},
        "role_protocol": "official temporal TRAIN intact; entire official VALID is SELECT; CAL/evaluate authority not created",
        "ignored_source_fields": {"weight": "native unweighted GCN/support and record supervision", "year": "authenticated temporal role custody; not model input"},
        "author_repository": AUTHOR_REPOSITORY, "author_commit": AUTHOR_COMMIT,
        "author_protocol_replication": False, "native_training_inputs_preserved": True,
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "torch-geometric", "ogb")},
        "trusted_legacy_numpy_pickle": "exact official file bytes authenticated before torch.load(weights_only=False)",
        "test_file_opened": False, "model_or_checkpoint_loaded": False, "predictive_metrics_computed": False,
        "original_paper_scores_changed": False, "science_ready": False,
    }
    if admission is not None:
        manifest["source_admission_receipt"] = receipt(project, admission_file)
        manifest["archive_admission"] = {k: admission[k] for k in ("provider", "archive", "official_files", "extracted_archive_members")}
    path = destination / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path, help="Verified four-member archive admission receipt, when freshly downloaded")
    parser.add_argument("--record-duplicate-policy", choices=("native_record_multiset", "require_unique_pairs"), default="native_record_multiset")
    args = parser.parse_args()
    result = prepare(project_root=args.project_root, cache_root=args.cache_root, authority_path=args.authority,
                     output=args.output, duplicate_policy=args.record_duplicate_policy, admission_path=args.admission)
    print(json.dumps({"status": "prepared", "dataset": result["dataset"], "output": str(args.output.resolve()),
                      "input_audit": result["input_audit"], "science_ready": False}, indent=2))


if __name__ == "__main__":
    main()
