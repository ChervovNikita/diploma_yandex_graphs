#!/usr/bin/env python3
"""Frozen-protocol, one-cell coordinator v3 for a NEW node-classification study.

Source preparation only until independently authorized. Importing this module imports
only the Python standard library. Dataset acquisition and model work are confined
to explicit commands; no test-label reader or test-scoring command exists here.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace


SCHEMA_VERSION = 1
DATASET_NAMES = {"CoauthorCS", "AmazonPhoto"}
ARMS = {"original", "factor", "coordinate", "permutation", "bias_only",
        "single", "untied", "heads", "gt_sep_single"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def absolute_path(value):
    require(isinstance(value, str) and Path(value).is_absolute(),
            "Every protocol path must be absolute")
    return Path(value).resolve()


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path == root or root in path.parents, "Path escapes its supplied phase root")
    return path


def phase_paths(protocol, mode):
    phase = protocol["phases"][mode]
    root = absolute_path(phase["root"])
    names = ["cache_root", "output_root"]
    if mode == "acquire":
        names += ["pyg_root", "public_root", "sealed_root"]
    paths = {"root": root}
    for name in names:
        paths[name] = within(absolute_path(phase[name]), root)
        require(paths[name] != root, name + " must be a subdirectory of the phase root")
    values = [paths[n] for n in names]
    for i, left in enumerate(values):
        for right in values[i + 1:]:
            require(left != right and left not in right.parents and right not in left.parents,
                    "Phase data/cache/output directories must be disjoint")
    return paths


def verify_frozen_json(path, expected_sha):
    path = absolute_path(str(path))
    require(len(expected_sha) == 64, "Supply an explicit SHA256 digest")
    actual = sha256(path)
    require(actual == expected_sha, "Frozen JSON hash mismatch: " + path.name)
    return json.loads(path.read_text(encoding="utf-8")), actual


def validate_recipe(recipe):
    for name in ("num_layers", "width", "members", "max_epochs", "min_epochs",
                 "patience", "eval_every"):
        require(type(recipe[name]) is int and recipe[name] > 0,
                "Recipe field must be a positive integer: " + name)
    require(recipe["min_epochs"] <= recipe["max_epochs"], "Invalid epoch bounds")
    require(0 <= recipe["dropout"] < 1, "Invalid dropout")
    require(recipe["checkpoint_metric"] == "validation_nll", "Only validation NLL selects checkpoints")
    require(recipe["checkpoint_ties"] == "earliest", "Checkpoint ties must retain earliest epoch")
    require(recipe["patience_unit"] == "epochs_since_improvement", "Patience must be explicit in epochs")
    require(recipe["scheduler"] == "constant", "This runner supports the declared constant scheduler")
    kwargs = recipe["model_kwargs"]
    require(set(kwargs) == {"hidden_dim_multiplier", "normalization", "num_heads"},
            "Specify all supported architecture kwargs explicitly")
    require(kwargs["hidden_dim_multiplier"] > 0 and kwargs["normalization"] == "layer",
            "Invalid model architecture kwargs")
    require(type(kwargs["num_heads"]) is int and kwargs["num_heads"] > 0, "Invalid num_heads")
    optimizer = recipe["optimizer"]
    require(optimizer["name"] in {"Adam", "AdamW"}, "Unsupported optimizer")
    require(optimizer["lr"] > 0 and optimizer["weight_decay"] >= 0, "Invalid optimizer scalars")
    require(len(optimizer["betas"]) == 2 and all(0 <= x < 1 for x in optimizer["betas"]),
            "Invalid optimizer betas")
    require(optimizer["eps"] > 0, "Invalid optimizer eps")


def validate_profiling(profile):
    require(profile["cold_calls"] == 1 and
            profile["scope"] == "full_graph_forward_probability_pool_and_development_gather",
            "Initial and warm profiling scope must be explicit")
    for specification in (profile, profile["selected_checkpoint_cold_deployment"]):
        require(type(specification["warmups"]) is int and specification["warmups"] >= 0 and
                type(specification["repeats"]) is int and specification["repeats"] > 0,
                "Freeze valid profiling warmups and repeats before execution")
    breakdown = profile["operation_breakdown"]
    require(type(breakdown["enabled"]) is bool and breakdown["operations"] == [
        "full_graph_member_logits_forward", "development_member_logits_gather", "development_probability_pool"],
        "Freeze the fixed operation breakdown")
    if breakdown["enabled"]:
        require(type(breakdown["warmups"]) is int and breakdown["warmups"] >= 0 and
                type(breakdown["repeats"]) is int and breakdown["repeats"] > 0,
                "Freeze valid operation warmups/repeats")


def verify_inputs(args):
    protocol, protocol_sha = verify_frozen_json(args.protocol, args.protocol_sha256)
    source_manifest, sources_sha = verify_frozen_json(args.sources, args.sources_sha256)
    require(protocol["schema_version"] == SCHEMA_VERSION, "Unsupported protocol version")
    require(protocol["frozen"] is True and protocol["study_kind"] == "new_study",
            "Require a supplied frozen NEW-study protocol")
    require(source_manifest["schema_version"] == SCHEMA_VERSION, "Unsupported source manifest")
    require(protocol["source_manifest_sha256"] == sources_sha, "Protocol/source manifest binding mismatch")
    require(set(protocol["pooling"]) == {"primary", "same_checkpoint_mean_logit_sensitivity"},
            "Pooling must be explicitly declared")
    require(protocol["pooling"]["primary"] == "arithmetic_mean_softmax" and
            protocol["pooling"]["same_checkpoint_mean_logit_sensitivity"] is True,
            "Require primary probability pooling and same-checkpoint sensitivity")
    require(protocol["test_policy"] == "sealed_independent_confirmation_only", "Invalid test policy")
    require(protocol["preprocessing"]["features"] in {"none", "row_sum"}, "Invalid feature preprocessing")
    require(protocol["preprocessing"]["edges"] == "as_pyg", "Graph topology must be as acquired by PyG")
    require(args.dataset in DATASET_NAMES and args.dataset in protocol["datasets"], "Dataset not frozen")
    files, seen = [], set()
    for entry in source_manifest["files"]:
        path = absolute_path(entry["path"])
        require(path not in seen, "Duplicate source path")
        seen.add(path)
        require(sha256(path) == entry["sha256"], "Source hash mismatch: " + path.name)
        files.append({"path": str(path), "sha256": entry["sha256"], "role": entry["role"]})
    model_entry = absolute_path(source_manifest["model_entry"])
    require(model_entry in seen and Path(__file__).resolve() in seen,
            "Hash both model entry and this coordinator in the source manifest")
    require(any(x["role"] == "author_model_dependency" for x in files),
            "Source manifest must include the pinned original author model dependency")
    require(any(x["role"] == "author_dataset_loader" for x in files),
            "Source manifest must include audited author dataset loader source")
    dataset = protocol["datasets"][args.dataset]
    require(len(dataset["pyg_loader_sha256"]) == 64, "Frozen PyG dataset loader hash required")
    require(type(dataset["test_holdout"]["seed"]) is int, "Explicit test holdout seed required")
    require(0 < dataset["test_holdout"]["fraction"] < 1, "Invalid test fraction")
    require(dataset["rounding"] == "floor_test_and_validation_per_class_train_remainder",
            "Explicit test/validation-floor, training-remainder allocation required")
    split_ids = set()
    require(len(dataset["splits"]) > 0, "At least one prospective development split is required")
    for split in dataset["splits"]:
        require(isinstance(split["id"], str) and split["id"] not in split_ids, "Split IDs must be unique strings")
        require(split["id"].replace("_", "").replace("-", "").isalnum(), "Unsafe split ID")
        split_ids.add(split["id"])
        require(type(split["seed"]) is int and 0 < split["train_fraction_of_development"] < 1,
                "Invalid prospective split")
    for mode in ("acquire", "qualify", "fit"):
        phase_paths(protocol, mode)
    if args.mode != "acquire":
        require(args.seed is not None and args.split is not None and args.arm is not None and args.recipe is not None,
                "Qualify/fit require one explicit seed, split, arm and recipe")
        matches = [cell for cell in protocol["cells"] if cell["dataset"] == args.dataset and
                   cell["seed"] == args.seed and cell["split"] == args.split and cell["arm"] == args.arm
                   and cell["recipe"] == args.recipe]
        require(len(matches) == 1, "Cell must appear exactly once in the frozen protocol")
        cell = matches[0]
        require(args.split in split_ids and args.arm in ARMS, "Invalid frozen cell")
        require(args.recipe.replace("_", "").replace("-", "").isalnum(), "Unsafe recipe ID")
        require(type(cell["permutation_seed"]) is int, "Explicit permutation seed required")
        recipe = protocol["recipes"][cell["recipe"]]
        validate_recipe(recipe)
        validate_profiling(protocol["profiling"])
        gate = cell["storage_gate"]
        require(type(gate["expected_model_tensor_bytes"]) is int and gate["expected_model_tensor_bytes"] > 0,
                "Freeze exact expected deployed model bytes for each cell")
        require(type(gate["expected_index_buffer_bytes"]) is int and gate["expected_index_buffer_bytes"] >= 0,
                "Freeze expected index-buffer bytes")
        require(protocol["storage_policy"] == {"parameter_dtype": "torch.float32", "index_dtype": "torch.int64"},
                "This study charges actual FP32 parameters and int64 indices")
        require(args.data_manifest and args.data_manifest_sha256, "Supply the sealed acquisition manifest digest")
    else:
        require(args.seed is None and args.split is None and args.arm is None and args.recipe is None,
                "Acquisition creates only the prospectively listed splits")
        cell, recipe = None, None
    return protocol, protocol_sha, source_manifest, sources_sha, files, model_entry, cell, recipe


def prepare_cache(paths):
    root = paths["cache_root"]
    root.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    # Set before importing torch/PyG; all framework caches remain in phase paths.
    for variable, child in {"TORCH_HOME": "torch", "XDG_CACHE_HOME": "xdg",
                            "CUDA_CACHE_PATH": "cuda", "TRITON_CACHE_DIR": "triton",
                            "TORCH_EXTENSIONS_DIR": "torch_extensions", "NUMBA_CACHE_DIR": "numba",
                            "HF_HOME": "huggingface", "MPLCONFIGDIR": "matplotlib"}.items():
        target = root / child
        target.mkdir(exist_ok=True)
        os.environ[variable] = str(target)


def new_output(paths, args):
    suffix = args.dataset if args.mode == "acquire" else f"{args.dataset}__{args.split}__seed{args.seed}__{args.arm}__{args.recipe}"
    directory = paths["output_root"] / suffix
    require(not directory.exists(), "Output already exists; do not overwrite or silently resume a cell")
    directory.mkdir(parents=True)
    return directory


def snapshot_inputs(output, args, files):
    shutil.copyfile(args.protocol, output / "frozen_protocol.json")
    shutil.copyfile(args.sources, output / "frozen_sources.json")
    require(sha256(output / "frozen_protocol.json") == args.protocol_sha256 and
            sha256(output / "frozen_sources.json") == args.sources_sha256,
            "Frozen input changed during snapshotting")
    snapshots = output / "verified_sources"
    snapshots.mkdir()
    mapping = []
    for i, entry in enumerate(files):
        target = snapshots / f"{i:03d}_{Path(entry['path']).name}"
        shutil.copyfile(entry["path"], target)
        require(sha256(target) == entry["sha256"], "Source snapshot copy hash mismatch")
        mapping.append({**entry, "snapshot": str(target.relative_to(output))})
    write_json(output / "source_snapshots.json", mapping)


def artifact_entry(path, root):
    path = Path(path)
    return {"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": path.stat().st_size}


def artifact_manifest(output):
    entries = [artifact_entry(path, output) for path in sorted(output.rglob("*"))
               if path.is_file() and path.name != "artifacts.json"]
    write_json(output / "artifacts.json", {"schema_version": 1, "files": entries})


def split_indices(labels, dataset):
    """One shared held-out test set; classwise repartition only its complement."""
    classes = sorted(set(labels))
    require(classes == list(range(len(classes))), "Require contiguous zero-based class labels")
    by_class = {c: [i for i, label in enumerate(labels) if label == c] for c in classes}
    test_rng = random.Random(dataset["test_holdout"]["seed"])
    test, development = [], {}
    for c in classes:
        ids = by_class[c][:]
        test_rng.shuffle(ids)
        n_test = math.floor(len(ids) * dataset["test_holdout"]["fraction"])
        require(0 < n_test < len(ids) - 1, "Class cannot support frozen nonempty train/validation/test partitions")
        test += ids[:n_test]
        development[c] = ids[n_test:]
    splits = {}
    for spec in dataset["splits"]:
        rng = random.Random(spec["seed"])
        train, validation = [], []
        for c in classes:
            ids = development[c][:]
            rng.shuffle(ids)
            n_validation = math.floor(len(ids) * (1 - spec["train_fraction_of_development"]))
            n_train = len(ids) - n_validation
            require(0 < n_train < len(ids), "Class cannot support frozen nonempty development partitions")
            train += ids[:n_train]
            validation += ids[n_train:]
        train, validation = sorted(train), sorted(validation)
        require(not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test)),
                "Partition overlap")
        require(len(train) + len(validation) + len(test) == len(labels), "Partitions do not cover nodes")
        splits[spec["id"]] = {"spec": spec, "train": train, "validation": validation}
    return sorted(test), splits


def capture_loader_source(loader_class, protocol, dataset_name, torch, torch_geometric, output):
    """Acquire/qualify source receipt; never silently adapt an installed provider."""
    loader_path = Path(inspect.getsourcefile(loader_class)).resolve()
    snapshot = output / "actual_installed_dataset_loader.py"
    shutil.copyfile(loader_path, snapshot)
    inventory = {"torch_version": torch.__version__, "torch_geometric_version": torch_geometric.__version__,
                 "loader_source_path": str(loader_path), "loader_source_sha256": sha256(loader_path),
                 "snapshot": artifact_entry(snapshot, output),
                 "frozen_loader_sha256": protocol["datasets"][dataset_name]["pyg_loader_sha256"]}
    write_json(output / "runtime_provider_inventory.json", inventory)
    require(inventory["snapshot"]["sha256"] == inventory["loader_source_sha256"], "Loader snapshot mismatch")
    require(torch.__version__ == protocol["runtime"]["torch_version"] and
            torch_geometric.__version__ == protocol["runtime"]["torch_geometric_version"],
            "Actual framework versions differ; root must review the runtime inventory and adopt separately")
    require(inventory["loader_source_sha256"] == inventory["frozen_loader_sha256"],
            "Actual provider bytes differ; root must review/adopt the phase-retained loader snapshot separately")
    return inventory


def graph_audit(torch, x, edges):
    n = x.shape[0]
    codes = edges[0] * n + edges[1]
    unique = torch.unique(codes)
    unique_set = set(unique.tolist())
    missing_reverse = sum((int(v) * n + int(u)) not in unique_set for u, v in edges.t().tolist())
    return {"edge_entries": edges.shape[1], "unique_directed_edge_pairs": unique.numel(),
            "duplicate_edge_entries": edges.shape[1] - unique.numel(),
            "self_loop_entries": int((edges[0] == edges[1]).sum().item()),
            "entries_without_reverse": missing_reverse, "bidirectional": missing_reverse == 0,
            "feature_negative_entries": int((x < 0).sum().item()),
            "feature_zero_rows": int((x == 0).all(dim=1).sum().item()),
            "feature_min": float(x.min().item()), "feature_max": float(x.max().item()),
            "automatic_edge_transform_applied": False}


def acquire(args, protocol, protocol_sha, sources_sha, paths, output):
    import torch
    import torch_geometric
    from torch_geometric.datasets import Amazon, Coauthor

    loader_class = Coauthor if args.dataset == "CoauthorCS" else Amazon
    loader_inventory = capture_loader_source(loader_class, protocol, args.dataset, torch, torch_geometric, output)

    dataset_root = paths["pyg_root"] / args.dataset
    public = paths["public_root"] / args.dataset
    sealed = paths["sealed_root"] / args.dataset
    require(not public.exists() and not sealed.exists() and not dataset_root.exists(),
            "Acquisition requires new PyG, public and sealed dataset paths")
    public.mkdir(parents=True)
    sealed.mkdir(parents=True, mode=0o700)
    started = time.perf_counter()
    dataset = (Coauthor(root=str(dataset_root), name="CS") if args.dataset == "CoauthorCS"
               else Amazon(root=str(dataset_root), name="Photo"))
    require(len(dataset) == 1, "Expected one full node-classification graph")
    data = dataset[0]
    x, edge_index, labels_tensor = data.x.detach().cpu(), data.edge_index.detach().cpu(), data.y.detach().cpu()
    require(x.ndim == 2 and x.is_floating_point() and bool(torch.isfinite(x).all()), "Invalid features")
    require(labels_tensor.ndim == 1 and labels_tensor.numel() == x.shape[0], "Invalid label rows")
    require(labels_tensor.dtype == torch.long, "Expected integer class labels")
    require(edge_index.dtype == torch.long and edge_index.ndim == 2 and edge_index.shape[0] == 2,
            "Invalid edge_index")
    require(edge_index.numel() > 0 and int(edge_index.min()) >= 0 and int(edge_index.max()) < x.shape[0],
            "Invalid edge endpoints")
    if protocol["preprocessing"]["features"] == "row_sum":
        require(bool((x >= 0).all()), "Row-sum normalization requires nonnegative features")
        denominator = x.sum(dim=1, keepdim=True)
        x = x / torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    labels = labels_tensor.tolist()  # All labels are used only in acquisition/stratification.
    test_ids, splits = split_indices(labels, protocol["datasets"][args.dataset])
    num_nodes, num_classes = x.shape[0], len(set(labels))
    torch.save({"x": x.contiguous(), "edge_index": edge_index.contiguous()}, public / "graph.pt")
    test_tensor = torch.tensor(test_ids, dtype=torch.long)
    torch.save({"indices": test_tensor}, public / "test_indices.pt")
    # This is the sole test-label write. No qualification/fit code opens this file.
    torch.save({"indices": test_tensor, "labels": labels_tensor[test_tensor]}, sealed / "test_labels.pt")
    os.chmod(sealed / "test_labels.pt", 0o600)
    os.chmod(sealed, 0o700)
    split_receipts = {}
    for split_id, split in splits.items():
        split_dir = public / split_id
        split_dir.mkdir()
        for role in ("train", "validation"):
            ids = torch.tensor(split[role], dtype=torch.long)
            torch.save({"indices": ids, "labels": labels_tensor[ids]}, split_dir / f"{role}.pt")
        counts = {role: len(split[role]) for role in ("train", "validation")}
        split_receipts[split_id] = {"spec": split["spec"], "spec_sha256": canonical_sha(split["spec"]),
                                   "counts": counts,
                                   "mask_indices_sha256": {role: canonical_sha(split[role])
                                                           for role in ("train", "validation")},
                                   "train": artifact_entry(split_dir / "train.pt", public),
                                   "validation": artifact_entry(split_dir / "validation.pt", public)}
    raw_snapshots = sealed / "author_raw_snapshots"
    raw_snapshots.mkdir()
    raw_receipts = []
    raw_root = Path(dataset.raw_dir).resolve()
    within(raw_root, dataset_root)
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(raw_root)
        target = raw_snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        os.chmod(target, 0o600)
        raw_receipts.append({"pyg_raw_path": str(path), "source_sha256": sha256(path),
                             "copy_root": str(sealed), "copy": artifact_entry(target, sealed),
                             "access_policy": "acquisition_or_independent_confirmation_only"})
        require(raw_receipts[-1]["source_sha256"] == raw_receipts[-1]["copy"]["sha256"],
                "Author raw copy mismatch")
    require(raw_receipts, "No author raw source files found")
    processed = [artifact_entry(path, dataset_root) for path in sorted(Path(dataset.processed_dir).rglob("*"))
                 if path.is_file()]
    require(processed, "No PyG processed source receipt")
    manifest = {"schema_version": 1, "dataset": args.dataset, "protocol_sha256": protocol_sha,
                "source_manifest_sha256": sources_sha, "public_root": str(public),
                "data_definition_sha256": canonical_sha({"dataset": args.dataset,
                    "spec": protocol["datasets"][args.dataset], "preprocessing": protocol["preprocessing"]}),
                "num_nodes": num_nodes, "num_features": x.shape[1], "num_classes": num_classes,
                "num_edges": edge_index.shape[1], "feature_dtype": str(x.dtype),
                "preprocessing": protocol["preprocessing"],
                "split_algorithm": "stdlib_random_classwise_shuffle_floor_test_validation_train_remainder_v1",
                "graph_feature_audit": graph_audit(torch, x, edge_index),
                "test_holdout": protocol["datasets"][args.dataset]["test_holdout"],
                "test_count": len(test_ids), "graph": artifact_entry(public / "graph.pt", public),
                "test_indices_sha256": canonical_sha(test_ids),
                "test_indices": artifact_entry(public / "test_indices.pt", public),
                "splits": split_receipts, "author_raw": raw_receipts,
                "pyg_processed_root": str(dataset_root), "pyg_processed_files": processed,
                "versions": {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__},
                "executed_pyg_loader": loader_inventory,
                "sealed_test_labels": {"path": str(sealed / "test_labels.pt"),
                                       "sha256": sha256(sealed / "test_labels.pt"),
                                       "access_policy": "independent_confirmation_only"}}
    write_json(public / "data_manifest.json", manifest)
    receipt = {"status": "ACQUIRED_NOT_EXPERIMENT_ADMITTED", "created_utc": utc_now(),
               "data_manifest": str(public / "data_manifest.json"),
               "data_manifest_sha256": sha256(public / "data_manifest.json"),
               "acquisition_seconds": time.perf_counter() - started,
               "label_summary_acquisition_only": {str(c): labels.count(c) for c in range(num_classes)},
               "graph_summary": {k: manifest[k] for k in ("num_nodes", "num_features", "num_classes", "num_edges")},
               "graph_feature_audit": manifest["graph_feature_audit"],
               "split_counts": {sid: {**item["counts"], "test": len(test_ids)}
                                for sid, item in split_receipts.items()},
               "test_label_seal": manifest["sealed_test_labels"],
               "official_masks_used": False, "scientific_scoring_performed": False}
    write_json(output / "acquisition_receipt.json", receipt)
    return receipt


def public_artifact(manifest, entry):
    root = absolute_path(manifest["public_root"])
    path = within(root / entry["path"], root)
    require(sha256(path) == entry["sha256"] and path.stat().st_size == entry["bytes"],
            "Public acquisition artifact mismatch: " + path.name)
    return path


def load_development(args, protocol, torch, output):
    manifest, data_sha = verify_frozen_json(args.data_manifest, args.data_manifest_sha256)
    require(manifest["schema_version"] == 1 and manifest["dataset"] == args.dataset, "Wrong data manifest")
    binding = protocol["data_bindings"][args.dataset]
    require(binding["data_manifest_sha256"] == data_sha and
            binding["acquisition_protocol_sha256"] == manifest["protocol_sha256"] and
            binding["acquisition_source_manifest_sha256"] == manifest["source_manifest_sha256"],
            "Execution protocol must bind the exact acquisition manifest and its freeze")
    require(manifest["data_definition_sha256"] == canonical_sha({"dataset": args.dataset,
                "spec": protocol["datasets"][args.dataset], "preprocessing": protocol["preprocessing"]}),
            "Prospective data definition changed between acquisition and execution")
    acquire_paths = phase_paths(protocol, "acquire")
    require(absolute_path(manifest["public_root"]) == acquire_paths["public_root"] / args.dataset,
            "Public data root differs from the frozen phase path")
    require(manifest["preprocessing"] == protocol["preprocessing"], "Preprocessing changed")
    expected = next(s for s in protocol["datasets"][args.dataset]["splits"] if s["id"] == args.split)
    split = manifest["splits"][args.split]
    require(split["spec"] == expected and split["spec_sha256"] == canonical_sha(expected), "Split changed")
    require(manifest["test_holdout"] == protocol["datasets"][args.dataset]["test_holdout"], "Test holdout changed")
    shutil.copyfile(args.data_manifest, output / "frozen_data_manifest.json")
    require(sha256(output / "frozen_data_manifest.json") == data_sha, "Data manifest changed during snapshotting")
    # These four public files are the complete data-read allowlist. Raw/processed
    # PyG objects and sealed test labels are never opened in scientific phases.
    graph = torch.load(public_artifact(manifest, manifest["graph"]), map_location="cpu", weights_only=True)
    train = torch.load(public_artifact(manifest, split["train"]), map_location="cpu", weights_only=True)
    validation = torch.load(public_artifact(manifest, split["validation"]), map_location="cpu", weights_only=True)
    test = torch.load(public_artifact(manifest, manifest["test_indices"]), map_location="cpu", weights_only=True)
    require(set(graph) == {"x", "edge_index"} and set(train) == set(validation) == {"indices", "labels"}
            and set(test) == {"indices"}, "Unexpected keys in public data bundles")
    x, edge = graph["x"], graph["edge_index"]
    n = manifest["num_nodes"]
    require(x.shape == (n, manifest["num_features"]) and bool(torch.isfinite(x).all()), "Invalid feature bundle")
    require(edge.dtype == torch.long and edge.shape == (2, manifest["num_edges"]), "Invalid edge bundle")
    require(edge.numel() > 0 and int(edge.min()) >= 0 and int(edge.max()) < n, "Invalid edge endpoints")
    groups = [train["indices"], validation["indices"], test["indices"]]
    for ids in groups:
        require(ids.dtype == torch.long and ids.ndim == 1 and ids.numel() > 0, "Invalid node indices")
        require(int(ids.min()) >= 0 and int(ids.max()) < n and ids.unique().numel() == ids.numel(),
                "Duplicate or out-of-range indices")
    require(torch.cat(groups).unique().numel() == n and sum(ids.numel() for ids in groups) == n,
            "Masks must be disjoint and cover the entire graph")
    for pack in (train, validation):
        y = pack["labels"]
        require(y.dtype == torch.long and y.shape == pack["indices"].shape, "Invalid development labels")
        require(int(y.min()) >= 0 and int(y.max()) < manifest["num_classes"], "Invalid development classes")
    mask_hashes = {"train": canonical_sha(train["indices"].tolist()),
                   "validation": canonical_sha(validation["indices"].tolist()),
                   "test_indices_only": canonical_sha(test["indices"].tolist())}
    require(mask_hashes["train"] == split["mask_indices_sha256"]["train"] and
            mask_hashes["validation"] == split["mask_indices_sha256"]["validation"] and
            mask_hashes["test_indices_only"] == manifest["test_indices_sha256"], "Mask-index hash mismatch")
    # Discard test indices after mask integrity checks; they are not evaluation inputs.
    del test, groups
    return manifest, data_sha, graph, train, validation, mask_hashes


@contextlib.contextmanager
def one_gpu_job(protocol, device):
    if not device.startswith("cuda"):
        yield
        return
    path = absolute_path(protocol["runtime"]["gpu_lock_file"])
    within(path, phase_paths(protocol, "acquire")["root"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another coordinator GPU job holds the frozen single-job lock") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "utc": utc_now(), "device": device}) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_model_source(path):
    name = "coordinate_study_models_" + sha256(path)[:16]
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, "Cannot load frozen model source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    # Python bytecode is a phase cache concern; disable writes beside frozen sources.
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    require(callable(module.build_model) and callable(module.storage_report), "Invalid model source API")
    return module


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_and_runtime(torch, runtime, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(runtime["device"])
    require(device.type in {"cpu", "cuda"}, "Only CPU/CUDA runtime is supported")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "Frozen CUDA device is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    require(type(runtime["deterministic_algorithms"]) is bool and type(runtime["allow_tf32"]) is bool,
            "Runtime determinism/TF32 must be explicit")
    torch.use_deterministic_algorithms(runtime["deterministic_algorithms"])
    torch.backends.cuda.matmul.allow_tf32 = runtime["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["allow_tf32"]
    torch.backends.cudnn.benchmark = False
    require(type(runtime["cpu_threads"]) is int and runtime["cpu_threads"] > 0, "Invalid CPU thread count")
    torch.set_num_threads(runtime["cpu_threads"])
    return device


def check_logits(torch, logits, members, num_nodes, num_classes):
    require(tuple(logits.shape) == (members, num_nodes, num_classes), "Model violated [M,N,C] forward contract")
    require(bool(torch.isfinite(logits).all()), "Nonfinite member logits")


def member_losses(torch, logits, ids, labels):
    selected = logits.index_select(1, ids)
    m, n, c = selected.shape
    return torch.nn.functional.cross_entropy(selected.reshape(m * n, c), labels.repeat(m), reduction="none").view(m, n).mean(dim=1)


def pooled_log_probabilities(torch, selected_logits):
    return torch.logsumexp(torch.nn.functional.log_softmax(selected_logits, dim=-1), dim=0) - math.log(selected_logits.shape[0])


def development_metrics(torch, selected_logits, labels):
    pooled = pooled_log_probabilities(torch, selected_logits)
    sensitivity = torch.nn.functional.log_softmax(selected_logits.mean(dim=0), dim=-1)
    result = {"primary_probability_pool": {
        "nll": float(torch.nn.functional.nll_loss(pooled, labels).item()),
        "accuracy": float((pooled.argmax(dim=-1) == labels).float().mean().item()),
        "macro_f1": macro_f1(torch, pooled.argmax(dim=-1), labels, selected_logits.shape[-1])},
        "same_checkpoint_mean_logit_sensitivity": {
            "nll": float(torch.nn.functional.nll_loss(sensitivity, labels).item()),
            "accuracy": float((sensitivity.argmax(dim=-1) == labels).float().mean().item()),
            "macro_f1": macro_f1(torch, sensitivity.argmax(dim=-1), labels, selected_logits.shape[-1])},
        "member_argmax_disagreement": member_disagreement(torch, selected_logits),
        "members": []}
    for member in selected_logits:
        result["members"].append({"nll": float(torch.nn.functional.cross_entropy(member, labels).item()),
                                  "accuracy": float((member.argmax(dim=-1) == labels).float().mean().item()),
                                  "macro_f1": macro_f1(torch, member.argmax(dim=-1), labels, selected_logits.shape[-1])})
    return result


def macro_f1(torch, predictions, labels, num_classes):
    """Unweighted mean F1 across all C classes; zero denominator contributes 0."""
    confusion = torch.bincount(labels * num_classes + predictions,
                               minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    numerator = 2 * confusion.diag()
    scores = torch.where(denominator > 0,
                         numerator.to(torch.float64) / denominator.clamp_min(1).to(torch.float64),
                         torch.zeros(num_classes, dtype=torch.float64, device=labels.device))
    return float(scores.mean().item())


def member_disagreement(torch, selected_logits):
    """Argmax disagreement fraction on selected development nodes, no targets."""
    predictions = selected_logits.argmax(dim=-1)
    pairs = []
    for left in range(predictions.shape[0]):
        for right in range(left + 1, predictions.shape[0]):
            value = float((predictions[left] != predictions[right]).to(torch.float64).mean().item())
            pairs.append({"member_i": left, "member_j": right, "disagreement": value})
    return {"mean_pairwise": sum(pair["disagreement"] for pair in pairs) / len(pairs) if pairs else None,
            "pairs": pairs, "unordered_pair_count": len(pairs), "nodes": selected_logits.shape[1],
            "single_member_value": None, "argmax_ties": "first_class_index"}


def cpu_copy(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v, torch) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v, torch) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v, torch) for v in value)
    return value


def save_checkpoint(path, model, optimizers, torch, epoch, provenance):
    torch.save({"epoch": epoch, "model_state": cpu_copy(model.state_dict(), torch),
                "optimizer_states": [cpu_copy(optimizer.state_dict(), torch) for optimizer in optimizers],
                "provenance": provenance}, path)


def unique_tensor_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return sum(storages.values())


def profile_latency(torch, model, graph, x, ids, profile, device):
    warmups, repeats = profile["warmups"], profile["repeats"]
    require(type(warmups) is int and warmups >= 0 and type(repeats) is int and repeats > 0,
            "Profiling warmups/repeats must be fixed integers")
    require(profile["scope"] == "full_graph_forward_probability_pool_and_development_gather",
            "Unknown end-to-end profiling scope")
    model.eval()
    durations = []
    started = time.perf_counter()
    with torch.no_grad():
        for iteration in range(warmups + repeats):
            synchronize(torch, device)
            tick = time.perf_counter()
            # No label access; profile the useful selected-node probability output.
            logits = model(graph, x)
            selected = logits.index_select(1, ids)
            probabilities = pooled_log_probabilities(torch, selected).exp()
            synchronize(torch, device)
            if iteration >= warmups:
                durations.append(time.perf_counter() - tick)
            del probabilities, selected, logits
    ordered = sorted(durations)
    median = (ordered[len(ordered) // 2] if len(ordered) % 2 else
              (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2)
    return {"scope": profile["scope"], "input_transfer_included": False,
            "output_transfer_included": False, "warmups": warmups, "repeats": repeats,
            "seconds": durations, "median_seconds": median,
            "p95_seconds": ordered[math.ceil(0.95 * repeats) - 1], "percentile_method": "nearest_rank",
            "mean_seconds": sum(durations) / repeats,
            "total_profiling_seconds_including_warmups": time.perf_counter() - started,
            "cuda_synchronization": device.type == "cuda", "speedup_claim": None}


def cold_latency(torch, model, graph, x, ids, profile, device):
    require(profile["cold_calls"] == 1, "Freeze exactly one initial label-free cold evaluation")
    model.eval()
    synchronize(torch, device)
    tick = time.perf_counter()
    with torch.no_grad():
        logits = model(graph, x)
        probabilities = pooled_log_probabilities(torch, logits.index_select(1, ids)).exp()
        synchronize(torch, device)
        seconds = time.perf_counter() - tick
        del logits, probabilities
    return {"seconds": seconds, "calls": 1, "checkpoint": "initial_untrained",
            "scope": profile["scope"], "labels_or_metrics_used": False,
            "cuda_synchronization": device.type == "cuda"}


def timing_summary(values):
    require(len(values) > 0 and all(math.isfinite(value) and value >= 0 for value in values),
            "Invalid measured timing series")
    ordered = sorted(values)
    count = len(ordered)
    median = ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) / 2
    return {"seconds": values, "count": count, "mean_seconds": sum(values) / count,
            "median_seconds": median, "p95_seconds": ordered[math.ceil(0.95 * count) - 1],
            "percentile_method": "nearest_rank"}


def fixed_operation_breakdown(torch, model, graph, x, ids, specification, device):
    """Isolated fixed operations; synchronization overhead prevents additive use."""
    operations = ["full_graph_member_logits_forward", "development_member_logits_gather",
                  "development_probability_pool"]
    require(specification["operations"] == operations, "Operation breakdown must use the fixed declared operations")
    require(type(specification["enabled"]) is bool, "Explicit breakdown enabled flag required")
    if not specification["enabled"]:
        return {"status": "disabled_by_frozen_protocol", "operations": operations}
    warmups, repeats = specification["warmups"], specification["repeats"]
    require(type(warmups) is int and warmups >= 0 and type(repeats) is int and repeats > 0,
            "Operation timings require fixed warmups/repeats")
    started = time.perf_counter()
    model.eval()
    results = {}
    with torch.no_grad():
        for name in operations:
            fixture = None
            if name == "full_graph_member_logits_forward":
                operation = lambda: model(graph, x)
            elif name == "development_member_logits_gather":
                fixture = model(graph, x)
                operation = lambda: fixture.index_select(1, ids)
            else:
                full = model(graph, x)
                fixture = full.index_select(1, ids)
                del full
                operation = lambda: pooled_log_probabilities(torch, fixture).exp()
            durations = []
            for iteration in range(warmups + repeats):
                synchronize(torch, device)
                tick = time.perf_counter()
                result = operation()
                synchronize(torch, device)
                if iteration >= warmups:
                    durations.append(time.perf_counter() - tick)
                del result
            results[name] = {**timing_summary(durations), "warmups": warmups,
                             "resident_fixture_bytes": fixture.numel() * fixture.element_size() if fixture is not None else 0}
            del fixture, operation
    return {"status": "timed_fixed_real_operations", "operations": results,
            "total_seconds_including_fixtures_and_warmups": time.perf_counter() - started,
            "labels_or_metrics_used": False, "cuda_synchronization": device.type == "cuda",
            "additive_decomposition": False, "speedup_claim": None,
            "scope_limit": "Isolated resident operations; fixtures and per-operation synchronization differ from end-to-end execution"}


def cold_deployment_profile(torch, source, factory_kwargs, checkpoint_path, cpu_graph,
                            cpu_gather_indices, specification, device, checkpoint_label):
    """Repeated fresh resident setup; caller must first release fitting GPU assets.

    No labels, optimizer, or train/validation target pack is accepted. The current
    process, CUDA context, source imports, and OS filesystem cache remain warm.
    """
    warmups, repeats = specification["warmups"], specification["repeats"]
    require(type(warmups) is int and warmups >= 0 and type(repeats) is int and repeats > 0,
            "Fresh deployment requires fixed warmups/repeats")
    require(set(cpu_graph) == {"x", "edge_index"} and all(value.device.type == "cpu" for value in cpu_graph.values())
            and cpu_gather_indices.device.type == "cpu", "Deployment setup accepts CPU graph/features/index tensors only")
    timings = []
    peaks = []
    started = time.perf_counter()
    for iteration in range(warmups + repeats):
        synchronize(torch, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None
        total_tick = time.perf_counter()
        tick = time.perf_counter()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(set(checkpoint) == {"model_state", "factory_kwargs", "epoch", "protocol_sha256", "source_manifest_sha256"},
                "Deployment checkpoint must contain model state and provenance only")
        require(checkpoint["factory_kwargs"] == factory_kwargs, "Deployment architecture differs from frozen cell")
        checkpoint_reload = time.perf_counter() - tick
        tick = time.perf_counter()
        fresh_model = source.build_model(**factory_kwargs)
        cpu_model_construction = time.perf_counter() - tick
        tick = time.perf_counter()
        fresh_model.load_state_dict(checkpoint["model_state"])
        cpu_state_application = time.perf_counter() - tick
        del checkpoint
        synchronize(torch, device)
        tick = time.perf_counter()
        fresh_model = fresh_model.to(device)
        synchronize(torch, device)
        model_transfer = time.perf_counter() - tick
        tick = time.perf_counter()
        fresh_x = cpu_graph["x"].to(device)
        fresh_graph = SimpleNamespace(edge_index=cpu_graph["edge_index"].to(device))
        fresh_ids = cpu_gather_indices.to(device)
        synchronize(torch, device)
        input_transfer = time.perf_counter() - tick
        fresh_model.eval()
        tick = time.perf_counter()
        with torch.no_grad():
            fresh_logits = fresh_model(fresh_graph, fresh_x)
            probabilities = pooled_log_probabilities(torch, fresh_logits.index_select(1, fresh_ids)).exp()
            synchronize(torch, device)
        forward_probability_pool = time.perf_counter() - tick
        total_seconds = time.perf_counter() - total_tick  # Direct per-repeat total, never a sum of medians.
        if iteration >= warmups:
            timings.append({"checkpoint_reload": checkpoint_reload, "cpu_model_construction": cpu_model_construction,
                            "cpu_state_application": cpu_state_application, "model_device_transfer": model_transfer,
                            "graph_features_gather_indices_device_transfer": input_transfer,
                            "forward_probability_pool": forward_probability_pool, "end_to_end": total_seconds})
            peaks.append({"allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                          "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                          "baseline_allocated_bytes": baseline_allocated} if device.type == "cuda" else None)
        del probabilities, fresh_logits, fresh_ids, fresh_graph, fresh_x, fresh_model
        synchronize(torch, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    components = {name: timing_summary([repeat[name] for repeat in timings]) for name in timings[0]}
    return {"scope": "within_process_fresh_model_checkpoint_reload_graph_features_indices_transfer_and_probability_output",
            "checkpoint": checkpoint_label, "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_bytes": checkpoint_path.stat().st_size,
            "warmups_fresh_setups": warmups, "measured_fresh_setups": repeats,
            "per_repeat_seconds": timings, "component_summaries": components,
            "end_to_end": components["end_to_end"], "per_repeat_gpu_peak_memory": peaks,
            "total_seconds_including_warmups_and_cleanup": time.perf_counter() - started,
            "labels_or_targets_in_setup": False, "optimizer_in_setup": False,
            "old_resident_gpu_model_graph_retained": False, "input_transfers_included": True,
            "cpu_graph_features_retained_between_repeats": True,
            "cpu_to_gpu_transfers_performed": device.type == "cuda",
            "validation_or_train_gather_indices_included": True, "output_host_transfer_included": False,
            "new_process": False, "process_import_initialization_included": False,
            "cuda_context_initialization_included": False, "filesystem_cache_cleared": False,
            "cuda_allocator_cache_cleared_before_each_setup": device.type == "cuda",
            "per_repeat_end_to_end_excludes_allocator_clear_and_cleanup": True,
            "cuda_synchronization": device.type == "cuda", "speedup_claim": None}


def scientific_phase(args, protocol, protocol_sha, sources_sha, files, model_entry, cell, recipe, output):
    runtime = protocol["runtime"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = runtime["cublas_workspace_config"]
    import torch
    import torch_geometric

    if args.mode == "qualify":
        from torch_geometric.datasets import Amazon, Coauthor
        capture_loader_source(Coauthor if args.dataset == "CoauthorCS" else Amazon,
                              protocol, args.dataset, torch, torch_geometric, output)
    require(torch.__version__ == runtime["torch_version"] and
            torch_geometric.__version__ == runtime["torch_geometric_version"],
            "Execution framework version differs from the frozen runtime")
    with one_gpu_job(protocol, runtime["device"]):
        total_started = time.perf_counter()
        tick = time.perf_counter()
        device = seed_and_runtime(torch, runtime, args.seed)
        torch.set_default_dtype(torch.float32)
        synchronize(torch, device)
        device_initialization_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        manifest, data_sha, pack, train, validation, masks = load_development(
            args, protocol, torch, output)
        cpu_data_load_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        for entry in files:
            require(sha256(entry["path"]) == entry["sha256"], "Frozen source changed before model import")
        source = load_model_source(model_entry)
        members = 1 if args.arm in {"single", "gt_sep_single"} else recipe["members"]
        factory_kwargs = {"arm": args.arm, "input_dim": manifest["num_features"], "hidden_dim": recipe["width"],
                          "output_dim": manifest["num_classes"], "seed": args.seed,
                          "permutation_seed": cell["permutation_seed"], "members": recipe["members"],
                          "num_layers": recipe["num_layers"], "dropout": recipe["dropout"], **recipe["model_kwargs"]}
        model = source.build_model(**factory_kwargs)
        cpu_source_model_construction_seconds = time.perf_counter() - tick
        synchronize(torch, device)
        tick = time.perf_counter()
        model = model.to(device)
        synchronize(torch, device)
        model_transfer_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        x = pack["x"].to(device)
        graph = SimpleNamespace(edge_index=pack["edge_index"].to(device))
        train_ids, train_y = train["indices"].to(device), train["labels"].to(device)
        # Qualification never transfers or scores validation labels.
        val_ids = val_y = None
        if args.mode == "fit":
            val_ids, val_y = validation["indices"].to(device), validation["labels"].to(device)
        synchronize(torch, device)
        input_transfer_seconds = time.perf_counter() - tick
        settings = recipe["optimizer"]
        optimizer_type = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[settings["name"]]
        if args.arm == "untied":
            require(len(model.models) == members, "Untied source must expose its independent member models")
            member_parameters = [list(member.parameters()) for member in model.models]
            identities = [id(parameter) for parameters in member_parameters for parameter in parameters]
            require(len(identities) == len(set(identities)), "Untied members unexpectedly share parameters")
            require(set(identities) == {id(parameter) for parameter in model.parameters()},
                    "Untied member optimizers do not cover every model parameter")
            parameter_groups = member_parameters
        else:
            parameter_groups = [list(model.parameters())]
        optimizers = [optimizer_type(parameters, lr=settings["lr"], weight_decay=settings["weight_decay"],
                                     betas=tuple(settings["betas"]), eps=settings["eps"])
                      for parameters in parameter_groups]
        optimizer_strategy = ("independent_member_optimizers_sum_member_ce_backward" if args.arm == "untied"
                              else "one_optimizer_mean_member_ce_backward")
        qualification_initial_parameters = (
            {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
            if args.mode == "qualify" else None)
        provenance = {"protocol_sha256": protocol_sha, "source_manifest_sha256": sources_sha,
                      "data_manifest_sha256": data_sha, "mask_indices_sha256": masks,
                      "development_target_artifact_sha256": {
                          role: manifest["splits"][args.split][role]["sha256"] for role in ("train", "validation")},
                      "graph_artifact_sha256": manifest["graph"]["sha256"],
                      "cell": cell, "recipe": recipe, "optimizer": settings,
                      "optimizer_strategy": optimizer_strategy, "optimizer_count": len(optimizers)}
        storage = source.storage_report(model)
        write_json(output / "model_storage.json", storage)
        gate = cell["storage_gate"]
        actual_bytes = unique_tensor_bytes(list(model.parameters()) + list(model.buffers()))
        require(storage["unique_storage_bytes"] == actual_bytes, "Independent model storage accounting differs")
        require(storage["total_model_tensor_bytes"] == gate["expected_model_tensor_bytes"] and
                storage["index_buffer_bytes"] == gate["expected_index_buffer_bytes"],
                "Actual deployed/index bytes differ from the prospective cell storage gate")
        require(all(row["dtype"] == "torch.float32" for row in storage["parameters"]) and
                all(row["dtype"] == "torch.int64" for row in storage["buffers"] if row["kind"] == "index"),
                "Actual parameter/index dtype differs from frozen storage policy")
        equal_byte = gate["equal_byte_reference"]
        if equal_byte is not None:
            require(equal_byte["bytes"] > 0 and 0 <= equal_byte["relative_tolerance"] <= 0.01,
                    "Invalid equal-byte comparison gate")
            require(abs(storage["total_model_tensor_bytes"] / equal_byte["bytes"] - 1) <= equal_byte["relative_tolerance"],
                    "Actual control exceeds the prospective equal-byte tolerance")
        tick = time.perf_counter()
        torch.save({"model_state": cpu_copy(model.state_dict(), torch), "provenance": provenance},
                   output / "initial_model_state.pt")
        initial_checkpoint_seconds = time.perf_counter() - tick
        cold_ids = val_ids if args.mode == "fit" else train_ids
        cold = cold_latency(torch, model, graph, x, cold_ids, protocol["profiling"], device)
        del cold_ids
        synchronize(torch, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        setup_seconds = time.perf_counter() - total_started
        training_seconds = validation_seconds = checkpoint_seconds = 0.0
        qualification_evaluation_seconds = 0.0
        curve_path = output / "training_curve.jsonl"
        best_nll, best_epoch, selected_metrics, final_metrics = math.inf, 0, None, None
        qualification_final_train_metrics = None
        max_epochs = recipe["max_epochs"] if args.mode == "fit" else protocol["qualification"]["complete_updates"]
        require(type(max_epochs) is int and max_epochs > 0, "Qualification must specify complete updates")
        stopping_reason = "max_epochs" if args.mode == "fit" else "qualification_complete_updates"
        completed_updates = 0
        with curve_path.open("w", encoding="utf-8") as curve:
            for epoch in range(1, max_epochs + 1):
                model.train()
                synchronize(torch, device)
                tick = time.perf_counter()
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(graph, x)
                check_logits(torch, logits, members, manifest["num_nodes"], manifest["num_classes"])
                losses = member_losses(torch, logits, train_ids, train_y)
                loss = losses.mean()
                loss_value = float(loss.detach().item())
                require(math.isfinite(loss_value), "Nonfinite training objective")
                objective = losses.sum() if args.arm == "untied" else loss
                objective.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                require(len(gradients) > 0, "No parameter gradients were produced")
                require(bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()),
                        "Nonfinite parameter gradient")
                gradient_count = int(torch.stack([(gradient != 0).any() for gradient in gradients]).to(torch.int64).sum().item())
                del gradients
                require(gradient_count > 0, "No nonzero gradient reached a trainable tensor")
                for optimizer in optimizers:
                    optimizer.step()
                require(bool(torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all()),
                        "Nonfinite parameter after complete update")
                del logits, loss, losses, objective
                synchronize(torch, device)
                step_seconds = time.perf_counter() - tick
                training_seconds += step_seconds
                completed_updates += 1
                row = {"epoch": epoch, "training_member_mean_ce": loss_value,
                       "complete_optimizer_update": True, "nonzero_gradient_tensors": gradient_count,
                       "optimizer_strategy": optimizer_strategy,
                       "train_step_seconds": step_seconds}
                evaluate = epoch % recipe["eval_every"] == 0 or epoch == max_epochs
                if evaluate:
                    model.eval()
                    synchronize(torch, device)
                    tick = time.perf_counter()
                    with torch.no_grad():
                        logits = model(graph, x)
                        check_logits(torch, logits, members, manifest["num_nodes"], manifest["num_classes"])
                        eval_ids, eval_y = (val_ids, val_y) if args.mode == "fit" else (train_ids, train_y)
                        selected_logits = logits.index_select(1, eval_ids)
                        metrics = development_metrics(torch, selected_logits, eval_y)
                        if args.mode == "fit":
                            final_metrics = metrics
                            val_cpu = selected_logits.detach().cpu().clone()
                        else:
                            qualification_final_train_metrics = metrics
                        del logits, selected_logits, eval_ids, eval_y
                    synchronize(torch, device)
                    evaluation_seconds = time.perf_counter() - tick
                    if args.mode == "fit":
                        validation_seconds += evaluation_seconds
                        current_nll = final_metrics["primary_probability_pool"]["nll"]
                        improved = current_nll < best_nll  # Exact ties retain the earliest checkpoint.
                        row.update({"validation": final_metrics, "validation_seconds": evaluation_seconds,
                                    "checkpoint_improved": improved})
                        if improved:
                            best_nll, best_epoch, selected_metrics = current_nll, epoch, final_metrics
                            tick = time.perf_counter()
                            save_checkpoint(output / "selected_state.pt", model, optimizers, torch, epoch, provenance)
                            torch.save({"epoch": epoch, "node_ids": validation["indices"], "member_logits": val_cpu,
                                        "provenance": provenance}, output / "selected_validation_member_logits.pt")
                            checkpoint_seconds += time.perf_counter() - tick
                        if epoch >= recipe["min_epochs"] and epoch - best_epoch >= recipe["patience"]:
                            stopping_reason = "validation_nll_patience"
                    else:
                        qualification_evaluation_seconds += evaluation_seconds
                        row.update({"qualification_train_evaluation": metrics,
                                    "qualification_train_evaluation_seconds": evaluation_seconds})
                curve.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                curve.flush()
                if stopping_reason == "validation_nll_patience":
                    break
        final_epoch = completed_updates
        tick = time.perf_counter()
        save_checkpoint(output / "final_state.pt", model, optimizers, torch, final_epoch, provenance)
        if args.mode == "fit":
            require(selected_metrics is not None and final_metrics is not None, "No validation checkpoint was selected")
            torch.save({"epoch": final_epoch, "node_ids": validation["indices"], "member_logits": val_cpu,
                        "provenance": provenance}, output / "final_validation_member_logits.pt")
        checkpoint_seconds += time.perf_counter() - tick
        fit_peak = {"allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.max_memory_reserved(device))} if device.type == "cuda" else None
        qualification_changed_parameters = None
        if args.mode == "qualify":
            qualification_changed_parameters = sum(
                not torch.equal(qualification_initial_parameters[name], parameter.detach().cpu())
                for name, parameter in model.named_parameters())
            require(qualification_changed_parameters > 0, "Complete qualification updates changed no model parameter")
            del qualification_initial_parameters
        # Profile only the selected checkpoint in fit; no extra tuning is possible.
        selected_checkpoint_reload_seconds = 0.0
        deployment_checkpoint_path = output / ("selected_deployment_state.pt" if args.mode == "fit"
                                               else "qualification_final_deployment_state.pt")
        if args.mode == "fit":
            tick = time.perf_counter()
            selected_checkpoint = torch.load(output / "selected_state.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(selected_checkpoint["model_state"])
            synchronize(torch, device)
            selected_checkpoint_reload_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            torch.save({"model_state": selected_checkpoint["model_state"], "factory_kwargs": factory_kwargs,
                        "epoch": best_epoch, "protocol_sha256": protocol_sha, "source_manifest_sha256": sources_sha},
                       deployment_checkpoint_path)
            deployment_checkpoint_preparation_seconds = time.perf_counter() - tick
            del selected_checkpoint
        else:
            tick = time.perf_counter()
            torch.save({"model_state": cpu_copy(model.state_dict(), torch), "factory_kwargs": factory_kwargs,
                        "epoch": final_epoch, "protocol_sha256": protocol_sha, "source_manifest_sha256": sources_sha},
                       deployment_checkpoint_path)
            deployment_checkpoint_preparation_seconds = time.perf_counter() - tick
        profile_ids = val_ids if args.mode == "fit" else train_ids
        # Inference profiling retains no optimizer state or target tensors on GPU.
        # The loop variable also points to the last optimizer and must be released.
        del optimizer, optimizers, train_y
        val_y = None
        model.zero_grad(set_to_none=True)
        del train_ids
        synchronize(torch, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        latency = profile_latency(torch, model, graph, x, profile_ids, protocol["profiling"], device)
        profile_peak = {"allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "reserved_bytes": int(torch.cuda.max_memory_reserved(device))} if device.type == "cuda" else None
        operation_breakdown = fixed_operation_breakdown(torch, model, graph, x, profile_ids,
                                                        protocol["profiling"]["operation_breakdown"], device)
        model_bytes = unique_tensor_bytes(list(model.parameters()) + list(model.buffers()))
        input_bytes = unique_tensor_bytes([x, graph.edge_index])
        gather_index_bytes = profile_ids.numel() * profile_ids.element_size()
        # Parameter-list aliases retain the GPU model. Audit comprehensions are
        # scoped and the gradient list is deleted each step, leaving no loop alias.
        del model, graph, x, profile_ids, val_ids, parameter_groups
        if args.arm == "untied":
            del member_parameters
        synchronize(torch, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        deployment_cpu_ids = validation["indices"] if args.mode == "fit" else train["indices"]
        cold_deployment = cold_deployment_profile(
            torch, source, factory_kwargs, deployment_checkpoint_path, pack, deployment_cpu_ids,
            protocol["profiling"]["selected_checkpoint_cold_deployment"], device,
            "selected" if args.mode == "fit" else "qualification_final")
        report = {"schema_version": 1, "status": "FIT_COMPLETE" if args.mode == "fit" else "QUALIFICATION_COMPLETE",
                  "created_utc": utc_now(), "mode": args.mode, "provenance": provenance,
                  "device": str(device), "torch_version": torch.__version__,
                  "cuda_version": torch.version.cuda,
                  "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "completed_updates": completed_updates, "final_epoch": final_epoch,
                  "qualification_changed_parameter_tensors": qualification_changed_parameters,
                  "selected_epoch": best_epoch if args.mode == "fit" else None,
                  "stopping_reason": stopping_reason,
                  "convergence": {"selected_validation_nll": best_nll if args.mode == "fit" else None,
                                  "epochs_since_selected": final_epoch - best_epoch if args.mode == "fit" else None,
                                  "minimum_epochs_satisfied": final_epoch >= recipe["min_epochs"] if args.mode == "fit" else None,
                                  "patience_exhausted": stopping_reason == "validation_nll_patience",
                                  "max_epochs_reached": args.mode == "fit" and final_epoch == recipe["max_epochs"],
                                  "claim": "Stopping diagnostics only; numerical convergence is not certified"},
                  "selected_validation": selected_metrics, "final_validation": final_metrics,
                  "qualification_final_train_evaluation": qualification_final_train_metrics,
                  "elapsed_seconds": {"training_complete_updates": training_seconds,
                                      "setup_total_including_initial_checkpoint_and_cold_evaluation": setup_seconds,
                                      "qualification_train_evaluation": qualification_evaluation_seconds,
                                      "validation": validation_seconds,
                                      "selected_final_checkpoint_io": checkpoint_seconds,
                                      "selected_checkpoint_reload": selected_checkpoint_reload_seconds,
                                      "deployment_checkpoint_preparation": deployment_checkpoint_preparation_seconds,
                                      "profiling": latency["total_profiling_seconds_including_warmups"],
                                      "operation_breakdown": operation_breakdown.get("total_seconds_including_fixtures_and_warmups", 0.0),
                                      "fresh_checkpoint_deployment_profiling": cold_deployment["total_seconds_including_warmups_and_cleanup"],
                                      "total_scientific_phase": time.perf_counter() - total_started},
                  "setup_components_seconds": {"device_initialization": device_initialization_seconds,
                                                "cpu_data_load_and_hash_checks": cpu_data_load_seconds,
                                                "cpu_source_and_model_construction": cpu_source_model_construction_seconds,
                                                "model_device_transfer": model_transfer_seconds,
                                                "graph_feature_target_device_transfer": input_transfer_seconds,
                                                "initial_checkpoint_io": initial_checkpoint_seconds,
                                                "cold_initial_evaluation": cold["seconds"]},
                  "latency": {**latency, "nodes": "validation" if args.mode == "fit" else "train",
                              "checkpoint": "selected" if args.mode == "fit" else "qualification_final"},
                  "cold_latency": cold,
                  "fresh_checkpoint_deployment": cold_deployment,
                  "fixed_operation_breakdown": {**operation_breakdown,
                                                "checkpoint": "selected" if args.mode == "fit" else "qualification_final",
                                                "nodes": "validation" if args.mode == "fit" else "train"},
                  "gpu_peak_memory": {"train_validation": fit_peak, "profiling": profile_peak,
                                      "fresh_checkpoint_deployment_repeats": cold_deployment["per_repeat_gpu_peak_memory"],
                                      "train_validation_includes_optimizer_state": True,
                                      "profiling_includes_optimizer_or_targets": False},
                  "deployed_tensor_bytes": {"model_parameters_and_all_buffers_including_indices": model_bytes,
                                            "full_graph_and_features": input_bytes,
                                            "model_plus_graph_features": model_bytes + input_bytes,
                                            "model_plus_graph_features_and_gather": model_bytes + input_bytes + gather_index_bytes,
                                            "evaluation_gather_indices": gather_index_bytes,
                                            "labels_and_optimizer_excluded": True},
                  "checkpoint_serialized_bytes": {"final": (output / "final_state.pt").stat().st_size,
                                                  "selected": (output / "selected_state.pt").stat().st_size if args.mode == "fit" else None},
                  "deployment_model_only_checkpoint_bytes": deployment_checkpoint_path.stat().st_size,
                  "metric_definitions": {"macro_f1": "Mean of 2TP/(2TP+FP+FN) over every output class, zero denominator contributes 0",
                                         "member_argmax_disagreement": "Mean unequal-argmax fraction over all unordered member pairs on the reported development nodes; null when M=1"},
                  "test_labels_read": False, "test_scores_or_logits_written": False,
                  "experiment_admission": "Independent root decision required"}
        write_json(output / "cell_report.json", report)
        return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("acquire", "qualify", "preflight", "fit"))
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--sources", required=True, type=Path)
    p.add_argument("--sources-sha256", required=True)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_NAMES))
    p.add_argument("--seed", type=int)
    p.add_argument("--split")
    p.add_argument("--arm", choices=sorted(ARMS))
    p.add_argument("--recipe")
    p.add_argument("--data-manifest", type=Path)
    p.add_argument("--data-manifest-sha256")
    return p


def main(argv=None):
    invocation_started = time.perf_counter()
    args = parser().parse_args(argv)
    if args.mode == "preflight":
        args.mode = "qualify"
    inputs = verify_inputs(args)
    protocol, protocol_sha, _, sources_sha, files, model_entry, cell, recipe = inputs
    paths = phase_paths(protocol, args.mode)
    prepare_cache(paths)
    output = new_output(paths, args)
    snapshot_inputs(output, args, files)
    write_json(output / "invocation.json", {"created_utc": utc_now(), "mode": args.mode,
                                           "dataset": args.dataset, "seed": args.seed,
                                           "split": args.split, "arm": args.arm,
                                           "recipe": args.recipe,
                                           "protocol_sha256": protocol_sha, "source_manifest_sha256": sources_sha,
                                           "data_manifest_sha256": args.data_manifest_sha256})
    try:
        result = (acquire(args, protocol, protocol_sha, sources_sha, paths, output) if args.mode == "acquire"
                  else scientific_phase(args, protocol, protocol_sha, sources_sha, files, model_entry, cell, recipe, output))
    except Exception as exc:
        write_json(output / "failure.json", {"status": "FAILED", "created_utc": utc_now(),
                                             "exception_type": type(exc).__name__, "message": str(exc),
                                             "mode": args.mode, "test_scoring_attempted": False})
        artifact_manifest(output)
        raise
    write_json(output / "wall_clock_receipt.json", {
        "invocation_seconds_before_final_manifest": time.perf_counter() - invocation_started,
        "includes_source_verification_input_snapshots_and_framework_imports": True,
        "excludes_final_artifact_hashing": True})
    artifact_manifest(output)
    print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
