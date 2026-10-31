"""Acquire two adopted complete public graphs, with training-target-only exports.

Real execution requires current repository supervision before third-party imports.
Provider construction materializes all public targets; this is explicitly disclosed.
This program does not create a CUDA context, model, prediction, or score.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
DESTINATION = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
ROUTE = PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json"
ADOPTION = PHASE / "propagation_cost_scout_v1/TASK_ADOPTION_ROOT_before_outcomes_v1.json"
TASKS = {
    "reddit2": {"public_name": "Reddit2", "class_count": 41,
                "node_count": 232965, "feature_count": 602,
                "class_module": "torch_geometric.datasets.reddit2", "class_name": "Reddit2"},
    "products": {"public_name": "ogbn-products", "class_count": 47,
                 "node_count": 2449029, "feature_count": 100,
                 "class_module": "ogb.nodeproppred.dataset_pyg", "class_name": "PygNodePropPredDataset"},
}
PRODUCTS_URLS = {
    "http://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
    "https://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
}
CACHE_NAMES = (
    "GNNM_PHASE_ROOT", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR",
    "CONDA_PKGS_DIRS", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "HF_HOME", "HF_DATASETS_CACHE",
    "NUMBA_CACHE_DIR", "WANDB_DIR",
)
DEPENDENCIES = (
    "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json",
    "protocols/run_authorized_v2.py", "protocols/run_logged.py", "protocols/repo_env.sh",
    "propagation_cost_scout_v1/TASK_ADOPTION_ROOT_before_outcomes_v1.json",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def confined(path, root=PHASE):
    """Reject traversal and symlink components, including aliases within root."""
    root = Path(root).resolve()
    lexical = Path(os.path.abspath(path))
    if not lexical.is_relative_to(root):
        raise ValueError("Path outside declared source/output root")
    cursor = root
    for part in lexical.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Symlinks are not permitted in acquisition paths")
    resolved = lexical.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Resolved path outside declared source/output root")
    return resolved


def file_sha(path):
    p = confined(path)
    before = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("File changed while hashing")
    return h.hexdigest()


def write_json(path, value):
    p = confined(path, HERE)
    with p.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_bindings():
    value = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    if value.get("schema") != "gnnm-propagation-data-source-bindings-v1":
        raise ValueError("Missing prospective source bindings")
    expected = {"propagation_data_v1/acquire.py", "propagation_data_v1/verify_source.py",
                "propagation_data_v1/README.md", *DEPENDENCIES}
    if set(value["sha256"]) != expected:
        raise ValueError("Incomplete acquisition source/dependency closure")
    for name, digest in value["sha256"].items():
        if file_sha(PHASE / name) != digest:
            raise ValueError("Sealed source changed: " + name)
    return value


def live_context(supervisor):
    """Stdlib-only; no CUDA context or idle requirement for CPU acquisition."""
    p = confined(supervisor)
    e = json.loads((p / "environment.json").read_text())
    c = json.loads((p / "command.json").read_text())
    route = json.loads(ROUTE.read_text())
    if (route.get("ssh_destination") != DESTINATION or route.get("visible_gpu_count") != 1
            or route.get("repository_pwd_matches") is not True
            or route.get("forbidden_destination_reconnect_allowed") is not False
            or os.environ.get("GNNM_SSH_DESTINATION") != DESTINATION
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"):
        raise ValueError("Incorrect authorized allocation route/environment")
    repo = Path(route["repository"])
    if (Path.cwd().resolve() != repo or e.get("repository") != str(repo)
            or c.get("cwd") != str(repo) or e.get("phase") != str(PHASE)):
        raise ValueError("Incorrect live repository or phase")
    if (Path(os.environ.get("GNNM_PHASE_ROOT", "")).resolve() != PHASE
            or e.get("supervisor_sha256") != file_sha(PHASE / "protocols/run_logged.py")
            or e.get("wrapper_sha256") != file_sha(PHASE / "protocols/repo_env.sh")
            or e.get("script_sha256") != file_sha(Path(__file__))):
        raise ValueError("Incorrect supervisor/cache wrapper/entry point")
    argv = c.get("argv", [])
    if len(argv) < 2 or Path(argv[1]).resolve() != Path(__file__).resolve():
        raise ValueError("Supervisor did not start this acquisition entry point")
    if not Path(os.path.abspath(argv[0])).is_relative_to(repo):
        raise ValueError("Interpreter entry point outside repository")
    recorded = e.get("cache_and_temp", {})
    if set(recorded) != set(CACHE_NAMES):
        raise ValueError("Supervisor cache/temp schema differs")
    for name in CACHE_NAMES:
        current = os.environ.get(name)
        if not current or recorded[name] != current:
            raise ValueError("Live cache/temp environment differs: " + name)
        confined(current)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    rows = [[s.strip() for s in row.split(",")] for row in query.stdout.splitlines() if row.strip()]
    if (len(rows) != 1 or len(rows[0]) != 4 or rows[0][0] != "0"
            or rows[0][2] != "NVIDIA A100-SXM4-80GB" or int(rows[0][3]) != 81920):
        raise ValueError("Only corrected one-A10080GB allocation permitted")
    adoption = json.loads(ADOPTION.read_text())
    if (adoption.get("tasks") != ["Reddit2", "ogbn-products"]
            or adoption.get("acquisition_and_processing_authorized") is not True
            or adoption.get("adopted_before_new_model_or_test_outcomes") is not True
            or adoption.get("test_target_indexing_or_summary_before_final_selection_allowed") is not False
            or adoption.get("remote_destination") != DESTINATION):
        raise ValueError("Missing complete prospective task adoption")
    return {"supervisor_directory": str(p.relative_to(PHASE)), "gpu_uuid": rows[0][1],
            "command_sha256": file_sha(p / "command.json"),
            "environment_sha256": file_sha(p / "environment.json"),
            "route_sha256": file_sha(ROUTE), "adoption_sha256": file_sha(ADOPTION),
            "entrypoint_sha256": file_sha(Path(__file__)),
            "authorization_status": "pending until guarded child exits; completed-chain audit required"}


def archive_file(source, output, records):
    """Copy exact returned download bytes before provider unlink/extraction cleanup."""
    p = confined(source, output / "provider")
    if not p.is_file():
        raise ValueError("Provider did not return a confined archive file")
    before = p.stat()
    destination = output / "archives" / ("download_%04d_" % (len(records) + 1) + p.name)
    digest = hashlib.sha256()
    count = 0
    with p.open("rb") as original, destination.open("xb") as preserved:
        for chunk in iter(lambda: original.read(1024 * 1024), b""):
            preserved.write(chunk)
            digest.update(chunk)
            count += len(chunk)
    after = p.stat()
    if count != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Downloaded archive changed during preservation")
    if file_sha(destination) != digest.hexdigest():
        raise RuntimeError("Preserved archive copy differs")
    record = {"original_path": str(p.relative_to(PHASE)),
              "preserved_path": str(destination.relative_to(PHASE)),
              "bytes": count, "sha256": digest.hexdigest(), "preserved_UTC": utc()}
    records.append(record)
    write_json(output / "archives" / ("download_%04d.json" % len(records)), record)


@contextmanager
def preserve_provider_downloads(provider_class, task, output, archives, confirmations):
    """Patch only the exact named provider download function's referenced globals."""
    method = provider_class.download
    namespace = method.__globals__
    referenced = set(method.__code__.co_names)
    names = [n for n in ("download_url", "download_google_url") if n in referenced and callable(namespace.get(n))]
    if len(names) != 1:
        raise ValueError("Unsupported provider download API; no fallback or download started")
    expected_reddit = getattr(provider_class, "url", None) if task == "reddit2" else None
    if task == "reddit2" and not isinstance(expected_reddit, str):
        raise ValueError("Unsupported Reddit2 official download declaration")
    replacements = {}
    for name in names:
        original = namespace[name]

        @functools.wraps(original)
        def intercept(*args, _original=original, _name=name, **kwargs):
            declared = args[0] if args else kwargs.get("url", kwargs.get("id"))
            if (task == "products" and declared not in PRODUCTS_URLS
                    or task == "reddit2" and declared != expected_reddit):
                raise ValueError("Download request differs from named official provider")
            folder = args[1] if len(args) > 1 else kwargs.get("folder")
            if folder is None:
                raise ValueError("Download needs an explicit confined provider folder")
            confined(folder, output / "provider")
            path = _original(*args, **kwargs)
            archive_file(path, output, archives)
            return path

        replacements[name] = (original, intercept)
    if task == "products":
        original_decide = namespace.get("decide_download")
        if "decide_download" not in referenced or not callable(original_decide):
            raise ValueError("Unsupported OGB confirmation API; no bypass or download started")

        @functools.wraps(original_decide)
        def authorized_products_confirmation(url, *args, **kwargs):
            if url not in PRODUCTS_URLS or args or kwargs:
                raise ValueError("OGB confirmation bypass authorized only for exact public products.zip URL")
            row = {"provider": "ogbn-products", "url": url, "UTC": utc(),
                   "decision": True, "authority": "explicit named public dataset acquisition flag plus prospective adoption"}
            confirmations.append(row)
            write_json(output / ("ogb_confirmation_%04d.json" % len(confirmations)), row)
            return True

        replacements["decide_download"] = (original_decide, authorized_products_confirmation)
    try:
        for name, (_, replacement) in replacements.items():
            namespace[name] = replacement
        yield {"intercepted_globals": sorted(replacements), "reddit2_provider_declared_url_or_id": expected_reddit,
               "products_confirmation_URL_allowlist": sorted(PRODUCTS_URLS) if task == "products" else []}
    finally:
        for name, (original, replacement) in replacements.items():
            if namespace[name] is not replacement:
                raise RuntimeError("Provider download global changed concurrently")
            namespace[name] = original


def tensor_record(tensor):
    """Canonical metadata plus streamed contiguous CPU tensor bytes; no huge copy."""
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("Export requires provider tensors already contiguous on CPU")
    array = tensor.detach().numpy()
    metadata = {"dtype": str(tensor.dtype), "numpy_dtype": array.dtype.str,
                "shape": list(tensor.shape), "layout": "contiguous_C_order", "byteorder": sys.byteorder}
    header = canonical(metadata)
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    byte_digest = hashlib.sha256()
    view = memoryview(array).cast("B")
    for start in range(0, view.nbytes, 8 * 1024 * 1024):
        chunk = view[start:start + 8 * 1024 * 1024]
        digest.update(chunk)
        byte_digest.update(chunk)
    return {**metadata, "numel": tensor.numel(), "bytes": view.nbytes,
            "canonical_sha256": digest.hexdigest(), "raw_tensor_bytes_sha256": byte_digest.hexdigest(),
            "hash_encoding": "8-byte big-endian JSON-header length, canonical JSON header, exact CPU C-order bytes"}


def file_inventory(directory):
    root = confined(directory)
    result = []
    if not root.exists():
        raise ValueError("Provider storage directory missing")
    for current, folders, filenames in os.walk(root, followlinks=False):
        for name in folders + filenames:
            p = Path(current) / name
            confined(p, root)
        for name in sorted(filenames):
            p = Path(current) / name
            if not p.is_file():
                raise ValueError("Unexpected nonregular provider storage object")
            result.append({"path": str(p.relative_to(PHASE)), "bytes": p.stat().st_size,
                           "sha256": file_sha(p), "hash_scope": "opaque whole public file; may contain held-out targets"})
    return sorted(result, key=lambda row: row["path"])


def cuda_uninitialized(torch):
    if torch.cuda.is_initialized():
        raise RuntimeError("Acquisition must not initialize a CUDA context")


def graph_export(torch, data, task, dataset):
    """Only one target-value access: index_select with official training IDs."""
    spec = TASKS[task]
    x, edge_index = data.x, data.edge_index
    if (x.device.type != "cpu" or x.ndim != 2
            or list(x.shape) != [spec["node_count"], spec["feature_count"]]):
        raise ValueError("Complete official graph node/feature dimensions differ")
    nodes = x.shape[0]
    if (edge_index.device.type != "cpu" or edge_index.dtype != torch.long
            or edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] == 0
            or edge_index.min().item() < 0 or edge_index.max().item() >= nodes):
        raise ValueError("Invalid complete provider edge_index")
    if task == "reddit2":
        masks = {"train": data.train_mask, "valid": data.val_mask, "test": data.test_mask}
        for mask in masks.values():
            if mask.device.type != "cpu" or mask.dtype != torch.bool or list(mask.shape) != [nodes]:
                raise ValueError("Invalid official Reddit2 split mask")
        ids = {name: mask.nonzero(as_tuple=False).view(-1) for name, mask in masks.items()}
        order = "Reddit2 official mask; IDs in ascending node order derived by nonzero"
    else:
        official = dataset.get_idx_split()
        if set(official) != {"train", "valid", "test"}:
            raise ValueError("Products official split names differ")
        ids = {name: official[name] for name in ("train", "valid", "test")}
        masks = {}
        for name, node_ids in ids.items():
            if node_ids.device.type != "cpu" or node_ids.dtype != torch.long or node_ids.ndim != 1:
                raise ValueError("Invalid official products split ID tensor")
            mask = torch.zeros(nodes, dtype=torch.bool)
            if node_ids.numel() == 0 or node_ids.min().item() < 0 or node_ids.max().item() >= nodes:
                raise ValueError("Invalid official products split node IDs")
            mask[node_ids] = True
            masks[name] = mask
        order = "Products original get_idx_split node ID order preserved; Boolean masks derived"
    coverage = torch.zeros(nodes, dtype=torch.int8)
    for name in ("train", "valid", "test"):
        if ids[name].numel() == 0 or masks[name].sum().item() != ids[name].numel():
            raise ValueError("Empty or duplicate official split IDs")
        coverage += masks[name].to(torch.int8)
    if not bool((coverage == 1).all()):
        raise ValueError("Official splits must disjointly cover the complete adopted graph")
    # No y shape/max/min/unique, held-out target access, or num_classes property.
    training_targets = data.y.index_select(0, ids["train"])
    if (training_targets.device.type != "cpu" or training_targets.shape[0] != ids["train"].numel()
            or training_targets.ndim not in (1, 2)
            or training_targets.ndim == 2 and training_targets.shape[1] != 1):
        raise ValueError("Training target shape differs")
    labels = training_targets.view(-1)
    if labels.is_floating_point():
        if not bool(torch.isfinite(labels).all()) or not bool((labels == labels.round()).all()):
            raise ValueError("Training targets must be finite integer class IDs")
    elif labels.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Training target dtype unsupported")
    if labels.min().item() < 0 or labels.max().item() >= spec["class_count"]:
        raise ValueError("Training target IDs outside fixed official class range")
    training_summary = {"scope": "official training indices only", "class_count_source": "fixed official task declaration",
                        "fixed_official_class_count": spec["class_count"],
                        "training_class_counts": torch.bincount(labels.to(torch.long), minlength=spec["class_count"]).tolist()}
    export = {"schema": "gnnm-complete-node-graph-training-target-export-v1", "public_dataset": spec["public_name"],
              "x": x, "edge_index": edge_index, "declared_official_class_count": spec["class_count"],
              "official_split_node_ids": ids, "official_split_masks": masks,
              "training_target_node_ids": ids["train"], "training_targets": training_targets}
    for name in ("edge_attr", "edge_weight"):
        value = getattr(data, name, None)
        if value is not None:
            if value.device.type != "cpu" or value.shape[0] != edge_index.shape[1]:
                raise ValueError("Provider optional edge field differs")
            export[name] = value
    tensors = {"x": x, "edge_index": edge_index,
               "training_target_node_ids": ids["train"], "training_targets": training_targets}
    tensors.update({"official_split_node_ids/" + name: value for name, value in ids.items()})
    tensors.update({"official_split_masks/" + name: value for name, value in masks.items()})
    tensors.update({name: export[name] for name in ("edge_attr", "edge_weight") if name in export})
    records = {name: tensor_record(value) for name, value in sorted(tensors.items())}
    structure = {"node_count": nodes, "edge_index_column_count": edge_index.shape[1],
                 "feature_count": x.shape[1], "graph_representation": "exact named PyG provider output; no sampling, edge change, feature normalization or transforms",
                 "split_ID_order": order, "split_node_counts": {name: ids[name].numel() for name in ids},
                 "all_nodes_in_exactly_one_official_split": True,
                 "heldout_target_values_read_by_exporter": False}
    return export, records, structure, training_summary


def acquire(task, output, live, bindings):
    # Imports are deliberately after source/route/cache/supervisor/output guards.
    import torch
    import numpy
    cuda_uninitialized(torch)
    spec = TASKS[task]
    if task == "reddit2":
        from torch_geometric.datasets import Reddit2
        provider_class = Reddit2
    else:
        from ogb.nodeproppred import PygNodePropPredDataset
        provider_class = PygNodePropPredDataset
    cuda_uninitialized(torch)
    if (provider_class.__module__ != spec["class_module"] or provider_class.__name__ != spec["class_name"]):
        raise ValueError("Named provider class identity differs")
    versions = {}
    for name in ("torch", "torch-geometric", "ogb", "numpy", "scipy", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    module = inspect.getmodule(provider_class)
    source = inspect.getsource(module).encode()
    source_path = output / "sources" / (task + "_provider_module.py")
    with source_path.open("xb") as stream:
        stream.write(source)
    write_json(output / "PRE_PROVIDER.json", {"schema": "gnnm-propagation-data-pre-provider-v1", "UTC": utc(),
               "task": spec, "package_versions": versions, "python": sys.version,
               "numpy_version_runtime": numpy.__version__, "provider_module": module.__name__,
               "provider_source": {"path": str(source_path.relative_to(PHASE)), "sha256": file_sha(source_path),
                                   "encoding": "inspect.getsource(module), UTF-8"},
               "live_supervisor": live, "source_bindings": bindings,
               "provider_all_public_targets_materialization_disclosed": True,
               "scope": "Public provider construction loads complete y into CPU memory and stores raw/processed held-out targets. Exporter reads y values only via official training IDs. No cryptographic label isolation is claimed."})
    archives, confirmations = [], []
    provider_root = output / "provider" / "reddit2" if task == "reddit2" else output / "provider"
    start = time.perf_counter()
    with preserve_provider_downloads(provider_class, task, output, archives, confirmations) as interception:
        write_json(output / "DOWNLOAD_INTERCEPTION.json", interception)
        if task == "reddit2":
            dataset = provider_class(root=str(provider_root), transform=None, pre_transform=None)
        else:
            dataset = provider_class(name="ogbn-products", root=str(provider_root), transform=None, pre_transform=None)
    provider_seconds = time.perf_counter() - start
    cuda_uninitialized(torch)
    if len(dataset) != 1:
        raise ValueError("Require one complete named node graph")
    if not archives:
        raise ValueError("New acquisition produced no preserved original download archive")
    data = dataset[0]
    cuda_uninitialized(torch)
    export, tensors, structure, training = graph_export(torch, data, task, dataset)
    write_json(output / "TENSOR_FINGERPRINTS.json", tensors)
    write_json(output / "GRAPH_AND_SPLITS.json", structure)
    write_json(output / "TRAINING_TARGET_CHECK.json", training)
    bundle = output / "graph_training_targets.pt"
    with bundle.open("xb") as stream:
        torch.save(export, stream)
    cuda_uninitialized(torch)
    raw = file_inventory(dataset.raw_dir)
    processed = file_inventory(dataset.processed_dir)
    write_json(output / "PROVIDER_STORAGE.json", {"raw": raw, "processed": processed,
               "archive_records": archives, "ogb_confirmation_bypass_records": confirmations,
               "hashing_scope": "Opaque whole-file bytes. Provider storage and archives include all public targets; hashing does not index or summarize target values."})
    record = {"schema": "gnnm-propagation-data-acquisition-v1", "UTC": utc(), "status": "completed_data_only_acquisition",
              "public_dataset": spec["public_name"], "graph": structure,
              "export": {"path": str(bundle.relative_to(PHASE)), "bytes": bundle.stat().st_size, "sha256": file_sha(bundle)},
              "provider_processing_seconds": provider_seconds, "package_versions": versions,
              "cuda_context_created": False, "models_or_predictions_created": False, "scores_computed": False,
              "heldout_target_values_read_by_exporter": False, "provider_all_public_targets_materialized": True,
              "label_isolation": "scope discipline in exporter; complete public target storage exists, not cryptographic isolation",
              "live_supervisor": live, "source_bindings_sha256": file_sha(HERE / "SOURCE_BINDINGS.json"),
              "qualification": "Data receipt only; completed authorization chain and tensor replay audit required before scientific admission"}
    write_json(output / "ACQUISITION.json", record)
    write_json(output / "artifact_manifest.json", {"schema": "gnnm-propagation-data-artifact-manifest-v1", "UTC": utc(),
               "files": [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size, "sha256": file_sha(p)}
                         for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({"status": record["status"], "dataset": spec["public_name"],
                      "output": str(output), "export_sha256": record["export"]["sha256"]}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--authorize-public-download", choices=("Reddit2", "ogbn-products"), required=True,
                        help="Explicit named public provider acquisition; construction materializes all public targets")
    args = parser.parse_args(argv)
    if args.authorize_public_download != TASKS[args.task]["public_name"]:
        raise ValueError("Public download authorization must match exact named task")
    output = confined(args.output, HERE)
    if output == HERE or output.exists():
        raise ValueError("Require a new exclusive acquisition directory under propagation_data_v1")
    bindings = source_bindings()
    live = live_context(args.supervisor)
    # No third-party import, download, output creation or GPU context above this line.
    output.mkdir(parents=True, exist_ok=False)
    for name in ("provider", "archives", "sources"):
        (output / name).mkdir()
    write_json(output / "START.json", {"schema": "gnnm-propagation-data-start-v1", "UTC": utc(),
               "task": args.task, "explicit_public_download_authorization": args.authorize_public_download,
               "live_supervisor": live, "source_bindings_sha256": file_sha(HERE / "SOURCE_BINDINGS.json"),
               "provider_target_storage_disclosure": "Named public provider loads all labels. Only official training target values are selected/exported by study code."})
    try:
        acquire(args.task, output, live, bindings)
    except BaseException as error:
        write_json(output / "FAILURE.json", {"schema": "gnnm-propagation-data-failure-v1", "UTC": utc(),
                   "task": args.task, "type": type(error).__name__, "message": str(error),
                   "partial_artifacts_preserved": True, "restart_or_resume_permitted": False,
                   "live_supervisor": live})
        raise


if __name__ == "__main__":
    main()
