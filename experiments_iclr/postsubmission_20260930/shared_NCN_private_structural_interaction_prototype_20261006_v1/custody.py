"""Exact source/input/host custody. Numerical imports are deferred to calls."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
REPO = ROOT.parents[2]
TARGETS = {
    "gpu77": {
        "repository": "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git",
        "hostname": "peptide",
        "gpu_uuids": ("GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998", "GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced"),
        "interpreter_relative": ".gnnm_runtime/private_transfer_cp311_cu118_20261005_v1/bin/python"},
    "allocation": {
        "repository": "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs",
        "hostname": "anogena-2-0",
        "gpu_uuids": ("GPU-44039938-fd82-41d2-fefd-de71514e2fac",),
        "interpreter_relative": "experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python"},
}
EXPECTED_VERSIONS = {"torch": "2.1.2+cu118", "numpy": "1.26.4", "torch_geometric": "2.7.0",
                     "torch_sparse": "0.6.18+pt21cu118", "torch_scatter": "2.1.2+pt21cu118", "CUDA": "11.8"}
DONOR = "citeseer_heart_ncn_trainval_runner_source_20261005_v1"
DONOR_MANIFEST = "efa95806d86e3cc261a8506042204d8d32e39506386d6faf90a625d52be12ff9"
GEOMETRY = "endpoint_episode_geometry_preparation_20261005_v1"
DEPENDENCIES = {
    DONOR + "/SOURCE_MANIFEST.json": DONOR_MANIFEST,
    GEOMETRY + "/SOURCE_MANIFEST.json": "f1b97911cc13076bc2f72d57f2e7f9999245c5678c7979fad5c22dd82af55f10",
    GEOMETRY + "/episode_geometry.py": "f5562c94c8c90b999065e6b570f5a0e7c734f3f4f949274f98eeaf9859bf2321",
    GEOMETRY + "/native_episode_cycle.py": "8cd5596459a389aeaf435dcf5ecec819ae94c19a734f24ecf27c96cb4b97c0b9",
    "shared_core_private_learning_float64_dense_oracle_preparation_20261005_v1/recursive_adjoint.py":
        "14bf734b74983e91dec78ad8bba1ef7dcd910dc71aa20a9e75da466d76362293",
}
AVAILABLE = {"train_pos.txt", "valid_pos.txt", "heart_valid_samples.npy", "gnn_feature"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while part := stream.read(1024*1024): value.update(part)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def phase_file(relative):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Require relative input/evidence paths inside the pinned phase")
    path = (PHASE/value).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve(strict=True)) or not path.is_file():
        raise ValueError("Input/evidence leaves the authorized phase")
    return path


def parser(description):
    value = argparse.ArgumentParser(description=description)
    value.add_argument("--job", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def authorize(program, args, purpose):
    job_path = args.job.resolve(strict=True)
    job = json.loads(job_path.read_text())
    target = TARGETS.get(job.get("target"))
    if target is None or platform.system() != "Linux" or str(REPO) != target["repository"]:
        raise ValueError("Only the two exact authorized Linux repositories may execute this packet")
    expected_phase = REPO/"experiments_iclr/postsubmission_20260930"
    if PHASE != expected_phase or not job_path.is_relative_to(PHASE.resolve(strict=True)):
        raise ValueError("Source/job leaves the pinned target phase")
    if job.get("purpose") != purpose or job.get("source_review_approved") is not True:
        raise ValueError("Exact source and execution scope need root approval")
    if socket.gethostname() != target["hostname"] or job.get("expected_hostname") != target["hostname"]:
        raise ValueError("Pinned target hostname differs")
    interpreter = REPO/target["interpreter_relative"]
    if Path(sys.executable).absolute() != interpreter or job.get("python_executable") != str(interpreter):
        raise ValueError("Existing target-native runtime interpreter differs")
    query = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=30)
    actual = tuple(query.stdout.split())
    if len(actual) != len(target["gpu_uuids"]) or set(actual) != set(target["gpu_uuids"]):
        raise ValueError("Pinned target physical GPU inventory differs")
    selected = job.get("physical_gpu_uuid")
    if selected not in target["gpu_uuids"] or os.environ.get("CUDA_VISIBLE_DEVICES") != selected:
        raise ValueError("Require exactly one prospectively bound physical GPU UUID")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(PHASE.resolve(strict=True)) or not output.parent.is_dir():
        raise ValueError("Require a fresh output in the pinned phase")
    if str(output) != job.get("output_directory"):
        raise ValueError("Exact output binding differs")
    if sha(program) != job.get("program_sha256") or sha(ROOT/"SOURCE_MANIFEST.json") != job.get("source_manifest_sha256"):
        raise ValueError("Reviewed program/source manifest changed")
    for row in json.loads((ROOT/"SOURCE_MANIFEST.json").read_text())["files"]:
        path = (ROOT/row["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT) or sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Reviewed packet bytes differ: "+row["path"])
    for relative, expected in DEPENDENCIES.items():
        if sha(phase_file(relative)) != expected: raise ValueError("Pinned dependency differs: "+relative)
    donor_manifest = json.loads(phase_file(DONOR+"/SOURCE_MANIFEST.json").read_text())
    for row in donor_manifest["files"]:
        if sha(phase_file(DONOR+"/"+row["path"])) != row["sha256"]:
            raise ValueError("Unchanged donor dependency differs: "+row["path"])
    for key in ("source_review", "runtime_qualification", "feature_authority", "negative_pool_authority"):
        record = job.get(key, {})
        if record.get("approved") is not True or not record.get("evidence"):
            raise ValueError("Root admission unresolved: "+key)
        for evidence in record["evidence"]:
            if sha(phase_file(evidence["path"])) != evidence["sha256"]:
                raise ValueError("Exact root admission evidence changed: "+key)
    if type(job.get("seed")) is not int or not 0 <= job["seed"] < 2**32:
        raise ValueError("Prospectively fixed native-compatible seed required")
    if type(job.get("factor_seed")) is not int or not 0 <= job["factor_seed"] < 2**63:
        raise ValueError("Prospectively fixed factor seed required")
    if job.get("TEST_access") is not False or job.get("retry") is not False:
        raise ValueError("TEST is closed; no implicit retry")
    if job.get("constant_adjacency_recursive_adjoint_authorized") is not True:
        raise ValueError("Exact constant-adjacency higher-order repair requires root authorization")
    if type(job.get("soft_seconds")) is not int or job["soft_seconds"] < 1 or job.get("external_hard_bound_confirmed") is not True:
        raise ValueError("Root-fixed soft limit and external hard bound required")
    if type(job.get("hard_seconds")) is not int or job["hard_seconds"] <= job["soft_seconds"]:
        raise ValueError("Existing owner must bind a larger exact hard bound")
    return job, output


def load_module(name, path):
    path = Path(path).resolve(strict=True)
    if name in sys.modules and Path(sys.modules[name].__file__).resolve() != path:
        raise ValueError("Pinned module shadowed: "+name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sources():
    donor = PHASE/DONOR
    sys.path.insert(0, str(donor/"vendor"))
    native = importlib.import_module("baseline_models.NCN.model")
    if Path(native.__file__).resolve() != donor/"vendor/baseline_models/NCN/model.py":
        raise ValueError("Native model shadowed")
    heads = load_module("private_transfer_donor_heads", donor/"heads.py")
    geometry = load_module("private_transfer_geometry", PHASE/GEOMETRY/"episode_geometry.py")
    cycle = load_module("private_transfer_native_cycle", PHASE/GEOMETRY/"native_episode_cycle.py")
    adjoint = load_module("private_transfer_recursive_adjoint", PHASE/"shared_core_private_learning_float64_dense_oracle_preparation_20261005_v1/recursive_adjoint.py")
    load_module("private_adam", ROOT/"private_adam.py")
    load_module("models_base", ROOT/"models_base.py")
    load_module("topology", ROOT/"topology.py")
    models = load_module("private_transfer_models", ROOT/"models.py")
    steps = load_module("private_transfer_steps", ROOT/"transfer_step.py")
    return native, heads, geometry, cycle, adjoint, models, steps


def runtime(job):
    import numpy as np
    import torch
    import torch_geometric
    import torch_sparse
    import torch_scatter
    versions = {"torch": str(torch.__version__), "numpy": np.__version__,
                "torch_geometric": torch_geometric.__version__, "torch_sparse": torch_sparse.__version__,
                "torch_scatter": torch_scatter.__version__, "CUDA": torch.version.cuda}
    if versions != EXPECTED_VERSIONS or versions != job["runtime_versions"] or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exact one-visible-GPU numerical runtime differs")
    if job.get("cpu_threads") != 2 or job.get("cpu_interop_threads") != 1:
        raise ValueError("Fixed native CPU thread contract differs")
    torch.set_num_threads(job["cpu_threads"])
    torch.set_num_interop_threads(job["cpu_interop_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    return torch, torch.device("cuda:0"), versions


def load_inputs(torch, job, *, include_valid):
    """Qualification/cost modes never open VALID values; TEST role impossible."""
    import inspect
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise ValueError("Safe feature load required")
    manifest_path = phase_file(job["available_manifest_relative"])
    if sha(manifest_path) != job["available_manifest_sha256"]:
        raise ValueError("Acquisition manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if set(manifest["files"]) != AVAILABLE or manifest.get("TEST_available_to_loader") is not False:
        raise ValueError("Exactly four TRAIN/VALID roles required; TEST loader forbidden")
    paths, identities = {}, {}
    for name in (AVAILABLE if include_valid else {"train_pos.txt", "gnn_feature"}):
        record = manifest["files"][name]
        if tuple(Path(record["relative_path"]).parts) != ("available", "citeseer", name):
            raise ValueError("Acquisition role/path mismatch")
        path = (manifest_path.parent/record["relative_path"]).resolve(strict=True)
        if not path.is_relative_to(PHASE) or sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Authenticated input bytes changed: "+name)
        paths[name] = path; identities[name] = {"sha256": record["sha256"], "bytes": record["bytes"]}
    supplied = torch.load(paths["gnn_feature"], map_location="cpu", weights_only=True)
    x = supplied.get("entity_embedding") if isinstance(supplied, dict) else None
    if not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.dtype != torch.float32 or tuple(x.shape) != (3327, 3703) or not bool(torch.isfinite(x).all()):
        raise ValueError("Native full feature geometry differs")
    rows = {}
    for split in (("train", "valid") if include_valid else ("train",)):
        pairs, raw, selfs = [], 0, 0
        with paths[split+"_pos.txt"].open() as stream:
            for line in stream:
                fields = line.strip().split("\t")
                if len(fields) != 2: raise ValueError("Native tab-separated positive format differs")
                u, v = map(int, fields); raw += 1
                if not 0 <= u < len(x) or not 0 <= v < len(x): raise ValueError("Invalid positive endpoint")
                if u == v: selfs += 1
                else: pairs.append((u, v))
        record = manifest["files"][split+"_pos.txt"]["counts"]
        if (raw, selfs, len(pairs)) != (record["raw_rows"], record["self_loops"], record["native_nonself_rows"]):
            raise ValueError("Native positive counts changed")
        if len({tuple(sorted(p)) for p in pairs}) != len(pairs): raise ValueError("Duplicate positive facts")
        rows[split] = torch.tensor(pairs, dtype=torch.long)
    if len(rows["train"]) != 3870: raise ValueError("Native full TRAIN population differs")
    if not include_valid: return x, rows["train"], None, None, identities
    import numpy as np
    valid = rows["valid"]
    if len(valid) != 227 or {tuple(sorted(p)) for p in valid.tolist()} & {tuple(sorted(p)) for p in rows["train"].tolist()}:
        raise ValueError("Native VALID population/overlap differs")
    pool = np.load(paths["heart_valid_samples.npy"], allow_pickle=False)
    positive = valid.numpy()
    if pool.shape != (227, 500, 2) or pool.dtype != np.dtype("<i8") or not pool.flags.c_contiguous:
        raise ValueError("Complete fixed500 VALID pool differs")
    if np.any((pool < 0)|(pool >= len(x))) or np.any(pool[:, :250, 0] != positive[:, 0, None]) or np.any(pool[:, 250:, 1] != positive[:, 1, None]):
        raise ValueError("VALID association/range differs")
    return x, rows["train"], valid, torch.from_numpy(pool), identities


@contextmanager
def native_adjoint(torch, native, factory):
    saved = native.spmm_add
    if saved is not importlib.import_module("torch_sparse.matmul").spmm_add:
        raise ValueError("Native sparse alias was already replaced")
    wrapped = factory(torch, saved)
    try:
        native.spmm_add = wrapped
        yield
    finally:
        native.spmm_add = saved
        if native.spmm_add is not saved: raise ValueError("Native alias restoration failed")


def make_pair(torch, geometry, cycle, train, nodes, seed, cycle_index, outer_size, inner_size, endpoint_streams, random_streams):
    """Full draw frozen independent of model; return all batches including tail."""
    pairs = train.cpu().tolist()
    negative, order = cycle.native_cycle(pairs, nodes, seed, cycle_index)
    geometry.validate_negative_bank(pairs, negative, nodes)
    full_graph = geometry.neighbors(pairs, nodes)
    episodes = []
    for ids in geometry.outer_batches(order, len(pairs), outer_size):
        endpoint = geometry.endpoint_episode(pairs, negative, ids, endpoint_streams, inner_size)
        if not endpoint["feasible"]: raise ValueError("Prospective endpoint episode infeasible; no redraw/skip/fallback")
        random = geometry.matched_random_episode(pairs, negative, endpoint, random_streams, full_graph)
        if not random["feasible"]: raise ValueError("Prospective matched-random episode infeasible; no substitute")
        removed = geometry.mask_indices(endpoint, random)
        kept, graph = geometry.support(pairs, nodes, removed)
        episodes.append((endpoint, random, kept, {
            "endpoint": geometry.describe_episode(pairs, negative, endpoint, full_graph, graph, removed),
            "matched_random": geometry.describe_episode(pairs, negative, random, full_graph, graph, removed)}))
    return torch.tensor(negative, dtype=torch.long), order, episodes


def queries(torch, train, negative, episode, device):
    def take(role, indices): return role[torch.tensor(indices, dtype=torch.long, device=role.device)].t().to(device)
    inner = [(take(train, row["pos_ids"]), take(negative, row["neg_ids"])) for row in episode["inner"]]
    outer = (take(train, episode["outer_pos_ids"]), take(negative, episode["outer_neg_ids"]))
    return inner, outer


def support_tensor(torch, train, kept, nodes, device):
    from torch_sparse import SparseTensor
    index = torch.tensor(kept, dtype=torch.long, device=train.device)
    result = SparseTensor.from_edge_index(train[index].t(), sparse_sizes=(nodes, nodes)).to_symmetric().coalesce().to(device)
    if result.sparse_sizes() != (nodes, nodes) or result.nnz() != 2*len(kept):
        raise ValueError("Paired union-mask support changed")
    return result
