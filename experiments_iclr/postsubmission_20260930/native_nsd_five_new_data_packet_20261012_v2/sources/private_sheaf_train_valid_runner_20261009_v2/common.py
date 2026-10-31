# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only public entry support; numerical packages are imported after release."""
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

PACKET = Path(__file__).resolve().parent
NATIVE_PACKET = PACKET.parent / "private_sheaf_native_source_design_20261009_v1"
ROLE_KEYS = {"x", "edge_index", "train_index", "train_y", "valid_index", "valid_y"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def verify_source_seal():
    seal_path = PACKET / "SOURCE_SEAL.json"
    seal = json.loads(seal_path.read_text())
    for relative, expected in seal["sha256"].items():
        if sha256_file(PACKET / relative) != expected:
            raise RuntimeError("Source seal mismatch: " + relative)
    return sha256_file(seal_path)


def require_release(args, action, input_path):
    """Return root activation only after explicit execute, seal and input binding."""
    if not args.execute:
        raise RuntimeError("Inactive: --execute and a matching root release are required")
    if not args.release:
        raise RuntimeError("A root release file is required")
    release = json.loads(Path(args.release).read_text())
    if (release.get("enabled") is not True or release.get("release_owner") != "root"
            or release.get("action") != action):
        raise RuntimeError("Release does not authorize this action")
    if not release.get("prior_exposure_audit_complete"):
        raise RuntimeError("Root task/role exposure audit is required")
    if release.get("source_seal_sha256") != verify_source_seal():
        raise RuntimeError("Release is not bound to this source packet")
    if release.get("input_sha256") != sha256_file(input_path):
        raise RuntimeError("Release input hash mismatch")
    output = Path(args.output).resolve()
    if str(output) != release.get("output_directory") or output.exists():
        raise RuntimeError("Release output must be a fresh exact directory")
    if action == "screen":
        role_metadata = Path(input_path).parent / "ROLE.json"
        if release.get("role_metadata_sha256") != sha256_file(role_metadata):
            raise RuntimeError("Release role metadata hash mismatch")
        if (release.get("runtime_qualification_passed") is not True
                or release.get("native_numerical_qualification_passed") is not True
                or release.get("native_placement_qualification_passed") is not True
                or release.get("test_truth_excluded") is not True):
            raise RuntimeError("Root runtime/native/role qualification is required")
        if release.get("device") != args.device:
            raise RuntimeError("Release device mismatch")
        if type(release.get("deterministic_algorithms")) is not bool:
            raise RuntimeError("Release must choose deterministic_algorithms explicitly")
        if release.get("allow_tf32") is not False:
            raise RuntimeError("This numerical screen requires TF32 disabled")
    output.mkdir(parents=True, exist_ok=False)
    return release, output


def load_adapter():
    """Execute the sealed adapter only inside an authorized numerical action."""
    path = NATIVE_PACKET / "adapter.py"
    spec = importlib.util.spec_from_file_location("sealed_nsd_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_hash(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def validate_roles(np, arrays, metadata):
    """Numeric-only TRAIN/VALID input; TEST truth has no accepted field."""
    if set(arrays) != ROLE_KEYS:
        raise ValueError("Role archive keys must be exactly x/edge_index/train_index/train_y/valid_index/valid_y")
    if metadata.get("schema") != "nsd_train_valid_roles_v1" or metadata.get("dataset") != "tolokers":
        raise ValueError("Unknown role schema/task")
    if metadata.get("official_split") != 0 or metadata.get("official_split_count") != 10:
        raise ValueError("This screen is bound to official split0 of ten")
    for name, array in arrays.items():
        if array.dtype.hasobject or array.dtype.kind not in "fiub":
            raise ValueError("Non-numeric/object role array: " + name)
        if array_hash(array) != metadata["array_sha256"][name]:
            raise ValueError("Role array content hash mismatch: " + name)
    x, edges = arrays["x"], arrays["edge_index"]
    if x.dtype != np.float32 or x.ndim != 2 or x.shape != (11758, 10) or not np.isfinite(x).all():
        raise ValueError("Expected finite official float32 Tolokers features [11758,10]")
    n = x.shape[0]
    if edges.dtype != np.int64 or edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] == 0:
        raise ValueError("Expected nonempty int64 directed support [2,E]")
    if (edges.min() < 0 or edges.max() >= n or np.any(edges[0] == edges[1])):
        raise ValueError("Invalid/nonloop native support")
    keys = edges[0] * n + edges[1]
    if np.any(keys[1:] <= keys[:-1]):
        raise ValueError("Canonical paired support must be row-sorted and unique")
    reverse = np.sort(edges[1] * n + edges[0])
    if not np.array_equal(keys, reverse):
        raise ValueError("Canonical support must contain one reverse per incidence")
    counts = metadata["support_counts"]
    if counts["canonical_directed"] != edges.shape[1] or counts["canonical_undirected"] * 2 != edges.shape[1]:
        raise ValueError("Actual canonical support count mismatch")
    if metadata.get("support_convention") != "pyg_2.6.1_to_undirected_row_coalesce":
        raise ValueError("Unqualified support convention")
    for role in ("train", "valid"):
        index, labels = arrays[role + "_index"], arrays[role + "_y"]
        if (index.dtype != np.int64 or labels.dtype != np.int64 or index.ndim != 1
                or labels.ndim != 1 or len(index) != len(labels) or len(index) == 0):
            raise ValueError("Invalid role index/label dimensions: " + role)
        if index.min() < 0 or index.max() >= n or np.any(index[1:] <= index[:-1]):
            raise ValueError("Role indices must be unique sorted full-graph row IDs")
        if not np.array_equal(np.unique(labels), np.array([0, 1], dtype=np.int64)):
            raise ValueError("Both binary classes are required in TRAIN and VALIDATION")
        if metadata["role_counts"][role] != len(index):
            raise ValueError("Actual role count mismatch")
    if np.intersect1d(arrays["train_index"], arrays["valid_index"]).size:
        raise ValueError("TRAIN and VALIDATION overlap")


def read_roles(np, archive):
    archive = Path(archive)
    metadata_path = archive.parent / "ROLE.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["archive_sha256"] != sha256_file(archive):
        raise ValueError("Role archive hash mismatch")
    with np.load(archive, allow_pickle=False) as loaded:
        if set(loaded.files) != ROLE_KEYS:
            raise ValueError("Unexpected role keys; no full y or TEST truth fields are accepted")
        arrays = {name: loaded[name] for name in loaded.files}
    validate_roles(np, arrays, metadata)
    return arrays, metadata, sha256_file(metadata_path)


def process_peak_rss_bytes():
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def failure_record(error, stage):
    return {"stage": stage, "exception_type": type(error).__name__, "message": str(error),
            "time_ns": time.time_ns(), "python": platform.python_version()}
