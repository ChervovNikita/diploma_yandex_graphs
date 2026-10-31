"""Deferred ordinary runtime reuse and TRAIN-only array loading."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from replay_common import file_sha, load_module, require


def runtime(context):
    # This only runs after the independent stdlib release gate in a fresh process.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sources = context["sources"]
    authority = json.loads(sources["data_authority"].read_text())
    admitted = json.loads(sources["runtime_authority"].read_text())
    require(admitted.get("ordinary_host_execution") is True, "Ordinary runtime authority required")
    require(Path(sys.executable).resolve() == Path(admitted["interpreter_path"]).resolve()
            and file_sha(sys.executable) == admitted["interpreter_sha256"], "Interpreter differs")
    versions = {name: importlib.metadata.version(name) for name in admitted["distribution_versions"]}
    require(versions == admitted["distribution_versions"], "Runtime versions differ")
    common = load_module("pilot_common", sources["pilot_common"])
    state = load_module("pilot_state", sources["pilot_state"])
    model = load_module("pilot_model", sources["pilot_model"])
    reference = load_module("native_reference", sources["native_reference"])
    graphs = load_module("graph_ops", sources["graph_ops"])
    data_module = load_module("pilot_data", sources["pilot_data"])
    device, sampler = model.ordinary_runtime({"release": context["release"], "authority": authority, "runtime": admitted})
    import torch
    before_profile = model.runtime_settings()
    before_rng = state.rng_digest(state.rng_state())
    require(before_profile["deterministic_algorithms"] is False and before_profile["deterministic_warn_only"] is False
            and before_profile["autocast_CPU"] is False, "Ordinary pre-transition profile differs")
    torch.use_deterministic_algorithms(True, warn_only=False)
    after_profile = model.runtime_settings()
    require(after_profile == dict(before_profile, deterministic_algorithms=True, deterministic_warn_only=False)
            and state.rng_digest(state.rng_state()) == before_rng, "Profile transition changed RNG or other flags")
    require(reference.verify_seal() == file_sha(sources["prototype_manifest"]), "Prototype manifest differs")
    utils = load_module("_ncnc_replay_native_utils", sources["native_utils"])
    require(Path(utils.__file__).resolve() == sources["native_utils"], "Native permutation source shadowed")
    runtime_identity = {"authority_sha256": file_sha(sources["runtime_authority"]), "versions": versions,
                        "profile": after_profile, "sampler_source_sha256": admitted["negative_sampler"]["sha256"],
                        "sampler_function_sha256": admitted["negative_sampler"]["function_sha256"], "byteorder": sys.byteorder}
    source_identity = {pin["key"]: pin["sha256"] for pin in context["bindings"]["sources"]}
    return {"torch": torch, "device": device, "sampler": sampler, "PermIterator": utils.PermIterator,
            "graphs": graphs, "state": state, "seed_all": model.seed_all, "tensor_sha": data_module.tensor_sha,
            "frozen_epoch_stream": data_module.epoch_stream, "authority": authority, "utils": utils,
            "source_paths": sources,
            "identity": {"provider_manifest_sha256": context["provider_manifest_sha256"],
                         "source": source_identity, "runtime": runtime_identity},
            "runtime_profile_transition": {"before": before_profile, "after": after_profile, "RNG_unchanged": True}}


def load_train_only(rt):
    import codecs
    import numpy as np
    import pandas as pd
    torch = rt["torch"]
    authority = rt["authority"]
    root = Path(authority["dataset_root"]).resolve()
    paths = {}
    # Deliberately no node-feature, VALID, TEST, model or checkpoint reader.
    for relative in ("split/time/train.pt", "raw/edge.csv.gz"):
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "TRAIN path escaped")
        pin = authority["files"][relative]
        require(path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"], "TRAIN file custody differs")
        paths[relative] = path
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.int64)), codecs.encode]
    with torch.serialization.safe_globals(allowed):
        stored = torch.load(paths["split/time/train.pt"], map_location="cpu", weights_only=True)
    specs = authority["expected_arrays"]["train"]
    require(type(stored) is dict and set(stored) == set(specs), "TRAIN split fields differ")
    for key, spec in specs.items():
        value = stored[key]
        require(type(value) is np.ndarray and str(value.dtype) == spec["dtype"] and list(value.shape) == spec["shape"], "TRAIN array contract differs")
        require(rt["tensor_sha"](torch.from_numpy(value)) == spec["sha256"], "TRAIN tensor differs")
    pairs = torch.from_numpy(stored["edge"])
    raw = pd.read_csv(paths["raw/edge.csv.gz"], compression="gzip", header=None).values.T.astype(np.int64)
    expanded = np.repeat(raw, 2, axis=1)
    expanded[0, 1::2] = expanded[1, 0::2]
    expanded[1, 1::2] = expanded[0, 0::2]
    raw_edges = torch.from_numpy(expanded)
    nodes = authority["full_node_count"]
    require(nodes == 235868 and tuple(pairs.shape) == (1179052, 2) and tuple(raw_edges.shape) == (2, 2358104), "Complete TRAIN geometry differs")
    require(rt["tensor_sha"](pairs) == authority["train_raw_tensor_digests"]["train_records"]
            and rt["tensor_sha"](raw_edges) == authority["train_raw_tensor_digests"]["ordered_raw_graph"], "TRAIN/raw identity differs")
    require(int(pairs.min()) >= 0 and int(pairs.max()) < nodes and int(raw_edges.min()) >= 0 and int(raw_edges.max()) < nodes, "TRAIN endpoints out of range")
    identity = {"nodes": nodes, "records": len(pairs), "ordered_raw_entries": raw_edges.shape[1],
                "pairs_sha256": rt["tensor_sha"](pairs), "raw_sha256": rt["tensor_sha"](raw_edges), "batch_size": 65536}
    return {"pairs": pairs.to(rt["device"]), "raw": raw_edges.to(rt["device"]), "nodes": nodes,
            "batch_size": 65536, "identity": dict(rt["identity"], TRAIN=identity)}
