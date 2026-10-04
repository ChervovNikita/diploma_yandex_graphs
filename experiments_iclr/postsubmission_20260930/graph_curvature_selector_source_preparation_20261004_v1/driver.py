"""Minimal ordinary runtime wrapper; no dataset loader, server or launch command.

Direct invocation only prints the source-preparation status. Root must finish
numerical and fresh-warm qualification before scientifically using these APIs.
No older source certificate, fitted state or closed-study allocation is inherited.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from selector import ARMS, FrozenConstants, bind_head_only, require, select_initializations


HERE = Path(__file__).resolve().parent
CELLS = {"Squirrel": "polyformer_mono", "Photo": "polynormer_r"}
SEEDS = (17, 29, 43)


def load_sources():
    """Ordinary file imports of unchanged pinned sources; numerical imports deferred."""
    bindings = json.loads((HERE/"SOURCE_BINDINGS.json").read_text())
    modules = {}
    for role, record in bindings["runtime_modules"].items():
        path = Path(record["path"])
        data = path.read_bytes()
        require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"],
                "Pinned runtime source differs: "+str(path))
        name = path.stem
        if name in sys.modules:
            module = sys.modules[name]
            require(Path(module.__file__).resolve() == path.resolve(), "Ordinary module-name collision")
        else:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[name]
                raise
        modules[role] = module
    return modules


def prepare_from_native_warm(native_teacher, native_optimizer, warm_metadata,
                             continuation_rng, graph_name, seed, source_split_index,
                             graph, train, canonical_edges, constants, sources=None):
    """Runtime preparation from the prescribed fresh, successful native endpoint.

    Metadata checks establish schedule compatibility, not donor provenance.
    Root must independently establish freshness and actual-warm qualification.
    Inputs contain TRAIN labels only; this function receives no validation labels.
    """
    import torch
    sources = load_sources() if sources is None else sources
    adapter, boundary, source = (sources[key] for key in ("integration", "boundary", "initializer"))
    backbone = CELLS.get(graph_name)
    require(backbone is not None and seed in SEEDS and source_split_index == SEEDS.index(seed),
            "Exact two-graph, three-seed split pairing required")
    specification = native_teacher.specification
    require(specification["backbone"] == backbone == graph.teacher_backbone
            and specification["seed"] == seed and specification["config"] == 0
            and specification["family"] == "single_author" and native_teacher.members == 1,
            "Fresh native single cfg0 specification required")
    require(warm_metadata["fixed_last_warm"] is True
            and warm_metadata["warm_stage_epoch"] == 50
            and warm_metadata["actual_updates"] == (50 if graph_name == "Squirrel" else 250)
            and warm_metadata["warm_stage"] == ("native" if graph_name == "Squirrel" else "global"),
            "Prescribed native warm endpoint required")
    if graph_name == "Photo":
        require(native_teacher.global_stage is True
                and warm_metadata["local_model_and_adam_restored"] is True
                and warm_metadata["native_local_rng_restored"] is False,
                "Photo local-best model/Adam with post-all200 RNG then50 global required")
    constants.validate()
    try:
        adapter.rng_restore(continuation_rng)
        native = copy.deepcopy(native_teacher)  # Audit may set eval; donor stays untouched.
        frozen_native = adapter.named_optimizer_snapshot(native_teacher, native_optimizer)
        k1 = adapter.clone_boundary(native, boundary, 1)
        k4 = adapter.clone_boundary(native, boundary, 4)
        k1.eval()
        identity = adapter.identity_logits_audit(native, k1, k4, graph)
        k4.train(native_teacher.training)
        optimizer, transport = adapter.transport_optimizer(native, frozen_native, k4)
        frozen_raw = adapter.named_optimizer_snapshot(k4, optimizer)
        model_args = adapter.raw_arguments(graph, backbone)
        theta0, logits_fn, binding = bind_head_only(k1, backbone, model_args)
        S = source.symmetric_normalized_adjacency(graph.teacher_input.shape[0], canonical_edges,
                                                 theta0.dtype, theta0.device)
        target_nodes = torch.arange(graph.teacher_input.shape[0], device=theta0.device,
                                    dtype=torch.int64)
        initialized, receipt = select_initializations(
            k4, frozen_raw, logits_fn, theta0, S, target_nodes, train.nodes, train.labels,
            backbone, model_args, continuation_rng, seed, constants, source, adapter)
        receipt.update(graph=graph_name, seed=seed, source_split_index=source_split_index,
                       configuration=0, identity_logits_audit=identity,
                       optimizer_transport=transport, head_binding=binding,
                       warm_metadata=copy.deepcopy(warm_metadata),
                       eligible_donor_provenance_certified_by_source=False,
                       actual_warm_trial_custody_qualification_pending=True)
        return initialized, receipt
    finally:
        adapter.rng_restore(continuation_rng)


def acquire_fresh_and_prepare(native_api, graph_name, seed, graph, train, validation,
                              canonical_edges, constants, trace, save_transition):
    """Future fresh acquisition, using the unchanged native warm implementation.

    Validation is used solely by the native Photo local-best handoff and native
    diagnostics. It is never supplied to the covariance selector.
    No acquisition or continuation is invoked by source preparation or the CLI.
    """
    sources = load_sources()
    binding = json.loads((HERE/"SOURCE_BINDINGS.json").read_text())["native_adapter"]
    path = Path(native_api.__file__).resolve()
    data = path.read_bytes()
    require(path == Path(binding["path"]).resolve() and len(data) == binding["bytes"]
            and hashlib.sha256(data).hexdigest() == binding["sha256"],
            "Unchanged bound native adapter required for fresh acquisition")
    backbone = CELLS[graph_name]
    spec = native_api.specification(backbone, "single_author", 0, seed)
    native, optimizer, checkpoint = sources["integration"].warm_native(
        native_api, spec, graph, train, validation, trace, save_transition)
    initialized, receipt = prepare_from_native_warm(
        native, optimizer, checkpoint["metadata"], checkpoint["rng"], graph_name,
        seed, SEEDS.index(seed), graph, train, canonical_edges, constants, sources)
    return initialized, receipt, checkpoint


def continue_one_arm(initialized, graph, train, validation, trace, save_logits):
    """Unchanged native continuation; caller retains every failed/null arm.

    Scientific execution is separately owned by root. This wrapper adds no
    shortened pilot, alternative endpoint, validation selector or launch policy.
    """
    adapter = load_sources()["integration"]
    adapter.rng_restore(initialized["rng"])
    return adapter.continuation(initialized["model"], initialized["optimizer"],
                                graph, train, validation, trace, save_logits)


if __name__ == "__main__":
    print(json.dumps({"status": "SOURCE_PREPARATION_ONLY_UNQUALIFIED", "arms": list(ARMS),
                      "scientific_execution": False, "numerical_imports": False}))
