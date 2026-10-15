"""Frozen ingredient custody and explicit fresh native assembly helpers."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PHASE = EXPANSION.parent


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load(relative):
    path = EXPANSION / relative
    name = "_explicit_interaction_" + pin(path)["sha256"]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
    return sys.modules[name]


def verify_package(folder, expected):
    actual = pin(folder / "MANIFEST.json")
    if actual["sha256"] != expected:
        raise ValueError("Exact frozen ingredient/interaction manifest required")
    rows = [actual]
    for row in json.loads((folder / "MANIFEST.json").read_text())["files"]:
        path = (folder / row["path"]).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Package source escaped its binding")
        value = pin(path)
        if any(value[k] != row[k] for k in ("bytes", "sha256")):
            raise ValueError("Frozen ingredient changed: " + str(path))
        rows.append(value)
    return rows


def source_bindings(root, expected, family):
    root = Path(root).resolve()
    if not HERE.is_relative_to(root) or not PHASE.is_relative_to(root):
        raise ValueError("Bind the actual absolute repository root")
    receipts = verify_package(HERE, expected)
    declaration = json.loads((HERE / "BINDINGS.json").read_text())
    for row in declaration["packages"]:
        receipts += verify_package(EXPANSION / row["path"], row["manifest_sha256"])
    for row in declaration["helper_files"]:
        value = pin(EXPANSION / row["path"])
        if any(value[k] != row[k] for k in ("bytes", "sha256")):
            raise ValueError("Frozen objective helper changed")
        receipts.append(value)
    portable = load("portable_private_input_v2/extension.py")
    receipts += portable._source_bindings(family, root,
        next(r["manifest_sha256"] for r in declaration["packages"] if r["path"] == "portable_private_input_v2"))
    return receipts, declaration


def ownership(session, references):
    current = dict(session.model.named_parameters())
    if any(current.get(n) is not p for n, p in references.items()):
        raise ValueError("Original native Parameter identity/registration changed")
    owners = [p for group in session.optimizer.param_groups for p in group["params"]]
    if len(owners) != len({id(p) for p in owners}) or {id(p) for p in owners} != {id(p) for p in current.values()}:
        raise ValueError("Complete disjoint native/BE/residual/readout optimizer ownership required")
    core = session.model
    assertion = getattr(core, "assert_native_parameter_identity", getattr(core, "assert_source_identity", None))
    assertion()


def attach_residual(session, family, seed):
    portable = load("portable_private_input_v2/extension.py")
    library = load("portable_private_input_v2/core.py")
    owner, native, core, sites, output = portable.source_mapping(session, family)
    torch = session.torch
    residuals = torch.nn.ModuleList(library.build_residual(torch, anchor, width=width,
        seed=seed + 901003 + 104729 * i) for i, (_, _, width, anchor) in enumerate(sites))
    core.add_module("interaction_input_residuals", residuals)
    details = []
    for (_, module, _, anchor), residual in zip(sites, residuals):
        def post(source, arguments, encoded, adapter=residual):
            member = core._active_member if family == "node" else core._member
            return None if member is None else adapter(encoded, int(member))
        # Appended after the unchanged complete input boundary's BE hook.
        core._handles.append(module.register_forward_hook(post))
        group = library.assign_private_parameters(session.optimizer, anchor, residual.parameters())
        details.append({"optimizer_group": group, **residual.metadata()})
    return residuals, [{"site": site[0], "position": "after unchanged complete native encoded input and BE",
        **detail} for site, detail in zip(sites, details)]


def structural_table(session, family):
    """Exact C2 target facts/table; interaction owns weights, native slot stays empty."""
    torch = session.torch
    tablelib = load("structural_class_responsibility_v1/table.py")
    if family == "node":
        ids, labels = session.train_ids.cpu().tolist(), session.targets.cpu().tolist()
        if ids != torch.nonzero(session.data.train_mask, as_tuple=False).flatten().cpu().tolist() or not torch.equal(
                session.targets, session.data.y[session.train_ids]):
            raise ValueError("Authentic complete native TRAIN node population required")
        edge = session.data.edge_index.detach().cpu()
        metrics = tablelib.native_incoming_counts(int(session.data.num_nodes), edge[0].tolist(), edge[1].tolist(), ids)
        table = tablelib.make_table(ids, labels, metrics, role="TRAIN", permutation_seed=None)
        weights = table.tensor_weights(torch, device=session.targets.device)
        target = "source edge_index incoming-record counts within TRAIN class; exact tied quartile intervals"
    else:
        ids, labels, metrics, positions, missing = [], [], [], [], []
        if getattr(session.train_view, "role", None) != "TRAIN" or len(session.train_view) != len(session.train_ids):
            raise ValueError("Complete native TRAIN graph view required")
        for i in range(len(session.train_view)):
            graph = session.train_view[i]
            graph_id = int(session.train_ids[i])
            if graph.smoke_graph_id.reshape(-1).tolist() != [graph_id] or graph.y.numel() != 1 \
                    or graph.x.ndim != 2 or graph.x.shape[1] != 9 or len(graph.x) != int(graph.num_nodes) or not len(graph.x):
                raise ValueError("Full native nine-field TRAIN graph/identity/target required")
            y = graph.y.reshape(-1)[0]
            if bool(torch.isnan(y)):
                missing.append(i)
                continue
            if not bool(torch.isfinite(y)) or float(y) not in (0., 1.):
                raise ValueError("Observed native binary class required")
            ids.append(graph_id); labels.append(int(y)); metrics.append(len(graph.x)); positions.append(i)
        table = tablelib.make_table(ids, labels, metrics, role="TRAIN", permutation_seed=None)
        weights = torch.ones(4, len(session.train_view), dtype=torch.float64)
        weights[:, torch.tensor(positions)] = table.tensor_weights(torch)
        target = "native atom-row count within observed TRAIN class; missing-label placeholder1 unused"
    return table, weights, {"target": target, "table": table.summary(), "record": table.record(),
        "law": "w=.5+2q; exact class/member mass; no minibatch renormalization", "permutation": None}


def geometry(session, family, seed):
    if family == "node":
        selector = load("task_geometry_node_port_v1/selection.py")
        positions, record, signature = selector.select(session.train_ids.cpu().tolist(),
            session.targets.cpu().tolist(), seed=seed, role="TRAIN")
        auxiliary = load("task_geometry_node_port_v1/objective.py").NodeGeometry("alignment_variance")
        return auxiliary, session.torch.tensor(positions, dtype=session.torch.long, device=session.targets.device), {
            "selection": record, "selection_sha256": signature, "objective": auxiliary.state_dict()}
    module = load("molecular/objectives.py")
    auxiliary = module.MolecularObjective(module.ObjectiveConfig(mode="alignment_variance", temperature=.2,
        coefficient=.05, std_fraction=.5, variance_epsilon=1e-8, normalization_epsilon=1e-12, gradient_scope="all"))
    return auxiliary, None, {"selection": "every original source minibatch graph; unchanged IDs/observed-label mask",
        "objective": auxiliary.state_dict()}
