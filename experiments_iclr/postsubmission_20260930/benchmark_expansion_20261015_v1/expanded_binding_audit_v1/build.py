"""Resolve proposed methods to source bytes; never create or launch fit jobs.

Missing native ports remain explicit null bindings. Existing source alone is
not CPU/GPU, provider, convergence, serving or resource qualification.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PHASE = EXPANSION.parent


def pin(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PHASE):
        raise ValueError("Source escapes research phase")
    raw = path.read_bytes()
    if path.suffix == ".py":
        ast.parse(raw, filename=str(path))
    return {"path_relative_to_phase": str(path.relative_to(PHASE)),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def source(relative, function):
    path = EXPANSION / relative
    tree = ast.parse(path.read_text())
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if function not in names:
        raise ValueError("Missing actual exported function: " + relative + ":" + function)
    return {**pin(path), "function": function}


def prepared(cell, method, callback, *, procedure=None, transform=None, package=None, settings=None, pending=()):
    row = {"cell_id": cell["id"], "method_id": method,
           "runtime_procedure": procedure or method,
           "callback": callback, "session_transform": transform,
           "base_procedure": "gnnm_boundary_native", "method_settings": settings or {},
           "status": "source_prepared_not_runtime_admitted",
           "scientific_fits_represented": cell["cohorts"],
           "pending": list(pending), "runnable": False}
    if package:
        manifest = pin(EXPANSION / package / "MANIFEST.json")
        row["source_manifest"] = manifest
        row["method_settings"]["manifest_sha256"] = manifest["sha256"]
    return row


def missing(cell, method, reason):
    return {"cell_id": cell["id"], "method_id": method, "status": "native_port_missing",
            "callback": None, "session_transform": None,
            "scientific_fits_represented": cell["cohorts"], "pending": [reason], "runnable": False}


def resolve(cell, method, native):
    family = cell["task_family"]
    native_callback = {**native["callback"], "function": "create_adapter" if family == "link" else "create_session"}
    native_callback["path_relative_to_phase"] = native_callback.pop("path").split("/experiments_iclr/postsubmission_20260930/", 1)[1]
    native_callback.pop("exists", None)
    base_rules = {"gnnm_boundary_native", "boundary_private_half_pool", "boundary_private_fixed_responsibility",
                  "boundary_own_within_route_supcon"}
    if method in base_rules:
        return prepared(cell, method, native_callback)
    if family in {"node", "ogb"}:
        if method.startswith("task_geometry_"):
            suffix = "alignment_variance" if method.endswith("alignment_variance") else "alignment"
            if family == "node":
                package = "task_geometry_node_port_v1"
                return prepared(cell, method, native_callback,
                                procedure="task_node_class_" + suffix + "_v1",
                                transform=source(package + "/extension.py", "transform"), package=package)
            if suffix == "alignment":
                package = "task_geometry_molecular_alignment_control_v1"
                return prepared(cell, method, native_callback, procedure="task_graph_class_alignment_v1",
                                transform=source(package + "/extension.py", "transform"), package=package)
            dependencies = {name: pin(EXPANSION / name)["sha256"]
                            for name in ("molecular/objectives.py", "adapters/input_residual.py")}
            return prepared(cell, method, native_callback,
                            transform=source("molecular/session_extension.py", "transform"),
                            settings={"input_residual": False, "auxiliary": True, "adapter_rank": 8,
                                      "dependency_sha256": dependencies},
                            pending=["bind the molecular parent/helper source manifest as well as direct callback hash"])
        if method.startswith("structural_class_quartile_"):
            package = "structural_class_responsibility_node_port_v2" if family == "node" else "structural_class_responsibility_v1"
            return prepared(cell, method, native_callback, transform=source(package + "/extension.py", "transform"), package=package)
        if method.startswith("context_affinity_"):
            package = "context_affinity_node_port_v1" if family == "node" else "context_affinity_molecular_port_v1"
            task = "node" if family == "node" else "molecular"
            mode = method.removeprefix("context_affinity_")
            return prepared(cell, method, native_callback, procedure="context_affinity_" + task + "_" + mode + "_v1",
                            transform=source(package + "/extension.py", "transform"), package=package)
        if method.startswith("portable_input_"):
            package = "portable_private_input_v2"
            return prepared(cell, method, native_callback, transform=source(package + "/extension.py", "transform"), package=package)
        if method.startswith("aggregation_"):
            package = "conditional_aggregation_node_port_v1" if family == "node" else "conditional_aggregation_v1"
            name = "transform.py" if family == "node" else "molecular_transform.py"
            return prepared(cell, method, native_callback, transform=source(package + "/" + name, "transform"), package=package,
                            pending=["production selected same-bank mean export integration and selected-state authority"])
        if method.startswith("interaction_"):
            package = "interaction_node_molecular_v1"
            return prepared(cell, method, native_callback, transform=source(package + "/extension.py", "transform"), package=package,
                            pending=["production selected same-bank mean export integration and selected-state authority"])
        if method == "gnnm_private_bn_state_v3":
            return prepared(cell, method, native_callback,
                            transform=source("private_bn_state_v3/session_extension.py", "transform"),
                            settings={"banks_sha256": pin(EXPANSION / "private_bn_state_v3/banks.py")["sha256"]},
                            pending=["new selected BN scientific checkpoint replay after parent study finishes"])
    if family == "link":
        architecture = cell["id"].removeprefix("collab-").upper()
        geometry = {"link_same_query_cosine": "link-same-query-cosine-private",
                    "link_same_query_global": "link-global-opposite-label-private",
                    "link_same_query_endpoint": "link-endpoint-contrast-private"}
        if method in geometry:
            return prepared(cell, method, native_callback, procedure=geometry[method],
                            settings={"candidate_id": geometry[method]},
                            pending=["exact architecture-specific native lifecycle and GPU qualification; NCN CPU gate does not establish other ports"])
        if method.startswith("portable_input_"):
            package = "portable_private_input_v2"
            return prepared(cell, method, source(package + "/link_callback.py", "create_adapter"), package=package,
                            pending=["native architecture-specific context, CUDA restore and complete epoch qualification"])
        if method.startswith("aggregation_") or method in {"link_endpoint_responsibility_v2", "link_query_responsibility_v2"}:
            package = {"NCN": "link_expansion_ports_v1", "NCNC": "link_expansion_ports_NCNC_v1",
                       "BUDDY": "link_expansion_ports_BUDDY_v2"}[architecture]
            actual = {"link_endpoint_responsibility_v2": "link_endpoint_responsibility_private_v2",
                      "link_query_responsibility_v2": "link_query_iid_responsibility_private_v2"}.get(method, method)
            pending = ["native architecture-specific full-epoch, CUDA restore and serving qualification"]
            if architecture == "BUDDY":
                pending.append("required context_manifest/context_manifest_sha256 not bound until source year2007/native sampler context passes; other seeds remain unqualified")
            return prepared(cell, method, source(package + "/callback.py", "create_adapter"), procedure=actual, package=package, pending=pending)
        if method.startswith("context_affinity_") and architecture == "NCN":
            package = "context_affinity_link_port_v1"
            return prepared(cell, method, source(package + "/callback.py", "create_adapter"),
                            procedure="context_affinity_link_" + method.removeprefix("context_affinity_") + "_v1", package=package,
                            pending=["v1 CPU qualification only; device-safe sampler reconstruction needs separate new source generation and CUDA lifecycle gate"])
        if method.startswith("context_affinity_"):
            return missing(cell, method, "C3 native NCNC recursion/BUDDY joint-batch context port is not implemented")
        if method.startswith("interaction_"):
            return missing(cell, method, "separate native link composition factory is not implemented; do not chain incompatible standalone transforms")
        if method == "gnnm_private_bn_state_v3":
            return missing(cell, method, "private_bn_state_v3 is molecular-only; BUDDY branch/native BN state ownership requires its own port")
    raise ValueError("Unresolved roster method: " + cell["id"] + "/" + method)


def build():
    roster_path = EXPANSION / "expanded_study_design_v1/PROPOSED_ROSTER.json"
    bindings_path = EXPANSION / "job_binding_preparation_v1/CELL_BINDINGS.json"
    roster = json.loads(roster_path.read_text())
    bindings = json.loads(bindings_path.read_text())["cells"]
    entries = [resolve(cell, method, bindings[cell["id"]])
               for cell in roster["cells"] for method in cell["shared_method_ids"]]
    assert len({(r["cell_id"], r["method_id"]) for r in entries}) == len(entries)
    assert sum(r["scientific_fits_represented"] for r in entries) == roster["totals"]["shared"]
    totals = collections.Counter()
    for row in entries:
        totals[row["status"]] += row["scientific_fits_represented"]
    return {"schema": "expanded-method-source-binding-audit-v1", "roster": pin(roster_path),
            "native_binding_authority": pin(bindings_path), "physical_shared_fit_projection": dict(totals),
            "method_preset_bindings": len(entries), "entries": entries,
            "separate_native_source_reference_fits": 270, "matched_native_body_fits": 1080,
            "global_pending": ["concrete production runtime generation and all-source job descriptors",
                               "complete native epochs/selection/resume/GPU qualification",
                               "per-split/provider and actual selected serving qualification",
                               "measured CPU/GPU/disk profiles and resource-fit concurrency",
                               "prospective statistical/access-ledger freeze", "user full-campaign GO",
                               "77 remains unauthorized and undeployed"],
            "runnable": False, "scientific_jobs_created": 0, "scientific_jobs_started": 0,
            "full_campaign_GO": False, "actual_native_models_constructed": 0,
            "source_readiness_is_execution_readiness": False}


if __name__ == "__main__":
    value = build()
    target = HERE / "BINDING_AUDIT.json"
    raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text() != raw:
        raise ValueError("Preserve audited source projection; create a new generation for changed sources")
    target.write_text(raw)
    print(json.dumps({k: value[k] for k in ("method_preset_bindings", "physical_shared_fit_projection", "scientific_jobs_started", "runnable")}))
