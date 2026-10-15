"""C2 TRAIN-class incoming-degree responsibility for unchanged native PolyFormer."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent.parent
UPSTREAM = HERE.parent / "structural_class_responsibility_v1"
UPSTREAM_MANIFEST_SHA256 = "2fe768218083169f3973f78f5a9face1c2f79d75fb2a25cd2256a02fe4d25e98"
TABLE_SHA256 = "322c543d8c52b72eb62f28d1f1d9cec896c04b2794179c272b58844922ac2505"
METHODS = {"structural_class_quartile_private_v1": False,
           "structural_class_quartile_permuted_private_v1": True}


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _verify_package(package, expected):
    manifest = pin(package / "MANIFEST.json")
    if manifest["sha256"] != expected:
        raise ValueError("Frozen responsibility package manifest differs")
    receipts = [manifest]
    for row in json.loads((package / "MANIFEST.json").read_text())["files"]:
        path = (package / row["path"]).resolve()
        if not path.is_relative_to(package):
            raise ValueError("Pinned source escaped its responsibility package")
        actual = pin(path)
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen responsibility source changed: " + row["path"])
        receipts.append(actual)
    return receipts


def source_bindings(root, expected):
    root = Path(root).resolve()
    if not HERE.is_relative_to(root) or not PHASE.is_relative_to(root) or not UPSTREAM.is_relative_to(root):
        raise ValueError("Explicit repository binding must contain actual sources and phase")
    if not isinstance(expected, str):
        raise ValueError("Freeze this node package before execution")
    receipts = _verify_package(HERE, expected) + _verify_package(UPSTREAM, UPSTREAM_MANIFEST_SHA256)
    if pin(UPSTREAM / "table.py")["sha256"] != TABLE_SHA256:
        raise ValueError("C2 exact table helper changed")
    for row in json.loads((HERE / "SOURCES.json").read_text())["source_files"]:
        path = (PHASE / row["path_relative_to_phase"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Native source escaped its explicit repository")
        actual = pin(path)
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen native node responsibility source changed")
        receipts.append(actual)
    return receipts


def table_module():
    path = UPSTREAM / "table.py"
    if pin(path)["sha256"] != TABLE_SHA256:
        raise ValueError("C2 table helper must retain its frozen source")
    name = "structural_class_table_" + TABLE_SHA256
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class _RulesCapture:
    """Per-session F/W logging; all numerical work stays in frozen source rules."""
    def __init__(self, native):
        self.native, self.capture = native, []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def step_private_responsibility(self, torch, core, optimizer, *, F, F_weighted, before_step=None):
        self.capture.append((float(F.detach()), float(F_weighted.detach())))
        return self.native.step_private_responsibility(torch, core, optimizer,
            F=F, F_weighted=F_weighted, before_step=before_step)


def transform(session, job):
    if job.get("family") != "node" or job.get("procedure") not in METHODS or job.get("TEST_access") is not False:
        raise ValueError("This bridge requires explicit native node C2 method with closed TEST role")
    root_value = job.get("recipe", {}).get("_repository_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("Launcher must explicitly bind absolute recipe._repository_root")
    settings = job.get("method_settings", {})
    if set(settings) != {"manifest_sha256"}:
        raise ValueError("Fixed C2 method has no adjustable quartile/weight/LR settings")
    receipts = source_bindings(root_value, settings["manifest_sha256"])
    expected_source = PHASE / "reference_wrapper_redesign_20261013_v1/smoke_20261014/node_port.py"
    actual_source = Path(sys.modules[type(session).__module__].__file__).resolve()
    if actual_source != expected_source.resolve() or session.source_metadata.get("source_class") != "PolyFormer":
        raise ValueError("Only the actual pinned native PolyFormer NodeSession is supported")
    if job.get("policy", {}).get("track") == "source_reference" or session.procedure != "gnnm_boundary_native" \
            or session.updates != 0 or session.responsibility is not None:
        raise ValueError("Attach only to fresh unextended native boundary GNNM")
    if set(session.model._modules) != {"native", "input_bank", "output_bank"} or session.model is not session.core:
        raise ValueError("Retain exactly the original native four-route boundary wrapper")
    torch = session.torch
    ids = session.train_ids.detach().cpu().tolist()
    labels = session.targets.detach().cpu().tolist()
    if ids != torch.nonzero(session.data.train_mask, as_tuple=False).flatten().cpu().tolist() \
            or not torch.equal(session.targets, session.data.y[session.train_ids]):
        raise ValueError("Authentic complete source TRAIN IDs/classes in native order required")
    edge = session.data.edge_index.detach().cpu().contiguous()
    if edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype != torch.int64:
        raise ValueError("Actual post-source-preprocessing native int64 edge_index required")
    module = table_module()
    metrics = module.native_incoming_counts(int(session.data.num_nodes), edge[0].tolist(), edge[1].tolist(), ids)
    permuted = METHODS[job["procedure"]]
    seed = int(job["seed"])
    if seed != int(session.recipe.get("_seed", 0)):
        raise ValueError("Responsibility control must bind the native fit seed")
    table = module.make_table(ids, labels, metrics, role="TRAIN", permutation_seed=seed if permuted else None)
    weights = table.tensor_weights(torch, device=session.targets.device)
    source_parameters = dict(session.source_parameter_references)
    parameters = dict(session.model.named_parameters())
    if any(parameters.get(name) is not parameter for name, parameter in source_parameters.items()):
        raise ValueError("Native source Parameter identity differs before attachment")
    owners = [id(p) for group in session.optimizer.param_groups for p in group["params"]]
    if len(owners) != len(set(owners)) or set(owners) != {id(p) for p in session.model.parameters()}:
        raise ValueError("Exact native optimizer ownership required")
    metadata = {"schema": "structural-class-responsibility-native-node-v1", "method_id": job["procedure"],
        "method_manifest_sha256": settings["manifest_sha256"], "upstream_manifest_sha256": UPSTREAM_MANIFEST_SHA256,
        "table_helper_sha256": TABLE_SHA256, "table": table.summary(),
        "role": "TRAIN", "TRAIN_ids_classes_metric_sha256": module.canonical_hash({"ids": ids, "labels": labels, "incoming_counts": metrics}),
        "assignment": "incoming-degree quartiles within each observed source TRAIN class",
        "incoming_degree": "count actual directed edge_index target records after native source preprocessing; duplicates retained, existing self-loop records count once; no added/coalesced/normalized edges",
        "topology": {"num_nodes": int(session.data.num_nodes), "edge_index_shape": list(edge.shape), "dtype": str(edge.dtype),
            "sha256": hashlib.sha256(edge.numpy().tobytes()).hexdigest(), "hash_encoding": "contiguous CPU int64 edge_index bytes in native row and column order"},
        "topology_role": "full public transductive native graph; only TRAIN IDs/labels receive assignments",
        "ties": "same exact fractional tied rank-interval overlap as frozen C2 molecular helper",
        "permutation": "one whole four-member vector permuted within each TRAIN class using fit seed+7030031; no redraw" if permuted else None,
        "rule": "w=.5+2q; exact member/class mass N_c and per-record mean member weight1",
        "source_recipe_id": session.recipe["id"], "source_fixed_config": copy.deepcopy(session.recipe["native_fixed_config"]),
        "source_split_bindings": copy.deepcopy(session.recipe["official_split_bindings"]), "fit_seed": seed,
        "shared_credit": "unweighted full native TRAIN CE F", "private_credit": "positive fixed-table weighted full native TRAIN CE W",
        "native_reduction": "mean over all authentic TRAIN nodes within each member, then mean over four members",
        "native_dispatch": "boundary_private_fixed_responsibility set only after replacing table; no exponential table drawn",
        "activation_checkpointing": "unchanged native _routes(True), nonreentrant, preserve_rng_state=True",
        "source_Adam_groups_options_changed": False, "source_parameters_replaced": 0, "new_learned_parameters": 0,
        "reverse_collections_per_update": 2, "optimizer_transitions_per_update": 1,
        "global_scalar_objective_claim": False, "current_loss_or_VALID_assignment": False,
        "prediction": "unchanged mean raw logits", "TEST_accessed": False,
        "novelty_or_accuracy_claim": False, "source_receipts": receipts}
    session.responsibility = weights.clone()
    session.procedure = "boundary_private_fixed_responsibility"
    session.structural_class_responsibility_metadata = copy.deepcopy(metadata)
    session.source_metadata["structural_class_responsibility"] = copy.deepcopy(metadata)
    rules = _RulesCapture(session.rules); session.rules = rules
    original_update, original_state, original_load = session.train_update, session.state, session.load_state
    active = [False]

    def train_update(self):
        if active[0] or not torch.equal(self.responsibility, weights):
            raise ValueError("Complete updates require the immutable C2 TRAIN table")
        active[0] = True; rules.capture.clear()
        try:
            value = original_update()
            if len(rules.capture) != 1:
                raise ValueError("Exactly one unchanged native private-responsibility transition required")
            f, w = rules.capture[0]
            value.update(native_unweighted_F=f, private_weighted_W=w,
                loss_semantics="source loss is unweighted F; shared uses F and private uses W; no global scalar objective")
            return value
        finally:
            active[0] = False; rules.capture.clear()

    def state(self):
        if active[0]: raise ValueError("Checkpoint only between complete updates")
        return {"schema": "structural-class-responsibility-native-node-state-v1", "method": copy.deepcopy(metadata),
            "table_record": table.record(), "native_state": original_state()}

    def load_state(self, value):
        if active[0] or value.get("schema") != "structural-class-responsibility-native-node-state-v1" \
                or value.get("method") != metadata or value.get("table_record") != table.record():
            raise ValueError("Checkpoint C2 source, method, role, topology, classes or table changed")
        saved = value["native_state"]
        if not torch.equal(saved["responsibility"].cpu(), weights.cpu()):
            raise ValueError("Checkpoint native responsibility differs from exact frozen table")
        original_load(copy.deepcopy(saved))
        if any(dict(self.model.named_parameters()).get(name) is not p for name, p in source_parameters.items()):
            raise ValueError("Native source Parameter identity changed on restore")

    session.train_update = types.MethodType(train_update, session)
    session.state, session.load_state = types.MethodType(state, session), types.MethodType(load_state, session)
    return session


extend_session = transform
