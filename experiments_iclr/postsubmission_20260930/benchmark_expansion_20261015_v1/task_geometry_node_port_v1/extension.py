"""Task-aware multiclass geometry on live native PolyFormer TRAIN preheads."""
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
MOLECULAR_HELPER = HERE.parent / "molecular/objectives.py"
MOLECULAR_HELPER_SHA256 = "75e7d59f8360aa2688b9cd974e9e28863e475c683f3dd073e2eacd6665aff71a"
METHODS = {"task_node_class_alignment_v1": "alignment",
           "task_node_class_alignment_variance_v1": "alignment_variance"}


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _verify_package(package, expected):
    manifest = pin(package / "MANIFEST.json")
    if manifest["sha256"] != expected:
        raise ValueError("Frozen geometry package manifest differs")
    receipts = [manifest]
    for row in json.loads((package / "MANIFEST.json").read_text())["files"]:
        path = (package / row["path"]).resolve()
        if not path.is_relative_to(package):
            raise ValueError("Pinned source escaped its geometry package")
        actual = pin(path)
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen geometry source changed: " + row["path"])
        receipts.append(actual)
    return receipts


def source_bindings(root, expected):
    root = Path(root).resolve()
    if not HERE.is_relative_to(root) or not PHASE.is_relative_to(root) or not MOLECULAR_HELPER.is_relative_to(root):
        raise ValueError("Explicit repository binding must contain actual sources and phase")
    if not isinstance(expected, str):
        raise ValueError("Freeze this node package before execution")
    receipts = _verify_package(HERE, expected)
    helper = pin(MOLECULAR_HELPER)
    if helper["sha256"] != MOLECULAR_HELPER_SHA256:
        raise ValueError("Frozen molecular ancestor helper changed")
    receipts.append(helper)
    for row in json.loads((HERE / "SOURCES.json").read_text())["source_files"]:
        path = (PHASE / row["path_relative_to_phase"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Native source escaped its explicit repository")
        actual = pin(path)
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen native node geometry source changed")
        receipts.append(actual)
    return receipts



def _module(filename):
    path = HERE/filename
    name = "task_node_geometry_"+filename.replace(".", "_")+"_"+pin(path)["sha256"]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec); sys.modules[name] = result
        spec.loader.exec_module(result)
    return sys.modules[name]


def transform(session, job):
    if job.get("family") != "node" or job.get("procedure") not in METHODS or job.get("TEST_access") is not False:
        raise ValueError("Explicit node class geometry method and closed TEST role required")
    root = job.get("recipe", {}).get("_repository_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("Launcher must explicitly bind absolute recipe._repository_root")
    settings = job.get("method_settings", {})
    if set(settings) != {"manifest_sha256"}:
        raise ValueError("Fixed C1 methods expose only their frozen manifest pin")
    receipts = source_bindings(root, settings["manifest_sha256"])
    expected_source = PHASE/"reference_wrapper_redesign_20261013_v1/smoke_20261014/node_port.py"
    if Path(sys.modules[type(session).__module__].__file__).resolve() != expected_source.resolve() \
            or session.source_metadata.get("source_class") != "PolyFormer":
        raise ValueError("Only actual pinned native PolyFormer NodeSession is supported")
    if session.procedure != "gnnm_boundary_native" or session.updates != 0 or session.responsibility is not None \
            or job.get("policy", {}).get("track") == "source_reference":
        raise ValueError("Attach only to fresh native boundary GNNM, outside paper-reproduction track")
    if session.model is not session.core or set(session.core._modules) != {"native", "banks"} \
            or [(b.binding.module_path, b.binding.kind, b.binding.role) for b in session.core.banks] \
                != [("lin1", "linear", "input"), ("lin3", "linear", "output")]:
        raise ValueError("Retain exactly the native two-boundary four-route wrapper")
    torch = session.torch
    ids, labels = session.train_ids.cpu().tolist(), session.targets.cpu().tolist()
    if session.targets.dtype != torch.long or not torch.equal(session.targets, session.data.y[session.train_ids]) \
            or ids != torch.nonzero(session.data.train_mask, as_tuple=False).flatten().cpu().tolist():
        raise ValueError("Authentic complete observed native categorical TRAIN population required")
    seed = job["seed"]
    if type(seed) is not int or seed != int(session.recipe.get("_seed", 0)):
        raise ValueError("Auxiliary cohort must bind the actual native fit seed")
    selection = _module("selection.py")
    positions, cohort, cohort_sha = selection.select(ids, labels, seed=seed, role="TRAIN")
    positions_tensor = torch.tensor(positions, dtype=torch.long, device=session.targets.device)
    auxiliary = _module("objective.py").NodeGeometry(METHODS[job["procedure"]])
    original_parameters = dict(session.source_parameter_references)
    if any(dict(session.model.named_parameters()).get(n) is not p for n,p in original_parameters.items()):
        raise ValueError("Native source Parameter identity differs")
    owners = [id(p) for g in session.optimizer.param_groups for p in g["params"]]
    if len(owners) != len(set(owners)) or set(owners) != {id(p) for p in session.model.parameters()}:
        raise ValueError("Unchanged disjoint native+boundary optimizer ownership required")
    edge = session.data.edge_index.detach().cpu().contiguous()
    metadata = {"schema": "task-node-class-geometry-method-v1", "method_id": job["procedure"],
        "manifest_sha256": settings["manifest_sha256"], "ancestor_helper_sha256": MOLECULAR_HELPER_SHA256,
        "ancestor_relation": "explicit multiclass extension of frozen binary alignment/variance formulas; binary tensor equality tested; molecular source unmodified",
        "objective": auxiliary.state_dict(), "selection": cohort, "selection_sha256": cohort_sha,
        "source_recipe_id": session.recipe["id"], "source_fixed_config": copy.deepcopy(session.recipe["native_fixed_config"]),
        "source_split_bindings": copy.deepcopy(session.recipe["official_split_bindings"]), "fit_seed": seed,
        "topology": {"num_nodes": int(session.data.num_nodes), "edge_index_shape": list(edge.shape),
            "edge_index_bytes_sha256": hashlib.sha256(edge.numpy().tobytes()).hexdigest()},
        "native_CE": "unweighted source CE mean over complete TRAIN within each of four routes, then member mean",
        "update": "native F+.05*alignment, or F+.05*(alignment+variance), ordinary all-block backward and one source Adam step",
        "gradient_scope": "all parameters that the actual live computation reaches; no block rerouting",
        "gradient_support": "preheads precede lin3 output bank; auxiliary has no direct path to lin3 or final factors; native CE still trains them",
        "activation_checkpointing": "unchanged native _routes(True), nonreentrant preserve_rng_state=True",
        "new_forward_calls": 0, "new_learned_parameters": 0, "native_operators_changed": 0,
        "native_CE_population_changed": False, "source_optimizer_options_changed": False,
        "prediction": "unchanged mean raw logits", "route_repulsion": False,
        "TEST_accessed": False, "novelty_or_accuracy_claim": False, "source_receipts": receipts}
    session.task_geometry_metadata = copy.deepcopy(metadata)
    session.source_metadata["task_geometry"] = copy.deepcopy(metadata)
    original_state, original_load = session.state, session.load_state
    active = [False]

    def train_update(self):
        if active[0]: raise ValueError("No nested native geometry update")
        active[0] = True
        try:
            self.model.train(); self.optimizer.zero_grad()
            routes = self._routes(True)
            native_loss = lambda z: torch.nn.functional.cross_entropy(z[self.train_ids], self.targets)
            terms = self.rules.native_loss_terms(torch, [r.prediction for r in routes], [r.raw_logits for r in routes],
                native_loss=native_loss, pooled_native_loss=native_loss, native_regularizer=routes[0].raw_logits.new_zeros(()))
            hidden = torch.stack([r.prehead[self.train_ids] for r in routes])
            if not hidden.requires_grad or hidden.shape[0] != 4 or hidden.shape[1] != len(ids):
                raise ValueError("Live native [4,all TRAIN nodes,H] preheads required")
            extra = auxiliary(prehead=hidden[:, positions_tensor], labels=self.targets[positions_tensor],
                node_ids=self.train_ids[positions_tensor], role="TRAIN")
            loss = terms["F"]+extra["loss"]
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite native geometry update")
            loss.backward(); self.optimizer.step(); self.core.assert_native_parameter_identity(); self.updates += 1
            return {"loss": float(loss.detach()), "native_unweighted_F": float(terms["F"].detach()),
                "auxiliary_loss": float(extra["loss"].detach()), "alignment": float(extra["alignment"].detach()),
                "variance": float(extra["variance"].detach()), "auxiliary_diagnostics": extra["diagnostics"],
                "train_records": len(ids), "auxiliary_nodes": len(positions), "whole_native_train": True,
                "graph_nodes": int(self.data.num_nodes), "optimizer_updates": 1, "logical_update": self.updates,
                "work": {"reverse_collections": 1, "optimizer_transitions": 1, "extra_forward_views": 0},
                "native_route_forward_views": 4, "activation_checkpointing": True,
                "checkpoint_recomputation_work_not_counted": True}
        finally:
            active[0] = False

    def state(self):
        if active[0]: raise ValueError("Checkpoint only between complete native updates")
        return {"schema": "task-node-class-geometry-state-v1", "method": copy.deepcopy(metadata),
            "auxiliary_state": auxiliary.state_dict(), "native_state": original_state()}

    def load_state(self, value):
        if active[0] or value.get("schema") != "task-node-class-geometry-state-v1" or value.get("method") != metadata:
            raise ValueError("Checkpoint method, source, topology, TRAIN role/classes or selected cohort changed")
        auxiliary.load_state_dict(value["auxiliary_state"])
        if value["native_state"]["responsibility"] is not None:
            raise ValueError("Geometry checkpoint cannot introduce private responsibility weights")
        original_load(copy.deepcopy(value["native_state"]))
        if any(dict(self.model.named_parameters()).get(n) is not p for n,p in original_parameters.items()):
            raise ValueError("Native source Parameter identity changed on restore")

    session.train_update = types.MethodType(train_update, session)
    session.state, session.load_state = types.MethodType(state, session), types.MethodType(load_state, session)
    return session


extend_session = transform

