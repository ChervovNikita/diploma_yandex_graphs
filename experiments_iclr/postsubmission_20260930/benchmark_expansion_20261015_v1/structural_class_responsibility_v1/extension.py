"""Fixed TRAIN atom-count responsibility bridge for native molecular GNNM."""
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
METHODS = {"structural_class_quartile_private_v1": False,
           "structural_class_quartile_permuted_private_v1": True}


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def source_bindings(root, expected):
    root = Path(root).resolve()
    if not HERE.is_relative_to(root) or not PHASE.is_relative_to(root):
        raise ValueError("Explicit repository binding must contain actual phase and method source")
    manifest = pin(HERE/"MANIFEST.json")
    if not isinstance(expected, str) or manifest["sha256"] != expected:
        raise ValueError("Freeze the actual responsibility package manifest before execution")
    receipts = [manifest]
    for row in json.loads((HERE/"MANIFEST.json").read_text())["files"]:
        if row["path"].endswith(".py") or row["path"] in ("SOURCES.json", "METHODS.json"):
            path = (HERE/row["path"]).resolve()
            if not path.is_relative_to(HERE): raise ValueError("Method source escaped its package")
            actual = pin(path)
            if any(actual[key] != row[key] for key in ("bytes", "sha256")):
                raise ValueError("Responsibility method source changed: "+row["path"])
            receipts.append(actual)
    for row in json.loads((HERE/"SOURCES.json").read_text())["source_files"]:
        path = (PHASE/row["path_relative_to_phase"]).resolve()
        if not path.is_relative_to(root): raise ValueError("Native source escaped its repository")
        actual = pin(path)
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen native responsibility source changed")
        receipts.append(actual)
    return receipts


def table_module():
    path = HERE/"table.py"; name = "structural_class_table_"+pin(path)["sha256"]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def transform(session, job):
    if job.get("procedure") not in METHODS or job.get("TEST_access") is not False:
        raise ValueError("Explicit structural-class responsibility method and closed TEST role required")
    if job.get("family") != "ogb":
        raise NotImplementedError("Native molecular bridge only; node incoming-degree rule is defined but its source port remains pending")
    root_value = job.get("recipe", {}).get("_repository_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("Launcher must explicitly bind recipe._repository_root")
    settings = job.get("method_settings", {})
    if set(settings) != {"manifest_sha256"}:
        raise ValueError("This candidate has no adjustable weights, quartiles or hyperparameters")
    receipts = source_bindings(Path(root_value).resolve(), settings["manifest_sha256"])
    if job.get("policy", {}).get("track") == "source_reference" or session.procedure != "gnnm_boundary_native" \
            or session.updates != 0 or session.full_responsibilities is not None:
        raise ValueError("Attach only to fresh unextended native four-route own-loss GNNM")
    if set(session.model._modules) != {"native", "input_bank", "output_bank"}:
        raise ValueError("This isolated candidate retains exactly the native source BE wrapper")
    torch = session.torch
    view = session.train_view
    if getattr(view, "role", None) != "TRAIN" or len(view) != len(session.train_ids):
        raise ValueError("Complete source TRAIN graph view required")
    ids, labels, atoms, positions = [], [], [], []
    missing = []
    for index in range(len(view)):
        graph = view[index]
        graph_id = int(session.train_ids[index])
        if graph.smoke_graph_id.reshape(-1).tolist() != [graph_id] or graph.y.numel() != 1 \
                or graph.x.ndim != 2 or graph.x.shape[1] != 9 or int(graph.num_nodes) != len(graph.x) or not len(graph.x):
            raise ValueError("Full nine-field native molecular TRAIN graph and one target required")
        target = graph.y.reshape(-1)[0]
        if bool(torch.isnan(target)):
            missing.append(index); continue
        if not bool(torch.isfinite(target)) or float(target) not in (0., 1.):
            raise ValueError("Only observed binary source TRAIN classes define responsibility groups")
        ids.append(graph_id); labels.append(int(target)); atoms.append(int(graph.x.shape[0])); positions.append(index)
    module = table_module()
    permuted = METHODS[job["procedure"]]
    table = module.make_table(ids, labels, atoms, role="TRAIN", permutation_seed=int(job["seed"]) if permuted else None)
    weights = torch.ones(4, len(view), dtype=torch.float64)
    weights[:, torch.tensor(positions, dtype=torch.long)] = table.tensor_weights(torch)
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("Native supervision weights must stay positive and finite")
    source_parameters = dict(session.source_parameter_references)
    owner_ids = [id(parameter) for group in session.optimizer.param_groups for parameter in group["params"]]
    if len(owner_ids) != len(set(owner_ids)) or set(owner_ids) != {id(parameter) for parameter in session.model.parameters()}:
        raise ValueError("Native source optimizer ownership differs")
    metadata = {"schema": "fixed-structural-class-responsibility-v1", "method_id": job["procedure"],
        "assignment": "within observed TRAIN class native atom-count quartiles",
        "atom_count": "row count of actual source TRAIN graph.x, equal to native num_nodes; all nine atom fields unchanged",
        "ties": "equal fractional rank-interval overlap for every graph sharing the same atom count",
        "rule": "q mass1 per graph and class mass N/4 per member; w=.5+2q",
        "table": table.summary(), "observed_TRAIN_graph_ids": ids, "observed_TRAIN_positions": positions,
        "missing_TRAIN_positions": missing, "missing_target_weight": "unused placeholder1; native observed-label mask unchanged",
        "permutation": "one whole four-member vector moved among observed objects within each class; no redraw" if permuted else None,
        "shared_credit": "unweighted native F", "private_credit": "positive weighted native W",
        "global_scalar_objective_claim": False, "native_reduction": "original mixed-batch observed BCE mean, then mean across routes",
        "minibatch_normalization": "none; fixed full-TRAIN class masses do not imply minibatch class masses",
        "native_dispatch_label": "boundary_private_fixed_responsibility reuses unchanged source block-gradient branch; no exponential table was drawn",
        "source_parameters_replaced": 0, "new_learned_parameters": 0, "source_optimizer_options_changed": False,
        "one_optimizer_transition": True, "current_loss_or_VALID_assignment": False,
        "TEST_accessed": False, "novelty_or_accuracy_claim": False, "source_receipts": receipts}
    session.full_responsibilities = weights.clone()
    session.responsibility_metadata = copy.deepcopy(metadata)
    session.procedure = "boundary_private_fixed_responsibility"
    session.structural_class_responsibility_metadata = metadata
    original_loss, original_update = session._native_loss, session.train_update
    collecting, captured = [False], []
    def native_loss(self, prediction, batch):
        value = original_loss(prediction, batch)
        if collecting[0]: captured.append(value.detach())
        return value
    def train_update(self):
        if collecting[0]: raise ValueError("Nested responsibility update")
        captured.clear(); collecting[0] = True
        try:
            value = original_update()
            if len(captured) != 5:
                raise ValueError("Frozen native source must compute four own losses and one pooled diagnostic")
            value.update(native_unweighted_F=float(torch.stack(captured[:4]).mean()),
                         native_pool_loss=float(captured[4]), private_weighted_W=float(value["loss"]),
                         loss_semantics="private weighted W; shared block uses unweighted F; no global scalar objective")
            return value
        finally:
            collecting[0] = False; captured.clear()
    session._native_loss, session.train_update = types.MethodType(native_loss, session), types.MethodType(train_update, session)
    original_state, original_load, original_metadata = session.state, session.load_state, session.metadata
    def state(self):
        if collecting[0]: raise ValueError("Checkpoint only between complete native updates")
        return {"schema": "structural-class-responsibility-session-v1", "method": copy.deepcopy(metadata),
                "table_record": table.record(), "native_state": original_state()}
    def load_state(self, value):
        if value.get("schema") != "structural-class-responsibility-session-v1" or value.get("method") != metadata \
                or value.get("table_record") != table.record():
            raise ValueError("Checkpoint assignment, source, TRAIN object/class/size or control permutation changed")
        saved = value["native_state"]
        if not torch.equal(saved["responsibilities"].cpu(), weights):
            raise ValueError("Checkpoint fixed TRAIN weights differ from the frozen table")
        original_load(copy.deepcopy(saved))
        if any(dict(self.model.named_parameters()).get(name) is not parameter for name, parameter in source_parameters.items()):
            raise ValueError("Native source Parameter identity changed on restore")
    def combined_metadata(self):
        return {**original_metadata(), "structural_class_responsibility": metadata}
    session.state, session.load_state = types.MethodType(state, session), types.MethodType(load_state, session)
    session.metadata = types.MethodType(combined_metadata, session)
    return session


extend_session = transform
