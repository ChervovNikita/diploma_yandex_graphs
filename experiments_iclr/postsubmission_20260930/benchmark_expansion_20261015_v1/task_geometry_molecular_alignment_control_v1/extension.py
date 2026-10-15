"""Missing molecular C1 alignment-only control; original parent/helper untouched."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent.parent
MOLECULAR_HELPER=HERE.parent/'molecular/objectives.py'
MOLECULAR_HELPER_SHA256='75e7d59f8360aa2688b9cd974e9e28863e475c683f3dd073e2eacd6665aff71a'
METHODS={'task_graph_class_alignment_v1':'alignment'}

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



def transform(session, job):
    if job.get("family") != "ogb" or job.get("procedure") not in METHODS or job.get("TEST_access") is not False:
        raise ValueError("Explicit molecular alignment-only method with closed TEST role required")
    root = job.get("recipe", {}).get("_repository_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("Explicit absolute actual repository root required")
    settings = job.get("method_settings", {})
    if set(settings) != {"manifest_sha256"}: raise ValueError("Only the frozen control manifest is adjustable")
    receipts = source_bindings(root, settings["manifest_sha256"])
    expected = PHASE/"reference_wrapper_redesign_20261013_v1/smoke_20261014/ogb_port/session.py"
    if Path(sys.modules[type(session).__module__].__file__).resolve() != expected.resolve() \
            or session.procedure != "gnnm_boundary_native" or session.updates != 0 \
            or session.full_responsibilities is not None or job.get("policy", {}).get("track") == "source_reference":
        raise ValueError("Only fresh pinned native molecular boundary session outside source-reference track")
    if set(session.model._modules) != {"native", "input_bank", "output_bank"} \
            or getattr(session.train_view, "role", None) != "TRAIN":
        raise ValueError("Original source molecular wrapper and TRAIN role required")
    spec = importlib.util.spec_from_file_location("molecular_alignment_control_"+MOLECULAR_HELPER_SHA256,MOLECULAR_HELPER)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    auxiliary = module.MolecularObjective(module.ObjectiveConfig(mode="alignment", temperature=.2,
        coefficient=.05, std_fraction=.5, variance_epsilon=1e-8, normalization_epsilon=1e-12, gradient_scope="all"))
    torch = session.torch
    original_parameters = dict(session.source_parameter_references)
    owners = [id(p) for g in session.optimizer.param_groups for p in g["params"]]
    if len(owners) != len(set(owners)) or set(owners) != {id(p) for p in session.model.parameters()}:
        raise ValueError("Unchanged disjoint original native+boundary optimizer ownership required")
    metadata = {"schema":"task-graph-class-alignment-control-v1", "method_id":job["procedure"],
        "manifest_sha256":settings["manifest_sha256"], "helper_sha256":MOLECULAR_HELPER_SHA256,
        "objective":auxiliary.state_dict(), "native_task_loss_changed":False,
        "native_loss":"same source FP32 observed-label BCE mean within each original mixed batch, then mean over4members",
        "update":"native F+.05*alignment; ordinary all-block backward and one original Adam step",
        "variance_in_update":False, "variance_diagnostics_computed_by_frozen_helper":True,
        "gradient_support":"same live graph preheads as existing molecular parent; no direct auxiliary gradient to final output bank",
        "batching":"unchanged original source sampler/batch32/skip rules; no auxiliary subsampling",
        "source_bn_policy":"unchanged shared native buffers with sequential route forwards, same as existing parent callback",
        "TRAIN_ids":session.train_ids.cpu().tolist(), "VALID_ids":session.valid_ids.cpu().tolist(),
        "source_config":copy.deepcopy(session.config), "source_parameters_replaced":0,
        "source_optimizer_options_changed":False, "new_learned_parameters":0,"extra_native_forwards":0,
        "TEST_accessed":False,"novelty_or_accuracy_claim":False,"source_receipts":receipts}
    session.molecular_alignment_control_metadata=copy.deepcopy(metadata)
    original_state,original_load,original_metadata=session.state,session.load_state,session.metadata
    active=[False]

    def train_update(self):
        if active[0]:raise ValueError("No nested molecular control update")
        active[0]=True
        try:
            self.model.train()
            batch,positions,skipped=self._next_train_batch();self.last_train_positions=positions.clone()
            self.optimizer.zero_grad()
            routes=[self.model.run_member(member,batch) for member in range(4)]
            own_losses=torch.stack([self._native_loss(route[0],batch) for route in routes]);own=own_losses.mean()
            extra=auxiliary(prehead=torch.stack([route[2] for route in routes]),labels=batch.y,
                graph_ids=batch.smoke_graph_id.reshape(-1),role="TRAIN")
            loss=extra.scalar_total(own)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite molecular alignment-only control")
            loss.backward();self.optimizer.step();self.model.assert_source_identity();self.updates+=1
            return {"loss":float(loss.detach()),"native_own_loss":float(own.detach()),
                "native_member_losses":own_losses.detach().cpu().tolist(),"auxiliary_loss":float(extra.loss.detach()),
                "alignment":float(extra.terms["alignment"].detach()),"variance_diagnostic_only":float(extra.terms["variance"].detach()),
                "optimizer_updates":1,"update":self.updates,"TRAIN_graph_ids":self.train_ids[positions].tolist(),
                "native_skipped_batches":skipped,"native_batch_graphs":len(positions),"native_route_forwards":4,
                "work":{"optimizer_transitions":1,"reverse_collections":1,"extra_backbone_forwards":0},
                "auxiliary_diagnostics":extra.diagnostics}
        finally:active[0]=False

    def state(self):
        if active[0]:raise ValueError("Checkpoint only between complete updates")
        return {"schema":"task-graph-class-alignment-control-state-v1","method":copy.deepcopy(metadata),
            "auxiliary_state":auxiliary.state_dict(),"native_state":original_state()}

    def load_state(self,value):
        if active[0] or value.get("schema")!="task-graph-class-alignment-control-state-v1" or value.get("method")!=metadata:
            raise ValueError("Molecular control method/source/objective/data role changed")
        auxiliary.load_state_dict(value["auxiliary_state"])
        if value["native_state"]["responsibilities"] is not None:
            raise ValueError("Molecular alignment control cannot add responsibility table")
        original_load(copy.deepcopy(value["native_state"]))
        if any(dict(self.model.named_parameters()).get(n)is not p for n,p in original_parameters.items()):
            raise ValueError("Native source Parameter identity changed")

    def metadata_method(self):return {**original_metadata(),"molecular_alignment_control":metadata}
    session.train_update=types.MethodType(train_update,session)
    session.state,session.load_state=types.MethodType(state,session),types.MethodType(load_state,session)
    session.metadata=types.MethodType(metadata_method,session)
    return session


extend_session=transform

