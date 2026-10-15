"""Native selected node feature/one-hop context affinity, common versus split."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import time

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent.parent
CORE=HERE.parent/'context_affinity_v1'
CORE_MANIFEST_SHA256='fe6b26f875ece1173be06d8656b2d6442ac216e7afcef0e6d40fb7a5b36cf448'
CORE_SHA256='c510c0dbf02236c6345df99d2906534866cab9a65c8562557994e907303baa3b'
TARGET_SPEC=HERE.parent/'expanded_study_design_v1/TASK_CONTEXT_SPEC.md'
TARGET_SPEC_SHA256='671173b3bcf5e2ba4e56c5ee9c1bfebd45f7494c596510bb3cabf182a73577ec'
SELECTION=HERE.parent/'task_geometry_node_port_v1'
SELECTION_MANIFEST_SHA256='e7121f4775f029b366bf9e8b4bc5f68362f26f98ab98fc4431dfaced9cbbd763'
SELECTION_SHA256='9695425698705862c9d2c069fbfd1d2cefc4eedcf8d80ac76ed537b81ef0fab4'
METHODS={'context_affinity_node_common_v1':'common','context_affinity_node_split_v1':'split'}


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
    if not HERE.is_relative_to(root) or not PHASE.is_relative_to(root) or not CORE.is_relative_to(root) or not TARGET_SPEC.is_relative_to(root):
        raise ValueError("Explicit repository binding must contain actual sources and phase")
    if not isinstance(expected, str):
        raise ValueError("Freeze this molecular context package before execution")
    receipts = _verify_package(HERE, expected)
    receipts += _verify_package(CORE,CORE_MANIFEST_SHA256)
    receipts += _verify_package(SELECTION,SELECTION_MANIFEST_SHA256)
    if pin(SELECTION/"selection.py")["sha256"]!=SELECTION_SHA256:raise ValueError("Frozen C1 TRAIN cohort helper changed")
    spec=pin(TARGET_SPEC)
    if spec["sha256"]!=TARGET_SPEC_SHA256 or pin(CORE/"core.py")["sha256"]!=CORE_SHA256:raise ValueError("Frozen root context core or targetspec changed")
    receipts.append(spec)
    for row in json.loads((HERE / "SOURCES.json").read_text())["source_files"]:
        path = (PHASE / row["path_relative_to_phase"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Native source escaped its explicit repository")
        actual = pin(path)
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen native molecular source changed")
        receipts.append(actual)
    return receipts



def _load(path,name):
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def core_module():return _load(CORE/"core.py","node_context_core_"+CORE_SHA256)


def transform(session,job):
    if job.get("family")!="node"or job.get("procedure")not in METHODS or job.get("TEST_access")is not False:
        raise ValueError("Explicit native node context method with closed TEST role required")
    root=job.get("recipe",{}).get("_repository_root");settings=job.get("method_settings",{})
    if not isinstance(root,str)or not Path(root).is_absolute()or set(settings)!={"manifest_sha256"}:
        raise ValueError("Explicit absolute repository root and fixed method manifest required")
    receipts=source_bindings(root,settings["manifest_sha256"])
    expected=PHASE/"reference_wrapper_redesign_20261013_v1/smoke_20261014/node_port.py"
    if Path(sys.modules[type(session).__module__].__file__).resolve()!=expected.resolve() \
            or session.source_metadata.get("source_class")!="PolyFormer"or session.procedure!="gnnm_boundary_native" \
            or session.updates!=0 or session.responsibility is not None or job.get("policy",{}).get("track")=="source_reference":
        raise ValueError("Fresh actual pinned native PolyFormer boundary session required")
    if session.model is not session.core or set(session.core._modules)!={"native","banks"} \
            or[(b.binding.module_path,b.binding.kind,b.binding.role)for b in session.core.banks]!=[("lin1","linear","input"),("lin3","linear","output")]:
        raise ValueError("Original native node boundary wrapper required")
    torch=session.torch
    ids,labels=session.train_ids.cpu().tolist(),session.targets.cpu().tolist()
    if session.targets.dtype!=torch.long or not torch.equal(session.targets,session.data.y[session.train_ids]) \
            or ids!=torch.nonzero(session.data.train_mask,as_tuple=False).flatten().cpu().tolist():
        raise ValueError("Authentic complete source TRAIN IDs/classes required")
    seed=job["seed"]
    if type(seed)is not int or seed!=int(session.recipe.get("_seed",0)):raise ValueError("Actual native fit seed required")
    selector=_load(SELECTION/"selection.py","context_node_frozen_selection_"+SELECTION_SHA256)
    positions,cohort,cohort_sha=selector.select(ids,labels,seed=seed,role="TRAIN")
    selected=session.train_ids[torch.tensor(positions,dtype=torch.long,device=session.train_ids.device)]
    target_labels=session.targets[torch.tensor(positions,dtype=torch.long,device=session.targets.device)]
    context=_load(HERE/"context.py","native_node_context_"+pin(HERE/"context.py")["sha256"])
    started=time.perf_counter();a_cpu,b_cpu,target_receipt=context.build(torch,session.data.x,session.data.edge_index,selected)
    support=context.target_diagnostics(torch,a_cpu,b_cpu,target_labels.cpu(),selected.cpu())
    preparation_seconds=time.perf_counter()-started
    a,b=a_cpu.to(session.targets.device),b_cpu.to(session.targets.device)
    positions_tensor=torch.tensor(positions,dtype=torch.long,device=session.targets.device)
    original_parameters=dict(session.source_parameter_references)
    owners=[id(p)for g in session.optimizer.param_groups for p in g["params"]]
    if len(owners)!=len(set(owners))or set(owners)!={id(p)for p in session.model.parameters()}:
        raise ValueError("Original disjoint native+boundary optimizer ownership required")
    mode=METHODS[job["procedure"]];core=core_module()
    metadata={"schema":"context-affinity-native-node-v1","method_id":job["procedure"],"mode":mode,
        "manifest_sha256":settings["manifest_sha256"],"core_manifest_sha256":CORE_MANIFEST_SHA256,
        "core_sha256":CORE_SHA256,"target_spec_sha256":TARGET_SPEC_SHA256,
        "selection_manifest_sha256":SELECTION_MANIFEST_SHA256,"selection_helper_sha256":SELECTION_SHA256,
        "cohort":cohort,"cohort_sha256":cohort_sha,"target_receipt":target_receipt,"target_support":support,
        "context_A":"actual source-preprocessed native features at selected TRAIN node IDs",
        "context_B":"only selected destination rows of symmetric degree-normalized nonself native adjacency+one self; native nonself multiplicities/direction retained",
        "degree_convention":"incoming row degrees from ALL original nonself records plus one self per node; same degree vector on both operator sides",
        "full_propagated_N_by_D_cache":False,
        "target_population":"public full transductive topology/features; only source TRAIN IDs/classes define anchors/donors",
        "assignment":"allroutesfit.5qA+.5qB"if mode=="common"else"routes0/1 qA, routes2/3 qB; no assignment search",
        "target_rule":"detached factual cosine/.2 softmax over distinct same-class TRAIN donors; zero context uniform",
        "auxiliary_reduction":"class-balanced supported-anchor mean then4route mean",
        "native_CE":"unweighted complete native TRAIN CE mean within route, then4route mean",
        "update":"F+.05A all-block scalar backward, one original sourceAdamtransition",
        "source_recipe_id":session.recipe["id"],"source_fixed_config":copy.deepcopy(session.recipe["native_fixed_config"]),
        "source_split_bindings":copy.deepcopy(session.recipe["official_split_bindings"]),"fit_seed":seed,
        "source_operators_changed":0,"new_head_or_parameters":0,"extra_native_forwards":0,"source_optimizer_options_changed":False,
        "activation_checkpointing":"unchanged native _routes(True),nonreentrant,preserve_rng_state=True",
        "gradient_support":"live native preheads; no direct auxiliary path to lin3/final boundary bank",
        "prediction":"unchanged mean native raw logits; no factual context used at inference",
        "TEST_accessed":False,"novelty_or_accuracy_claim":False,"source_receipts":receipts}
    session.context_affinity_metadata=copy.deepcopy(metadata)
    session.context_preparation_host_seconds=preparation_seconds
    session.source_metadata["context_affinity"]=copy.deepcopy(metadata)
    original_state,original_load=session.state,session.load_state;active=[False]

    def train_update(self):
        if active[0]:raise ValueError("No nested node context update")
        active[0]=True
        try:
            self.model.train();self.optimizer.zero_grad();routes=self._routes(True)
            native_loss=lambda z:torch.nn.functional.cross_entropy(z[self.train_ids],self.targets)
            terms=self.rules.native_loss_terms(torch,[r.prediction for r in routes],[r.raw_logits for r in routes],
                native_loss=native_loss,pooled_native_loss=native_loss,native_regularizer=routes[0].raw_logits.new_zeros(()))
            hidden=torch.stack([r.prehead[self.train_ids]for r in routes])
            if not hidden.requires_grad or hidden.shape[0]!=4 or hidden.shape[1]!=len(ids):raise ValueError("Live full native TRAIN preheads required")
            auxiliary_started=time.perf_counter()
            extra=core.context_auxiliary(torch,preheads=hidden[:,positions_tensor],labels=target_labels,
                object_ids=selected,context_a=a,context_b=b,mode=mode,reduction="class_balanced")
            auxiliary_seconds=time.perf_counter()-auxiliary_started;loss=terms["F"]+extra["loss"]
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite native node context update")
            loss.backward();self.optimizer.step();self.core.assert_native_parameter_identity();self.updates+=1
            return {"loss":float(loss.detach()),"native_unweighted_F":float(terms["F"].detach()),"auxiliary_loss":float(extra["loss"].detach()),
                "context_A_loss":float(extra["A"].detach()),"auxiliary_diagnostics":extra["diagnostics"],
                "context_preparation_host_seconds":preparation_seconds,"context_auxiliary_host_seconds":auxiliary_seconds,
                "train_records":len(ids),"auxiliary_nodes":len(positions),"whole_native_train":True,
                "graph_nodes":int(self.data.num_nodes),"optimizer_updates":1,"logical_update":self.updates,
                "work":{"reverse_collections":1,"optimizer_transitions":1,"extra_forward_views":0},
                "native_route_forward_views":4,"activation_checkpointing":True,"checkpoint_recomputation_work_not_counted":True}
        finally:active[0]=False

    def state(self):
        if active[0]:raise ValueError("Checkpoint only between complete updates")
        return {"schema":"context-affinity-native-node-state-v1","method":copy.deepcopy(metadata),
            "targets":{"A":a_cpu.clone(),"B":b_cpu.clone()},"native_state":original_state()}

    def load_state(self,value):
        if active[0]or value.get("schema")!="context-affinity-native-node-state-v1"or value.get("method")!=metadata:
            raise ValueError("Checkpoint method/source/topology/features/cohort/target construction changed")
        if not torch.equal(value["targets"]["A"].cpu(),a_cpu)or not torch.equal(value["targets"]["B"].cpu(),b_cpu):
            raise ValueError("Checkpoint actual factual target values differ")
        if value["native_state"]["responsibility"]is not None:raise ValueError("Context checkpoint cannot introduce responsibility table")
        original_load(copy.deepcopy(value["native_state"]))
        if any(dict(self.model.named_parameters()).get(n)is not p for n,p in original_parameters.items()):raise ValueError("Native Parameter identity changed")

    session.train_update=types.MethodType(train_update,session)
    session.state,session.load_state=types.MethodType(state,session),types.MethodType(load_state,session)
    return session


extend_session=transform
