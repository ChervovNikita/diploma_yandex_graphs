"""Native molecular atom/bond context affinity, common versus split targets."""
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
METHODS={'context_affinity_molecular_common_v1':'common','context_affinity_molecular_split_v1':'split'}


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


def core_module():return _load(CORE/"core.py","molecular_context_core_"+CORE_SHA256)


def transform(session,job):
    if job.get("family")!="ogb" or job.get("procedure")not in METHODS or job.get("TEST_access")is not False:
        raise ValueError("Explicit molecular context method with closed TEST role required")
    root=job.get("recipe",{}).get("_repository_root")
    if not isinstance(root,str)or not Path(root).is_absolute():raise ValueError("Explicit absolute repository root required")
    settings=job.get("method_settings",{})
    if set(settings)!={"manifest_sha256"}:raise ValueError("Only frozen context package pin accepted")
    receipts=source_bindings(root,settings["manifest_sha256"])
    expected=PHASE/"reference_wrapper_redesign_20261013_v1/smoke_20261014/ogb_port/session.py"
    if Path(sys.modules[type(session).__module__].__file__).resolve()!=expected.resolve() \
            or session.procedure!="gnnm_boundary_native"or session.updates!=0 or session.full_responsibilities is not None \
            or job.get("policy",{}).get("track")=="source_reference":
        raise ValueError("Fresh unchanged pinned native molecular boundary session required")
    if set(session.model._modules)!={"native","input_bank","output_bank"}or getattr(session.train_view,"role",None)!="TRAIN":
        raise ValueError("Original molecular boundary wrapper and TRAIN custody required")
    context=_load(HERE/"context.py","native_molecular_context_"+pin(HERE/"context.py")["sha256"])
    core=core_module();torch=session.torch
    atom_vocab,bond_vocab=context.native_vocabularies(session)
    encoder_receipts=[]
    for row in session.encoding_dependencies.values():
        path=Path(row["path"]).resolve()
        if not path.is_relative_to(Path(root).resolve())or pin(path)["sha256"]!=row["sha256"]:
            raise ValueError("Actual native encoder dependency differs or escapes repository")
        encoder_receipts.append(pin(path))
    original_parameters=dict(session.source_parameter_references)
    owners=[id(p)for g in session.optimizer.param_groups for p in g["params"]]
    if len(owners)!=len(set(owners))or set(owners)!={id(p)for p in session.model.parameters()}:
        raise ValueError("Original native+boundary optimizer ownership required")
    mode=METHODS[job["procedure"]]
    metadata={"schema":"context-affinity-native-molecular-v1","method_id":job["procedure"],"mode":mode,
        "manifest_sha256":settings["manifest_sha256"],"core_manifest_sha256":CORE_MANIFEST_SHA256,
        "core_sha256":CORE_SHA256,"target_spec_sha256":TARGET_SPEC_SHA256,
        "atom_vocab":list(atom_vocab),"bond_vocab":list(bond_vocab),"native_encoder_receipts":encoder_receipts,
        "vocab_source":"num_embeddings in actual native AtomEncoder and every layer's BondEncoder; all source bond vocabularies equal",
        "context_A":"concatenated per-field native atom-category frequency histograms",
        "context_B":"concatenated per-field native directed bond-category frequency histograms; edgeless zero vector",
        "context_construction":"canonical CPU FP32 using every actual native batch record; no native feature/edge changes",
        "target":"detached cosine/.2 softmax over distinct same observed TRAIN class donors; no same-object donor",
        "assignment":"every route .5qA+.5qB"if mode=="common"else"routes0/1 qA, routes2/3 qB; no assignment search",
        "auxiliary_reduction":"mean supported anchors within observed class, then mean over supported classes, then mean members",
        "update":"original full native observed-label mixed-batch BCE member mean F+.05A, one all-block backward and original Adam step",
        "source_config":copy.deepcopy(session.config),"TRAIN_ids":session.train_ids.tolist(),"VALID_ids":session.valid_ids.tolist(),
        "missing_labels":"excluded from auxiliary only; original source BCE mask unchanged",
        "zero_contexts":"uniform target when zero; zero contexts and unsupported anchors explicitly reported",
        "gradient_support":"live native graph preheads; no direct auxiliary path to final output bank",
        "new_learned_parameters":0,"extra_native_forwards":0,"source_optimizer_options_changed":False,
        "source_bn_policy":"unchanged shared native buffers with sequential route forwards",
        "inference":"unchanged mean native raw logits; no factual context input at serving",
        "TEST_accessed":False,"novelty_or_accuracy_claim":False,"source_receipts":receipts}
    session.context_affinity_metadata=copy.deepcopy(metadata)
    original_state,original_load,original_metadata=session.state,session.load_state,session.metadata
    last_receipt=[None];active=[False]

    def contexts(batch):
        a,b,receipt=context.build(torch,batch,atom_vocab,bond_vocab)
        labels=batch.y.detach().cpu().reshape(-1)
        if torch.isinf(labels).any():raise ValueError("Infinite native TRAIN label")
        observed=~torch.isnan(labels)
        if not ((labels[observed]==0)|(labels[observed]==1)).all():raise ValueError("Observed binary native TRAIN classes required")
        ids=batch.smoke_graph_id.detach().cpu().reshape(-1)
        target_info=context.target_diagnostics(torch,a[observed],b[observed],labels[observed].long(),ids[observed])
        receipt.update(role="TRAIN",missing_labels=int((~observed).sum()),observed_labels=labels[observed].long().tolist(),
            observed_graph_ids=ids[observed].tolist(),target_diagnostics=target_info)
        receipt.pop("sha256");receipt["sha256"]=context.canonical_hash(receipt)
        return a,b,observed,labels,ids,receipt

    def train_update(self):
        if active[0]:raise ValueError("No nested molecular context update")
        active[0]=True
        try:
            self.model.train();batch,positions,skipped=self._next_train_batch();self.last_train_positions=positions.clone()
            if batch.smoke_graph_id.reshape(-1).cpu().tolist()!=self.train_ids[positions].tolist():
                raise ValueError("Actual source TRAIN batch identities differ")
            self.optimizer.zero_grad();routes=[self.model.run_member(m,batch)for m in range(4)]
            own_losses=torch.stack([self._native_loss(r[0],batch)for r in routes]);own=own_losses.mean()
            context_started=time.perf_counter()
            a,b,observed,labels,ids,receipt=contexts(batch);last_receipt[0]=receipt
            context_seconds=time.perf_counter()-context_started
            hidden=torch.stack([r[2]for r in routes])
            if not hidden.requires_grad:raise ValueError("Live native molecular preheads required")
            keep=observed.to(hidden.device)
            auxiliary_started=time.perf_counter()
            extra=core.context_auxiliary(torch,preheads=hidden[:,keep],labels=labels[observed].long().to(hidden.device),
                object_ids=ids[observed].to(hidden.device),context_a=a[observed].to(hidden),context_b=b[observed].to(hidden),
                mode=mode,reduction="class_balanced")
            auxiliary_seconds=time.perf_counter()-auxiliary_started
            loss=own+extra["loss"]
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite context update")
            loss.backward();self.optimizer.step();self.model.assert_source_identity();self.updates+=1
            return {"loss":float(loss.detach()),"native_own_loss":float(own.detach()),"auxiliary_loss":float(extra["loss"].detach()),
                "native_member_losses":own_losses.detach().cpu().tolist(),"context_A_loss":float(extra["A"].detach()),
                "auxiliary_diagnostics":extra["diagnostics"],"context_receipt":copy.deepcopy(receipt),
                "context_construction_host_seconds":context_seconds,"context_auxiliary_host_seconds":auxiliary_seconds,
                "optimizer_updates":1,"update":self.updates,"TRAIN_graph_ids":self.train_ids[positions].tolist(),
                "native_skipped_batches":skipped,"native_batch_graphs":len(positions),"native_route_forwards":4,
                "work":{"optimizer_transitions":1,"reverse_collections":1,"extra_backbone_forwards":0}}
        finally:active[0]=False

    def state(self):
        if active[0]:raise ValueError("Checkpoint only between complete context updates")
        return {"schema":"context-affinity-native-molecular-state-v1","method":copy.deepcopy(metadata),
            "last_context_receipt":copy.deepcopy(last_receipt[0]),"native_state":original_state()}

    def load_state(self,value):
        if active[0]or value.get("schema")!="context-affinity-native-molecular-state-v1"or value.get("method")!=metadata:
            raise ValueError("Checkpoint method/source/vocabulary/context construction/data role changed")
        saved=value["native_state"];receipt=value["last_context_receipt"];positions=saved["last_train_positions"]
        if saved["responsibilities"]is not None:raise ValueError("Context checkpoint cannot introduce responsibility table")
        if (receipt is None)!=(positions is None):raise ValueError("Context signature/native last batch differs")
        if receipt is not None:
            from torch_geometric.data import Batch
            batch=Batch.from_data_list([self.train_view[int(i)]for i in positions])
            if contexts(batch)[-1]!=receipt:raise ValueError("Checkpoint factual TRAIN target signature changed")
        original_load(copy.deepcopy(saved));last_receipt[0]=copy.deepcopy(receipt)
        if any(dict(self.model.named_parameters()).get(n)is not p for n,p in original_parameters.items()):
            raise ValueError("Native source Parameter identity changed")

    def metadata_method(self):return {**original_metadata(),"context_affinity":metadata}
    session.train_update=types.MethodType(train_update,session)
    session.state,session.load_state=types.MethodType(state,session),types.MethodType(load_state,session)
    session.metadata=types.MethodType(metadata_method,session)
    return session


extend_session=transform

