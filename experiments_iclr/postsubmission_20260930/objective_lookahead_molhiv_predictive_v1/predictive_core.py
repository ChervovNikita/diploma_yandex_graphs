"""Executable training/replay engine for a FUTURE separately admitted wrapper.

No OGB dataset provider is constructed. Inputs are admitted train/valid exports.
Current runner CLI never calls these real-data functions.
"""
import copy
import csv
import importlib.util
import inspect
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from contract import HERE, PHASE, RECIPE, LAYOUTS, OBJECTIVES, sha, digest, inside, write_new, sources, checked, audit_future_binding, live_job_context, authorized_completed


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


sys.path.insert(0, str(PHASE/"new_tasks"))
INHERITED = module("primary_molhiv_inherited_pilot", PHASE/"new_tasks/molhiv_pilot.py")
LOWRANK = module("primary_molhiv_declared_lowrank", PHASE/"closest_controls_v1/trainable_low_rank_gine_v2/adapter.py")
GINE = LOWRANK.GINE


def configure():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before process/CUDA initialization")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def tensor_hash(t):
    return INHERITED.tensor_hash(t)


def state_hash(model):
    return digest({n: tensor_hash(t) for n, t in sorted(model.state_dict().items())})


def objective(logits, targets, family):
    if family not in OBJECTIVES:
        raise ValueError("Unknown training objective")
    return INHERITED.graph_member_bce(logits if family == "own" else logits.mean(0)[None], targets)


def model_for(seed, layout, rank=None):
    if layout not in LAYOUTS or seed not in range(3):
        raise ValueError("Outside prescribed primary matrix")
    # Construct exactly the canonical fresh source. Privacy consumes no RNG.
    INHERITED.set_seed(seed)
    base = GINE.MolecularProjectorEnsemble(GINE.gine_options(), members=4)
    canonical_state = state_hash(base)
    canonical_cpu_rng = torch.get_rng_state().clone()
    if layout == "declared_low_rank":
        if rank is None:
            model, definition = LOWRANK.make_model(base, 29001+1000*seed)
            if not definition["rank_plan"]["within_one_percent"]:
                raise ValueError("Declared structural rank match exceeds1percent")
        else:
            model = LOWRANK.TrainableLowRankGINE(base, rank, 29001+1000*seed)
            definition = {"rank": rank, "counts": LOWRANK.counts(model)}
    else:
        index = {"tied": None, "first_private": 0, "middle_private": 1, "last_private": 2}[layout]
        model = copy.deepcopy(base) if index is None else GINE.copied_private(base, [index])
        definition = {"private_indices": [] if index is None else [index], "counts": LOWRANK.counts(model)}
    if not torch.equal(torch.get_rng_state(), canonical_cpu_rng):
        raise ValueError("Private construction consumed common/dropout RNG")
    definition.update(canonical_initial_state_sha256=canonical_state, initialization="fresh canonical seed; never P25", actual_training_loss_recorded_separately=True)
    return model, definition


def optimizer_for(model):
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., betas=(.9,.999), eps=1e-8)
    actual = [p for group in opt.param_groups for p in group["params"]]
    expected = [p for p in model.parameters() if p.requires_grad]
    if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != {id(p) for p in expected} or opt.state:
        raise ValueError("Optimizer has duplicate/orphan/missing parameters or nonempty initial moments")
    return opt


def evaluate(model, graphs, ids, device):
    metrics, values, targets = INHERITED.evaluate(model, graphs, device)
    if values.shape != (4, len(ids), 1) or targets.shape != (len(ids), 1):
        raise ValueError("Full validation member evidence shape differs")
    return metrics, values, targets


def fingerprints(graphs, ids):
    return digest([{ "id": int(i), "num_nodes": int(g.num_nodes),
        **{n: tensor_hash(getattr(g,n)) for n in ("x","edge_index","edge_attr","y")}}
        for g,i in zip(graphs,ids)])


def load_export(reference, split):
    # Parent authorization/source and binary hashes must pass before torch.load.
    path = checked(reference["bundle"])
    authorized_completed(PHASE/reference["supervisor_directory"])
    data = torch.load(path, map_location="cpu", weights_only=False)
    expected = "gnnm-molhiv-official-"+split+"-only-bundle-v1"
    scope=data.get("export_scope",{})
    if (data.get("schema") != expected or data.get("dataset")!="ogbg-molhiv"
        or scope.get("test_target_indexing_or_summary") is not False or scope.get("scores_read") is not False
        or scope.get("graph_targets_exported") != "official "+("training" if split=="train" else "validation")+" only"):
        raise ValueError("Require exact split-only export without test graphs/targets")
    id_key = "official_train_indices" if split == "train" else "official_validation_indices"
    ids = torch.as_tensor(data[id_key], dtype=torch.long)
    graphs = [Data(**g) if isinstance(g,dict) else g for g in data["graphs"]]
    if len(graphs) != len(ids) or len(set(ids.tolist())) != len(ids):
        raise ValueError("Incomplete/duplicated official split export")
    for graph in graphs:
        GINE._atoms(graph, graph.x)
        if graph.y.shape != (1,1) or not bool(torch.isfinite(graph.y).all()) or not bool(((graph.y==0)|(graph.y==1)).all()):
            raise ValueError("Invalid official molecular target")
    return graphs, ids


def admitted_inputs(binding):
    audit_future_binding(binding)
    # This semantic report must be produced by root's separate sealed admission,
    # not by this source-only runner. Its hash is bound before deserialization.
    report = json.loads(checked(binding["semantic_admission_report"]).read_text())
    checks = ("P25_checkpoint_record_replay_identity", "all_six_method_locks_before_validation",
        "all_six_P25_cells_qualified", "unchanged_masks101_102", "train_tensor_and_ID_identity",
        "validation_export_matches_full_official_split", "global_own_pooled_lock_replay",
        "correct_route_all_parents", "fresh_GPU_model_and_low_rank_gates")
    if report.get("status") != "passed" or any(report.get(k) is not True for k in checks):
        raise ValueError("Incomplete separately qualified semantic admission")
    if report.get("predictive_sources") != sources():
        raise ValueError("Semantic admission binds different predictive source")
    train, train_ids = load_export(binding["train_export"], "train")
    # No validation object is constructed until ALL training-only locks pass.
    val, val_ids = load_export(binding["validation_export"], "validation")
    if set(train_ids.tolist()) & set(val_ids.tolist()):
        raise ValueError("Official training/validation overlap")
    masks, summary = INHERITED.train_budgets(train, train_ids)
    freeze = json.loads(checked(binding["data_freeze"]).read_text())
    actual = {"official_train_indices_sha256":tensor_hash(train_ids),
        "official_validation_indices_sha256":tensor_hash(val_ids),
        "full_official_train_tensor_sha256":fingerprints(train,train_ids),
        "full_official_validation_tensor_sha256":fingerprints(val,val_ids),
        "train_budget_manifest":summary}
    if any(freeze.get(k) != v for k,v in actual.items()):
        raise ValueError("Accepted full split/mask hashes differ from actual export")
    return train, val, train_ids, val_ids, masks["0.1"], actual


def save(path, value):
    path = inside(path,HERE)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        torch.save(value,stream)


def archive(output, binding, env):
    for relative, expected in sources().items():
        target=output/"source"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PHASE/relative,target)
        if sha(target)!=expected:raise ValueError("Source changed during per-fit archive")
    write_new(output/"admission_binding.json",binding)
    write_new(output/"environment.json",env)
    from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
    from torch_geometric.nn import GINEConv
    for cls in (AtomEncoder,BondEncoder,GINEConv,GINE.GINEBlock,GINE.MolecularProjectorEnsemble,LOWRANK.MemberLowRankLinear):
        p=output/"source/runtime_classes"/(cls.__name__+".py.txt");p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(inspect.getsource(cls))
    for key in ("data_freeze","runtime_freeze","semantic_admission_report","coordinator_approval"):
        shutil.copyfile(checked(binding[key]),output/(key+".json"))


def artifact_manifest(output):
    write_new(output/"artifact_manifest.json",{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name!="artifact_manifest.json"})


def real128_preflight(binding, output, supervisor):
    """Called only by later approved entry point; one real TRAIN batch, no val."""
    audit_future_binding(binding)
    context=live_job_context(supervisor)
    if context["entrypoint_sha256"]!=binding["admission_entrypoint"]["sha256"]:raise ValueError("Unapproved real128 entrypoint")
    INHERITED.gpu_guard(SimpleNamespace(device="cuda:0",physical_gpu=0))
    train,ids=load_export(binding["train_export"],"train")
    masks,summary=INHERITED.train_budgets(train,ids)
    frozen=json.loads(checked(binding["data_freeze"]).read_text())
    actual={"official_train_indices_sha256":tensor_hash(ids),"full_official_train_tensor_sha256":fingerprints(train,ids),"train_budget_manifest":summary}
    if any(frozen.get(k)!=v for k,v in actual.items()):raise ValueError("Real128 accepted full training/mask hashes differ")
    positions=masks["0.1"][:128]
    if len(positions)!=128:raise ValueError("Need128 complete observed training graphs")
    batch=next(iter(DataLoader([train[int(i)] for i in positions],batch_size=128,shuffle=False,num_workers=0))).to("cuda:0")
    rows=[];reference=None
    for layout in LAYOUTS:
        model,definition=model_for(0,layout);model.to("cuda:0").eval()
        with torch.no_grad():initial=model(batch,batch.x)
        if reference is None:reference=initial.detach().clone()
        elif not torch.allclose(initial,reference,atol=1e-6,rtol=0):raise ValueError("Real128 canonical initial member function changed")
        for family in OBJECTIVES:
            fitted,_=model_for(0,layout);fitted.to("cuda:0").train();opt=optimizer_for(fitted)
            before=state_hash(fitted);torch.cuda.manual_seed(0);torch.cuda.synchronize();start=time.perf_counter()
            loss=objective(fitted(batch,batch.x),batch.y.float(),family);loss.backward()
            if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in fitted.parameters()):raise ValueError("Nonfinite/missing real128 gradients")
            opt.step();torch.cuda.synchronize()
            if state_hash(fitted)==before:raise ValueError("Real128 optimizer did not learn")
            rows.append({"layout":layout,"objective":family,"counts":definition["counts"],"loss":float(loss.detach()),"seconds":time.perf_counter()-start})
            del fitted,opt
        del model
    write_new(output,{"status":"passed","scope":"128 official observed training graphs; engineering only","official_ids":ids[positions].tolist(),"mask":summary,"rows":rows,"validation_or_test_loaded":False,"job_context":context,"sources":sources()})


def predictive_environment(guard):
    env=INHERITED.environment()
    env.pop("cuda_context_created_for_capture",None)
    env.update(gpu_uuid=guard["uuid"],device="cuda:0",threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    return env


def runtime_freeze(output, supervisor):
    """Synthetic structural/runtime preparation; no dataset/checkpoint opened."""
    context=live_job_context(supervisor)
    guard=INHERITED.gpu_guard(SimpleNamespace(device="cuda:0",physical_gpu=0))
    env=predictive_environment(guard)
    definitions={}
    for layout in LAYOUTS:
        model,definition=model_for(0,layout);definitions[layout]=definition;del model
    private=[definitions[k]["counts"]["parameter_scalars"] for k in ("first_private","middle_private","last_private")]
    if len(set(private))!=1:raise ValueError("Private placements have different actual parameter counts")
    write_new(output,{"status":"prospective_runtime_and_structural_freeze","predictive_environment_sha256":digest(env),"environment":env,"recipe":RECIPE,"sources":sources(),"definitions":definitions,"job_context":context,"data_accessed":False,"predictive_result":False})


def run_fit(binding, seed, family, layout, output, supervisor):
    """Complete100epoch primary fit; future admitted wrapper supplies binding."""
    if (seed,family,layout) not in {(s,o,l) for s in range(3) for o in OBJECTIVES for l in LAYOUTS}:raise ValueError("Outside prescribed30matrix")
    context=live_job_context(supervisor)
    if context["entrypoint_sha256"]!=binding["admission_entrypoint"]["sha256"]:raise ValueError("Unapproved predictive entrypoint")
    guard=INHERITED.gpu_guard(SimpleNamespace(device="cuda:0",physical_gpu=0))
    train,val,train_ids,val_ids,positions,data_identity=admitted_inputs(binding)
    selected_train=[train[int(i)].clone() for i in positions]
    # Supply immutable IDs on copies only; record the DataLoader's actual order.
    for graph,official in zip(selected_train,train_ids[positions]):graph.gnnm_official_id=torch.tensor([int(official)])
    model,definition=model_for(seed,layout);model.to("cuda:0")
    optimizer=optimizer_for(model)
    torch.cuda.manual_seed(seed)
    env=predictive_environment(guard)
    runtime=json.loads(checked(binding["runtime_freeze"]).read_text())
    if runtime.get("predictive_environment_sha256")!=digest(env):raise ValueError("Exact prospective environment differs")
    if runtime["definitions"][layout]["counts"]!=definition["counts"]:raise ValueError("Actual model capacity differs from qualified prospective count")
    output=inside(output,HERE)
    if output.exists():raise ValueError("Exclusive new fit output required")
    output.mkdir(parents=True);archive(output,binding,env)
    contract={"schema":"gnnm-primary-molhiv-predictive-fit-v1","seed":seed,"training_loss":family,"layout":layout,"label_fraction":.1,"recipe":RECIPE,"definition":definition,"data_identity":data_identity,"sources":sources(),"environment_sha256":digest(env),"job_context":context,"all_training_locks_before_first_validation":True,"test_subset_or_score":False}
    write_new(output/"contract.json",contract);contract_sha=sha(output/"contract.json")
    save(output/"initial.pt",{"state_dict":model.state_dict(),"canonical_seed":seed,"model_state_sha256":state_hash(model),"rng":INHERITED.rng_state(torch.device("cuda:0")),"contract_sha256":contract_sha})
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();trace=[]
    try:
        for epoch in range(1,101):
            loader=DataLoader(selected_train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(19709+1000*seed+epoch),num_workers=0)
            model.train();streams=[];sums={"own":0.,"pooled":0.};seen=0
            for graph in loader:
                streams.append(graph.gnnm_official_id.view(-1).tolist());graph=graph.to("cuda:0")
                optimizer.zero_grad(set_to_none=True);values=model(graph,graph.x)
                losses={name:objective(values,graph.y.float(),name) for name in OBJECTIVES}
                losses[family].backward()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()) or any(not bool(torch.isfinite(v)) for v in losses.values()):raise ValueError("Nonfinite or missing actual training gradient/loss")
                optimizer.step();seen+=graph.num_graphs
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError("Nonfinite AdamW parameter update")
                for name in OBJECTIVES:sums[name]+=float(losses[name].detach())*graph.num_graphs
            if sorted(x for b in streams for x in b)!=sorted(train_ids[positions].tolist()):raise ValueError("Training stream omitted/duplicated graph")
            metrics,logits,targets=evaluate(model,val,val_ids,torch.device("cuda:0"))
            row={"epoch":epoch,"train_own_bce":sums["own"]/seen,"train_pooled_bce":sums["pooled"]/seen,**metrics,"seconds":time.perf_counter()-start};trace.append(row)
            write_new(output/"streams"/f"epoch_{epoch:03d}.json",{"batches":streams,"generator_seed":19709+1000*seed+epoch,"stream_sha256":digest(streams)})
            save(output/"epochs"/f"epoch_{epoch:03d}.pt",{"epoch":epoch,"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"model_state_sha256":state_hash(model),"rng":INHERITED.rng_state(torch.device("cuda:0")),"contract_sha256":contract_sha})
            save(output/"evidence"/f"epoch_{epoch:03d}.pt",{"epoch":epoch,"member_logits":logits,"targets":targets,"official_validation_indices":val_ids,"metrics":metrics,"contract_sha256":contract_sha})
        best=max(trace,key=lambda r:INHERITED.selection_key(r,r["epoch"]))
        write_new(output/"trace.json",trace);write_new(output/"selected.json",best)
        selected_epoch=best["epoch"]
        shutil.copyfile(output/"epochs"/f"epoch_{selected_epoch:03d}.pt",output/"selected.pt")
        shutil.copyfile(output/"evidence"/f"epoch_{selected_epoch:03d}.pt",output/"selected_evidence.pt")
        elapsed=time.perf_counter()-start
        write_new(output/"result.json",{"status":"complete_pending_completed_authorization_and_replay","selected":best,"epoch100":trace[-1],"CUDA_peak_allocated_bytes":torch.cuda.max_memory_allocated(),"CUDA_peak_reserved_bytes":torch.cuda.max_memory_reserved(),"elapsed_seconds":elapsed,"training_graphs":len(selected_train),"training_graph_visits":100*len(selected_train),"effective_end_to_end_training_graph_visits_per_second":100*len(selected_train)/elapsed,"epoch_checkpoint_bytes":sum(p.stat().st_size for p in (output/"epochs").glob("*.pt")),"epoch_validation_evidence_bytes":sum(p.stat().st_size for p in (output/"evidence").glob("*.pt")),"throughput_scope":"includes full validation after every epoch and evidence serialization; observed for this fit only","test_scored":False,"adaptive_aliases":"none selected from validation; separately locked identities only"})
    except Exception as exc:
        write_new(output/"failure.json",{"type":type(exc).__name__,"message":str(exc),"completed_epochs":len(trace),"test_scored":False})
        raise
    finally:artifact_manifest(output)


def replay(run, binding, output, completed_supervisor, supervisor):
    """Same-device selected and epoch100 forward; all epoch metrics audited."""
    context=live_job_context(supervisor)
    if context["entrypoint_sha256"]!=binding["admission_entrypoint"]["sha256"]:raise ValueError("Unapproved replay entrypoint")
    authorized_completed(completed_supervisor)
    run=inside(run,HERE);files=json.loads((run/"artifact_manifest.json").read_text())
    actual={str(p.relative_to(run)) for p in run.rglob("*") if p.is_file() and p.name!="artifact_manifest.json"}
    if set(files)!=actual or any(sha(run/name)!=h for name,h in files.items()):raise ValueError("Fit artifact coverage/hash changed")
    c=json.loads((run/"contract.json").read_text());trace=json.loads((run/"trace.json").read_text())
    parent=inside(completed_supervisor)
    if str(parent.relative_to(PHASE))!=c["job_context"]["supervisor_directory"] or sha(parent/"command.json")!=c["job_context"]["command_sha256"] or sha(parent/"environment.json")!=c["job_context"]["environment_sha256"]:raise ValueError("Completed authorization is not this fitted job")
    if c["sources"]!=sources() or c["recipe"]!=RECIPE:raise ValueError("Replay runner/model source or recipe differs")
    if digest(binding)!=digest(json.loads((run/"admission_binding.json").read_text())):raise ValueError("Replay admission binding differs from the fitted binding")
    train,val,train_ids,val_ids,positions,identity=admitted_inputs(binding)
    if identity!=c["data_identity"]:raise ValueError("Replay full data identity differs")
    if [r["epoch"] for r in trace]!=list(range(1,101)):raise ValueError("Incomplete epoch evidence")
    targets_ref=torch.cat([graph.y.float() for graph in val],dim=0)
    for row in trace:
        e=torch.load(run/"evidence"/f"epoch_{row['epoch']:03d}.pt",map_location="cpu",weights_only=False)
        cp=torch.load(run/"epochs"/f"epoch_{row['epoch']:03d}.pt",map_location="cpu",weights_only=False)
        if e["epoch"]!=row["epoch"] or cp["epoch"]!=row["epoch"] or e["contract_sha256"]!=sha(run/"contract.json") or cp["contract_sha256"]!=sha(run/"contract.json"):raise ValueError("Epoch/contract binding changed")
        if not torch.equal(e["official_validation_indices"],val_ids):raise ValueError("Validation order changed")
        if not torch.equal(targets_ref,e["targets"]):raise ValueError("Retained validation targets differ from the full official validation export")
        m=INHERITED.score_evidence(e["member_logits"],e["targets"])
        if any(abs(m[k]-row[k])>1e-9 for k in m):raise ValueError("Per-epoch metric replay differs")
    best=max(trace,key=lambda r:INHERITED.selection_key(r,r["epoch"]))
    if best!=json.loads((run/"selected.json").read_text()):raise ValueError("Selection does not replay")
    result=json.loads((run/"result.json").read_text())
    if result.get("selected")!=best or result.get("epoch100")!=trace[-1]:raise ValueError("Selected/fixed-budget result differs from the complete trace")
    if sha(run/"selected.pt")!=sha(run/"epochs"/f"epoch_{best['epoch']:03d}.pt") or sha(run/"selected_evidence.pt")!=sha(run/"evidence"/f"epoch_{best['epoch']:03d}.pt"):raise ValueError("Selected checkpoint/evidence differs from its selected epoch")
    guard=INHERITED.gpu_guard(SimpleNamespace(device="cuda:0",physical_gpu=0))
    if digest(predictive_environment(guard))!=c["environment_sha256"]:raise ValueError("Same-device replay runtime differs from the fitted runtime")
    model,_=model_for(c["seed"],c["layout"]);model.to("cuda:0")
    endpoints={}
    for epoch in sorted({best["epoch"],100}):
        cp=torch.load(run/"epochs"/f"epoch_{epoch:03d}.pt",map_location="cpu",weights_only=False)
        model.load_state_dict(cp["state_dict"],strict=True)
        if state_hash(model)!=cp["model_state_sha256"]:raise ValueError("Loaded endpoint state hash differs")
        m,values,targets=evaluate(model,val,val_ids,torch.device("cuda:0"))
        retained=torch.load(run/"evidence"/f"epoch_{epoch:03d}.pt",map_location="cpu",weights_only=False)
        if not torch.equal(targets,retained["targets"]) or not torch.allclose(values,retained["member_logits"],atol=1e-6,rtol=0):raise ValueError("Full endpoint checkpoint member-logit replay failed at epoch"+str(epoch))
        if any(abs(m[k]-trace[epoch-1][k])>1e-9 for k in m):raise ValueError("Full endpoint forward metric differs at epoch"+str(epoch))
        endpoints[epoch]={"epoch":epoch,"metrics":m,"maximum_member_logit_error":float((values-retained["member_logits"]).abs().max()),"checkpoint_sha256":sha(run/"epochs"/f"epoch_{epoch:03d}.pt"),"evidence_sha256":sha(run/"evidence"/f"epoch_{epoch:03d}.pt")}
    write_new(output,{"status":"passed","scope":"all100epoch retained metrics/selection plus full official-validation selected and epoch100 checkpoint forwards","selected_epoch":best["epoch"],"maximum_member_logit_error":endpoints[best["epoch"]]["maximum_member_logit_error"],"epoch100_reuses_selected_forward":best["epoch"]==100,"full_forward_endpoints":{"selected":endpoints[best["epoch"]],"epoch100":endpoints[100]},"replay_atol":1e-6,"fit_artifact_manifest_sha256":sha(run/"artifact_manifest.json"),"authorization":authorized_completed(completed_supervisor),"replay_job_context":context,"test_scored":False})
