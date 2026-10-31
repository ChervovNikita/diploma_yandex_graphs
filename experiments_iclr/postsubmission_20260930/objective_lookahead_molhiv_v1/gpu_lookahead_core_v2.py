"""Source-only deterministic GPU actual-AdamW lookahead; synthetic fixtures only.

No CPU molecular sibling is required. Future real execution needs a separately
versioned admission API with correct-route authorization and GPU P25 evidence.
"""
import math
import os
import torch
from torch_geometric.data import Batch

from lookahead_contract import RECIPE, RULE, PLACEMENTS, LOSSES
from lookahead_core import (checked_units, initial_models, objective_values,
                            population_summary, state_hash, tensor_hash, update_schedule)

SOURCE_ONLY = True
GPU_RECIPE = {**RECIPE,"device":"one authorized GPU exposed as cuda:0",
    "dtype":"float32","AMP":False,"deterministic_algorithms":True,
    "matmul_tf32":False,"cudnn_tf32":False,"cudnn_benchmark":False,
    "cublas_workspace_config":":4096:8",
    "GPU_input_source":"newly qualified correct-route GPU P-only epoch25 checkpoint",
    "CPU_molecular_sibling_required":False}


def gpu_environment():
    if torch.cuda.device_count()!=1:raise ValueError("Expose exactly one authorized GPU")
    if not torch.are_deterministic_algorithms_enabled():raise ValueError("Require deterministic GPU algorithms; no fallback")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("Require TF32 disabled")
    if torch.backends.cudnn.benchmark:raise ValueError("Require cudnn benchmark disabled")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before any CUDA initialization")
    if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
        raise ValueError("Require one intra-op/inter-op thread")
    properties=torch.cuda.get_device_properties(0)
    return {"device":"cuda:0","visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name":properties.name,"total_memory_bytes":properties.total_memory,
        "torch":torch.__version__,"compiled_cuda":torch.version.cuda,
        "deterministic_algorithms":True,"matmul_tf32":False,"cudnn_tf32":False,
        "cudnn_benchmark":False,"cublas_workspace_config":":4096:8"}


def score_gpu(model,units):
    model.eval();pooled=[];own=[];ids=[];member_logits=[]
    with torch.no_grad():
        ordered=sorted(units,key=lambda u:u.unit_id)
        for start in range(0,len(ordered),RECIPE["batch_size"]):
            part=ordered[start:start+RECIPE["batch_size"]]
            batch=Batch.from_data_list([u.graph for u in part]).to("cuda:0")
            logits=model(batch,batch.x)
            _,rows=objective_values(logits,batch.y.float())
            pooled.extend(rows["pooled"].double().cpu().tolist())
            own.extend(rows["own"].double().cpu().tolist())
            ids.extend(u.unit_id for u in part)
            member_logits.append(logits.detach().cpu())
    logits=torch.cat(member_logits,dim=1)
    return {"unit_ids":ids,"pooled_bce_by_graph":pooled,"own_bce_by_graph":own,
        "pooled_bce":math.fsum(pooled)/len(pooled),"own_bce":math.fsum(own)/len(own),
        "member_logits_sha256":tensor_hash(logits)},logits


def paired_gpu_lookahead(common,update,screen,source_fit_groups,seed,*,fixture_mode=False,include_null=False):
    """Return report and final (model,optimizer,logits) artifacts for a fixture.

    Real inputs are refused before device inspection. The future admitted API
    must preserve this implementation rather than monkeypatching this flag.
    """
    if not fixture_mode and SOURCE_ONLY:
        raise ValueError("Real GPU MolHIV inputs not admitted in source-only v2")
    if seed not in (0,1,2):raise ValueError("Only three declared source seeds")
    environment=gpu_environment()
    if any(p.device!=torch.device("cuda:0") or p.dtype!=torch.float32 for p in common.parameters()):
        raise ValueError("Native float32 common source on cuda:0 required")
    groups=checked_units(update,screen,source_fit_groups,fixture_mode)
    before=state_hash(common)
    def inputs():
        return {u.unit_id:{k:tensor_hash(getattr(u.graph,k)) for k in ("x","edge_index","edge_attr","y")}
                for u in [*update,*screen]}
    input_hashes=inputs();schedule=update_schedule(update,seed)
    seen={u.unit_id:u for step in schedule for u in step}
    coverage={"update_fold":population_summary(update),"screen_fold":population_summary(screen),
        "seen_update_unique":population_summary(list(seen.values())),
        "update_graph_presentations":sum(len(step) for step in schedule),
        "update_unique_graph_fraction":len(seen)/len(update)}
    rows={};artifacts={}
    with torch.random.fork_rng(devices=[0]):
        states,block_count=initial_models(common,include_null)
        probe=Batch.from_data_list([u.graph for u in schedule[0]]).to("cuda:0")
        initial_logits=None
        for name,(model,optimizer) in states.items():
            model.eval()
            with torch.no_grad():logits=model(probe,probe.x)
            if initial_logits is None:initial_logits=logits.detach().clone()
            elif not torch.equal(initial_logits,logits):
                raise ValueError("Private GPU layout changes initial source function")
        for name,(model,optimizer) in states.items():
            loss=name.split("_")[0];initial,_=score_gpu(model,screen);trace=[]
            for step,units in enumerate(schedule):
                model.train();torch.manual_seed(27431+1000*seed+step)
                batch=Batch.from_data_list([u.graph for u in units]).to("cuda:0")
                optimizer.zero_grad(set_to_none=True)
                values,_=objective_values(model(batch,batch.x),batch.y.float())
                if not bool(torch.isfinite(values[loss])):raise ValueError("Nonfinite GPU update loss")
                values[loss].backward()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise ValueError("Missing/nonfinite actual GPU gradient")
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise ValueError("Nonfinite GPU AdamW update")
                trace.append({"step":step+1,"unit_ids":[u.unit_id for u in units],
                    "dropout_seed":27431+1000*seed+step,
                    "own_loss":float(values["own"].detach()),"pooled_loss":float(values["pooled"].detach())})
            final,logits=score_gpu(model,screen)
            steps=sorted({float(s["step"]) for s in optimizer.state.values()})
            if steps!=[float(RECIPE["steps"])]:raise ValueError("GPU optimizer did not perform exactly K updates")
            rows[name]={"training_loss":loss,"initial_screen":initial,"final_screen":final,
                "steps":trace,"initial_optimizer_empty":True,"optimizer_step_counts":steps,
                "final_state_sha256":state_hash(model),"parameter_count":sum(p.numel() for p in model.parameters()),
                "private_block_parameter_count":block_count}
            artifacts[name]=(model,optimizer,logits)
        for name,row in rows.items():
            tied=rows[row["training_loss"]+"_tied"]["final_screen"]
            row["paired_pooled_bce_gain_vs_actual_tied_after_K"]=tied["pooled_bce"]-row["final_screen"]["pooled_bce"]
            row["paired_graph_gains_vs_actual_tied_after_K"]=[x-y for x,y in zip(
                tied["pooled_bce_by_graph"],row["final_screen"]["pooled_bce_by_graph"])]
    torch.cuda.synchronize(0)
    if state_hash(common)!=before or inputs()!=input_hashes:raise ValueError("GPU lookahead changed source or CPU input tensors")
    if gpu_environment()!=environment:raise ValueError("GPU runtime flags changed during lookahead")
    return {"schema":"gnnm-molhiv-GPU-actual-optimizer-lookahead-cell-v2",
        "scope":"synthetic GPU engineering fixture" if fixture_mode else "training-only GPU mechanism screen",
        "source_only":SOURCE_ONLY,"seed":seed,"source_state_sha256":before,
        "groups":groups,"coverage":coverage,"recipe":GPU_RECIPE,"rule":RULE,
        "environment":environment,"trajectories":rows,"input_hashes":input_hashes,
        "source_and_inputs_preserved":True,"screen_scored_completely":True,
        "predictive_accuracy_claimed":False,"novelty_claimed":False,
        "CPU_molecular_sibling_required":False,"validation_or_test_loaded":False},artifacts
