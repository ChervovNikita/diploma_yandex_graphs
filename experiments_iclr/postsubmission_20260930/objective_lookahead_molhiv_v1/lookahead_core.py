"""Paired real AdamW updates and pooled training-graph scoring, fixture-ready."""
import copy
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
from functools import lru_cache
import math
import random

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from lookahead_contract import (HERE, PHASE, SOURCE_ONLY, PLACEMENTS, LOSSES, RECIPE, RULE,
                                canonical, verify_model_sources)


def source_module():
    verify_model_sources()
    return _load_source_module()


@lru_cache(maxsize=1)
def _load_source_module():
    spec=importlib.util.spec_from_file_location("gnnm_lookahead_exact_GINE_source_v1",PHASE/"new_tasks/gine_adapters.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@dataclass
class Unit:
    unit_id: str
    group_id: str
    graph: object
    split: str = "train"
    synthetic: bool = False


def tensor_hash(value):
    x=value.detach().cpu().contiguous()
    meta={"dtype":str(x.dtype),"shape":list(x.shape)}
    return hashlib.sha256(canonical(meta)+x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_hash(model):
    return hashlib.sha256(canonical({k:tensor_hash(v) for k,v in model.state_dict().items()})).hexdigest()


def objective_values(logits,targets):
    if logits.ndim!=3 or logits.shape[0]!=4 or targets.shape!=logits.shape[1:]:
        raise ValueError("Require logits[4,G,T] and graph targets[G,T]")
    if not bool(torch.isfinite(logits).all()):raise ValueError("Nonfinite logits")
    if bool(torch.isinf(targets).any()):raise ValueError("Infinite targets are not missing labels")
    finite=torch.isfinite(targets);counts=finite.sum(-1)
    if not bool((counts>0).all()):raise ValueError("Wholly hidden graphs cannot enter loss")
    if not bool(((targets[finite]==0)|(targets[finite]==1)).all()):raise ValueError("Nonbinary observed targets")
    safe=torch.where(finite,targets,torch.zeros_like(targets))
    own=F.binary_cross_entropy_with_logits(logits,safe[None].expand_as(logits),reduction="none")
    own_by_member_graph=(own*finite[None]).sum(-1)/counts[None]
    pooled_logits=logits.mean(0)
    pool=F.binary_cross_entropy_with_logits(pooled_logits,safe,reduction="none")
    pool_by_graph=(pool*finite).sum(-1)/counts
    return {"own":own_by_member_graph.mean(),"pooled":pool_by_graph.mean()}, {
        "own":own_by_member_graph.mean(0),"pooled":pool_by_graph}


def checked_units(update,screen,source_fit_groups,fixture_mode):
    if not update or not screen or not source_fit_groups:raise ValueError("Nonempty P/update/screen groups required")
    combined=[*update,*screen]
    if len({u.unit_id for u in combined})!=len(combined):raise ValueError("Repeated unit crosses update/screen")
    for u in combined:
        if not isinstance(u.unit_id,str) or not u.unit_id or not isinstance(u.group_id,str) or not u.group_id:
            raise ValueError("Nonempty unit and group identifiers required")
        if u.split!="train":raise ValueError("Training-only units required")
        if u.synthetic is not fixture_mode:raise ValueError("Fixture/real unit scope mismatch")
        if not all(torch.is_tensor(t) for t in (u.graph.x,u.graph.edge_index,u.graph.edge_attr,u.graph.y)):
            raise ValueError("Complete graph tensors required")
        if u.graph.y.shape!=(1,1) or not bool(torch.isfinite(u.graph.y).all()):raise ValueError("Use observed MolHIV graph targets only")
        if not bool(((u.graph.y==0)|(u.graph.y==1)).all()):raise ValueError("Binary graph target required")
        if any(t.device.type!="cpu" for t in (u.graph.x,u.graph.edge_index,u.graph.edge_attr,u.graph.y)):
            raise ValueError("CPU graph tensors only")
        if u.graph.x.dtype!=torch.long or u.graph.x.ndim!=2 or u.graph.x.shape[1]!=9:
            raise ValueError("Original OGB categorical atom fields required")
        if u.graph.edge_index.dtype!=torch.long or u.graph.edge_index.ndim!=2 or u.graph.edge_index.shape[0]!=2:
            raise ValueError("Original OGB edge indices required")
        if u.graph.edge_attr.dtype!=torch.long or u.graph.edge_attr.shape!=(u.graph.edge_index.shape[1],3):
            raise ValueError("Original OGB categorical bond fields required")
        if u.graph.y.dtype!=torch.float32:raise ValueError("Native float32 graph targets required")
    ug,sg={u.group_id for u in update},{u.group_id for u in screen}
    if ug&sg or ug&set(source_fit_groups) or sg&set(source_fit_groups):raise ValueError("Molecule/scaffold groups cross P/update/screen")
    if not fixture_mode and min(len(ug),len(sg))<RULE["minimum_groups_per_update_and_screen_fold"]:
        raise ValueError("Too few grouped graph units for proposed mechanism screen")
    return {"update_groups":sorted(ug),"screen_groups":sorted(sg),"source_fit_groups":sorted(source_fit_groups)}


def update_schedule(units,seed):
    ordered=sorted(units,key=lambda u:u.unit_id);result=[];cycle=0;position=0;permutation=[]
    for step in range(RECIPE["steps"]):
        if position==len(permutation):
            permutation=list(range(len(ordered)))
            random.Random(21109+1000*seed+cycle).shuffle(permutation);position=0;cycle+=1
        indices=permutation[position:position+RECIPE["batch_size"]];position+=len(indices)
        result.append([ordered[i] for i in indices])
    return result


def population_summary(units):
    groups=Counter(u.group_id for u in units)
    sizes=sorted(groups.values())
    positives=sum(int(u.graph.y.item()) for u in units)
    return {"graphs":len(units),"groups":len(groups),"positive_graphs":positives,
            "negative_graphs":len(units)-positives,"group_sizes":dict(sorted(groups.items())),
            "minimum_group_graphs":min(sizes),"maximum_group_graphs":max(sizes),
            "largest_group_graph_fraction":max(sizes)/len(units)}


def score(model,units):
    model.eval();pooled=[];own=[];ids=[]
    with torch.no_grad():
        ordered=sorted(units,key=lambda u:u.unit_id)
        for start in range(0,len(ordered),RECIPE["batch_size"]):
            part=ordered[start:start+RECIPE["batch_size"]]
            batch=Batch.from_data_list([u.graph for u in part]);logits=model(batch,batch.x)
            _,rows=objective_values(logits,batch.y.float())
            pooled.extend(rows["pooled"].double().tolist());own.extend(rows["own"].double().tolist())
            ids.extend(u.unit_id for u in part)
    return {"unit_ids":ids,"pooled_bce_by_graph":pooled,"own_bce_by_graph":own,
            "pooled_bce":math.fsum(pooled)/len(pooled),"own_bce":math.fsum(own)/len(own)}


def initial_models(common,include_null=False):
    module=source_module()
    if type(common) is not module.MolecularProjectorEnsemble:
        raise ValueError("Build the exact tied GINE through source_module()")
    if common.private_blocks or common.members!=4:raise ValueError("Common source must be tied, four members")
    expected=module.gine_options(output_dim=1,hidden_dim=128,num_layers=3,dropout=.1,hidden_dim_multiplier=2)
    if common.options!=expected:raise ValueError("Unchanged h128/three-layer recipe required")
    states={};counts=[]
    with torch.random.fork_rng(devices=[]):
        for index in range(3):counts.append(sum(p.numel() for p in common.blocks[index].parameters()))
        if len(set(counts))!=1:raise ValueError("First/middle/last capacities differ")
        for loss in LOSSES:
            for placement in PLACEMENTS:
                model=copy.deepcopy(common)
                if placement!="tied":model.privatize((PLACEMENTS.index(placement)-1,))
                model.zero_grad(set_to_none=True)
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.,eps=1e-8,betas=(.9,.999))
                if opt.state:raise ValueError("Fresh AdamW state required")
                extra=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in common.parameters())
                if extra!=(0 if placement=="tied" else 3*counts[0]):raise ValueError("Wrong block copy storage")
                states[loss+"_"+placement]=(model,opt)
            if include_null:
                model=copy.deepcopy(common);model.zero_grad(set_to_none=True)
                states[loss+"_null_tied_copy"]=(model,torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.,eps=1e-8,betas=(.9,.999)))
    return states,counts[0]


def paired_lookahead(common,update,screen,source_fit_groups,seed,*,fixture_mode=False,include_null=False):
    if not fixture_mode and SOURCE_ONLY:raise ValueError("Real MolHIV inputs not admitted in source-only v1")
    if seed not in (0,1,2):raise ValueError("Only three declared source seeds")
    if torch.cuda.is_initialized():raise ValueError("No CUDA context permitted")
    if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:raise ValueError("One-thread CPU recipe required")
    if any(p.device.type!="cpu" or p.dtype!=torch.float32 for p in common.parameters()):raise ValueError("Native CPU float32 source required")
    groups=checked_units(update,screen,source_fit_groups,fixture_mode)
    before=state_hash(common);input_hashes={u.unit_id:{"x":tensor_hash(u.graph.x),"edge_index":tensor_hash(u.graph.edge_index),
                      "edge_attr":tensor_hash(u.graph.edge_attr),"y":tensor_hash(u.graph.y)} for u in [*update,*screen]}
    schedule=update_schedule(update,seed);rows={}
    seen={u.unit_id:u for step in schedule for u in step}
    coverage={"update_fold":population_summary(update),"screen_fold":population_summary(screen),
              "seen_update_unique":population_summary(list(seen.values())),
              "update_graph_presentations":sum(len(step) for step in schedule),
              "update_unique_graph_fraction":len(seen)/len(update)}
    with torch.random.fork_rng(devices=[]):
        states,block_count=initial_models(common,include_null)
        probe_batch=Batch.from_data_list([u.graph for u in schedule[0]])
        initial_logits=None
        for name,(model,opt) in states.items():
            model.eval()
            with torch.no_grad():logits=model(probe_batch,probe_batch.x)
            if initial_logits is None:initial_logits=logits.detach().clone()
            elif not torch.equal(initial_logits,logits):raise ValueError("Initial copied layout changes source function")
        for name,(model,opt) in states.items():
            loss=name.split("_")[0];initial=score(model,screen);step_rows=[]
            for step,units in enumerate(schedule):
                model.train();torch.manual_seed(27431+1000*seed+step)
                batch=Batch.from_data_list([u.graph for u in units]);opt.zero_grad(set_to_none=True)
                logits=model(batch,batch.x);values,_=objective_values(logits,batch.y.float())
                if not bool(torch.isfinite(values[loss])):raise ValueError("Nonfinite training loss")
                values[loss].backward()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise ValueError("Missing/nonfinite actual optimizer gradient")
                opt.step()
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError("Nonfinite AdamW update")
                step_rows.append({"step":step+1,"unit_ids":[u.unit_id for u in units],"dropout_seed":27431+1000*seed+step,
                                  "own_loss":float(values["own"].detach()),"pooled_loss":float(values["pooled"].detach())})
            final=score(model,screen)
            steps=sorted({float(s["step"]) for s in opt.state.values()})
            if steps!=[float(RECIPE["steps"])]:raise ValueError("Actual optimizer step count differs")
            rows[name]={"training_loss":loss,"initial_screen":initial,"final_screen":final,"steps":step_rows,
                        "initial_optimizer_empty":True,"optimizer_step_counts":steps,"final_state_sha256":state_hash(model),
                        "parameter_count":sum(p.numel() for p in model.parameters()),"private_block_parameter_count":block_count}
        for name,row in rows.items():
            tied=rows[row["training_loss"]+"_tied"]["final_screen"]
            row["paired_pooled_bce_gain_vs_actual_tied_after_K"]=tied["pooled_bce"]-row["final_screen"]["pooled_bce"]
            row["paired_graph_gains_vs_actual_tied_after_K"]=[x-y for x,y in zip(tied["pooled_bce_by_graph"],row["final_screen"]["pooled_bce_by_graph"])]
    if state_hash(common)!=before:raise ValueError("Common source changed")
    for u in [*update,*screen]:
        now={"x":tensor_hash(u.graph.x),"edge_index":tensor_hash(u.graph.edge_index),"edge_attr":tensor_hash(u.graph.edge_attr),"y":tensor_hash(u.graph.y)}
        if now!=input_hashes[u.unit_id]:raise ValueError("Training graph input mutated")
    if torch.cuda.is_initialized():raise ValueError("CUDA context created")
    return {"schema":"gnnm-molhiv-actual-optimizer-lookahead-cell-v1","scope":"synthetic engineering fixture" if fixture_mode else "training-only mechanism screen",
            "source_only":SOURCE_ONLY,"seed":seed,"source_state_sha256":before,"groups":groups,
            "recipe":RECIPE,"rule":RULE,"trajectories":rows,"input_hashes":input_hashes,"coverage":coverage,
            "source_and_inputs_preserved":True,"update_unique_graphs":len({u.unit_id for step in schedule for u in step}),
            "update_fold_graphs":len(update),"screen_fold_graphs":len(screen),"screen_scored_completely":True,
            "predictive_accuracy_claimed":False,"novelty_claimed":False,"validation_or_test_loaded":False}


def mechanism_rule(gains):
    """Aggregate all48 trajectories before choosing; no accuracy promotion."""
    expected={(seed,fraction,loss,placement) for seed in range(3) for fraction in (.1,1.)
              for loss in LOSSES for placement in PLACEMENTS}
    if set(gains)!=expected or any(not math.isfinite(x) for x in gains.values()):
        raise ValueError("Complete six-cell/eight-arm finite gain matrix required")
    if any(value!=0. for key,value in gains.items() if key[3]=="tied"):
        raise ValueError("Actual tied-after-K baseline gains must be exactly zero")
    result={}
    for loss in LOSSES:
        rows={}
        for placement in PLACEMENTS[1:]:
            primary=[gains[s,.1,loss,placement] for s in range(3)]
            adverse=[gains[s,1.,loss,placement] for s in range(3)]
            eligible=(min(primary)>=RULE["primary_each_seed_minimum_gain"]+RULE["numerical_margin"]
                      and math.fsum(primary)/3>=RULE["primary_mean_minimum_gain"]+RULE["numerical_margin"]
                      and min(adverse)>=-RULE["full_label_each_seed_maximum_harm"]+RULE["numerical_margin"])
            rows[placement]={"primary_seed_gains":primary,"full_label_seed_gains":adverse,
                             "mean_primary_gain":math.fsum(primary)/3,"mechanism_screen_eligible":eligible}
        eligible=[p for p in PLACEMENTS[1:] if rows[p]["mechanism_screen_eligible"]]
        chosen=max(eligible,key=lambda p:rows[p]["mean_primary_gain"]) if eligible else None
        result[loss]={"placements":rows,"candidate_for_separate_full_predictive_study":chosen,"abstain":chosen is None}
    return {"loss_families":result,"accuracy_promotion_allowed":False,
            "screening_is_not_independent_confirmation":True,"all_48_gains":{str(k):v for k,v in gains.items()},"rule":RULE}
