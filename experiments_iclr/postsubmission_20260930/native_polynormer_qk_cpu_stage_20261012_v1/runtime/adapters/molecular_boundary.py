"""Original boundary GNNM blocks adapted to scalar molecular graph prediction.

Source BEBlock init/equations and native member BCE/4 backwards/AdamW0/mean
logits are retained. OGB atom/bond encoding, shared GINE, graph pooling,
minibatching, scaffold roles and graph budgets are declared new-task changes.
"""
from __future__ import annotations

import copy
import math

from .molecular import (MolecularAdapter,MolecularEpochMeans,_positive_integer,_finite_number,_receipt,
                        bound_file,load_source,validate_pack,validate_role_authorities,validate_cost_contract)
from .node import source_module
try:
    from ..engine import AdapterStepError
except ImportError:
    from engine import AdapterStepError

ARM="original_gnnm_binary_graph"


def validate_boundary_config(config,run):
    required={"molecular_inputs","molecular_source","boundary_source","model","optimizer","training","arm","fusion",
              "initialization_seed","dropout_seed","shuffle_seed","head_policy","contrastive","target_policy","is_original_gnnm","molecular_cost","train_minibatches_per_epoch","members"}
    optional={"maximum_cuda_bytes","stage","reservation_mib","estimated_seconds","selected_config_from","resolved_selection"}
    if not required<=set(config) or set(config)-required-optional or config["arm"]!=ARM or config["is_original_gnnm"] is not True:
        raise ValueError("Declare the explicit original boundary GNNM binary graph adaptation")
    if (config["members"]!=4 or config["fusion"]!="mean_logits" or config["head_policy"]!="source_boundary_scalar_graph_adaptation"
            or config["target_policy"]!="complete_binary" or config["model"]["outputs"]!=1):
        raise ValueError("Boundary graph adaptation requires complete scalar binary targets and mean raw logits")
    if set(config["molecular_inputs"])!={"train","select"} or set(config["molecular_source"])!={"models","factors","objectives"}:
        raise ValueError("Boundary fit opens only TRAIN/SELECT and exact pinned source modules")
    for name,row in {**config["molecular_inputs"],**config["molecular_source"],"boundary":config["boundary_source"]}.items():_receipt(row,name)
    if set(config["model"])!={"hidden","layers","dropout","outputs"}:
        raise ValueError("Declare exact GINE model shape")
    for key in ["hidden","layers","outputs"]:_positive_integer(config["model"][key],key)
    _finite_number(config["model"]["dropout"],"dropout",maximum=1.)
    if config["training"]!={"batch_size":128,"eval_batch_size":128,"views_per_update":1}:
        # Small synthetic fixtures may declare their own batch size, preserving the same native view law.
        if (set(config["training"])!={"batch_size","eval_batch_size","views_per_update"}
                or config["training"]["views_per_update"]!=1):
            raise ValueError("Original boundary own law has one forward/member per minibatch")
        for key in ["batch_size","eval_batch_size"]:_positive_integer(config["training"][key],key)
    optimizer=config["optimizer"]
    if (set(optimizer)!={"name","learning_rate","weight_decay","eps","betas"} or optimizer["name"]!="AdamW"
            or optimizer["weight_decay"]!=0 or optimizer["eps"]!=1e-8 or optimizer["betas"]!=[.9,.999]):
        raise ValueError("Retain source one AdamW update with zero weight decay")
    _finite_number(optimizer["learning_rate"],"learning rate",strict=True)
    c=config["contrastive"]
    if (set(c)!={"max_objects","alignment_weight","residual_weight","temperature"}
            or c["alignment_weight"]!=0 or c["residual_weight"]!=0 or c["temperature"]!=.2):
        raise ValueError("Original reference has no auxiliary or ineffective coefficient search")
    for key in ["initialization_seed","dropout_seed","shuffle_seed"]:
        if type(config[key]) is not int or not 0<=config[key]<2**32:
            raise ValueError("Explicit uint32 original reference stream seed required")
    if run["budget"]["unit"]!="epochs" or run["budget"]["policy"]!="full_budget":
        raise ValueError("Graph-task epoch budget is an explicitly standardized adaptation")
    _positive_integer(run["budget"]["maximum"],"graph epoch budget")
    if "stage" in config and config["stage"] not in {"search","comparative"}:
        raise ValueError("Original graph reference belongs only to declared search/comparative stages")
    if (run["selection"]["metric"]!="rocauc" or run["selection"]["direction"]!="max"
            or run["selection"]["ties"]!="first" or run["selection"].get("save_every",1)!=1):
        raise ValueError("Bind the standardized graph-task SELECT ROC-AUC selector")
    validate_cost_contract(config,run)
    return copy.deepcopy(config)


def build_boundary_model(torch,models,native,options):
    class MemberBoundary(torch.nn.Module):
        def __init__(self,input_dim,output_dim,first):
            super().__init__();self.block=native.BEBlock(input_dim,output_dim,first,"cpu",4);self.member=0
        def forward(self,x):return self.block(x,self.member)
    class BoundaryBody(torch.nn.Module):
        def __init__(self):
            super().__init__();self.body=models.MoleculeBackbone(options);self.member=0
            self.body.input=MemberBoundary(options["hidden"],options["hidden"],True)
            self.body.head=MemberBoundary(options["hidden"],1,False)
            assert self.body.message_factors is None
        def forward(self,batch):
            self.body.input.member=self.body.head.member=self.member
            return self.body(batch)
    return BoundaryBody()


class BoundaryAdapter(MolecularAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.members=4;self.streams=[]
        for member in range(4):
            seed=self.config["dropout_seed"]+1009*member
            stream={"cpu":self.t.Generator(device="cpu").manual_seed(seed).get_state()}
            if self.cuda_devices:stream["cuda"]=self.t.Generator(device=self.device).manual_seed(seed).get_state()
            self.streams.append(stream)

    def _one_member(self,batch,member,counts):
        stream=self.streams[member]
        with self.t.random.fork_rng(devices=self.cuda_devices):
            self.t.set_rng_state(stream["cpu"])
            if self.cuda_devices:self.t.cuda.set_rng_state(stream["cuda"],self.device)
            old=self.model.member;self.model.member=member
            try:
                counts["forward_attempts"]+=1
                logits,hidden=self.model(batch)
                counts["model_forwards"]+=1;counts["route_predictions"]+=len(logits)
            finally:self.model.member=old
            stream["cpu"]=self.t.get_rng_state().clone()
            if self.cuda_devices:stream["cuda"]=self.t.cuda.get_rng_state(self.device).clone()
        return logits,hidden

    def train_step(self,epoch):
        self.model.train();accumulator=MolecularEpochMeans()
        counts={"optimizer_updates":0,"optimizer_update_attempts":0,"model_forwards":0,"forward_attempts":0,
                "backward_calls":0,"backward_attempts":0,"supervised_examples":0,"observed_targets":0,
                "route_predictions":0,"training_label_opportunities":0}
        try:
            generator=self.t.Generator(device="cpu").manual_seed(self.config["shuffle_seed"]+19709+1000*epoch)
            order=self.t.randperm(len(self.train["ids"]),generator=generator)
            for positions in order.split(self.config["training"]["batch_size"]):
                batch,labels=self._batch(self.train,positions);self.optimizer.zero_grad(set_to_none=True)
                objective=0.
                for member in range(4):
                    logits,hidden=self._one_member(batch,member,counts)
                    if not bool(self.t.isfinite(logits).all()) or not bool(self.t.isfinite(hidden).all()):
                        raise FloatingPointError("Nonfinite original boundary TRAIN state")
                    loss=self.F.binary_cross_entropy_with_logits(logits,labels)/4
                    objective+=self._number(loss)
                    counts["backward_attempts"]+=1;loss.backward();counts["backward_calls"]+=1
                active=[p for p in self.model.parameters() if p.grad is not None]
                if not active or any(not bool(self.t.isfinite(p.grad).all()) for p in active):
                    raise FloatingPointError("Nonfinite or absent original boundary gradient")
                counts["optimizer_update_attempts"]+=1;self.optimizer.step();counts["optimizer_updates"]+=1;self._finite_state()
                graphs=len(positions);observed=int(labels.numel())
                counts["supervised_examples"]+=graphs;counts["observed_targets"]+=observed;counts["training_label_opportunities"]+=4*observed
                accumulator.add({"objective":objective,"ce":objective,"regularization":0.,"alignment":0.,"residual":0.},graphs=graphs,observed_targets=observed)
            result=accumulator.finish()
            if result["sample_count"]!=len(self.train["ids"]) or counts["optimizer_updates"]!=self.config["molecular_cost"]["minibatches_per_epoch"]:
                raise ValueError("Original boundary epoch missed its exact include-tail TRAIN roster")
            result.update(operation_counts=counts,loss_timing="old-state native four member BCE/4 backwards then one AdamW0 update; one own view per route")
            self._epoch=epoch;return result
        except BaseException as error:raise AdapterStepError(str(error),operation_counts=counts) from error


def make_adapter(*,torch,project_root,study,run,device):
    config=validate_boundary_config(run["config"],run);validate_role_authorities(study,config)
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from torch_geometric.data import Batch,Data
    from ogb.utils.features import get_atom_feature_dims,get_bond_feature_dims
    if torch.get_default_dtype()!=torch.float32:
        raise ValueError("The declared original graph adaptation uses float32 arithmetic")
    packs={}
    for role,row in config["molecular_inputs"].items():
        pack=validate_pack(torch,torch.load(bound_file(project_root,row),map_location="cpu",weights_only=True),1,"complete_binary")
        for key,limits in [("x",get_atom_feature_dims()),("edge_attr",get_bond_feature_dims())]:
            for column,limit in enumerate(limits):
                if bool(((pack[key][:,column]<0)|(pack[key][:,column]>=limit)).any()):
                    raise ValueError("Original categorical molecular fields must stay in vocabulary")
        packs[role]=pack
    if bool(torch.isin(packs["train"]["ids"],packs["select"]["ids"]).any()) or len(packs["train"]["ids"])!=config["molecular_cost"]["train_graphs"]:
        raise ValueError("Boundary roles overlap or actual TRAIN count changed")
    models,factors,losses=load_source(project_root,config["molecular_source"])
    bound_file(project_root,config["boundary_source"]);native=source_module(project_root,config["boundary_source"]["path"])
    options={key:config["model"][key] for key in ["hidden","layers","dropout"]}
    with models.seeded(config["initialization_seed"]):model=build_boundary_model(torch,models,native,options)
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["optimizer"]["learning_rate"],weight_decay=0,
                               eps=1e-8,betas=(.9,.999))
    provenance={"run_id":run["run_id"],"body":run["body"],"dataset":study["dataset"],"source":study["source"],
        "data":study["data"],"config":config,"budget":copy.deepcopy(run["budget"]),"selection":copy.deepcopy(run["selection"]),
        "implementation":"pinned original BEBlock input/output only around shared plain archived GINE",
        "author_protocol_replication":False,"new_graph_task_adaptation":True,
        "original_native_laws":["first R Rademacher; later R/S1; private B0; shared Xavier W","native BCE/4 separate backwards before one AdamW0 step","mean raw logits then sigmoid"],
        "declared_changes":["OGB Atom/BondEncoder","shared residual virtual-node GINE","boundary input after AtomEncoder","graph global_mean_pool before scalar boundary head","graph minibatching/scaffold four roles/100 or300 epoch budget/SELECT ROC-AUC"],
        "constructor_order":"archived GINE body construction then source boundary constructors; discarded ordinary boundary maps are not model parameters"}
    return BoundaryAdapter(torch,model,optimizer,factors=factors,losses=losses,train=packs["train"],select=packs["select"],
                           config=config,selector=run["selection"],provenance=provenance,data_class=Data,batch_class=Batch,auc_function=roc_auc_score,numpy=np)
