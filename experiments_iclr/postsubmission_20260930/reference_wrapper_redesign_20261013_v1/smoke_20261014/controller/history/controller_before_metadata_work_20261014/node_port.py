"""Real full-graph PolyFormer sessions with unchanged author model operators.

Only needed literal source definitions are executed, avoiding unused optional
imports. No source model/function AST body is rewritten. No TEST evaluator.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import math
import os
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace


def _module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def _definitions(path, names, namespace):
    parsed=ast.parse(path.read_text(),filename=str(path))
    selected=[node for node in parsed.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    assert {node.name for node in selected}==set(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return {name:namespace[name] for name in names}


def _seed(torch,np,seed):
    random.seed(seed);np.random.seed(seed % 2**32);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def _source_environment(redesign):
    import torch
    import numpy as np
    import scipy.sparse as sp
    import torch_geometric.utils as U
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    native=redesign/'native_sources/polyformer/node_classification'
    scope={'__name__':'polyformer_author_smoke','torch':torch,'nn':torch.nn,
           'F':torch.nn.functional,'np':np,'math':math,'sp':sp,'os':os,'pickle':pickle,
           'Linear':torch.nn.Linear,'gcn_norm':gcn_norm,
           'to_scipy_sparse_matrix':U.to_scipy_sparse_matrix,'get_laplacian':U.get_laplacian,
           'add_self_loops':U.add_self_loops,'to_undirected':U.to_undirected,'is_undirected':U.is_undirected}
    wanted=['cheby','init_temp','index_to_mask','take_rest','random_splits',
            'sparse_mx_to_torch_sparse_tensor','mono_base','cheb_base','get_base']
    _definitions(native/'utils.py',wanted,scope)
    _definitions(native/'layers/PolyFormerBlock.py',['PolyAttn','FFNNetwork','FFN','PolyFormerBlock'],scope)
    _definitions(native/'mymodels.py',['PolyFormer'],scope)
    return torch,np,U,scope


def _prepare_data(recipe,data_root,redesign,scope,torch):
    from torch_geometric.datasets import Planetoid,Coauthor
    from torch_geometric.transforms import NormalizeFeatures
    dataset=recipe['dataset']
    names={'citeseer':'CiteSeer','pubmed':'PubMed','coauthor-cs':'CS','coauthor-physics':'Physics'}
    cache=Path(data_root)/dataset
    repo=redesign.parents[2]
    assert cache.resolve().is_relative_to(repo), 'Dataset cache must remain inside project'
    name=names[dataset]
    processed=cache/name/'processed/data.pt'
    assert processed.is_file(), 'Missing real native dataset store: '+str(processed)
    factory=Planetoid if dataset in ('citeseer','pubmed') else Coauthor
    source=factory(root=str(cache),name=name,transform=NormalizeFeatures())
    data=source[0]
    if not scope['is_undirected'](data.edge_index):data.edge_index=scope['to_undirected'](data.edge_index)
    assert scope['is_undirected'](data.edge_index)
    config=recipe['native_fixed_config']
    basis=recipe['id'].rsplit('-',1)[-1]
    store=redesign/'smoke_20261014/working_data/node_bases'
    store.mkdir(parents=True,exist_ok=True)
    path=store/(recipe['id']+'.pt')
    if path.exists():
        packet=torch.load(path,map_location='cpu',weights_only=False)
        assert packet['dataset']==dataset and packet['basis']==basis and packet['K']==int(config['K'])
        assert packet['shape']==list(data.x.shape) and packet['edges']==data.edge_index.shape[1]
        data.list_mat=packet['list_mat']
    else:
        data.list_mat=scope['get_base'](basis,int(config['K']),data.x,data.edge_index,data.edge_attr)
        torch.save({'dataset':dataset,'basis':basis,'K':int(config['K']),'shape':list(data.x.shape),
                    'edges':data.edge_index.shape[1],'list_mat':data.list_mat,
                    'source_utils_sha256':hashlib.sha256((redesign/'native_sources/polyformer/node_classification/utils.py').read_bytes()).hexdigest()},path)
    split_seed=recipe['official_split_bindings']['0']['source_split_seed']
    data=scope['random_splits'](data,source.num_classes,
            int(round(.6*len(data.y)/source.num_classes)),int(round(.2*len(data.y))),split_seed)
    assert not (data.train_mask & data.val_mask).any()
    return source,data,basis


def _chunked_supcon(torch,hidden,labels):
    """Same full-TRAIN objective, checkpointed blocks, no anchor subsampling."""
    from torch.utils.checkpoint import checkpoint
    normalized=torch.nn.functional.normalize(hidden,p=2,dim=-1,eps=1e-12)
    count=labels.numel()
    counts=torch.bincount(labels)[labels]-1
    assert (counts>0).all()
    sums=[]
    for member in range(4):
        values=normalized[member]
        for begin in range(0,count,256):
            end=min(count,begin+256)
            def block(z,lo=begin,hi=end):
                score=(z[lo:hi]@z.T)/.2
                rows=torch.arange(lo,hi,device=z.device)
                cols=torch.arange(count,device=z.device)
                allowed=rows[:,None]!=cols[None,:]
                positive=labels[lo:hi,None].eq(labels[None,:]) & allowed
                denominator=torch.logsumexp(score.masked_fill(~allowed,float('-inf')),dim=-1)
                positive_mean=score.masked_fill(~positive,0).sum(-1)/counts[lo:hi]
                return (denominator-positive_mean).sum()
            sums.append(checkpoint(block,values,use_reentrant=False,preserve_rng_state=True))
    return torch.stack(sums).sum()/(4*count)


class NodeSession:
    def __init__(self,recipe,procedure,data_root,device):
        self.recipe=copy.deepcopy({k:v for k,v in recipe.items() if k!='_shared_node_session'});self.procedure=procedure
        redesign=Path(recipe['_redesign_root'])
        torch,np,U,scope=_source_environment(redesign)
        self.torch=torch;self.np=np;self.device=device;self.updates=0
        shared=recipe.get('_shared_node_session')
        if shared is None:
            source,data,basis=_prepare_data(recipe,data_root,redesign,scope,torch)
            self.data=data.to(device)
        else:
            assert shared.recipe['id']==recipe['id'] and shared.device==device
            source,basis=shared.source,shared.basis
            self.data=shared.data
        self.source,self.basis=source,basis
        config=copy.deepcopy(recipe['native_fixed_config'])
        config.update(dataset={'coauthor-cs':'cs','coauthor-physics':'physics'}.get(recipe['dataset'],recipe['dataset']),
                      base=basis,num_features=source.num_features,num_classes=source.num_classes,
                      n_channel=config['hidden'],device=torch.device(device))
        args=SimpleNamespace(**config)
        _seed(torch,np,int(recipe.get('_seed',0)))
        native=scope['PolyFormer'](source,args).to(device)
        before=dict(native.named_parameters())
        groups=[]
        for name,p in native.named_parameters():
            attention='attnmodule' in name
            groups.append({'params':[p],'lr':args.attn_lr if attention else args.lr,
                           'weight_decay':args.attn_wd if attention else args.weight_decay})
        self.optimizer=torch.optim.Adam(groups)
        self.optimizers=[self.optimizer]
        self.rules=_module(redesign/'runtime/training_rules.py','node_smoke_training_rules')
        self.core=None
        if procedure!='native_single':
            corelib=_module(redesign/'runtime/boundary_core.py','node_smoke_boundary_core')
            self.core=corelib.attach_boundary_core(torch,native,[corelib.BoundaryBinding('lin1','linear','input'),
                                corelib.BoundaryBinding('lin3','linear','output')],factor_seed=int(recipe.get('_seed',0))+700001)
            corelib.extend_native_optimizer_groups(self.core,self.optimizer)
            self.model=self.core
            self.source_parameter_references={'native.'+name:p for name,p in before.items()}
        else:
            self.model=native
            self.source_parameter_references=before
        self.replaced_source_parameter_names=[]
        self.allow_missing_grad_names=[]
        self.allow_nontrainable_source_names=[]
        self.train_ids=torch.nonzero(self.data.train_mask,as_tuple=False).flatten()
        self.targets=self.data.y[self.train_ids]
        self.responsibility=None
        if procedure=='boundary_private_fixed_responsibility':
            self.responsibility,_=self.rules.draw_fixed_responsibilities(torch,self.targets,seed=int(recipe.get('_seed',0)))
        self.source_metadata={'source_class':'PolyFormer','dataset':recipe['dataset'],'full_graph_nodes':int(self.data.num_nodes),
            'train_nodes':len(self.train_ids),'valid_nodes':int(self.data.val_mask.sum()),'basis':basis,
            'source_split_seed':split_seed if (split_seed:=recipe['official_split_bindings']['0']['source_split_seed']) else 0,
            'test_scoring':False,'boundary_paths':[] if self.core is None else ['lin1','lin3'],
            'native_operator_changes':0,'native_parameter_replacements':0,
            'source_functions':'unchanged original AST definitions; unused optional imports omitted',
            'activation_checkpointing':self.core is not None,
            'supcon_complete_train':procedure=='boundary_own_within_route_supcon'}

    def _routes(self,training):
        if not training:
            return [self.core.run_member(m,self.data) for m in range(4)]
        from torch.utils.checkpoint import checkpoint
        result=[]
        for m in range(4):
            def call(features,member=m):
                out=self.core.run_member(member,self.data)
                return out.raw_logits,out.prehead
            raw,hidden=checkpoint(call,self.data.list_mat[0],use_reentrant=False,preserve_rng_state=True)
            result.append(SimpleNamespace(prediction=raw,raw_logits=raw,prehead=hidden))
        return result

    def train_update(self):
        torch=self.torch;F=torch.nn.functional
        self.model.train();self.optimizer.zero_grad()
        if self.core is None:
            logits=self.model(self.data)
            loss=F.cross_entropy(logits[self.train_ids],self.targets)
            loss.backward();self.optimizer.step()
        else:
            out=self._routes(True)
            native_loss=lambda z:F.cross_entropy(z[self.train_ids],self.targets)
            terms=self.rules.native_loss_terms(torch,[r.prediction for r in out],[r.raw_logits for r in out],
                native_loss=native_loss,pooled_native_loss=native_loss,native_regularizer=out[0].raw_logits.new_zeros(()))
            loss=terms['F']
            if self.procedure=='gnnm_boundary_native':
                loss.backward();self.optimizer.step()
            elif self.procedure=='boundary_global_half_pool':
                self.rules.step_global_half_pool(torch,self.core,self.optimizer,F=terms['F'],L=terms['L']);loss=terms['J']
            elif self.procedure=='boundary_private_half_pool':
                self.rules.step_private_half_pool(torch,self.core,self.optimizer,F=terms['F'],L=terms['L']);loss=terms['J']
            elif self.procedure=='boundary_private_fixed_responsibility':
                records=torch.stack([F.cross_entropy(r.raw_logits[self.train_ids],self.targets,reduction='none') for r in out])
                weighted=self.rules.weighted_native_own_loss(torch,records,self.responsibility,
                            native_reduce=lambda v:v.mean(),native_regularizer=records.new_zeros(()))
                self.rules.step_private_responsibility(torch,self.core,self.optimizer,F=terms['F'],F_weighted=weighted)
            elif self.procedure=='boundary_own_within_route_supcon':
                hidden=torch.stack([r.prehead[self.train_ids] for r in out])
                auxiliary=_chunked_supcon(torch,hidden,self.targets)
                self.rules.step_own_supcon(torch,self.core,self.optimizer,F=terms['F'],A=auxiliary,task_kind='node_categorical')
                loss=terms['F']+.05*auxiliary
            else:raise ValueError('Unsupported frozen procedure: '+self.procedure)
        self.updates+=1
        return {'loss':float(loss.detach()),'optimizer_updates':1,'logical_update':self.updates,
                'train_records':len(self.train_ids),'graph_nodes':int(self.data.num_nodes),'whole_native_train':True}

    def validate(self):
        torch=self.torch;self.model.eval()
        with torch.no_grad():
            if self.core is None:logits=self.model(self.data)
            else:logits=torch.stack([r.raw_logits for r in self._routes(False)]).mean(0)
            outputs=logits[self.data.val_mask]
            targets=self.data.y[self.data.val_mask]
            loss=torch.nn.functional.cross_entropy(outputs,targets)
            return {'outputs':outputs,'loss':float(loss),'metrics':{},'role':'VALID','test_scoring':False}

    def state(self):
        return {'model':copy.deepcopy(self.model.state_dict()),'optimizers':[copy.deepcopy(self.optimizer.state_dict())],
                'updates':self.updates,'responsibility':None if self.responsibility is None else self.responsibility.detach().clone()}

    def load_state(self,state):
        self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizers'][0])
        self.updates=state['updates']
        if state['responsibility'] is not None:self.responsibility=state['responsibility'].to(self.device)


def create_session(recipe,procedure,data_root,device):
    return NodeSession(recipe,procedure,Path(data_root),device)
