"""Pinned public Session + literal audited O/I/P/G old-state gradient programs."""
import copy
import math
from pathlib import Path
import sys
from adapters.classifier import ClassifierAdapter
from adapters.source_common import bound, module, roles, Streams, OptimizerBank
from engine import AdapterStepError
from oipg_typing import validate_oipg_run

class OIPGAdapter(ClassifierAdapter):
    def raw_logits(self):
        self.model.eval()
        logits,_=self.session.forward({'x':self.x,'edge_index':self.edge,'ids':self.t.arange(len(self.x),device=self.x.device)})
        return logits
    def train_step(self,step):
        members=self.session.model.members;reverse=1 if self.control is None else 3 if self.control.policy=='P' else 2
        counts={'optimizer_updates':0,'model_forwards':0,'route_predictions':0,'backward_calls':0,'supervised_examples':0,'source_local_best_restorations':0}
        try:
            if step==101:
                if self.local_selected is None:raise ValueError('Missing native local-best state')
                # Literal source transition changes only model/optimizer/mode.
                # The live end-local member/global RNG state remains untouched.
                before=self.cpu(self.session.streams)
                self.session.core['selection'].local_transition(self.model,self.session.optimizers,lambda _:self.local_selected,False)
                if not all(self.t.equal(a['cpu'],b['cpu']) and (self.session.cuda_index is None or self.t.equal(a['cuda'],b['cuda'])) for a,b in zip(before,self.session.streams)):
                    raise ValueError('Local transition altered live member RNG')
                self.transitions+=1;counts['source_local_best_restorations']=1
            batch={'x':self.x,'edge_index':self.edge,'ids':self.train_ids}
            value=self.session.train_step(batch,self.train_y) if self.control is None else self.control.train_step(batch,self.train_y)
            counts.update(optimizer_updates=len(self.session.optimizers),model_forwards=2*members,route_predictions=2*members,backward_calls=reverse,
                          source_old_state_gradient_collections=reverse,supervised_examples=2*members*len(self.train_y))
            own=float(value['own_mean']);aux=float(value.get('auxiliary',self.t.tensor(0.)))
            self.last_source_metrics={key:None if item is None else float(item) for key,item in value.items() if key in ('own_mean','pool_mean','mixture_mean','alignment','auxiliary','internal_loss')}
            return {'objective':own+aux,'ce':own,'regularization':aux,'source_objectives':self.last_source_metrics,'operation_counts':counts}
        except BaseException as error:raise AdapterStepError(str(error),operation_counts=counts) from error
    def validate(self,step):
        with self.t.no_grad():
            logits=self.raw_logits();pool=logits.softmax(-1).mean(0);rows=pool[self.select_ids];correct=int(rows.argmax(-1).eq(self.select_y).sum())
            ce=self.t.stack([self.F.cross_entropy(z[self.select_ids],self.select_y) for z in logits]).mean()
            lp=self.t.logsumexp(logits[:,self.select_ids].log_softmax(-1),0)-math.log(len(logits))
            nll=-lp[self.t.arange(len(self.select_y),device=self.select_y.device),self.select_y].mean()
            metric=correct/len(self.select_y)
            if step<=100 and (self.local_best is None or metric>self.local_best):
                self.local_best=metric
                self.local_selected=self.cpu({'model':self.model.state_dict(),'optimizers':[o.state_dict() for o in self.session.optimizers],
                                             'epoch':step,'metric':metric})
            return {'objective':float(ce),'ce':float(ce),'regularization':0.,'accuracy':metric,'pooled_nll':float(nll),
                    'selection_metric':metric,'source_global_mode':bool(self.model.models[0].body._global),
                    'operation_counts':{'optimizer_updates':0,'model_forwards':len(logits),'route_predictions':len(logits),'backward_calls':0,'supervised_examples':0}}
    def cpu(self,value):
        if self.t.is_tensor(value):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:self.cpu(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return type(value)(self.cpu(v) for v in value)
        return copy.deepcopy(value)
    def capture_state(self):
        value=super().capture_state();value['source_OIPG']=self.cpu({'streams':self.session.streams,'session_steps':self.session.steps,
            'global_modes':[body.body._global for body in self.model.models],'global_rng':self.rng.capture(),
            'policy_counters':None if self.control is None else self.control.counters,'local_best':self.local_best,'local_selected':self.local_selected,
            'transitions':self.transitions,'last_source_metrics':self.last_source_metrics})
        return value
    def restore_state(self,value):
        super().restore_state(value);state=value['source_OIPG'];self.session.streams=copy.deepcopy(state['streams']);self.session.steps=state['session_steps']
        for body,flag in zip(self.model.models,state['global_modes']):body.set_global(flag)
        self.rng.restore(state['global_rng'])
        if self.control is not None:self.control.counters=copy.deepcopy(state['policy_counters'])
        for name in ('local_best','local_selected','transitions','last_source_metrics'):setattr(self,name,copy.deepcopy(state[name]))

def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
    import numpy as np
    from torch.nn import functional as F
    c=copy.deepcopy(run['config'])
    if study['stage']!='engineering':validate_oipg_run(run)
    root=Path(project_root);sources=c['oipg_sources']
    if sources['method']['sha256']!='c4f879f803e6ec3b2216b6cdadb400cede9beeee401ad5aa14324c091322d6cd':raise ValueError('Exact OIPG source method required')
    for row in sources['files']:bound(root,row)
    public=module(root,sources['portable'],'oipg_public_portable');method=module(root,sources['method'],'oipg_exact_allocation_method')
    package=root/sources['auditor_directory'];name='oipg_pinned_auditor'
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,package/'__init__.py',submodule_search_locations=[str(package)])
    audit=importlib.util.module_from_spec(spec);sys.modules[name]=audit;spec.loader.exec_module(audit)
    session=public.Session('wikics',c['base_arm'],c['initialization_seed'],device,polynormer=bound(root,sources['native']))
    control=None if c['oipg_policy'] in ('native_M1','fourbody_coupled') else method.PolicyAdapter(session,audit,policy=c['oipg_policy'],lambda_value=c['allocation_lambda'])
    packs=roles(torch,root,c);x=packs['graph']['x'].to(device);edge=packs['graph']['edge_index'].to(device)
    if study['stage']!='engineering' and (tuple(x.shape)!=(11701,300) or edge.shape!=(2,442907)):raise ValueError('Complete source WikiCS feature/support scope required')
    adapter=OIPGAdapter(torch,F,session.model,OptimizerBank(session.optimizers) if len(session.optimizers)>1 else session.optimizers[0],logits_fn=lambda *_:None,
        train_ids=packs['train']['ids'].to(device),train_y=packs['train']['y'].to(device),select_ids=packs['select']['ids'].to(device),select_y=packs['select']['y'].to(device),
        regularizer=lambda _:torch.zeros((),device=device),fusion='mean_probabilities',
        selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':1101,'save_every':1},
        provenance={'run_id':run['run_id'],'config':c,'source':study['source'],'actual_model_seed':c['initialization_seed'],'actual_parent':None,'actual_H0_hash':None})
    adapter.session,adapter.control,adapter.x,adapter.edge=session,control,x,edge
    adapter.rng=Streams(torch,np,device);adapter.local_best=None;adapter.local_selected=None;adapter.transitions=0;adapter.last_source_metrics=None
    adapter.logits_fn=lambda *_:adapter.raw_logits()
    return adapter

def choose_lambda(*,project_root,study,run,output_root,output_dir,registry,device):
    from spec import fingerprint
    rows=[]
    for cell in run['config']['configurations']:
        scores=[]
        for identifier in cell['run_ids']:
            row=registry.run(study['study_id'],identifier)
            if row['state']!='complete' or row['result']['completed_steps']!=1100:raise ValueError('Both complete source lambda horizons required; no survivor choice')
            scores.append(row['result']['selected']['validation']['accuracy'])
        rows.append({**cell,'mean_SELECT_accuracy':sum(scores)/len(scores)})
    rows.sort(key=lambda r:(-r['mean_SELECT_accuracy'],r['parameters']['allocation_lambda']))
    choice={k:copy.deepcopy(rows[0][k]) for k in ('config_id','parameters','run_ids')}
    return {'status':'complete','selected_configuration':choice,'choice_hash':fingerprint(choice),
            'ranking_source':'prospective_complete_source_lambda_GRID_TRAIN_SELECT_only','ranking':rows,
            'failed_search_trajectories':[],'CAL_read':False,'evaluate_read':False,'scientific_backbone_fits':0}
