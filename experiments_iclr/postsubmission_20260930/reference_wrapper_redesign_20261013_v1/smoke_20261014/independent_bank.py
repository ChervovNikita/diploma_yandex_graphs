"""Smoke four independently owned native models and mean-logit serving."""
import copy
import importlib.util
import random
import sys
from pathlib import Path


def _rng(torch,np):
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore(torch,np,state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def _metadata(session):
    value=getattr(session,'source_metadata',None)
    if value is None:value=getattr(session,'metadata',None)
    return value() if callable(value) else value


def _validation_evidence(result):
    return {k:result[k] for k in ('role','VALID_scope','VALID_graphs','VALID_nodes',
            'complete_VALID_positive_records','complete_VALID_negative_records',
            'native_skipped_batches','TEST_accessed','TEST_opened','TEST_scored','test_scoring','work')
            if k in result}


class IndependentBank:
    def __init__(self,recipe,data_root,device):
        import torch
        import numpy as np
        self.torch,self.np=torch,np
        callback=Path(recipe['_native_callback'])
        spec=importlib.util.spec_from_file_location('independent_native_factory',callback)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        self.members=[];self.member_rng=[]
        parent=_rng(torch,np)
        for index,offset in enumerate(recipe['native_seed_offsets']):
            cfg=copy.deepcopy(recipe);cfg['_seed']=int(recipe.get('_seed',0))+offset
            cfg['seed']=cfg['_seed']
            random.seed(cfg['_seed']);np.random.seed(cfg['_seed'] % 2**32)
            torch.manual_seed(cfg['_seed'])
            if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['_seed'])
            if self.members and recipe['task']=='node_classification':cfg['_shared_node_session']=self.members[0]
            member=module.create_session(cfg,'native_single',Path(data_root),device)
            self.members.append(member);self.member_rng.append(_rng(torch,np))
        _restore(torch,np,parent)
        self.model=torch.nn.ModuleList([s.model for s in self.members])
        self.optimizers=[o for s in self.members for o in s.optimizers]
        assert len(self.optimizers)==4
        self.source_parameter_references={str(index)+'.'+name:p for index,s in enumerate(self.members)
                                          for name,p in s.source_parameter_references.items()}
        self.replaced_source_parameter_names=[]
        self.allow_missing_grad_names=[str(i)+'.'+name for i,s in enumerate(self.members)
                                       for name in getattr(s,'allow_missing_grad_names',[])]
        self.allow_nontrainable_source_names=[str(i)+'.'+name for i,s in enumerate(self.members)
                                              for name in getattr(s,'allow_nontrainable_source_names',[])]
        self.is_independent4=True;self.bank_rule='mean_logits';self.updates=0
        self.source_metadata={'bank':'four genuine independent native models','native_backbones_unchanged':True,
                              'member_seeds':[int(recipe.get('_seed',0))+offset for offset in recipe['native_seed_offsets']],
                              'private_member_rng':True,'shared_training_weights':False,'shared_fixed_graph_inputs':recipe['task']=='node_classification',
                              'members':[_metadata(s) for s in self.members]}

    def _own(self,index,method):
        parent=_rng(self.torch,self.np)
        try:
            _restore(self.torch,self.np,self.member_rng[index])
            result=getattr(self.members[index],method)()
            self.member_rng[index]=_rng(self.torch,self.np)
            return result
        finally:_restore(self.torch,self.np,parent)

    def train_update(self):
        results=[self._own(i,'train_update') for i in range(4)]
        self.updates+=1
        return {'loss':sum(float(x['loss']) for x in results)/4,'optimizer_updates':4,
                'logical_update':self.updates,'member_updates':results,'independent_optimizers':True}

    def validate(self):
        torch=self.torch
        results=[self._own(i,'validate') for i in range(4)]
        members=torch.stack([x['outputs'] for x in results])
        return {'outputs':members.mean(0),'member_outputs':members,
                'loss':sum(float(x['loss']) for x in results)/4,'metrics':{},'role':'VALID',
                'member_validation_evidence':[_validation_evidence(x) for x in results],
                'loss_semantics':'mean independent member validation losses; aggregation is checked on raw logits',
                'test_scoring':False}

    def state(self):
        return {'members':[s.state() for s in self.members],'member_rng':copy.deepcopy(self.member_rng),'updates':self.updates}

    def load_state(self,state):
        for member,value in zip(self.members,state['members']):member.load_state(value)
        self.member_rng=copy.deepcopy(state['member_rng']);self.updates=state['updates']


def create_session(recipe,procedure,data_root,device):
    assert procedure=='independent4'
    return IndependentBank(recipe,data_root,device)
