"""Literal source factories on synthetic tensors only; no actual role files."""
import copy,gc,hashlib,json,os
from pathlib import Path
import sys,tempfile,time
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent;sys.path.insert(0,str(OUT/'runtime'))
import torch
import numpy as np
from adapters.physics_joint import make_adapter as physics_factory
from adapters.oipg import make_adapter as oipg_factory
from adapters.source_endpoints import fit_temperature
from adapters.classifier import ClassifierAdapter
from tabm_stopping import budget,selection,validate_tabm_run

def receipt(path):
    raw=path.read_bytes();return {'path':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def bindings():
    names={'native':'native_models.py','author':'author_model.py','joint':'joint_forward.py','samplers':'samplers.py','independent':'independent_controller.py'}
    physics={key:receipt(OUT/'sources/physics'/value) for key,value in names.items()}
    directory=OUT/'sources/portable_internal_be_public_interface_20261007_v2';files=[receipt(p) for p in sorted((OUT/'sources').rglob('*')) if p.is_file()]
    oipg={'portable':receipt(directory/'portable.py'),'method':receipt(OUT/'sources/public_internal_be_allocation_controls_20261007_v1/method.py'),
          'native':receipt(OUT/'sources/native_polynormer/model.py'),'auditor_directory':str((OUT/'sources/public_internal_be_private_steering_adapter_20261007_v1').relative_to(ROOT)),'files':files}
    return physics,oipg
def fixture(directory,features,classes):
    n=12;g=torch.Generator().manual_seed(2321);x=torch.randint(0,2,(n,features),generator=g).float();ids=torch.arange(n)
    edge=torch.stack((torch.cat((ids,ids,ids)),torch.cat((ids,(ids+1)%n,(ids-1)%n))))
    rows={}
    for role,value in {'graph':{'x':x,'edge_index':edge},'train':{'ids':ids[:6],'y':ids[:6]%classes},'select':{'ids':ids[6:],'y':ids[6:]%classes}}.items():
        p=directory/(str(features)+'_'+role+'.pt');torch.save(value,p);rows[role]=receipt(p)
    return rows
def snapshot(adapter):return copy.deepcopy((adapter.model.state_dict(),adapter.optimizer.state_dict(),adapter.capture_state()))
def replay(adapter,step):
    state=snapshot(adapter);a=adapter.train_step(step);first=snapshot(adapter)
    adapter.model.load_state_dict(state[0]);adapter.optimizer.load_state_dict(state[1]);adapter.restore_state(state[2]);b=adapter.train_step(step);second=snapshot(adapter)
    if not equal(first,second) or a!=b:raise AssertionError('Exact next source update/state replay failed')
    return a
def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU synthetic checks require hidden CUDA')
    torch.set_num_threads(2);torch.set_default_dtype(torch.float32);start=time.monotonic();pbind,obind=bindings();proof=[];updates=0
    with tempfile.TemporaryDirectory(dir=OUT/'tests') as temporary:
        folder=Path(temporary);small=fixture(folder,7,5);wide=fixture(folder,8415,5);wiki=fixture(folder,300,3)
        arms=['O4_stem_IID','O4_stem_SIGN','O4_graph_IID','O4_graph_SIGN','O4_graph_BALANCED','factor_M1_stem','factor_M1_graph','MS4_IID','MS4_SIGN','coupled_I4','original_gnnm','source_meanSAGE']
        for arm in arms:
            source=arm=='source_meanSAGE';owners=4 if arm=='coupled_I4' else 1;members=4 if arm.startswith('O4') or arm in ('coupled_I4','original_gnnm') else 1
            cfg={'physics_arm':arm,'physics_placement':'stem_pre_gelu' if 'stem' in arm or arm=='original_gnnm' else 'first_normalized_sage_input',
                 'physics_sources':pbind,'initialization_seed':6211,'classes':5,'node_inputs':wide if source else small,
                 'architecture':'source_TABM_joint_dropout','backbone':'SAGE','depth':2,'width':32 if source else 128,'dropout':.4 if source else .2,
                 'learning_rate':.001,'weight_decay':0,'members':members,'physical_slow_bodies':owners,'source_optimizer_count':owners,
                 'source_TRAIN_forwards_per_round':4 if arm.startswith('O4') or arm.startswith('MS4') or arm in ('coupled_I4','original_gnnm') else 1,
                 'historical_state_inherited':False,'fusion':'mean_logits'}
            run={'run_id':'synthetic-'+arm,'kind':'model_fit','seed':6211,'body':0,'config':cfg,'budget':budget(arm),'selection':selection(arm)}
            validate_tabm_run(run);study={'stage':'engineering','source':{'files':list(pbind.values())}}
            adapter=physics_factory(torch=torch,project_root=ROOT,study=study,run=run,device='cpu')
            first=adapter.train_step(1);updates+=first['operation_counts']['optimizer_updates'];checked=replay(adapter,2);updates+=2*checked['operation_counts']['optimizer_updates']
            adapter.validate(2)
            if arm=='coupled_I4':
                adapter.active=[True,False,True,False];old=[copy.deepcopy(body.state_dict()) for body in adapter.model.bodies];value=adapter.train_step(3);updates+=2
                if value['operation_counts']['optimizer_updates']!=2 or value['operation_counts']['source_stopped_companion_prefix_forwards']!=2:raise AssertionError('Inactive companion accounting')
                if not equal(old[1],adapter.model.bodies[1].state_dict()) or not equal(old[3],adapter.model.bodies[3].state_dict()):raise AssertionError('Stopped companions updated')
            proof.append({'factory':arm,'optimizer_owners':owners,'served_routes':members,'first_counts':first['operation_counts'],'next_update_exact_replay':True})
            del adapter,first,checked;gc.collect()
        for policy in ('native_M1','O','I','P','G','fourbody_coupled'):
            owners=4 if policy=='fourbody_coupled' else 1;members=1 if policy=='native_M1' else 4
            cfg={'oipg_policy':policy,'base_arm':'single' if policy=='native_M1' else 'independent4_contrastive' if owners==4 else 'be_unit',
                 'allocation_lambda':0. if policy in ('O','native_M1','fourbody_coupled') else .5,'initialization_seed':6413,
                 'oipg_sources':obind,'node_inputs':wiki}
            run={'run_id':'synthetic-'+policy,'kind':'model_fit','seed':6413,'body':0,'config':cfg};study={'stage':'engineering','source':{'files':obind['files']}}
            adapter=oipg_factory(torch=torch,project_root=ROOT,study=study,run=run,device='cpu');first=adapter.train_step(1);updates+=owners
            adapter.validate(1);checked=replay(adapter,2);updates+=2*owners
            if first['operation_counts']['optimizer_updates']!=owners or first['operation_counts']['backward_calls']!=(3 if policy=='P' else 1 if policy in ('native_M1','fourbody_coupled') else 2):raise AssertionError('OIPG exact source reverse/optimizer ownership')
            adapter.train_step(101);updates+=owners
            if adapter.transitions!=1 or not all(body.body._global for body in adapter.model.models):raise AssertionError('Literal source local-best transition')
            proof.append({'factory':'OIPG-'+policy,'optimizer_owners':owners,'source_reverse_calls':first['operation_counts']['backward_calls'],'next_update_exact_replay':True,'localbest_transition_and_live_RNG_preserved':True})
            del adapter,first,checked;gc.collect()
        # Off-cadence improvements reset stale without advancing saved-best.
        selector=ClassifierAdapter.__new__(ClassifierAdapter);selector.selector={'direction':'max','ties':'first','save_every':10,'patience_steps':300};selector._best=None;selector._best_step=None;selector._stale=0;selector._last_validation_step=0;selector._scalar=float
        decisions=[]
        for step in range(1,311):decisions.append(selector.observe_validation(step,{'selection_metric':1. if step<=10 else 0.}))
        if decisions[8]['save_selected'] or not decisions[9]['save_selected'] or not decisions[-1]['stop_requested'] or selector._best_step!=10:raise AssertionError('Native saved-best cadence/p300 law')
        if fit_temperature(torch,torch.zeros((4,1)),torch.zeros(4,dtype=torch.long))!=1.:raise AssertionError('Fixed500 zero-gradient CAL endpoint')
    result={'schema':'Physics-OIPG-literal-source-synthetic-CPU-proof-v1','factories':proof,'factories_checked':len(proof),
            'actual_synthetic_optimizer_maps':updates,'native_selector_fixture_stop':310,'native_selector_fixture_selected':10,
            'CAL_synthetic_scalar_maps':500,'wall_seconds':time.monotonic()-start,'actual_input_model_constructions':0,
            'actual_input_backbone_updates':0,'actual_input_scores':0,'scientific_fits':0,'GO':False,'CUDA_initialized':torch.cuda.is_initialized()}
    p=OUT/'SYNTHETIC_CPU_PROOF.json'
    if p.exists():raise FileExistsError('Preserve prior proof')
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='factories'}))
if __name__=='__main__':main()
