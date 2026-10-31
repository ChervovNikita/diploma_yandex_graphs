"""Tiny zero-update CPU selected-state transformation proof, no deletion."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if (socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']):raise ValueError('Exact authorized route required')
    os.environ['CUDA_VISIBLE_DEVICES']='';root=Path(a.project_root).resolve();out=root/a.output
    if out.exists() or not out.resolve().is_relative_to(root):raise ValueError('Fresh confined tiny QA output required')
    out.mkdir();import torch
    sys.path.insert(0,str(Path(__file__).resolve().parent));from checkpoint_retention import export_serving_checkpoint,restore_serving_checkpoint,receipt
    sys.path.insert(0,str(root/'master_controller_source_20261012_v14/runtime'));import engine
    torch.set_num_threads(2);torch.manual_seed(88201)
    class Adapter:
        def __init__(self):
            linear=torch.nn.Linear(3,2)
            self.model=torch.nn.ModuleList([linear,linear,torch.nn.BatchNorm1d(2)])
            self.model[0].bias.requires_grad_(False)
            self.optimizer=torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad])
            self.model.eval();self.model[2].train()
            self.state={'selector':{'best':.25,'step':7},'member_rng':[torch.get_rng_state().clone()],
                'custom_tensor':torch.arange(6.).reshape(2,3),'private_counter':13,'source_nonpersistent':{'factor':torch.tensor([.5])}}
            # Materialize deterministic synthetic Adam fields without invoking
            # step, a functional optimizer or a scientific loss.
            for p in self.model.parameters():
                if p.requires_grad:self.optimizer.state[p]={'step':torch.tensor(7.),'exp_avg':torch.full_like(p,.1),'exp_avg_sq':torch.full_like(p,.2)}
        def capture_state(self):return copy.deepcopy(self.state)
        def restore_state(self,state):self.state=copy.deepcopy(state)
    adapter=Adapter();source=out/'synthetic_complete_selected.pt'
    payload=engine.capture_checkpoint(torch,adapter,fit_id='synthetic-serving',step=7,validation={'objective':.25},decision={'save_selected':True,'stop_requested':False})
    payload['recovery']={'selected_snapshots':{'selected.pt':b'synthetic-duplicate-bytes'},'trace':'synthetic'}
    torch.save(payload,source);source_receipt=receipt(source,root)
    manifest=export_serving_checkpoint(torch,engine,source,out/'synthetic_selected_serving.pt',project_root=root,expected_source_receipt=source_receipt,expected_fit_id='synthetic-serving')
    expected=engine._clone_tree(torch,{k:payload[k] for k in ('model_state','module_modes','requires_grad','adapter_state')})
    for p in adapter.model.parameters():p.data.add_(3)
    adapter.model.train();adapter.state['private_counter']=999
    before=engine._clone_tree(torch,adapter.optimizer.state_dict())
    selected=restore_serving_checkpoint(torch,engine,out/'synthetic_selected_serving.pt',adapter,project_root=root,expected_receipt=manifest['serving_checkpoint'],expected_fit_id='synthetic-serving')
    actual={'model_state':engine._clone_tree(torch,adapter.model.state_dict()),'module_modes':{n:bool(m.training) for n,m in adapter.model.named_modules()},'requires_grad':{n:bool(p.requires_grad) for n,p in adapter.model.named_parameters()},'adapter_state':engine._clone_tree(torch,adapter.capture_state())}
    assert engine._equal_tree(torch,expected,actual) and engine._equal_tree(torch,before,adapter.optimizer.state_dict())
    assert receipt(source,root)==source_receipt and source.is_file() and selected['resumable'] is False
    saved=torch.load(out/'synthetic_selected_serving.pt',map_location='cpu',weights_only=False)
    assert saved['model_state']['0.weight'].untyped_storage().data_ptr()==saved['model_state']['1.weight'].untyped_storage().data_ptr()
    negatives=[]
    for name,kwargs in [('wrong_fit',{'expected_fit_id':'another-fit'}),('wrong_hash',{'expected_receipt':{**manifest['serving_checkpoint'],'sha256':'0'*64}})]:
        options={'project_root':root,'expected_receipt':manifest['serving_checkpoint'],'expected_fit_id':'synthetic-serving'};options.update(kwargs)
        try:restore_serving_checkpoint(torch,engine,out/'synthetic_selected_serving.pt',adapter,**options)
        except ValueError:negatives.append(name)
        else:raise ValueError('Negative authority path did not reject '+name)
    try:engine.restore_checkpoint(torch,out/'synthetic_selected_serving.pt',adapter,expected_fit_id='synthetic-serving')
    except ValueError:negatives.append('training_restore_refuses_serving_schema')
    else:raise ValueError('Serving payload became a resumable training authority')
    assert not torch.cuda.is_initialized()
    result={'schema':'prospective-serving-state-tiny-CPU-proof-v1','status':'passed','exact_state_restore':True,'alias_preserved':True,'mixed_modes_and_flags_preserved':True,'complete_adapter_state_preserved':True,'optimizer_unchanged':True,'source_full_checkpoint_unchanged':True,'training_restore_rejected_serving':True,'negative_cases':negatives,'manifest':manifest,'source_files':[receipt(Path(__file__),root),receipt(Path(__file__).parent/'checkpoint_retention.py',root)],'synthetic_optimizer_moments_materialized_without_steps':True,'full_geometry_or_custom_source_families_qualified':False,'actual_optimizer_updates':0,'optimizer_kernels':0,'scientific_fits':0,'predictive_scores':0,'CUDA':False,'delete_executed':False,'GO':False}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','exact_state_restore':True,'negative_cases':negatives,'delete_executed':False,'CUDA':False,'GO':False}))


if __name__=='__main__':main()
