"""Synthetic CPU unit trajectory, compact bank/lease/negative authority proof.

Three actual tiny CPU Adam maps are explicit engineering operations. No real
dataset, scientific fit, heldout score, GPU execution or output deletion occurs.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
    from adapters.classifier import ClassifierAdapter
    torch.manual_seed(89201)
    model=torch.nn.ModuleList([torch.nn.Linear(3,2)]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    x=torch.arange(24,dtype=torch.float32).reshape(8,3)/24
    return ClassifierAdapter(torch,torch.nn.functional,model,optimizer,
        logits_fn=lambda models,training,step:torch.stack([m(x.to(device)) for m in models]),
        train_ids=torch.arange(4).to(device),train_y=torch.tensor([0,1,0,1]).to(device),
        select_ids=torch.tensor([4,5]).to(device),select_y=torch.tensor([0,1]).to(device),
        regularizer=lambda unused:torch.tensor(0.).to(device),fusion='mean_probabilities',
        selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':5,'save_every':1},
        provenance={'actual_model_seed':89201,'synthetic_engineering_only':True})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:raise ValueError('Exact one-GPU route required before project operations')
    os.environ['CUDA_VISIBLE_DEVICES']='';root=Path(a.project_root).resolve();stage=Path(__file__).resolve().parent;out=root/a.output
    if out.exists() or not out.resolve().is_relative_to(root):raise ValueError('Fresh confined CPU engineering output required')
    out.mkdir();sys.path.insert(0,str(stage/'runtime'))
    import torch,engine
    from registry import Registry
    from retention_lifecycle import declare_policy,close_fit,authority,consumer_leases,inference_export,release_candidates,validate_plan_policy
    from checkpoint_retention import receipt,restore_serving_checkpoint
    from run_study import load_factory,owner_identity,paid_parent_checkpoints
    torch.set_num_threads(2)
    ids={}
    for role,id in [('cal',6),('evaluate',7)]:
        target=out/(role+'_ids.pt');torch.save(torch.tensor([id],dtype=torch.long),target);ids[role+'_ids']=str(target.relative_to(root))
        target=out/(role+'_targets.pt');torch.save({'ids':torch.tensor([id]),'y':torch.tensor([id%2])},target);ids[role+'_targets']=str(target.relative_to(root))
    common={'method':'synthetic_engine','split':0,'seed':89201,'body':0,'parents':[]}
    fit={**common,'run_id':'tiny-fit','kind':'model_fit','adapter':str(Path(__file__).relative_to(root))+':make_adapter',
        'config':{'stage':'comparative','members':1},'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':3},
        'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False}}
    bank={**common,'run_id':'tiny-bank','kind':'ensemble_assembly','parents':['tiny-fit'],
        'adapter':str((stage/'runtime/adapters/node_endpoints.py').relative_to(root))+':export_bank',
        'config':{'stage':'bank_export','fit_parents':['tiny-fit'],'endpoint_inputs':ids,'fusion':'mean_probabilities'},'budget':{},'selection':{}}
    cal={**common,'run_id':'tiny-cal','kind':'calibration','parents':['tiny-bank'],'adapter':bank['adapter'],
        'config':{'stage':'calibration'},'budget':{},'selection':{}}
    study={'schema_version':1,'study_id':'synthetic-retention-engineering','stage':'engineering','hypothesis':'Exact selected representation and conservative full-state lifetime engineering',
        'dataset':'synthetic-8-rows-not-a-research-dataset','data':{'task':'synthetic_unit','preprocessing':'fixed synthetic array','roles':{k:{} for k in ('train','select','cal','evaluate')},'prior_exposure':'synthetic unit','evidence':{'scientific':False}},
        'source':{'git_commit':'synthetic-unit-no-Git-mutation','files':[receipt(Path(__file__),root)]},
        'runs':[fit,bank,cal],'comparisons':[{'references':['synthetic_engine'],'candidates':['synthetic_engine']}],
        'release':{'policy':'whole_roster','primary_endpoint':'synthetic unit only'},'resources':{'wall_seconds':60}}
    declare_policy(root,study);registry=Registry(out/'unit.sqlite');registry.register(study);registry.claim(study['study_id'],fit['run_id'],owner_identity())
    adapter=make_adapter(torch=torch,project_root=root,study=study,run=fit,device='cpu')
    spec={'fit_id':fit['run_id'],'budget_steps':3,'budget_unit':'optimizer_updates','validation_interval':1,'validate_initial':False,
        'prefixes':[],'inherited_fit_ids':[],'adapter_spec':{'run':fit}}
    folder=out/'runs'/study['study_id']/fit['run_id']
    result=engine.run_trajectory(torch,adapter,spec,folder);assert result['trajectory_operation_counts']['optimizer_updates']==3
    result=close_fit(root,study,fit,folder,result);registry.finish(study['study_id'],fit['run_id'],result,success=True)
    row,compact=authority(root,registry,study,fit['run_id']);before=consumer_leases(study,registry,fit['run_id'])
    assert before['closed'] is False and before['outstanding_full_state_consumers']==['tiny-bank']
    assert not release_candidates(root,study,registry,fit['run_id'],owner_terminal=False)['release_candidates']
    factory=load_factory(root,bank['adapter']);bank_out=out/'runs'/study['study_id']/bank['run_id'];bank_out.mkdir(parents=True)
    registry.claim(study['study_id'],bank['run_id'],owner_identity())
    denied_target=False
    with inference_export(root,study,bank,registry,out/'runs',factory):
        try:torch.load(root/ids['cal_targets'],weights_only=True)
        except ValueError:denied_target=True
        exported=factory(project_root=root,study=study,run=bank,output_root=out/'runs',output_dir=bank_out,registry=registry,device='cpu')
    assert denied_target and exported['CAL_targets_read'] is False and exported['evaluation_targets_read'] is False
    registry.finish(study['study_id'],bank['run_id'],exported,success=True)
    after=consumer_leases(study,registry,fit['run_id']);assert after['closed'] is True
    released=release_candidates(root,study,registry,fit['run_id'],owner_terminal=True)
    assert compact['source_full_selected'] in released['release_candidates'] and (folder/'selected.pt').exists()
    clone=make_adapter(torch=torch,project_root=root,study=study,run=fit,device='cpu');b=compact['serving']['serving_checkpoint']
    restore_serving_checkpoint(torch,engine,root/b['path'],clone,project_root=root,expected_receipt=b,expected_fit_id=fit['run_id'])
    with torch.no_grad():assert torch.equal(adapter.logits_fn(adapter.model,False,3),clone.logits_fn(clone.model,False,3))
    negatives=['target_read_denied']
    try:engine.restore_checkpoint(torch,root/b['path'],clone,expected_fit_id=fit['run_id'])
    except ValueError:negatives.append('training_restore_rejects_compact')
    else:raise ValueError('Compact became a training authority')
    bad=out/'bad_recovery';bad.mkdir();(bad/'latest_trajectory.pt').write_bytes((root/b['path']).read_bytes())
    (bad/'spec.json').write_text(json.dumps(spec)+'\n')
    try:engine.resume_trajectory(torch,clone,spec,bad,admission={'previous_owner_terminal':True,'previous_owner':{'synthetic':True}})
    except ValueError:negatives.append('recovery_rejects_compact')
    else:raise ValueError('Compact became a recovery authority')
    child={**fit,'run_id':'unexecuted-child','parents':[fit['run_id']],'config':{**fit['config'],'lifecycle':{'schema':'paid-parent-continuation-v1','parent_fit_ids':[fit['run_id']],'new_optimizer':True,'parameter_initialization':'selected_parent_no_optimizer_reuse'}}}
    class WrongAuthority:
        def run(self,*args):
            v=copy.deepcopy(registry.run(*args));v['result']['selected_checkpoint']={**b,'fit_id':fit['run_id'],'step':1,'path':'selected_serving.pt'};return v
    try:paid_parent_checkpoints(root,study,child,WrongAuthority(),out/'runs')
    except ValueError:negatives.append('paid_parent_rejects_compact')
    else:raise ValueError('Compact became a paid source authority')
    assert not torch.cuda.is_initialized() and len(negatives)==4
    output={'schema':'prospective-retention-CPU-lifecycle-unit-v1','status':'passed','synthetic_CPU_optimizer_maps':3,'synthetic_validation_calls':result['validation_calls'],
        'scientific_fits':0,'scientific_predictive_scores':0,'CUDA':False,'existing_outputs_deleted':False,'actual_deletion_executed':False,
        'complete_engine_to_compact_to_audited_bank':True,'exact_selected_logits_restored':True,'CAL_and_EVALUATE_targets_denied':True,
        'paid_and_recovery_negative_cases':negatives,'leases_before':before,'leases_after':after,
        'full_selected_still_present':True,'pending_CAL_still_registered':registry.run(study['study_id'],'tiny-cal')['state']=='registered',
        'full_geometry_and_all_source_families_qualified':False,'GO':False}
    registry.close();(out/'RESULT.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n');print(json.dumps(output))


if __name__=='__main__':main()
