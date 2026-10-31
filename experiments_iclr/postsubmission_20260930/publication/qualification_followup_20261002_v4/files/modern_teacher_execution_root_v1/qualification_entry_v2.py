"""Confined environment export or report-ineligible full-graph qualification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode=True
PHASE=Path(__file__).resolve().parents[1]
REPO=PHASE.parents[1]
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def confined(value):
    path=Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to(PHASE) or path.resolve()==PHASE:
        raise ValueError('Require a confined phase path')
    cursor=PHASE
    for part in path.relative_to(PHASE).parts:
        cursor/=part
        if cursor.is_symlink():
            raise ValueError('Symlink paths forbidden')
    return path

def bound(record):
    path=confined(record['path'])
    if sha(path)!=record['sha256']:
        raise ValueError('Bound file changed')
    return path

def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True)
    parser.add_argument('--supervisor',required=True)
    args=parser.parse_args()
    if Path.cwd().resolve()!=REPO or os.environ.get('GNNM_PHASE_ROOT')!=str(PHASE) or os.environ.get('GNNM_SSH_DESTINATION')!=LOGIN:
        raise ValueError('Wrong repository/environment/route')
    import subprocess
    gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()
    if gpu!=[UUID]:
        raise ValueError('Wrong allocation')
    request_path=confined(args.request);request_sha=sha(request_path)
    request=json.loads(request_path.read_text())
    if request['root_admitted'] is not True or request['full_fit_admitted'] or request['heldout_scoring_admitted']:
        raise ValueError('Only environment/qualification admission accepted')
    for item in request['protected_files']:
        bound(item)
    packet=confined(request['source_packet'])
    manifest=json.loads(bound(request['source_manifest']).read_text())
    for item in manifest['payload']:
        path=packet/item['path']
        if not path.resolve().is_relative_to(packet) or sha(path)!=item['sha256']:
            raise ValueError('Immutable source packet mismatch')
    sys.path.insert(0,str(packet/'prototype'))
    import correction_screen_driver as core
    out=confined(request['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    write(out/'START.json',dict(request_sha256=request_sha,entry_source_sha256=sha(Path(__file__)),
                              UTC=datetime.now(timezone.utc).isoformat(),action=request['action']))
    try:
        if request['action']=='environment':
            _,_,environment=core.runtime('cuda:0')
            write(out/'ENVIRONMENT.json',environment)
            result=dict(environment_sha256=sha(out/'ENVIRONMENT.json'),labels_read=False,training_steps=0)
        elif request['action']=='qualification':
            admission=bound(request['teacher_admission'])
            import torch
            import modern_teacher_adapter as adapter
            import modern_teacher_driver as driver
            audit=dict(optimizer_groups=[],steps=[],restores=[],model_restores=[])
            from full_parity_checks_v1 import qualification_parity
            original_parity=driver.qualification_parity
            driver.qualification_parity=qualification_parity
            original_factory=adapter.optimizer_for

            def equal_state(expected,actual):
                if torch.is_tensor(expected):
                    if not torch.is_tensor(actual) or expected.dtype!=actual.dtype or expected.shape!=actual.shape:
                        raise ValueError('Restored optimizer tensor type/shape differs')
                    torch.testing.assert_close(expected.detach().cpu(),actual.detach().cpu(),rtol=0,atol=0)
                elif isinstance(expected,dict):
                    if set(expected)!=set(actual):
                        raise ValueError('Restored optimizer keys differ')
                    for key in expected:equal_state(expected[key],actual[key])
                elif isinstance(expected,(tuple,list)):
                    if len(expected)!=len(actual):raise ValueError('Restored optimizer length differs')
                    for a,b in zip(expected,actual):equal_state(a,b)
                elif expected!=actual:
                    raise ValueError('Restored optimizer scalar differs')

            def audited_factory(model):
                optimizer=original_factory(model)
                named=dict(model.named_parameters())
                flattened=[p for group in optimizer.param_groups for p in group['params']]
                if len(flattened)!=len(named) or len({id(p) for p in flattened})!=len(flattened) or {id(p) for p in flattened}!={id(p) for p in named.values()}:
                    raise ValueError('Optimizer membership is not exactly the live model parameters')
                names={id(p):name for name,p in named.items()}
                spec=model.specification;n=spec['native'];mult=spec['lr_multiplier']
                for group in optimizer.param_groups:
                    for p in group['params']:
                        name=names[id(p)]
                        expected_lr=mult*(n['attention_lr'] if 'attnmodule' in name else n['lr']) if spec['backbone']=='polyformer_mono' else n['lr']*mult
                        expected_wd=(n['attention_weight_decay'] if 'attnmodule' in name else n['weight_decay']) if spec['backbone']=='polyformer_mono' else n['weight_decay']
                        if group['lr']!=expected_lr or group['weight_decay']!=expected_wd:
                            raise ValueError('Native optimizer grouping differs')
                audit['optimizer_groups'].append(dict(backbone=spec['backbone'],family=spec['family'],parameters=len(named),groups=len(optimizer.param_groups),live_membership_exact=True))
                original_step=optimizer.step;original_load=optimizer.load_state_dict
                original_model_load=model.load_state_dict
                def audited_model_load(state,*a,**kw):
                    result=original_model_load(state,*a,**kw)
                    equal_state(state,model.state_dict())
                    audit['model_restores'].append(dict(exact_model_state_restored=True,tensors=len(state),global_stage_before_load=model.global_stage))
                    return result
                model.load_state_dict=audited_model_load

                def audited_step(*a,**kw):
                    member_energy=[0.0]*spec['members'];active=0
                    for name,p in named.items():
                        if p.grad is None:continue
                        if not bool(torch.isfinite(p.grad).all()):
                            raise ValueError('Nonfinite parameter gradient: '+name)
                        active+=1
                        if spec['family']=='gnnm_boundary_4' and name.startswith('boundary.') and name.rsplit('.',1)[-1] in ('R','S','B'):
                            if p.grad.shape[0]!=spec['members']:raise ValueError('Wrong private member axis')
                            for m in range(spec['members']):member_energy[m]+=float(p.grad[m].double().square().sum())
                        elif spec['family']!='gnnm_boundary_4':
                            for m in range(spec['members']):
                                if name.startswith('models.'+str(m)+'.'):member_energy[m]+=float(p.grad.double().square().sum())
                    if active==0 or not all(v>0 for v in member_energy):
                        raise ValueError('A complete member has no nonzero training gradient')
                    audit['steps'].append(dict(global_stage=model.global_stage,finite_gradient_tensors=active,member_gradient_energy=member_energy))
                    return original_step(*a,**kw)

                def audited_load(state):
                    result=original_load(state)
                    equal_state(state,optimizer.state_dict())
                    audit['restores'].append(dict(exact_optimizer_state_restored=True,entries=len(state['state']),step_counters_checked=True))
                    return result
                optimizer.step=audited_step;optimizer.load_state_dict=audited_load
                return optimizer

            adapter.optimizer_for=audited_factory
            try:
                driver.fit_command(SimpleNamespace(command='teacher-qualify',admission=str(admission),output=str(out/'cell')))
            finally:
                adapter.optimizer_for=original_factory
                driver.qualification_parity=original_parity
            import numpy as np
            saved=np.load(out/'cell/teacher_member_logits.npy',allow_pickle=False)
            if not np.isfinite(saved).all():raise ValueError('Nonfinite full-graph selected qualification logits')
            audit['selected_full_graph_logits_finite']=True
            teacher=json.loads(admission.read_text())
            expected_steps=4 if teacher['backbone']=='polynormer_r' else 3
            if len(audit['steps'])!=expected_steps:
                raise ValueError('Qualification update count differs')
            if teacher['backbone']=='polynormer_r' and len(audit['restores'])!=1:
                raise ValueError('Photo local optimizer handoff was not verified')
            if len(audit['model_restores'])!=(2 if teacher['backbone']=='polynormer_r' else 1):
                raise ValueError('Expected local/final model-state restores were not verified')
            audit['exact_input_custody']={key:teacher[key] for key in ('role_freeze','source_labels','environment','backbone','family','implementation_sha256')}
            audit['exact_input_custody']['source_manifest']=request['source_manifest']
            roles=json.loads(bound(teacher['role_freeze']).read_text())
            audit['exact_input_custody']['graph_input']=roles['graph_input']
            write(out/'EXTERNAL_OPTIMIZER_AUDIT.json',audit)
            result=dict(qualification_receipt_sha256=sha(out/'cell/QUALIFICATION_RECEIPT.json'),
                        optimizer_audit_sha256=sha(out/'EXTERNAL_OPTIMIZER_AUDIT.json'),
                        training_steps=expected_steps,report_eligible=False,final_labels_read=False)
        else:raise ValueError('Unknown qualification action')
        for item in request['protected_files']:bound(item)
        if sha(request_path)!=request_sha:raise ValueError('Root request changed')
        write(out/'TERMINAL.json',dict(complete=True,seconds=time.monotonic()-started,request_sha256=request_sha,**result))
    except Exception as error:
        write(out/'FAILED_ATTEMPT.json',dict(error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,automatic_retry=False))
        raise

if __name__=='__main__':main()
