"""Remaining native correction branches, zero own dose and tiny finite trials.

Synthetic CPU only. Native correction forwards/VJPs run at full width. The
finite trial rollback check uses the exact helper on an explicitly tiny
ownership fixture; it does not qualify native predictive behavior.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:raise ValueError('Exact authorized allocation required')
    root=Path(a.project_root).resolve();out=root/a.output;stage=Path(__file__).resolve().parent
    if out.exists() or not out.resolve().is_relative_to(root):raise ValueError('Fresh confined QA output required')
    os.environ['CUDA_VISIBLE_DEVICES']='';out.mkdir();sys.path.insert(0,str(stage/'runtime'))
    import torch
    from source_utils import bind_source
    from adapters.sehgnn import make_adapter,SUPPLY
    torch.set_num_threads(2);src=bind_source(root);custody=json.loads((root/'sehgnn_typed_newdata_custody_20261012_v2/CUSTODY.json').read_text());N=12;g=torch.Generator().manual_seed(71901)
    full={k:torch.randn(N,s[-1],generator=g) for k,s in custody['feature_shapes'].items()};protocol=json.loads((root/src['metadata']['protocol']['path']).read_text());labs={k:torch.randn(N,5,generator=g) for k in protocol['label_paths']}
    fv={'full':full};lv={'full':labs}
    for family,symbol in (('actor','A'),('director','D'),('keyword','K')):
        fv[family]={k:torch.zeros_like(v) if symbol in k else v.clone() for k,v in full.items()};lv[family]={k:torch.zeros_like(v) if symbol in k else v.clone() for k,v in labs.items()}
    y=((torch.arange(N)[:,None]+torch.arange(5)[None])%2).float();fixture={'features':fv,'labels':lv,'train':{'ids':torch.arange(6),'y':y[:6]},'select':{'ids':torch.arange(6,10),'y':y[6:10]}}
    cfg={'sehgnn_arm':'shared_own_only','sehgnn_epochs':1000,'historical_state_inherited':False,'learning_rate':.001,'dropout':.5,'sehgnn_sources':src}
    study={'stage':'engineering','study_id':'synthetic_sehgnn_correction','data':{'synthetic':True},'source':{'files':[]}};run={'run_id':'CPU_correction','seed':81901,'body':0,'split':0,'config':cfg};rows=[]
    try:
        adapter=make_adapter(torch=torch,project_root=root,study=study,run=run,device='cpu',execution_context={'engineering_synthetic':True,'fixture':fixture});s=adapter.session;h=adapter.source_modules['helper'];policy=adapter.source_modules['policy']
        zero={m:tuple(torch.zeros_like(p) for p in s.ownership.private[m]) for m in range(4)}
        for arm in SUPPLY[1:]:
            scheduled=policy.ScheduledSession(s,arm,{'synthetic_correction':True});scheduled.completed_own_epochs=15
            before=copy.deepcopy(s.counters);old_make,old_acc,old_trial=h.make_credit_plan,h.accumulate_replay,h.try_private_step;counts={'leaf_reverse':0,'private_VJP':0,'trials':0}
            def make(*args,**kwargs):
                plan=old_make(*args,**kwargs);counts['leaf_reverse']+=len(plan.cotangents);return plan
            def acc(plan,key,scores):
                counts['private_VJP']+=sum(key in v for v in plan.cotangents.values()) if plan.assignments[key.member] is not None else 0;return old_acc(plan,key,scores)
            def trial(*args,**kwargs):counts['trials']+=1;return old_trial(*args,**kwargs)
            h.make_credit_plan,h.accumulate_replay,h.try_private_step=make,acc,trial
            try:value=scheduled.correct(zero)
            finally:h.make_credit_plan,h.accumulate_replay,h.try_private_step=old_make,old_acc,old_trial
            delta={k:s.counters[k]-before.get(k,0) for k in s.counters if type(s.counters[k]) is int};expected=15 if arm in ('native_pool_credit','source_view_supervision') else 18
            if value['status']!='zero' or value['private_L2_dose']!=0 or delta['source_reference_forwards']!=20 or delta['source_replay_forwards']!=11 or counts!={'leaf_reverse':12,'private_VJP':expected,'trials':0}:raise ValueError('Native correction wiring/cost changed')
            rows.append({'arm':arm,'native_correction_zero_own_dose':'pass','costs':delta,'helper_calls':counts,'protected_state_checks':'exact original source finally guards passed'})
        # Independent helper trial fixture: all outputs depend on genuine
        # private parameters; no native resizing or pseudo scientific result.
        class Toy(torch.nn.Module):
            def __init__(self,shared,m):super().__init__();self.shared=shared;self.private=torch.nn.Parameter(torch.ones(5)*(.2+.01*m))
        shared=torch.nn.Parameter(torch.zeros(5));members=[Toy(shared,m) for m in range(4)];own=h.verify_ownership(members,('shared',),('private',));hc=h.Config(enabled=True,score_mode='bernoulli_marginal_logits');assign=('actor','director','keyword',None)
        views={x:h.SourceView(x,'toy-'+x,(x,),True,True,True,True) for x in assign[:3]};keys=[h.OutputKey(kind,m) for kind in ('train_full','eval_full') for m in range(4)]+[h.OutputKey('train_probe',m,x) for x in assign[:3] for m in range(4)];truth=torch.ones(2,5);tokens={k:(k.member,k.source,k.kind) for k in keys}
        def forward(m,source,mode,token):return (members[m].private+members[m].shared).expand(2,5)
        outputs={k:forward(k.member,k.source,'eval',tokens[k]).detach().clone() for k in keys};plan=h.make_credit_plan(hc,own,assign,views,'toy-full',outputs,truth,tokens,target_role='TRAIN')
        # An intentionally inconsistent descent slope makes the Armijo guard
        # reject; exact original helper must roll private values back.
        direction=h.Direction({m:tuple(-torch.ones_like(v)*.1 for v in own.private[m]) for m in plan.active()},-1.,False,id(plan));before=[p.detach().clone() for p in own.all_parameters];calls=[0]
        def count_forward(*args):calls[0]+=1;return forward(*args)
        decision=h.try_private_step(plan,direction,1.,count_forward)
        if decision.accepted or calls[0]!=11 or any(not torch.equal(p,q) for p,q in zip(own.all_parameters,before)):raise ValueError('Finite private rejected trial/cost/rollback changed')
        rows.append({'tiny_private_trial':'rejected_with_exact_rollback','forwards':11,'scope':'helper finite-trial engineering fixture only'})
        if torch.cuda.is_initialized():raise ValueError('CPU correction QA initialized CUDA')
        result={'status':'passed','checks':rows,'scope':'Remaining five source policy native reference/replay/VJP wiring at zero own displacement; exact tiny helper rejected trial','fits':0,'optimizer_updates':0,'real_data_model_input':False,'CUDA':False,'GO':False};(out/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','native_correction_policies':5,'tiny_rejected_trials':1,'fits':0,'CUDA':False}))
    except BaseException as e:
        (out/'FAILURE.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc(),'completed':rows,'fits':0,'GO':False},indent=2)+'\n');raise


if __name__=='__main__':main()
