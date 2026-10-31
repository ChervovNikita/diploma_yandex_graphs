"""Source scalar derivatives and pure cadence costs; no native fit/model."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact intended allocation route required')
    root=Path(args.project_root).resolve();stage=Path(__file__).resolve().parent;out=root/args.output
    if out.exists() or not out.resolve().is_relative_to(root):raise ValueError('Fresh confined source scalar/cost QA required')
    os.environ['CUDA_VISIBLE_DEVICES']='';out.mkdir();sys.path.insert(0,str(stage/'runtime'))
    import torch
    from source_utils import bind_source,load
    source=bind_source(root);helper=load(root,source['modules']['supply'],'supply');policy=load(root,source['modules']['policy'],'policy');checks=[]
    try:
        F=torch.nn.functional;g=torch.Generator().manual_seed(73301);target=(torch.arange(4)[:,None]+torch.arange(5)[None])%2
        truth=target.float();base=helper.Config(enabled=True,score_mode='bernoulli_marginal_logits')
        keys=[helper.OutputKey(kind,m) for kind in ('train_full','eval_full') for m in range(4)]
        keys += [helper.OutputKey('train_probe',m,a) for a in ('actor','director','keyword') for m in range(4)]
        raw={k:torch.randn(4,5,generator=g) for k in keys}
        def observed(x):return torch.where(truth.bool(),F.logsigmoid(x),F.logsigmoid(-x))
        def pool(xs):return -(torch.logsumexp(torch.stack([observed(x) for x in xs]),0)-__import__('math').log(len(xs))).mean()
        for arm in policy.ARMS:
            assignments=('actor','director','keyword',None) if arm!='COMMON_cycle' else ('actor','actor','actor',None)
            leaves={k:v.clone().requires_grad_(True) for k,v in raw.items()};peers={k:v.detach().clone() for k,v in raw.items()}
            with policy.objective_scope(helper,arm):terms=helper._terms(leaves,peers,truth,assignments,base)
            if arm=='native_pool_credit':oracle=pool([leaves[helper.OutputKey('train_full',m)] for m in range(4)])
            elif arm=='source_view_supervision':oracle=sum(F.binary_cross_entropy_with_logits(leaves[helper.OutputKey('train_probe',m,a)],truth) for m,a in enumerate(assignments[:3]))/3
            elif arm=='uncoupled_source_contrast':oracle=sum(F.binary_cross_entropy_with_logits(leaves[helper.OutputKey('train_full',m)],truth)-F.binary_cross_entropy_with_logits(leaves[helper.OutputKey('train_probe',m,a)],truth) for m,a in enumerate(assignments[:3]))/3
            else:
                oracle=sum(pool([leaves[helper.OutputKey('train_full',m)],*[peers[helper.OutputKey('train_probe',k,a)] for k in range(4) if k!=m]])
                    -pool([leaves[helper.OutputKey('train_probe',m,a)],*[peers[helper.OutputKey('train_probe',k,a)] for k in range(4) if k!=m]]) for m,a in enumerate(assignments[:3]))/3
            if not torch.allclose(terms['J'],oracle,rtol=3e-6,atol=3e-6):raise ValueError('Source policy scalar/sign differs from independent Bernoulli oracle')
            risk_names=[*('full_own:'+str(m) for m in range(4)),'full_pool',*('probe_own:'+str(m) for m in range(3)),*('absent:'+str(m) for m in range(3))]
            credits={}
            for label in ('J',*risk_names):
                grads=torch.autograd.grad(terms[label],[leaves[k] for k in keys],allow_unused=True,retain_graph=True)
                credits[label]={k:v for k,v in zip(keys,grads) if v is not None}
            replay=[*(helper.OutputKey('train_full',m) for m in range(4)),*(helper.OutputKey('train_probe',m,a) for m,a in enumerate(assignments[:3])),*(helper.OutputKey('eval_full',m) for m in range(4))]
            model_vjps=sum(sum(k in credit for credit in credits.values()) for k in replay if k.member<3)
            expected=15 if arm in ('native_pool_credit','source_view_supervision') else 18
            if model_vjps!=expected or len(replay)!=11 or len(keys)!=20:raise ValueError('Exact source reference/private-VJP cost support changed')
            if arm in ('native_pool_credit','source_view_supervision','uncoupled_source_contrast') and any(not torch.equal(terms['source_j:'+str(m)],terms['J']) for m in range(3)):
                raise ValueError('Matched control source-J guard must repeat its global objective')
            checks.append({'arm':arm,'source_scalar_and_credit_support':'pass','nonown_reference_forwards_per_opportunity':20,
                'nonown_replay_forwards_per_opportunity':11,'scalar_leaf_reverse_calls_per_opportunity':12,'private_model_VJP_calls_per_opportunity':model_vjps,
                'source_own_arm_has_no_actual_opportunities':arm=='shared_own_only','trial_forward_calls_range_per_opportunity':[0,44]})
        schedules=[]
        for horizon in (200,500,1000,5000):
            due=[e for e in range(1,horizon+1) if e>=15 and e%5==0];common={a:0 for a in ('actor','director','keyword')}
            for i,e in enumerate(due):common[('actor','director','keyword')[i%3]]+=3
            schedules.append({'complete_epoch_ceiling':horizon,'own_complete_batches_per_body':horizon,'opportunities':len(due),
                'assigned_recipient_family_exposures':{a:len(due) for a in common},'COMMON_recipient_family_exposures':common,
                'trial_attempt_upper_bound_per_fit':4*len(due),'trial_model_forward_upper_bound_per_fit':44*len(due)})
        if torch.cuda.is_initialized():raise ValueError('CPU scalar QA initialized CUDA')
        result={'status':'passed','scope':'Synthetic output scalar autograd and exact policy cadence metadata only; no native constructor, native update, warm acquisition or score',
            'checks':checks,'schedules':schedules,'primary_horizon':1000,'source_historical_horizon':200,'optional_stress_horizon':5000,
            'source_bindings':source,'fits':0,'native_optimizer_updates':0,'predictive_scores':0,'CUDA':False,'GO':False}
        (out/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'status':'passed','source_policy_scalars':6,'primary_complete_epochs':1000,'primary_source_opportunities':198,'fits':0,'CUDA':False,'GO':False}))
    except BaseException as e:
        (out/'FAILURE.json').write_text(json.dumps({'status':'failed','error':str(e),'traceback':traceback.format_exc(),'completed':checks,'fits':0,'GO':False},indent=2)+'\n');raise


if __name__=='__main__':main()
