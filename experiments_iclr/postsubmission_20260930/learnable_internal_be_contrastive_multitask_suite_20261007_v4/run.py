"""Disabled predictive runner; one prospective task/arm/seed cell per release.

No TEST loader. Validation values remain in closed output until whole-family
closure; stdout prints no predictive metric. No active queue integration.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time
from runtime import admit, runtime_versions, PHASE, ROOT


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temp, path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args=parser.parse_args()
    started=time.monotonic()
    job, config, output = admit(args.job)
    output.mkdir()
    trace=[];steps=0
    try:
        versions=runtime_versions()
        import numpy as np
        import torch
        from ogb.linkproppred import Evaluator as LinkEvaluator
        from ogb.graphproppred import Evaluator as GraphEvaluator
        from models import Ensemble, native_sources
        from data import load_projection, batches, valid_batches, bound
        from objectives import objective, own_supervision
        from factors import factor_counts
        from selection import local_transition,finite_predictions,finite_state
        seed=job['seed']; task=job['task']; recipe=config['training']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        train, valid, authority=load_projection(PHASE, job['data_manifest'], task, True)
        source_binding=json.loads((ROOT/'DEPENDENCIES.json').read_text())
        sources=native_sources(PHASE, source_binding)
        model=Ensemble(task, job['arm'], seed, config['model'], sources).to('cuda:0')
        ordinary_independent = job['arm']=='independent4'
        model.set_global(False) if task=='wikics' else None
        def optimizer(body):
            if task=='collab':
                return torch.optim.Adam([{'params':body.encoder.parameters(), 'lr':.0082},
                                         {'params':body.decoder.parameters(), 'lr':.0037}], weight_decay=0.)
            return torch.optim.Adam(body.parameters(), lr=recipe['lr'], weight_decay=0., eps=1e-8)
        opts=[optimizer(body) for body in model.models]
        # Each predictor has a persistent independent dropout RNG stream. Data order
        # is deliberately common across arms/members; contrastive pairs align IDs.
        streams=[]
        for m in range(model.members):
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed+1009*m+300001); torch.cuda.manual_seed(seed+1009*m+300001)
                streams.append({'cpu':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state()})
        def forward(batch):
            rows=[]
            for m, stream in enumerate(streams):
                with torch.random.fork_rng(devices=[0]):
                    torch.set_rng_state(stream['cpu']); torch.cuda.set_rng_state(stream['cuda'])
                    rows.append(model.member_forward(batch,m))
                    stream.update(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
            return torch.stack([r[0] for r in rows]),torch.stack([r[1] for r in rows])
        def serving(logits):
            return logits.softmax(-1).mean(0) if task=='wikics' else logits.mean(0)
        def evaluate():
            model.eval(); chunks=[]; truth=[]; member_chunks=[]
            with torch.no_grad():
                for batch,y in valid_batches(task, train, valid, recipe, 'cuda:0'):
                    logits,_=forward(batch)
                    pooled=serving(logits)
                    finite_predictions(logits,pooled)
                    chunks.append(pooled.cpu());member_chunks.append(logits.cpu());truth.append(y.cpu())
            prediction=torch.cat(chunks); members=torch.cat(member_chunks,1)
            if task=='wikics':
                y=torch.cat(truth)
                metric=float((prediction.argmax(-1)==y).float().mean())
                per=[float((x.argmax(-1)==y).float().mean()) for x in members]
            elif task=='molhiv':
                y=torch.cat(truth).reshape(-1,1)
                ev=GraphEvaluator(name='ogbg-molhiv')
                metric=float(ev.eval({'y_true':y.numpy(),'y_pred':prediction.numpy()})['rocauc'])
                per=[float(ev.eval({'y_true':y.numpy(),'y_pred':x.numpy()})['rocauc']) for x in members]
            else:
                n=len(valid['positive']); ev=LinkEvaluator(name='ogbl-collab');ev.K=50
                def hits(p):return float(ev.eval({'y_pred_pos':p[:n].flatten(),'y_pred_neg':p[n:].flatten()})['hits@50'])
                metric=hits(prediction);per=[hits(x) for x in members]
            if not np.isfinite(metric) or not np.isfinite(per).all():
                raise ValueError('Complete selector metric nonfinite')
            return metric,per
        def cpu_tree(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
            if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return type(value)(cpu_tree(v) for v in value)
            return value
        def save(path, epoch, metric, per):
            torch.save({'model':cpu_tree(model.state_dict()),'optimizers':cpu_tree([o.state_dict() for o in opts]),
                'streams':cpu_tree(streams),'epoch':epoch,'global':task=='wikics' and model.models[0].body._global,
                'selected_VALID':metric,'member_VALID':per,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'job':job,'config':config},path)
        best=-float('inf');best_local=-float('inf')
        own_best=[-float('inf')]*model.members;own_local=[-float('inf')]*model.members
        for epoch in range(1,recipe['epochs']+1):
            if time.monotonic()-started>job['soft_seconds']:
                raise TimeoutError('Preserved failure; no shortened fit or retry')
            if task=='wikics' and epoch==recipe['local_epochs']+1:
                local_transition(model,opts,lambda name:torch.load(output/name,map_location='cuda:0',weights_only=False),ordinary_independent)
            model.train()
            last=None
            for batch,y in batches(task,train,recipe,epoch,seed,'cuda:0'):
                for opt in opts:opt.zero_grad(set_to_none=True)
                la,ha=forward(batch);lb,hb=forward(batch)
                finite_predictions(la,serving(la));finite_predictions(lb,serving(lb))
                if not torch.isfinite(ha).all() or not torch.isfinite(hb).all():raise FloatingPointError("TRAIN representation")
                # Deterministically bounded contrastive batch; full labels still CE.
                size=min(len(y),config['contrastive']['max_objects'])
                index=torch.linspace(0,len(y)-1,steps=size,device=y.device).long()
                from objectives import alignment_loss,residual_member_contrast
                own=.5*(own_supervision(la,y,task)+own_supervision(lb,y,task))
                aux=own.sum()*0
                if model.contrastive:
                    identity = (batch['query'][index] if task=='collab' else None)
                    aux=.05*alignment_loss(ha[:,index],hb[:,index],y[index],task,identities=identity)+.05*residual_member_contrast(ha[:,index],hb[:,index],y[index])
                loss=own.sum()+model.members*aux if model.independent else own.mean()+aux
                if not torch.isfinite(loss):raise FloatingPointError('TRAIN loss')
                loss.backward()
                active=[p for p in model.parameters() if p.grad is not None]
                if not active or any(not torch.isfinite(p.grad).all() for p in active):raise FloatingPointError('TRAIN gradient')
                for opt in opts:opt.step()
                finite_state(model,opts)
                steps+=1;last={'own_mean':float(own.mean().detach()),'aux':float(aux.detach())}
            metric,per=evaluate()
            if not ordinary_independent and metric>best:
                best=metric;save(output/'selected.pt',epoch,metric,per)
            if task=='wikics' and epoch<=recipe['local_epochs'] and metric>best_local:
                best_local=metric;save(output/'selected_local.pt',epoch,metric,per)
            if model.independent:
                for m,body in enumerate(model.models):
                    row={'model':cpu_tree(body.state_dict()),'optimizer':cpu_tree(opts[m].state_dict()),
                         'epoch':epoch,'global':task=='wikics' and body.body._global,'metric':per[m]}
                    if per[m]>own_best[m]:
                        own_best[m]=per[m];torch.save(row,output/('own_best_'+str(m)+'.pt'))
                    if task=='wikics' and epoch<=recipe['local_epochs'] and per[m]>own_local[m]:
                        own_local[m]=per[m];torch.save(row,output/('own_local_'+str(m)+'.pt'))
            trace.append({'epoch':epoch,'VALID_selector':metric,'members':per,'TRAIN':last,'seconds':time.monotonic()-started})
            atomic_json(output/'CLOSED_TRACE.json',trace)
            atomic_json(output/'PROGRESS.json',{'epoch':epoch,'complete_epochs':recipe['epochs'],'steps':steps,'scores_closed':True})
        # Ordinary independent ensemble serves the four OWN-selected models.
        # Pooled metrics never select any of its member checkpoints or bank.
        # Stage flags can differ, so preserve per-body serving modes explicitly.
        if ordinary_independent:
            for m,body in enumerate(model.models):
                saved=torch.load(output/('own_best_'+str(m)+'.pt'),map_location='cuda:0',weights_only=False)
                body.load_state_dict(saved['model'])
                if task=='wikics':body.set_global(saved['global'])
            metric,per=evaluate()
            atomic_json(output/'CLOSED_OWN_BEST_BANK.json',{'VALID':metric,'members':per})
            # Evaluation-only assembled bank; own checkpoints retain each
            # selected optimizer. No joint resume from mixed epochs is claimed.
            torch.save({'model':cpu_tree(model.state_dict()),'body_global':[getattr(getattr(x,'body',None),'_global',False) for x in model.models],
                'evaluation_only':True,'candidate':'individual_best_bank_only','selected_VALID':metric,'job':job,'config':config},output/'selected.pt')
        atomic_json(output/'FREEZE.json',{'complete':True,'task':task,'arm':job['arm'],'seed':seed,'epochs':recipe['epochs'],
            'steps':steps,'TEST_access':False,'scores_closed':True,'source_manifest_sha256':job['source_manifest_sha256'],
            'selected_sha256':__import__('runtime').sha(output/'selected.pt'),'versions':versions,
            'parameter_counts':factor_counts(model),'seconds':time.monotonic()-started})
    except Exception as error:
        atomic_json(output/'FAILURE.json',{'complete':False,'error_type':type(error).__name__,'error':str(error),
            'epochs_completed':len(trace),'steps':steps,'scores_closed':True,'seconds':time.monotonic()-started,'automatic_retry':False,'phase':'startup_or_training'})
        raise
    print(json.dumps({'complete':True,'scores_closed':True,'output':str(output)}))


if __name__=='__main__':main()
