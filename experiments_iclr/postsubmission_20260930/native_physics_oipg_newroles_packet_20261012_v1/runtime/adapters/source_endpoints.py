"""Native serving authority and separated CAL/development endpoint jobs."""
import gc,math
from pathlib import Path
from adapters.node_endpoints import receipt,load_bound
from engine import restore_checkpoint
from run_study import load_factory,resolve_selected_config
from spec import confined,fingerprint

def endpoint_input(torch,root,config,key):
    row=config['endpoint_input_receipts'][key]
    if row['path']!=config['endpoint_inputs'][key]:raise ValueError('Exact endpoint path/receipt mismatch')
    return load_bound(torch,root,row)

def export_bank(*,project_root,study,run,output_root,output_dir,registry,device):
    import torch
    c=run['config'];fits=c['fit_parents']
    if c['fusion'] not in ('mean_logits','mean_probabilities'):raise ValueError('Literal native serving pool required')
    if len(fits) not in (1,4) or len(fits)!=len(set(fits)) or not set(fits)<=set(run['parents']):raise ValueError('One source bank or four own selected source fits')
    ids={r:endpoint_input(torch,project_root,c,r+'_ids') for r in ('cal','evaluate')}
    if any(v.dtype!=torch.long or v.ndim!=1 or not len(v) or len(v.unique())!=len(v) or int(v.min())<0 for v in ids.values()):raise ValueError('Nonempty unique int64 endpoint IDs required')
    if torch.isin(ids['cal'],ids['evaluate']).any():raise ValueError('CAL and development evaluation overlap')
    values={r:[] for r in ids};ancestry=[];configs=[];constructors=optimizers=selected_routes=0
    for fit in fits:
        row=registry.run(study['study_id'],fit)
        if row['state']!='complete' or row['result']['status']!='complete':raise ValueError('Whole own source fit required')
        actual=resolve_selected_config(study,row['contract'],registry);adapter=load_factory(project_root,actual['adapter'])(torch=torch,project_root=project_root,study=study,run=actual,device=device)
        authority=row['result']['selected_checkpoint'];selected=row['result']['selected'];checkpoint=Path(output_root)/study['study_id']/fit/authority['path']
        got=receipt(checkpoint,project_root)
        if got['bytes']!=authority['bytes'] or got['sha256']!=authority['sha256']:raise ValueError('Exact selected checkpoint changed')
        restored=restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=fit)
        if restored['step']!=selected['step']:raise ValueError('Selected checkpoint step differs')
        adapter.model.eval()
        with torch.no_grad():
            logits=adapter.logits_fn(adapter.model,False,selected['step'])
            if any(int(v.max())>=logits.shape[1] for v in ids.values()):raise ValueError('Endpoint IDs outside full graph')
            if logits.dtype!=torch.float32 or logits.ndim!=3 or not torch.isfinite(logits).all() or len(fits)==4 and len(logits)!=1:raise ValueError('Native complete bank logits and genuine own I4')
            if getattr(adapter,'coupled',False):
                replay=[int(z[adapter.select_ids].argmax(-1).eq(adapter.select_y).sum()) for z in logits]
                if replay!=[s['SELECT_correct'] for s in adapter.own_selected]:raise ValueError('Coupled own selected checkpoints do not replay')
                own_steps=[s['selected_step'] for s in adapter.own_selected]
            else:own_steps=[selected['step']]
            for role,index in ids.items():values[role].append(logits[:,index.to(logits.device)].detach().cpu())
        ancestry.append({'run_id':fit,'checkpoint':got,'selected_container_step':selected['step'],'own_selected_steps':own_steps,
                         'actual_model_seed':adapter.provenance['actual_model_seed'],'config_hash':fingerprint(actual['config'])})
        configs.append(actual['config']);constructors+=actual['config']['physical_slow_bodies'];optimizers+=actual['config']['source_optimizer_count'];selected_routes+=len(logits)
        del adapter,logits;gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    if len(fits)==4 and len({row['actual_model_seed'] for row in ancestry})!=4:raise ValueError('Four distinct own model seeds')
    files={}
    for role,parts in values.items():
        logits=torch.cat(parts).to(device)
        # Capture the source serving operator in FP32 on the execution device
        # before transporting endpoint values to float64 CPU metric routines.
        if c['fusion']=='mean_logits':
            scores=logits.mean(0);decision=scores.argmax(-1)
        else:
            probability=logits.softmax(-1).mean(0);decision=probability.argmax(-1)
            scores=torch.logsumexp(logits.log_softmax(-1),0)-math.log(len(logits))
        pack={'ids':ids[role],'logits':logits.cpu(),'native_scores':scores.cpu(),'native_decision':decision.cpu(),'fusion':c['fusion']}
        p=output_dir/(role+'_logits.pt');torch.save(pack,p);files[role]=receipt(p,project_root)
    return {'status':'complete','files':files,'ancestry':ancestry,'fusion':c['fusion'],'independent_complete_learners':len(fits)==4,
            'CAL_targets_read':False,'evaluation_targets_read':False,'scientific_backbone_fits':0,
            'operation_counts':{'optimizer_updates':0,'source_export_body_constructors':constructors,'source_export_optimizer_constructors':optimizers,
                                'source_selected_route_forwards':selected_routes}}

def fit_temperature(torch,scores,labels):
    parameter=torch.nn.Parameter(torch.zeros((),dtype=torch.float64));optimizer=torch.optim.Adam([parameter],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0)
    for step in range(500):
        optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(scores.double()/parameter.exp(),labels)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fixed CAL objective')
        loss.backward()
        if not torch.isfinite(parameter.grad):raise FloatingPointError('Nonfinite CAL gradient')
        optimizer.step()
    value=float(parameter.detach().exp())
    if not math.isfinite(value) or value<=0:raise FloatingPointError('Nonfinite selected final temperature')
    return value

def scalar_calibration(*,project_root,study,run,output_root,output_dir,registry,device):
    import torch
    c=run['config'];row=registry.run(study['study_id'],c['export_parent'])
    if row['state']!='complete' or c['export_parent'] not in run['parents']:raise ValueError('Completed label-free source export required')
    bank=load_bound(torch,project_root,row['result']['files']['cal']);targets=endpoint_input(torch,project_root,c,'cal_targets')
    if set(targets)!={'ids','y'} or targets['y'].dtype!=torch.long or targets['y'].shape!=targets['ids'].shape or not torch.equal(bank['ids'],targets['ids']):raise ValueError('CAL ordering/schema differs')
    if c['placement']!='after_serving_pool' or c['steps']!=500 or c['learning_rate']!=.01:raise ValueError('Fixed source/prospective CAL500 common scalar')
    return {'status':'complete','temperature':fit_temperature(torch,bank['native_scores'],targets['y']),
            'placement':c['placement'],'fusion':c['fusion'],'CAL_examples':len(targets['y']),'scalar_optimizer_updates':500,
            'endpoint':'fixed_final500_no_clamp_no_CAL_selector','export_parent':c['export_parent'],'evaluation_targets_read':False,'scientific_backbone_fits':0}

def evaluate_bank(*,project_root,study,run,output_root,output_dir,registry,device):
    import torch
    c=run['config'];erow=registry.run(study['study_id'],c['export_parent']);crow=registry.run(study['study_id'],c['calibration_parent'])
    if erow['state']!='complete' or crow['state']!='complete' or not {c['export_parent'],c['calibration_parent']}<=set(run['parents']):raise ValueError('Exact completed export and CAL dependencies required')
    exported=erow['result'];cal=crow['result']
    if cal['export_parent']!=c['export_parent'] or cal['fusion']!=c['fusion']:raise ValueError('CAL belongs to another source bank')
    bank=load_bound(torch,project_root,exported['files']['evaluate']);targets=endpoint_input(torch,project_root,c,'evaluate_targets')
    if set(targets)!={'ids','y'} or targets['y'].dtype!=torch.long or targets['y'].shape!=targets['ids'].shape or not torch.equal(targets['ids'],bank['ids']):raise ValueError('Development endpoint role order/schema')
    labels=targets['y'];scores=bank['native_scores'].double();metrics={}
    for name,temp in (('raw',1.),('calibrated',cal['temperature'])):
        lp=(scores/temp).log_softmax(-1);p=lp.exp();correct=bank['native_decision'].eq(labels)
        metrics[name]={'accuracy':float(correct.double().mean()),'correct':int(correct.sum()),'examples':len(labels),
                       'nll':float(-lp.gather(1,labels[:,None]).mean()),'brier_sum_classes':float((p-torch.nn.functional.one_hot(labels,p.shape[1])).square().sum(-1).mean())}
    return {'status':'complete','metrics':metrics,'member_accuracies':[float(z.argmax(-1).eq(labels).double().mean()) for z in bank['logits']],
            'ancestry':exported['ancestry'],'fusion':c['fusion'],'evaluation_scope':study['data']['prior_exposure'],
            'statistical_inference':'whole_case_seed_cohort_required_no_node_bootstrap','scientific_backbone_fits':0}
