"""Four separate own-selected landmark bodies; standardized CAL after serving pool."""
import gc
from pathlib import Path

try:
    from .node_endpoints import receipt, scalar_calibration, evaluate_bank as _evaluate_bank, load_bound
    from ..engine import restore_checkpoint
    from ..run_study import load_factory, resolve_selected_config
    from ..spec import confined, fingerprint
except ImportError:
    from adapters.node_endpoints import receipt, scalar_calibration, evaluate_bank as _evaluate_bank, load_bound
    from engine import restore_checkpoint
    from run_study import load_factory, resolve_selected_config
    from spec import confined, fingerprint


I4_CONTRACT={'schema':'four-own-one-view-v1','bodies':4,'views_per_body':1,'served_views':4}


def export_bank(*,project_root,study,run,output_root,output_dir,registry,device):
    import torch
    c=run['config']; fits=c['fit_parents']; independent=len(fits)==4; grouped=c.get('landmark_memory_I4',False)
    if len(fits) not in (1,4) or len(set(fits))!=len(fits) or not set(fits).issubset(run['parents']): raise ValueError('Declared complete selected own learners required')
    if type(grouped)!=bool or (grouped and not independent): raise ValueError('Explicit four-own one-view4 bank contract required')
    ids={role:torch.load(confined(project_root,c['endpoint_inputs'][role+'_ids']),map_location='cpu',weights_only=True) for role in ('cal','evaluate')}
    if any(not torch.is_tensor(v) or v.ndim!=1 or v.dtype!=torch.long for v in ids.values()) or torch.isin(ids['cal'],ids['evaluate']).any(): raise ValueError('Separate ordered target-free endpoint IDs required')
    outputs={role:[] for role in ids}; ancestry=[]; contracts=[]; mapping=[]
    for body,identifier in enumerate(fits):
        row=registry.run(study['study_id'],identifier)
        if row['state']!='complete' or row['contract']['kind']!='model_fit': raise ValueError('Complete individually selected own fit required')
        actual=resolve_selected_config(study,row['contract'],registry); factory=load_factory(project_root,actual['adapter'])
        adapter=factory(torch=torch,project_root=project_root,study=study,run=actual,device=device,execution_context={'registry':registry,'output_root':output_root})
        selected=row['result']['selected']; authority=row['result']['selected_checkpoint']; rel=Path(authority['path'])
        if rel.is_absolute() or '..' in rel.parts or authority['fit_id']!=identifier or authority['step']!=selected['step']: raise ValueError('Safe own selected checkpoint authority required')
        checkpoint=Path(output_root)/study['study_id']/identifier/rel
        if checkpoint.is_symlink(): raise ValueError('Selected checkpoint symlink excluded')
        observed=receipt(checkpoint,project_root)
        if any(observed[k]!=authority[k] for k in ('bytes','sha256')): raise ValueError('Selected export checkpoint changed')
        restored=restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=identifier)
        if restored['step']!=selected['step']: raise ValueError('Own selected export state differs')
        adapter.model.eval()
        with torch.no_grad(): logits=adapter.logits_fn(adapter.model,False,selected['step'])
        expected=1
        if logits.ndim!=3 or not torch.isfinite(logits).all() or (independent and (logits.shape[0]!=expected or actual['body']!=body)): raise ValueError('Four disjoint own learners with declared actual views and body order required')
        if grouped and actual['config']['landmark_arm'] not in ('S64','FS64'): raise ValueError('Memory I4 contains four complete selected S64/FS64 one-view learners only')
        for role,value in ids.items(): outputs[role].append(logits[:,value.to(logits.device)].detach().cpu())
        mapping += [body]*len(logits)
        ancestry.append({'run_id':identifier,'checkpoint':observed,'selected_step':selected['step'],'actual_model_seed':adapter.provenance['actual_model_seed'],
            'actual_paid_parent':adapter.provenance['actual_parent'],'actual_H_hash':adapter.provenance['actual_H_hash'],'actual_U_hash':adapter.provenance['actual_U_hash'],'config_hash':fingerprint(actual['config']),'served_views':len(logits),'body':actual['body']})
        contracts.append(actual['config']); del adapter,logits; gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    if independent:
        if len({v['actual_model_seed'] for v in ancestry})!=4: raise ValueError('Four own initialization seeds required')
        ancestors=[v['actual_paid_parent'] for v in ancestry]
        if any(v is not None for v in ancestors) and (any(v is None for v in ancestors) or len({v['fit_id'] for v in ancestors})!=4): raise ValueError('I4 continuation requires four distinct own paid source acquisitions')
        excluded={'initialization_seed','resolved_selection','reservation_mib','estimated_seconds','lifecycle'}
        normalized=[{k:v for k,v in value.items() if k not in excluded} for value in contracts]
        if any(v!=normalized[0] for v in normalized[1:]): raise ValueError('Independent learners have different data/source/model/hyperparameter choices')
    files={}
    for role,values in outputs.items():
        path=output_dir/(role+'_logits.pt'); torch.save({'ids':ids[role],'logits':torch.cat(values),'view_body_ids':torch.tensor(mapping)},path); files[role]=receipt(path,project_root)
    return {'status':'complete','files':files,'ancestry':ancestry,'fusion':c['fusion'],'independent_complete_learners':independent,
        'independent_bodies':len(fits),'served_views':len(mapping),'view_body_ids':mapping,'landmark_memory_I4':grouped,
        'CAL_targets_read':False,'evaluation_targets_read':False,'scientific_backbone_fits':0}


def evaluate_bank(**kwargs):
    result=_evaluate_bank(**kwargs)
    import torch
    root,study,run,registry=kwargs['project_root'],kwargs['study'],kwargs['run'],kwargs['registry']
    config=run['config']; exported=registry.run(study['study_id'],config['export_parent'])['result']
    bank=load_bound(torch,root,exported['files']['evaluate'])
    targets=torch.load(confined(root,config['endpoint_inputs']['evaluate_targets']),map_location='cpu',weights_only=True)
    if not torch.equal(bank['ids'],targets['ids']): raise ValueError('Body diagnostic target order differs')
    mapping=bank['view_body_ids']; logits=bank['logits'].double(); body_ids=sorted(set(mapping.tolist()))
    body_probability=torch.stack([logits[mapping==body].softmax(-1).mean(0) for body in body_ids])
    correct=body_probability.argmax(-1).eq(targets['y'][None])
    result['body_diagnostics']={'body_ids':body_ids,'views_per_body':[int((mapping==body).sum()) for body in body_ids],
        'accuracies':[float(v.double().mean()) for v in correct],'oracle_body_accuracy_descriptive_only':float(correct.any(0).double().mean()),
        'diagnostic_unit':'complete separately selected own learner probability mean; distinct from individual landmark-view metrics'}
    return result
