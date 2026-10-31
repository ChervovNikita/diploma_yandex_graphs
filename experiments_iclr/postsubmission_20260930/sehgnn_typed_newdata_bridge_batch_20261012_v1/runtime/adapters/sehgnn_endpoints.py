"""Native5 selected-state banks and role-gated Bernoulli endpoints."""
import gc
import json
import math
from pathlib import Path

try:
    from ..source_utils import receipt,bound
    from ..engine import restore_checkpoint
    from ..run_study import load_factory,resolve_selected_config
    from ..spec import confined,fingerprint
except ImportError:
    from source_utils import receipt,bound
    from engine import restore_checkpoint
    from run_study import load_factory,resolve_selected_config
    from spec import confined,fingerprint


def selected_authority(root,output_root,study,row):
    contract,result=row['contract'],row['result'];identifier=contract['run_id']
    if row['state']!='complete' or result['status']!='complete' or result['completed_steps']!=1000 or contract['budget']['maximum']!=1000:raise ValueError('Complete independently paid1000epoch fit required')
    authority=result['selected_checkpoint'];selected=result['selected'];folder=Path(output_root)/study['study_id']/identifier
    if authority['path']!='selected.pt' or authority['fit_id']!=identifier or not 0<authority['step']<=1000 or selected['step']!=authority['step']:raise ValueError('Positive own SELECT endpoint with exact fit authority required')
    path=folder/authority['path'];observed=receipt(root,path)
    if any(observed[k]!=authority[k] for k in ('bytes','sha256')):raise ValueError('Full selected checkpoint bytes changed')
    prefixes=result['prefixes'];wanted={200,500};diagnostics=[]
    if len(prefixes)!=2 or {p['cap_step'] for p in prefixes}!=wanted:raise ValueError('Both declared convergence prefixes required')
    for prefix in prefixes:
        cap=prefix['cap_step'];endpoint=prefix['selected'];certificate=folder/'prefixes'/prefix['prefix_id']/'certificate.json'
        if prefix['closed'] is not True or prefix['closure_step']!=cap or prefix['closure_reason']!='cap' or not 0<endpoint['step']<=cap or json.loads(certificate.read_text())!=prefix:raise ValueError('Closed positive prefix certificate required; diagnostics never serve heldout predictions')
        diagnostics.append({'cap':cap,'certificate':receipt(root,certificate),'checkpoint':receipt(root,folder/endpoint['path']),'selected_step':endpoint['step'],'heldout_predictions_computed':False})
    return path,observed,diagnostics


def export_bank(*,project_root,study,run,output_root,output_dir,registry,device):
    import torch
    c=run['config'];fits=c['fit_parents'];roles=('cal','evaluate')
    if len(fits) not in (1,4) or not set(fits)<=set(run['parents']):raise ValueError('Coherent bank or genuine four selected single learners required')
    ids={r:torch.load(confined(project_root,c['endpoint_inputs'][r+'_ids']),map_location='cpu',weights_only=True) for r in roles}
    if any(v.ndim!=1 or v.dtype!=torch.long or len(v)!=len(torch.unique(v)) for v in ids.values()) or torch.isin(ids['cal'],ids['evaluate']).any():raise ValueError('Disjoint ordered label-free endpoint IDs required')
    outputs={r:[] for r in roles};ancestry=[];contracts=[]
    for fit in fits:
        row=registry.run(study['study_id'],fit);path,authority,prefixes=selected_authority(project_root,output_root,study,row);actual=resolve_selected_config(study,row['contract'],registry)
        factory=load_factory(project_root,actual['adapter']);adapter=factory(torch=torch,project_root=project_root,study=study,run=actual,device=device,execution_context={'registry':registry,'endpoint_reconstruction':True})
        selected=restore_checkpoint(torch,path,adapter,expected_fit_id=fit);adapter.model.eval()
        with torch.no_grad():
            for role in roles:
                z=adapter.row_logits_fn(ids[role]);
                if z.ndim!=3 or z.shape[1:]!=(len(ids[role]),5) or not torch.isfinite(z).all() or len(fits)==4 and len(z)!=1:raise ValueError('Complete finite native five-label bank geometry required')
                outputs[role].append(z.detach().cpu())
        ancestry.append({'run_id':fit,'checkpoint':authority,'selected_step':selected['step'],'prefix_diagnostics':prefixes,'actual_model_seed':adapter.provenance['actual_model_seed'],'config_hash':fingerprint(actual['config'])});contracts.append(actual)
        del adapter,z;gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    if len(fits)==4:
        if len({r['actual_model_seed'] for r in ancestry})!=4 or sorted(r['body'] for r in contracts)!=[0,1,2,3]:raise ValueError('Four disjoint separately selected one-route learners required')
        excluded={'initialization_seed','resolved_selection','reservation_mib','estimated_seconds'};views=[{k:v for k,v in r['config'].items() if k not in excluded} for r in contracts]
        if any(v!=views[0] for v in views[1:]):raise ValueError('Genuine I4 source/objective/domain differ')
    files={}
    for role in roles:
        path=Path(output_dir)/(role+'_logits.pt');torch.save({'ids':ids[role],'logits':torch.cat(outputs[role])},path);files[role]=receipt(project_root,path)
    return {'status':'complete','files':files,'fusion':c['fusion'],'ancestry':ancestry,'independent_complete_learners':len(fits)==4,'CAL_targets_read':False,'evaluation_targets_read':False,'scientific_backbone_fits':0,'endpoint_model_forward_calls':sum(len(v) for v in outputs.values())}


def pooled_log_odds(torch,z,fusion):
    if z.ndim!=3 or z.shape[-1]!=5 or not torch.isfinite(z).all():raise ValueError('Finite [members,examples,5] logits required')
    if fusion=='mean_logits':return z.mean(0)
    if fusion=='mean_probabilities':
        F=torch.nn.functional
        return torch.logsumexp(F.logsigmoid(z),0)-torch.logsumexp(F.logsigmoid(-z),0)
    raise ValueError('Explicit Bernoulli serving law required')


def bank(torch,root,row):return torch.load(bound(root,row),map_location='cpu',weights_only=True)


def targets(torch,root,c,role,ids):
    value=torch.load(confined(root,c['endpoint_inputs'][role+'_targets']),map_location='cpu',weights_only=True)
    if set(value)!={'ids','y'} or not torch.equal(value['ids'],ids) or value['y'].shape!=(len(ids),5) or not ((value['y']==0)|(value['y']==1)).all():raise ValueError('Ordered complete hard Bernoulli role targets required')
    return value['y'].double()


def scalar_calibration(*,project_root,study,run,output_root,output_dir,registry,device):
    import torch
    c=run['config'];export=c['export_parent'];row=registry.run(study['study_id'],export)
    if row['state']!='complete' or export not in run['parents'] or row['result']['fusion']!=c['fusion']:raise ValueError('Completed label-free bank dependency required')
    if c['placement']!='after_serving_pool' or c['steps']!=500 or c['learning_rate']!=.01:raise ValueError('Fixed500step common scalar calibration required')
    b=bank(torch,project_root,row['result']['files']['cal']);y=targets(torch,project_root,c,'cal',b['ids']);scores=pooled_log_odds(torch,b['logits'].double(),c['fusion']).detach();p=torch.nn.Parameter(torch.zeros((),dtype=torch.float64));opt=torch.optim.Adam([p],lr=.01)
    for step in range(500):
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(scores/p.exp(),y)
        if not torch.isfinite(loss):raise ValueError('Finite scalar CAL BCE required')
        loss.backward();opt.step()
        with torch.no_grad():p.clamp_(-4,4)
    return {'status':'complete','temperature':float(p.detach().exp()),'scalar_optimizer_updates':500,'CAL_examples':len(y),'label_events':y.numel(),'export_parent':export,'fusion':c['fusion'],'placement':c['placement'],'evaluation_targets_read':False,'scientific_backbone_fits':0}


def evaluate_bank(*,project_root,study,run,output_root,output_dir,registry,device):
    import torch
    from sklearn.metrics import f1_score,roc_auc_score
    c=run['config'];ei,ci=c['export_parent'],c['calibration_parent']
    if not {ei,ci}<=set(run['parents']):raise ValueError('Explicit completed bank and CAL ancestry required')
    rows=[registry.run(study['study_id'],i) for i in (ei,ci)]
    if any(r['state']!='complete' for r in rows):raise ValueError('Completed endpoint parents required')
    exported,calibrated=(r['result'] for r in rows)
    if calibrated['export_parent']!=ei or any(r['fusion']!=c['fusion'] for r in (exported,calibrated)):raise ValueError('Same-bank serving and calibration required')
    b=bank(torch,project_root,exported['files']['evaluate']);y=targets(torch,project_root,c,'evaluate',b['ids']);scores=pooled_log_odds(torch,b['logits'].double(),c['fusion']);metrics={}
    for name,temp in (('raw',1.),('calibrated',calibrated['temperature'])):
        probability=(scores/temp).sigmoid();pred=probability>=.5;yn=y.numpy();pn=probability.numpy()
        metrics[name]={'BCE':float(torch.nn.functional.binary_cross_entropy_with_logits(scores/temp,y)),'brier_per_label':float((probability-y).square().mean()),'micro_F1':float(f1_score(yn,pred.numpy(),average='micro',zero_division=0)),'macro_F1':float(f1_score(yn,pred.numpy(),average='macro',zero_division=0)),'macro_ROCAUC':float(roc_auc_score(yn,pn,average='macro')),'examples':len(y),'label_events':y.numel()}
    members={'micro_F1':[float(f1_score(y.numpy(),(z>=0).numpy(),average='micro',zero_division=0)) for z in b['logits']],'all_members_wrong_label_event_fraction':float((b['logits'].ge(0)!=y.bool()[None]).all(0).double().mean())}
    return {'status':'complete','metrics':metrics,'member_diagnostics':members,'fusion':c['fusion'],'ancestry':exported['ancestry'],'evaluation_scope':study['data']['prior_exposure'],'statistical_inference':'whole paired seed/mask cohort; node rows are not independent replicates','scientific_backbone_fits':0}
