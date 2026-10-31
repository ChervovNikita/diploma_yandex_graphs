"""Own-selected Bayesian posterior banks with complete draw-count evidence."""
import gc
import hashlib
from pathlib import Path
try:
    from ..run_study import load_factory
except ImportError:
    from run_study import load_factory
from .label_family_endpoints import scalar_calibration, evaluate_bank


def receipt(root,path):
    raw=path.read_bytes()
    return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def export_bank(*,project_root,study,run,output_root,output_dir,registry,device):
    import torch
    from engine import restore_checkpoint
    from bsnn_stopping import validate_bsnn_run
    root=Path(project_root).resolve();c=run['config'];inputs=c['endpoint_inputs']
    ids={role:torch.load(root/inputs[role+'_ids'],map_location='cpu',weights_only=True) for role in ('cal','evaluate')}
    views={role:[] for role in ids};draws={role:[] for role in ids};ancestry=[]
    count=c['posterior_draws_per_route']; mc16=c['posterior_count_control']
    if count not in (4,16) or mc16!=(count==16) or mc16 and len(c['fit_parents'])!=1:
        raise ValueError('Declared sourceMC4 or same-paid-M1 MC16 control only')
    for identifier in c['fit_parents']:
        row=registry.run(study['study_id'],identifier);fit=row['contract'];validate_bsnn_run(fit)
        result=row['result'];completed=result.get('completed_steps')
        if (row['state']!='complete' or not 201<=completed<=500
                or result.get('completion_reason') not in ('native_patience200','native_cap500')
                or result.get('actual_completed_optimizer_updates')!=completed
                or result.get('trajectory_operation_counts',{}).get('optimizer_updates')!=completed):
            raise ValueError('Complete own-selected native source stop/cap authority required')
        adapter=load_factory(root,fit['adapter'])(torch=torch,project_root=root,study=study,run=fit,device=device)
        checkpoint=Path(output_root)/study['study_id']/identifier/'selected.pt'
        restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=identifier);adapter.model.eval();step=result['selected']['step']
        with torch.no_grad():
            raw=adapter.serving_draws(step,count)
            members=adapter.members
            if raw.ndim!=3 or raw.shape[0]!=members*count or not torch.isfinite(raw).all():
                raise ValueError('Every source posterior draw must be returned, not copied')
            if len(c['fit_parents'])==4 and members!=1 or mc16 and members!=1:
                raise ValueError('Genuine I4 and MC16 require own native M1 posterior source')
            if adapter.fusion=='mean_probabilities':
                pooled=raw.exp().reshape(members,count,*raw.shape[1:]).mean(1).log()
            else:
                pooled=raw.reshape(members,count,*raw.shape[1:]).mean(1)
            for role in ids:
                views[role].append(pooled[:,ids[role].to(device)].cpu())
                draws[role].append(raw[:,ids[role].to(device)].cpu())
        ancestry.append({'fit_id':identifier,'checkpoint':receipt(root,checkpoint),'selected_step':step,
                         'completed_native_optimizer_maps':completed,'actual_model_seed':adapter.seed,
                         'source_selection_posterior_draws':4*members,'served_posterior_draws':count*members,
                         'native_posterior_layers':adapter.native_args['layers'],
                         'extra_forward_draws_without_reselection':12 if mc16 else 0,
                         'source_fit_retrained':False,'own_selected':True})
        del adapter;gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    if len(c['fit_parents'])==4 and len({a['actual_model_seed'] for a in ancestry})!=4:
        raise ValueError('Genuine I4 needs four different independently selected native bodies')
    files={};posterior_evidence={}
    for role in views:
        primary=Path(output_dir)/(role+'_logits.pt')
        torch.save({'ids':ids[role],'logits':torch.cat(views[role]),'source_FP32_posterior_pool_per_body_or_route':True},primary)
        evidence=Path(output_dir)/(role+'_all_posterior_draws.pt')
        torch.save({'ids':ids[role],'logits':torch.cat(draws[role]),'ancestry':ancestry,
                    'source_draws_are_samples_not_independent_parameter_members':True},evidence)
        files[role]=receipt(root,primary);posterior_evidence[role]=receipt(root,evidence)
    return {'status':'complete','files':files,'posterior_draw_evidence':posterior_evidence,'ancestry':ancestry,
            'fusion':c['fusion'],'source_posterior_full_forwards':sum(a['served_posterior_draws'] for a in ancestry),
            'source_posterior_layer_samples':sum(a['served_posterior_draws']*a['native_posterior_layers'] for a in ancestry),
            'scientific_backbone_fits':0,'MC16_reselection':False,'CAL_targets_read':False,'evaluation_targets_read':False,
            'endpoint_constructor_reads_own_SELECT_for_source_factory':True,'early_label_free_export_audited':False}
