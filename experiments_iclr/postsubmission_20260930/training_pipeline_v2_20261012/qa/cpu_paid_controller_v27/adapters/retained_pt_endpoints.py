"""PT selected export with explicit paid ancestry context; shared CAL/eval law."""
import gc
from pathlib import Path

try:
    from .node_endpoints import receipt, scalar_calibration, evaluate_bank
    from ..engine import restore_checkpoint
    from ..run_study import load_factory, resolve_selected_config
    from ..spec import confined, fingerprint
except ImportError:
    from adapters.node_endpoints import receipt, scalar_calibration, evaluate_bank
    from engine import restore_checkpoint
    from run_study import load_factory, resolve_selected_config
    from spec import confined, fingerprint


def export_bank(*, project_root, study, run, output_root, output_dir, registry, device):
    import torch
    c = run['config']; fits = c['fit_parents']; output_root = Path(output_root)
    if len(fits) not in (1,4) or len(set(fits)) != len(fits) or not set(fits).issubset(run['parents']):
        raise ValueError('One selected bank or four separate complete own learners required')
    ids = {role:torch.load(confined(project_root,c['endpoint_inputs'][role+'_ids']),map_location='cpu',weights_only=True) for role in ('cal','evaluate')}
    if any(not torch.is_tensor(v) or v.ndim != 1 or v.dtype != torch.long for v in ids.values()) or torch.isin(ids['cal'],ids['evaluate']).any():
        raise ValueError('Separate ordered label-free int64 export IDs required')
    outputs = {role:[] for role in ids}; ancestry = []; configs = []
    for fit in fits:
        row = registry.run(study['study_id'],fit)
        if row['state'] != 'complete' or row['contract']['kind'] != 'model_fit': raise ValueError('Completed selected fit required')
        actual = resolve_selected_config(study,row['contract'],registry)
        adapter = load_factory(project_root,actual['adapter'])(torch=torch,project_root=project_root,study=study,run=actual,device=device,
                    execution_context={'registry':registry,'output_root':output_root})
        selected = row['result']['selected']; authority = row['result']['selected_checkpoint']; relative = Path(authority['path'])
        if relative.is_absolute() or '..' in relative.parts or authority['fit_id'] != fit or authority['step'] != selected['step']:
            raise ValueError('Exact safe selected checkpoint authority required')
        checkpoint = output_root/study['study_id']/fit/relative; observed = receipt(checkpoint,project_root)
        if any(observed[k] != authority[k] for k in ('bytes','sha256')): raise ValueError('Selected export checkpoint changed')
        restored = restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=fit)
        if restored['step'] != selected['step']: raise ValueError('Selected export step differs')
        adapter.model.eval()
        with torch.no_grad(): logits = adapter.logits_fn(adapter.model,False,selected['step'])
        if logits.ndim != 3 or not torch.isfinite(logits).all() or (len(fits)==4 and logits.shape[0]!=1):
            raise ValueError('Actual complete own logits; genuine I4 requires four single-view learners')
        for role,v in ids.items(): outputs[role].append(logits[:,v.to(logits.device)].detach().cpu())
        ancestry.append({'run_id':fit,'checkpoint':observed,'selected_step':selected['step'],
            'actual_model_seed':adapter.provenance['actual_model_seed'],'config_hash':fingerprint(actual['config']),
            'actual_paid_parent':adapter.provenance['actual_parent'],'actual_H0_hash':adapter.provenance['actual_H0_hash']})
        configs.append(actual['config']); del adapter,logits; gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    if len(fits)==4:
        if len(set(v['actual_model_seed'] for v in ancestry)) != 4: raise ValueError('I4 seed ancestry differs')
        excluded = {'initialization_seed','resolved_selection','reservation_mib','estimated_seconds','lifecycle'}
        normalized = [{k:v for k,v in config.items() if k not in excluded} for config in configs]
        if any(v != normalized[0] for v in normalized[1:]): raise ValueError('I4 architecture/objective/role choices differ')
    files = {}
    for role,values in outputs.items():
        path = output_dir/(role+'_logits.pt'); torch.save({'ids':ids[role],'logits':torch.cat(values)},path); files[role] = receipt(path,project_root)
    return {'status':'complete','files':files,'ancestry':ancestry,'fusion':c['fusion'],
            'independent_complete_learners':len(fits)==4,'CAL_targets_read':False,'evaluation_targets_read':False,'scientific_backbone_fits':0}
