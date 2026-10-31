"""One paid own-native active-head capture and coherent selected-label export."""
import gc
from pathlib import Path

try:
    from .native_polynormer_qk import bound,load_module
    from .node_endpoints import receipt,scalar_calibration,evaluate_bank
    from ..engine import restore_checkpoint
    from ..run_study import load_factory,resolve_selected_config,paid_parent_checkpoints
    from ..spec import confined
except ImportError:
    from adapters.native_polynormer_qk import bound,load_module
    from adapters.node_endpoints import receipt,scalar_calibration,evaluate_bank
    from engine import restore_checkpoint
    from run_study import load_factory,resolve_selected_config,paid_parent_checkpoints
    from spec import confined


def capture_native(*,project_root,study,run,output_root,output_dir,registry,device):
    import torch
    parent=run['config']['native_fit_parent'];row=registry.run(study['study_id'],parent)
    if row['state']!='complete' or row['result']['status']!='complete' or row['result']['completed_steps']!=row['contract']['budget']['maximum']:
        raise ValueError('New complete paid native acquisition required')
    actual=resolve_selected_config(study,row['contract'],registry)
    if actual['config']['polynormer_qk_arm']!='bare_native_single':raise ValueError('One native own-selected bare Polynormer, not SAGE or a shared four-route bank')
    adapter=load_factory(project_root,actual['adapter'])(torch=torch,project_root=project_root,study=study,run=actual,device=device)
    authority=row['result']['selected_checkpoint'];path=Path(output_root)/study['study_id']/parent/authority['path']
    observed=receipt(path,project_root)
    if authority['path']!='selected.pt' or any(authority[k]!=observed[k] for k in ('bytes','sha256')) or authority['fit_id']!=parent:raise ValueError('Own-selected checkpoint byte authority differs')
    selected=restore_checkpoint(torch,path,adapter,expected_fit_id=parent)
    adapter.model.eval()
    for p in adapter.model.parameters():p.requires_grad_(False)
    source=load_module(project_root,run['config']['capture_source'],'label_native_active_capture')
    N=run['config']['nodes'];C=run['config']['classes'];session=adapter.source_session
    capture=source.NativeSingleCapture(session,N,512,C);before=adapter.capture_state()
    try:
        capture.begin('SERVE',1)
        with torch.no_grad():unused=adapter.logits_fn(adapter.model,False,selected['step']);H,base=capture.finish();del unused
    finally:capture.close()
    # Eval capture contains no dropout draw; exact source streams must stay live.
    from engine import _equal_tree
    if not _equal_tree(torch,before,adapter.capture_state()):raise ValueError('One frozen capture changed native RNG/source state')
    cache={'schema':'paid-native-active-head-capture-v1','native_fit_id':parent,'selected_checkpoint':authority,
           'native_selected_step':selected['step'],'native_logical_steps':session.steps,'native_global':bool(session.model.models[0].body._global),
           'H':H.detach().cpu().clone(),'native_logits':base.detach().cpu().clone(),'source_recipe':actual['config'],
           'whole_native_budget_paid':row['result']['completed_steps'],'native_acquisition_cost_is_not_selected_step':True}
    path=output_dir/'NATIVE_CAPTURE.pt';torch.save(cache,path)
    return {'status':'complete','cache':receipt(path,project_root),'native_fit_parent':parent,'native_capture_calls':1,
            'new_native_optimizer_updates':0,'native_acquisition_was_paid':True,'SELECT_targets_decoded_by_constructor':True,
            'SELECT_metrics_computed':False,'CAL_evaluate_targets_read':False,'historical_native_state_reused':False}


def export_bank(*,project_root,study,run,output_root,output_dir,registry,device):
    import torch
    fits=run['config']['fit_parents']
    if len(fits)!=1:raise ValueError('One coherent source label bank required; native I4 uses its own endpoint')
    fit=fits[0];row=registry.run(study['study_id'],fit)
    if row['state']!='complete' or row['result']['completed_steps']!=row['contract']['budget']['maximum']:raise ValueError('Complete paid label-stage fit required')
    actual=resolve_selected_config(study,row['contract'],registry)
    context={'registry':registry,'output_root':Path(output_root),'paid_parent_checkpoints':paid_parent_checkpoints(project_root,study,actual,registry,output_root)}
    adapter=load_factory(project_root,actual['adapter'])(torch=torch,project_root=project_root,study=study,run=actual,device=device,execution_context=context)
    authority=row['result']['selected_checkpoint'];path=Path(output_root)/study['study_id']/fit/authority['path'];observed=receipt(path,project_root)
    if any(observed[k]!=authority[k] for k in ('bytes','sha256')) or authority['fit_id']!=fit:raise ValueError('Label selected state byte custody differs')
    selected=restore_checkpoint(torch,path,adapter,expected_fit_id=fit)
    adapter.model.eval()
    with torch.no_grad():logits=adapter.logits_fn(adapter.model,False,selected['step'])
    files={}
    for role in ('cal','evaluate'):
        ids=torch.load(confined(project_root,run['config']['endpoint_inputs'][role+'_ids']),map_location='cpu',weights_only=True)
        path=output_dir/(role+'_logits.pt');torch.save({'ids':ids,'logits':logits[:,ids.to(logits.device)].detach().cpu()},path);files[role]=receipt(path,project_root)
    result={'status':'complete','files':files,'fusion':'mean_probabilities','ancestry':[{'run_id':fit,'checkpoint':observed,
                'selected_step':selected['step'],'actual_model_seed':adapter.provenance['actual_model_seed'],
                'paid_native_fit_id':adapter.provenance['paid_native_fit_id'],'paid_native_capture':adapter.provenance['paid_native_capture']}],
            'native_optimizer_updates':0,'native_forwards':0,'actual_endpoint_target_reads':False}
    del adapter,logits;gc.collect()
    return result
