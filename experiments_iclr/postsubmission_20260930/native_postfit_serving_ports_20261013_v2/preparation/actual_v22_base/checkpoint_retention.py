"""Prospective serving authority + release-plan draft, never delete any file.

This module is not installed into a live controller. Training/recovery always
requires a complete resumable checkpoint. Serving export creates a separate
immutable authority and keeps the input untouched.
"""
import copy
import hashlib
import json
from pathlib import Path

SCHEMA='gnnm-prospective-selected-serving-checkpoint-v1'
KEYS=('fit_id','step','validation','decision','model_state','module_modes','requires_grad','adapter_state')


def confined(root,path):
    root=Path(root).resolve();path=Path(path)
    path=path if path.is_absolute() else root/path
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Checkpoint operation must stay inside the explicit project root')
    return path


def receipt(path,project_root=None):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return {'path':str(path) if project_root is None else str(path.resolve().relative_to(Path(project_root).resolve())),
        'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def export_serving_checkpoint(torch,engine,source,target,*,project_root,expected_source_receipt,expected_fit_id):
    """Lossless selected inference state without optimizer/recovery snapshots."""
    source,target=confined(project_root,source),confined(project_root,target)
    if target.exists() or target.with_suffix(target.suffix+'.receipt.json').exists():
        raise FileExistsError('Serving authority is immutable; choose a fresh target')
    if receipt(source,project_root)!=expected_source_receipt:
        raise ValueError('Exact independently bound selected authority required')
    payload=torch.load(source,map_location='cpu',weights_only=False)
    if payload.get('schema_version')!=1 or payload.get('fit_id')!=expected_fit_id:
        raise ValueError('Only a complete checkpoint from this paid fit can supply serving state')
    if any(k not in payload for k in KEYS) or 'optimizer_state' not in payload:
        raise ValueError('Complete native source checkpoint required')
    result={'serving_schema':SCHEMA,'source_full_checkpoint':copy.deepcopy(expected_source_receipt),
        'resumable':False,**{k:engine._clone_tree(torch,payload[k]) for k in KEYS}}
    engine._finite_tree(torch,result,'serving export')
    target.parent.mkdir(parents=True,exist_ok=True)
    engine._atomic_checkpoint(torch,target,result)
    restored=torch.load(target,map_location='cpu',weights_only=False)
    if not engine._equal_tree(torch,result,restored) or receipt(source,project_root)!=expected_source_receipt:
        raise ValueError('Serving serialization/source preservation differs')
    manifest={'schema':SCHEMA,'source_full_checkpoint':expected_source_receipt,
        'serving_checkpoint':receipt(target,project_root),'exact_preserved_fields':list(KEYS),
        'optimizer_state_omitted':True,'recovery_snapshot_bytes_omitted':True,
        'source_modified':False,'full_checkpoint_released':False,'resumable':False}
    mp=target.with_suffix(target.suffix+'.receipt.json')
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def restore_serving_checkpoint(torch,engine,path,adapter,*,project_root,expected_receipt,expected_fit_id):
    """Restore exact inference/model/adapter state; reject training authority."""
    path=confined(project_root,path)
    if receipt(path,project_root)!=expected_receipt:raise ValueError('Immutable serving authority changed')
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value.get('serving_schema')!=SCHEMA or value.get('fit_id')!=expected_fit_id
            or value.get('resumable') is not False or 'optimizer_state' in value or 'recovery' in value):
        raise ValueError('Serving checkpoints are explicit inference authorities only')
    modules=dict(adapter.model.named_modules());parameters=dict(adapter.model.named_parameters())
    if set(modules)!=set(value['module_modes']) or set(parameters)!=set(value['requires_grad']):
        raise ValueError('Serving topology differs')
    optimizer_before=engine._clone_tree(torch,adapter.optimizer.state_dict())
    adapter.model.load_state_dict(value['model_state'],strict=True)
    for name,flag in value['requires_grad'].items():parameters[name].requires_grad_(flag)
    for p in adapter.model.parameters():p.grad=None
    for name,flag in value['module_modes'].items():modules[name].training=flag
    adapter.restore_state(engine._clone_tree(torch,value['adapter_state']))
    actual={'model_state':engine._clone_tree(torch,adapter.model.state_dict()),
        'module_modes':{n:bool(m.training) for n,m in adapter.model.named_modules()},
        'requires_grad':{n:bool(p.requires_grad) for n,p in adapter.model.named_parameters()},
        'adapter_state':engine._clone_tree(torch,adapter.capture_state())}
    if not engine._equal_tree(torch,actual,{k:value[k] for k in actual}):
        raise ValueError('Serving restore did not reproduce exact selected state')
    if not engine._equal_tree(torch,optimizer_before,adapter.optimizer.state_dict()):
        raise ValueError('Serving restore changed unrelated optimizer authority')
    return {'fit_id':value['fit_id'],'step':value['step'],'source_full_checkpoint':value['source_full_checkpoint'],
        'purpose':'inference_only','resumable':False}


def label_free_export_ready(record):
    """Prospective gate proposal: labels stay under the global endpoint gate."""
    return (record.get('phase')=='label_free_bank_export'
        and record.get('all_fit_parents_complete') is True
        and record.get('selected_authorities_verified') is True
        and record.get('reads_CAL_or_EVALUATE_targets') is False
        and record.get('emits_predictive_scores') is False
        and record.get('registered_endpoint_ids_only') is True)


def release_plan(record):
    """Return explicit bounded lifetime decisions; performs no filesystem IO."""
    required=('fit_id','purpose','stage','fit_status','owner_terminal','history_closed','serving_equivalence_verified',
        'HPO_group_closed','SELECT_history_saved','label_free_bank_closed','unfinished_state_dependents',
        'continuation_admissions_closed','full_checkpoint_receipts','serving_receipt')
    if any(k not in record for k in required):raise ValueError('Explicit complete retention lifecycle required')
    if record['purpose'] not in ('prospective_scientific','synthetic_engineering'):
        raise ValueError('Historical evidence cannot be released by this prospective proposal')
    blockers=[]
    if record['fit_status']!='complete' or record['owner_terminal'] is not True: blockers.append('fit_or_owner_unfinished')
    if record['history_closed'] is not True: blockers.append('history_not_closed')
    if record['serving_equivalence_verified'] is not True: blockers.append('selected_serving_not_equivalent')
    if record.get('stage')=='search' and (record['HPO_group_closed'] is not True or record['SELECT_history_saved'] is not True):blockers.append('whole_HPO_or_SELECT_history_open')
    if record.get('stage')!='search' and record['label_free_bank_closed'] is not True:blockers.append('selected_label_free_bank_open')
    if record['unfinished_state_dependents']:blockers.append('paid_state_dependents_open')
    if record['continuation_admissions_closed'] is not True:blockers.append('retry_or_continuation_admission_open')
    if not isinstance(record['full_checkpoint_receipts'],list) or not record['full_checkpoint_receipts']:
        raise ValueError('Individually bound complete checkpoint release candidates required')
    if record['serving_receipt'] in record['full_checkpoint_receipts']:
        raise ValueError('Preserved serving authority cannot be a release candidate')
    return {'schema':'gnnm-prospective-checkpoint-lifetime-draft-v1','fit_id':record['fit_id'],
        'eligible':not blockers,'blockers':blockers,'full_state_candidates':copy.deepcopy(record['full_checkpoint_receipts']),
        'preserved_selected_serving':copy.deepcopy(record['serving_receipt']),
        'preserve':['run_spec','source_and_data_custody','train_and_SELECT_curves','all_costs','result_and_failure_history',
            'selected_receipt_and_exact_serving_state','member_prediction_bank_and_export_receipt'],
        'delete_executed':False,'source_policy_installed':False,'GO':False}
