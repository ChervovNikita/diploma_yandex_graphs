"""Prospective compact authorities, conservative DAG leases and audited exports.

Only the new frozen release schema opts in. No existing output is migrated.
CAL/EVALUATE remain global-gated. All removal candidates have independent byte
receipts; ordinary training and recovery do not understand the serving schema.
"""
import builtins
from contextlib import contextmanager,ExitStack
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from checkpoint_retention import receipt,confined,export_serving_checkpoint,restore_serving_checkpoint

SCHEMA='gnnm-prospective-checkpoint-lifetimes-v1'
AUDITED={
 '5571953847f054b01e1784efb0cb35330794af0f3f8ea1cbee24a0afda5c121e':'node',
 '20144e4ca7cdc9304cf12becc1984b974e1c7ed99371ab2bc19c5ad86abbd981':'sehgnn',
 'f53a108553b1058af706270675ccab99802e4c0d15d1a0e191202b5f2b7a46a6':'nsd'}


def exporter_binding(root,run):
    if run['kind']!='ensemble_assembly' or run['config'].get('stage')!='bank_export':return None
    file,_,function=run['adapter'].partition(':')
    if function!='export_bank':return None
    binding=receipt(confined(root,file),root);family=AUDITED.get(binding['sha256'])
    return None if family is None else {'family':family,'factory':run['adapter'],'file':binding}


def declare_policy(root,study):
    exports={r['run_id']:b for r in study['runs'] if (b:=exporter_binding(root,r)) is not None}
    study['release']['checkpoint_retention_policy']={'schema':SCHEMA,'compact_selected_for_every_engine_fit':True,
        'audited_label_free_exports':exports,'unsupported_exporters_keep_full_states':True,
        'full_parent_and_recovery_authorities_required':True,'existing_output_migration':False}
    from unused_search_retention import declare
    study['release']['checkpoint_retention_policy']['unused_search_binary_release']=declare(root,study)
    return study


def enabled(study):
    policy=study['release'].get('checkpoint_retention_policy')
    if policy is None:return False
    if policy.get('schema')!=SCHEMA or policy.get('existing_output_migration') is not False:
        raise ValueError('Only a newly frozen prospective retention policy is supported')
    return True


def early_export_key(study,run):
    return study['study_id']+'/'+run['run_id'] if enabled(study) and run['run_id'] in study['release']['checkpoint_retention_policy']['audited_label_free_exports'] else None


def validate_plan_policy(root,plan,runs,studies):
    keys=set(plan.get('label_free_export_run_keys',[]))
    for key in keys:
        if key not in runs:raise ValueError('Unknown prospective export exception')
        item=runs[key];study=studies[item['study']]['spec'];run=item['run']
        if early_export_key(study,run)!=key or exporter_binding(root,run)!=study['release']['checkpoint_retention_policy']['audited_label_free_exports'][run['run_id']]:
            raise ValueError('Only the exact hash-audited label-free export can bypass the comparative phase barrier')


def close_fit(root,study,run,folder,result):
    if not enabled(study):return result
    import torch,engine
    folder=confined(root,folder)
    if result['status']!='complete':raise ValueError('Only complete selected trajectories compact')
    selected=folder/'selected.pt'
    source=receipt(selected,root)
    if any(source[k]!=result['selected_checkpoint'][k] for k in ('bytes','sha256')):raise ValueError('Closed selected authority changed before compaction')
    manifest=export_serving_checkpoint(torch,engine,selected,folder/'selected_serving.pt',project_root=root,
        expected_source_receipt=source,expected_fit_id=run['run_id'])
    diagnostic=[]
    for prefix in result.get('prefixes',[]):
        if prefix.get('selected') is not None:
            p=folder/prefix['selected']['path']
            diagnostic.append({'prefix_id':prefix['prefix_id'],'selected_step':prefix['selected']['step'],
                'full_checkpoint':receipt(p,root),'certificate':receipt(folder/'prefixes'/prefix['prefix_id']/'certificate.json',root),
                'diagnostic_only':bool(run['config'].get('prefixes_are_not_additional_heldout_endpoints'))})
    full=[receipt(p,root) for p in [selected,folder/'final_training.pt',folder/'latest_trajectory.pt'] if p.is_file()]
    full += [d['full_checkpoint'] for d in diagnostic]
    record={'schema':SCHEMA,'fit_id':run['run_id'],'source_full_selected':source,'serving':manifest,
        'prefixes':diagnostic,'full_state_receipts':full,'history_source_result':receipt(folder/'result.json',root),
        'source_policy':study['release']['checkpoint_retention_policy'],'source_modified':False,
        'existing_output_migration':False,'released':False}
    path=folder/'RETENTION_AUTHORITY.json'
    if path.exists():raise FileExistsError('Retained authority already exists; inspect owner/retry history')
    path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    value=copy.deepcopy(result);value['retention_authority']=receipt(path,root)
    value['selected_serving_checkpoint']=manifest['serving_checkpoint']
    return value


def authority(root,registry,study,identifier):
    row=registry.run(study['study_id'],identifier)
    if row['state']!='complete' or row['contract']['kind']!='model_fit':raise ValueError('Complete fitted parent required for serving')
    binding=row['result'].get('retention_authority')
    if binding is None:raise ValueError('This fit has no prospective serving authority')
    path=confined(root,binding['path'])
    if receipt(path,root)!=binding:raise ValueError('Immutable retention authority changed')
    value=json.loads(path.read_text())
    if value['schema']!=SCHEMA or value['fit_id']!=identifier:raise ValueError('Retention fit/schema differs')
    selected=row['result']['selected_checkpoint']
    if any(value['source_full_selected'][k]!=selected[k] for k in ('bytes','sha256')):raise ValueError('Serving authority did not descend from the actual selected state')
    compact=value['serving']['serving_checkpoint']
    if receipt(confined(root,compact['path']),root)!=compact:raise ValueError('Selected serving checkpoint changed')
    return row,value


def consumer_leases(study,registry,identifier):
    """Immutable within-study DAG, not timestamps or inferred stopped jobs."""
    runs={r['run_id']:r for r in study['runs']};children={k:[] for k in runs}
    for r in runs.values():
        for p in r['parents']:children[p].append(r['run_id'])
    seen=set();todo=list(children[identifier])
    while todo:
        k=todo.pop()
        if k in seen:continue
        seen.add(k);todo.extend(children[k])
    registered=registry.consumer_records(study['study_id'],identifier)
    if set(registered['transitive_ids'])!=seen or set(registered['direct_ids'])!=set(children[identifier]):raise ValueError('Retention DAG differs from registered immutable study')
    states=registered['states']
    paid_children=[k for k in children[identifier] if runs[k]['kind']=='model_fit']
    # A source parent stays full through every transitive fit/endpoint consumer
    # and open failure/retry. Unknown endpoint families remain conservative.
    if paid_children:
        outstanding=sorted(k for k in seen if states[k]!='complete')
    else:
        direct=children[identifier]
        supported=study['release']['checkpoint_retention_policy']['audited_label_free_exports']
        outstanding=sorted(k for k in direct if states[k]!='complete' or (runs[k]['kind']=='ensemble_assembly' and k not in supported))
    if runs[identifier]['config'].get('stage')=='search':
        hpo=[r['run_id'] for r in study['runs'] if r['config'].get('stage') in ('search','search_selection')]
        outstanding+=['HPO:'+k for k in hpo if registry.run(study['study_id'],k)['state']!='complete']
    return {'fit_id':identifier,'transitive_consumer_ids':sorted(seen),'outstanding_full_state_consumers':sorted(set(outstanding)),
        'paid_source_parent':bool(paid_children),'states':states,'closed':not outstanding}


@contextmanager
def inference_export(root,study,run,registry,output_root,factory):
    """Scoped endpoint-only authority selection; source/paid parents stay full."""
    import torch,engine
    binding=exporter_binding(root,run)
    expected=study['release']['checkpoint_retention_policy']['audited_label_free_exports'].get(run['run_id'])
    if binding!=expected:raise ValueError('Endpoint source is not the frozen audited exporter')
    fits=run['config']['fit_parents'];selected={}
    for identifier in fits:
        row,value=authority(root,registry,study,identifier)
        selected[(Path(output_root)/study['study_id']/identifier/'selected.pt').resolve()]=(identifier,row,value)
    module=sys.modules[factory.__module__];old_restore=engine.restore_checkpoint
    def restore(t,path,adapter,*,expected_fit_id=None,verify=True):
        item=selected.get(Path(path).resolve())
        if item is None:return old_restore(t,path,adapter,expected_fit_id=expected_fit_id,verify=verify)
        identifier,row,value=item
        if expected_fit_id!=identifier:raise ValueError('Selected inference authority belongs to another fit')
        b=value['serving']['serving_checkpoint']
        restore_serving_checkpoint(t,engine,confined(root,b['path']),adapter,project_root=root,expected_receipt=b,expected_fit_id=identifier)
        return {'fit_id':identifier,'step':row['result']['selected']['step'],
            'validation':row['result']['selected']['validation'],'decision':{},'authority_type':'selected_serving_inference_only'}
    old_receipt=getattr(module,'receipt',None)
    def selected_receipt(*args):
        path=next((Path(a) for a in args if isinstance(a,(str,Path)) and Path(a).resolve() in selected),None)
        if path is not None:return selected[path.resolve()][2]['serving']['serving_checkpoint']
        return old_receipt(*args)
    def sehgnn_selected(project_root,output,unused_study,row):
        result=row['result'];contract=row['contract']
        if (row['state']!='complete' or result['status']!='complete' or result['completed_steps']!=1000
                or contract['budget']['maximum']!=1000 or result['selected_checkpoint']['path']!='selected.pt'
                or not 0<result['selected']['step']<=1000 or result['selected_checkpoint']['step']!=result['selected']['step']):
            raise ValueError('Complete independently selected native1000-epoch authority required')
        identifier=row['contract']['run_id'];item=selected[(Path(output)/study['study_id']/identifier/'selected.pt').resolve()]
        value=item[2];diagnostics=[]
        for p in value['prefixes']:
            cp=confined(root,p['certificate']['path'])
            original=next(q for q in row['result']['prefixes'] if q['prefix_id']==p['prefix_id'])
            if (receipt(cp,root)!=p['certificate'] or json.loads(cp.read_text())!=original
                    or original['cap_step'] not in (200,500) or original['closed'] is not True
                    or original['closure_step']!=original['cap_step'] or original['closure_reason']!='cap'
                    or not 0<p['selected_step']<=original['cap_step']):raise ValueError('Source diagnostic prefix certificate changed')
            diagnostics.append({'cap':next(q['cap_step'] for q in row['result']['prefixes'] if q['prefix_id']==p['prefix_id']),
                'certificate':p['certificate'],'checkpoint':p['full_checkpoint'],'selected_step':p['selected_step'],
                'checkpoint_authority':'preserved original full-state byte receipt; diagnostic-only; serving does not read this full state',
                'heldout_predictions_computed':False})
        return Path(output)/study['study_id']/identifier/'selected.pt',value['serving']['serving_checkpoint'],diagnostics
    forbidden={confined(root,p).resolve() for k,p in run['config']['endpoint_inputs'].items() if k.endswith('_targets')}
    original_load=torch.load;original_open=builtins.open;original_path_open=Path.open
    def guard_path(value):
        if isinstance(value,(str,Path)) and Path(value).resolve() in forbidden:raise ValueError('Label-free export cannot access CAL/EVALUATE targets')
    def guarded_load(value,*a,**kw):guard_path(value);return original_load(value,*a,**kw)
    def guarded_open(value,*a,**kw):guard_path(value);return original_open(value,*a,**kw)
    def guarded_path_open(value,*a,**kw):guard_path(value);return original_path_open(value,*a,**kw)
    with ExitStack() as stack:
        stack.enter_context(patch.object(engine,'restore_checkpoint',restore))
        if hasattr(module,'restore_checkpoint'):stack.enter_context(patch.object(module,'restore_checkpoint',restore))
        if old_receipt is not None:stack.enter_context(patch.object(module,'receipt',selected_receipt))
        if binding['family']=='sehgnn':stack.enter_context(patch.object(module,'selected_authority',sehgnn_selected))
        stack.enter_context(patch.object(torch,'load',guarded_load));stack.enter_context(patch.object(builtins,'open',guarded_open));stack.enter_context(patch.object(Path,'open',guarded_path_open))
        yield


def release_candidates(root,study,registry,identifier,*,owner_terminal):
    """Bound prospective actions; callers prove child termination before removal."""
    if not enabled(study):return None
    row,value=authority(root,registry,study,identifier);leases=consumer_leases(study,registry,identifier)
    full=value['full_state_receipts'];selected=value['source_full_selected']
    completed_releases=[]
    folder=confined(root,row['result']['retention_authority']['path']).parent
    for p in folder.glob('RETENTION_RELEASE_*_COMPLETE.json'):
        completed=json.loads(p.read_text())
        if completed.get('schema')!=SCHEMA:raise ValueError('Unknown checkpoint release completion schema')
        intent=completed['intent'];ip=confined(root,intent['path'])
        if receipt(ip,root)!=intent:raise ValueError('Checkpoint release intent changed')
        completed_releases.extend(completed['released'])
    candidates=[]
    for r in full:
        name=Path(r['path']).name
        primary=r==selected or (name=='final_training.pt' and leases['paid_source_parent'])
        if primary and not leases['closed']:continue
        if name=='selected.pt' and '/prefixes/' in r['path']:
            p=next(p for p in value['prefixes'] if p['full_checkpoint']==r)
            if not p['diagnostic_only']:continue  # Unsupported served prefixes retain full authority.
        if not owner_terminal:continue
        path=confined(root,r['path'])
        if path.is_file():
            if receipt(path,root)!=r:raise ValueError('Complete-state release candidate changed')
            candidates.append(r)
        elif r not in completed_releases:
            raise ValueError('Checkpoint disappeared without a closed release receipt; inspect interrupted or foreign mutation')
    return {'schema':SCHEMA,'fit_id':identifier,'authority':row['result']['retention_authority'],
        'owner_terminal':owner_terminal,'leases':leases,'release_candidates':candidates,
        'preserved_serving':value['serving']['serving_checkpoint'],'scientific_history_preserved':True,'delete_executed':False}


def sweep_closed_owner(root,study,registry,trigger_id):
    """Future supervisor hook, only after its exact child has exited.

    Outputs without the new prospective authority are never migration targets.
    Immutable intent/completion receipts preserve every released byte authority.
    """
    if not enabled(study):return []
    runs={r['run_id']:r for r in study['runs']};affected=set();todo=[trigger_id]
    while todo:
        k=todo.pop()
        if k in affected:continue
        affected.add(k);todo.extend(runs[k]['parents'])
    reports=[]
    for identifier in sorted(affected):
        if runs[identifier]['kind']!='model_fit':continue
        row=registry.run(study['study_id'],identifier)
        if row['state']!='complete' or not row['result'].get('retention_authority'):continue
        # The triggering child is known exited by the supervisor. Earlier
        # complete fits are verified by Linux process identity, never status alone.
        from run_study import terminal_owner
        termination=terminal_owner(row['owner'])
        from unused_search_retention import sweep_one
        handled,bindings=sweep_one(root,study,registry,identifier,owner_terminal=termination['previous_owner_terminal'])
        reports.extend(bindings)
        if handled:continue  # Exact tombstones validate without requiring deleted serving bytes.
        plan=release_candidates(root,study,registry,identifier,owner_terminal=termination['previous_owner_terminal'])
        if not plan['release_candidates']:continue
        folder=confined(root,row['result']['retention_authority']['path']).parent
        index=0
        while (folder/('RETENTION_RELEASE_'+str(index)+'.json')).exists():index+=1
        intent=folder/('RETENTION_RELEASE_'+str(index)+'.json')
        plan['owner_termination']=termination;plan['existing_output_migration']=False
        intent.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
        removed=[]
        for r in plan['release_candidates']:
            p=confined(root,r['path'])
            if receipt(p,root)!=r:raise ValueError('Release candidate changed after intent closure')
            p.unlink();removed.append(r)
        done=folder/('RETENTION_RELEASE_'+str(index)+'_COMPLETE.json')
        done.write_text(json.dumps({'schema':SCHEMA,'intent':receipt(intent,root),'released':removed,
            'preserved_serving':plan['preserved_serving'],'scientific_history_preserved':True,
            'existing_output_migration':False},indent=2,sort_keys=True)+'\n')
        binding=receipt(done,root);registry.event(study['study_id'],identifier,'prospective_full_state_release',binding);reports.append(binding)
    return reports
