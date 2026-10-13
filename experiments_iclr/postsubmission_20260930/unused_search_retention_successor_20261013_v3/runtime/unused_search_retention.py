"""New-output-only release of search states with metadata-only direct users.

This changes no fit, source learning law, endpoint, selection rule or result.
The source pins cover the exact reviewed selector entry functions. Chosen
search configurations are refit from fresh initialization; their binary states
have no direct paid consumer. Failure/history/curves and metadata remain.
"""
import copy
import hashlib
import json
import os
from pathlib import Path

from checkpoint_retention import confined,receipt

SCHEMA='gnnm-prospective-unused-search-binary-release-v1'
SELECTORS={
 'a0c17d9584f0d9496f37c236ca97c2b859bae023bf419635ceccf009af9588dd':'select_configuration',
 'ccdc568ad63b22eac2945b1a6a82bcf3869382df5027ad276fafbf45ece32f00':'select_configuration',
 '499da68f8d7ab06a06beed6e7281de271bc9d95d0fb7e67b301c66122ac83a3e':'choose_lambda',
 '1cf551b17409cc84faa23082c00450642ec00ec568d254fe2ed27e1212cedd50':'choose_lambda',
}

def selector_binding(root,run):
    if run['kind']!='selection' or run['config'].get('stage')!='search_selection':return None
    file,_,function=run['adapter'].partition(':');bound=receipt(confined(root,file),root)
    if SELECTORS.get(bound['sha256'])!=function:return None
    return {'factory':run['adapter'],'source':bound,'reads_binary_states':False,
            'reads_TRAIN_SELECT_registry_metadata_only':True}

def declare(root,study):
    runs={r['run_id']:r for r in study['runs']};children={k:[] for k in runs}
    for r in runs.values():
        for p in r['parents']:children[p].append(r['run_id'])
    selectors={}
    for r in runs.values():
        binding=selector_binding(root,r)
        if binding is None:continue
        parents=r['parents']
        if not parents or any(runs[p]['kind']!='model_fit' or runs[p]['config'].get('stage')!='search' for p in parents):continue
        # Bind the real candidate roster as well as the dependency roster.
        declared=[p for cell in r['config']['configurations'] for p in cell['run_ids']]
        if len(declared)!=len(set(declared)) or set(declared)!=set(parents):continue
        selectors[r['run_id']]={'binding':binding,'fit_cohort_ids':sorted(parents)}
    fits={}
    for r in runs.values():
        key=r['run_id'];direct=children[key]
        if r['kind']=='model_fit' and r['config'].get('stage')=='search' and direct and all(k in selectors for k in direct):
            fits[key]={'direct_metadata_selection_ids':sorted(direct)}
    from protected_search_roster import PROTECTED
    protected={}
    for r in runs.values():
        key=r['run_id'];entry=PROTECTED.get(study['study_id']+'/'+key)
        if key in fits or r['kind']!='model_fit' or r['config'].get('stage')!='search' or entry is None:continue
        file,_,function=r['adapter'].partition(':');source=receipt(confined(root,file),root)
        direct=[{'run_id':runs[k]['run_id'],'kind':runs[k]['kind'],'stage':runs[k]['config'].get('stage'),
                 'adapter':runs[k]['adapter'],'parents':runs[k]['parents']} for k in children[key]]
        contract=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if (r['adapter']!=entry['adapter'] or source!=entry['source'] or contract!=entry['contract_sha256']
                or r['parents']!=entry['parents']
                or sorted(direct,key=lambda x:x['run_id'])!=sorted(entry['direct_children'],key=lambda x:x['run_id'])):
            raise ValueError('Exact protected paid search-source contract or consumer roster changed')
        protected[key]={'factory':r['adapter'],'source':source,'contract_sha256':contract,
                        'direct_state_consumer_ids':sorted(children[key]),'direct_state_consumer_contracts':direct,
                        'own_parent_ids':r['parents'],'full_state_required':True,
                        'release_under_unused_search_policy':False}
    return {'schema':SCHEMA,'existing_output_migration':False,
            'audited_metadata_selectors':selectors,'unused_search_fits':fits,
            'protected_search_source_fits':protected,
            'retained':['run_spec','source_and_data_receipts','seeds','TRAIN_SELECT_trace',
                        'result','ranking','selection','costs','failures','all_release_receipts']}

def policy(study):
    value=study['release']['checkpoint_retention_policy'].get('unused_search_binary_release')
    if value is None:return None
    if value.get('schema')!=SCHEMA or value.get('existing_output_migration') is not False:
        raise ValueError('Only explicitly frozen new-output search release is supported')
    return value

def closure(root,study,registry,identifier):
    value=policy(study)
    if value is None or identifier not in value['unused_search_fits']:return None
    runs={r['run_id']:r for r in study['runs']};fit=runs[identifier]
    if fit['kind']!='model_fit' or fit['config'].get('stage')!='search':raise ValueError('Frozen release fit changed')
    direct=sorted(r['run_id'] for r in runs.values() if identifier in r['parents'])
    frozen=value['unused_search_fits'][identifier]['direct_metadata_selection_ids']
    if direct!=frozen:raise ValueError('Unused search fit gained a direct nonmetadata consumer')
    consumers=registry.consumer_records(study['study_id'],identifier)
    if set(consumers['direct_ids'])!=set(direct):raise ValueError('Search registry DAG changed')
    descendants=set();todo=list(direct)
    while todo:
        k=todo.pop()
        if k in descendants:continue
        descendants.add(k);todo.extend(r['run_id'] for r in runs.values() if k in r['parents'])
    if set(consumers['transitive_ids'])!=descendants:raise ValueError('Search transitive registry DAG changed')
    # Transitive fresh comparative fits use chosen configuration metadata. The
    # only direct users of these tensors must be the exact no-tensor selectors.
    blocked=[];bindings=[]
    for key in direct:
        entry=value['audited_metadata_selectors'][key];run=runs[key]
        if selector_binding(root,run)!=entry['binding'] or sorted(run['parents'])!=entry['fit_cohort_ids']:
            raise ValueError('Source-pinned metadata selector/cohort changed')
        row=registry.run(study['study_id'],key)
        if row['state']!='complete' or row.get('result',{}).get('status')!='complete':blocked.append(key)
        for parent in entry['fit_cohort_ids']:
            if registry.run(study['study_id'],parent)['state'] not in ('complete','failed'):blocked.append(parent)
        bindings.append({'selection_id':key,'source':entry['binding'],'fit_cohort_ids':entry['fit_cohort_ids'],
                         'result':copy.deepcopy(row.get('result'))})
    return {'closed':not blocked,'blocked':sorted(set(blocked)),
            'exact_metadata_selection_closures':bindings,'no_direct_paid_state_consumers':True}

def _persist(path,value):
    # Durable exclusive intent. A stopped sweep resumes this same intent;
    # historical/foreign outputs never acquire a new authority retroactively.
    with path.open('x') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())

def _authority(root,study,registry,identifier):
    row=registry.run(study['study_id'],identifier);bound=row.get('result',{}).get('retention_authority')
    if row['state']!='complete' or not bound:return None
    path=confined(root,bound['path'])
    if receipt(path,root)!=bound:raise ValueError('Immutable fit retention authority changed')
    value=json.loads(path.read_text())
    if value.get('existing_output_migration') is not False or value['fit_id']!=identifier:
        raise ValueError('Existing or different fit outputs cannot release')
    if value['source_policy'].get('unused_search_binary_release')!=policy(study):
        raise ValueError('Fit did not opt into this frozen prospective search policy')
    selected=row['result']['selected_checkpoint']
    if any(value['source_full_selected'][k]!=selected[k] for k in ('bytes','sha256')):
        raise ValueError('Release authority differs from actual selected state')
    return row,value,path.parent

def _validate_tombstone(root,identifier,row,folder,candidates):
    intent=folder/'UNUSED_SEARCH_RELEASE_INTENT.json';done=folder/'UNUSED_SEARCH_RELEASE_COMPLETE.json'
    tomb=folder/'UNUSED_SEARCH_RELEASE_TOMBSTONE.json'
    if not tomb.is_file():return False
    tombstone=json.loads(tomb.read_text());binding=tombstone['completion']
    if tombstone.get('schema')!=SCHEMA or receipt(done,root)!=binding:
        raise ValueError('Released-search completion/schema changed')
    completion=json.loads(done.read_text());planned=json.loads(intent.read_text())
    if (completion.get('schema')!=SCHEMA or planned.get('schema')!=SCHEMA
            or receipt(intent,root)!=completion['intent'] or completion['released_binary_authorities']!=candidates
            or planned['release_binary_authorities']!=candidates
            or tombstone['fit_id']!=identifier or tombstone['authority']!=row['result']['retention_authority']):
        raise ValueError('Released-search tombstone is not bound to this exact fit')
    if any(confined(root,r['path']).exists() for r in candidates):raise ValueError('Released unused search binary reappeared')
    return True

def closed_tombstone(root,study,registry,identifier):
    """Read-only scheduler proof; incomplete/failed states retain frontier slots."""
    closed=closure(root,study,registry,identifier)
    if closed is None or not closed['closed']:return False
    found=_authority(root,study,registry,identifier)
    if found is None:return False
    row,value,folder=found
    candidates=copy.deepcopy(value['full_state_receipts'])+[copy.deepcopy(value['serving']['serving_checkpoint'])]
    return _validate_tombstone(root,identifier,row,folder,candidates)

def cohort_release_closed(root,study,registry,selector_id):
    """Nonmutating whole-cohort scheduler proof, including terminal owners."""
    value=policy(study)
    if value is None or selector_id not in value['audited_metadata_selectors']:
        return {'closed':False,'blocked':['unsupported_metadata_selector'],'tombstone_receipts':[]}
    run=next(r for r in study['runs'] if r['run_id']==selector_id)
    entry=value['audited_metadata_selectors'][selector_id]
    if selector_binding(root,run)!=entry['binding'] or sorted(run['parents'])!=entry['fit_cohort_ids']:
        raise ValueError('Exact prospective search selector/cohort changed')
    row=registry.run(study['study_id'],selector_id);blocked=[];bindings=[]
    if row['state']!='complete' or row.get('result',{}).get('status')!='complete':blocked.append(selector_id)
    from run_study import terminal_owner
    if row['state']=='complete' and not terminal_owner(row['owner'])['previous_owner_terminal']:
        blocked.append(selector_id+':owner_live')
    for identifier in entry['fit_cohort_ids']:
        fit=registry.run(study['study_id'],identifier)
        if fit['state']!='complete':blocked.append(identifier+':'+fit['state']);continue
        if not terminal_owner(fit['owner'])['previous_owner_terminal']:
            blocked.append(identifier+':owner_live');continue
        if not closed_tombstone(root,study,registry,identifier):blocked.append(identifier+':binary_frontier_open');continue
        found=_authority(root,study,registry,identifier)
        bindings.append(receipt(found[2]/'UNUSED_SEARCH_RELEASE_TOMBSTONE.json',root))
    return {'closed':not blocked,'blocked':blocked,'tombstone_receipts':bindings}

def sweep_one(root,study,registry,identifier,*,owner_terminal):
    """Return (handled, receipts); no live fit or existing output is migrated."""
    closed=closure(root,study,registry,identifier)
    if closed is None:return False,[]
    found=_authority(root,study,registry,identifier)
    if found is None or not closed['closed'] or not owner_terminal:return True,[]
    row,value,folder=found;intent=folder/'UNUSED_SEARCH_RELEASE_INTENT.json'
    done=folder/'UNUSED_SEARCH_RELEASE_COMPLETE.json';tomb=folder/'UNUSED_SEARCH_RELEASE_TOMBSTONE.json'
    candidates=copy.deepcopy(value['full_state_receipts'])+[copy.deepcopy(value['serving']['serving_checkpoint'])]
    paths=[r['path'] for r in candidates]
    if len(paths)!=len(set(paths)):raise ValueError('Search release has duplicate binary authorities')
    if tomb.exists():
        _validate_tombstone(root,identifier,row,folder,candidates)
        return True,[]
    planned={'schema':SCHEMA,'fit_id':identifier,'authority':row['result']['retention_authority'],
             'release_binary_authorities':candidates,'metadata_closure':closed,
             'owner_terminal':True,'existing_output_migration':False,'scientific_history_preserved':True}
    if intent.exists():
        actual=json.loads(intent.read_text())
        if actual!=planned:raise ValueError('Pending exact unused-search deletion intent changed')
    else:
        # All candidate byte receipts must exist before the first intent.
        for bound in candidates:
            if receipt(confined(root,bound['path']),root)!=bound:raise ValueError('Search state changed before release intent')
        _persist(intent,planned)
    if not done.exists():
        removed=[];absent=[]
        for bound in candidates:
            p=confined(root,bound['path'])
            if p.exists():
                if receipt(p,root)!=bound:raise ValueError('Search state changed after immutable deletion intent')
                p.unlink();removed.append(bound)
            else:absent.append(bound)  # Permitted only by the already durable exact intent.
        _persist(done,{'schema':SCHEMA,'intent':receipt(intent,root),
            'released_binary_authorities':candidates,'removed_by_this_sweep':removed,
            'already_absent_under_same_durable_intent':absent,
            'existing_output_migration':False,'scientific_history_preserved':True})
    completion=json.loads(done.read_text())
    if receipt(intent,root)!=completion['intent'] or completion['released_binary_authorities']!=candidates:
        raise ValueError('Search deletion completion does not bind exact intent')
    _persist(tomb,{'schema':SCHEMA,'fit_id':identifier,'authority':row['result']['retention_authority'],
                   'completion':receipt(done,root),'purpose':'no binary users; metadata and history retained',
                   'existing_output_migration':False,'scientific_history_preserved':True})
    binding=receipt(tomb,root);registry.event(study['study_id'],identifier,'prospective_unused_search_binary_release',binding)
    return True,[binding]
