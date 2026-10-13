"""Stdlib-only exact protected-source custody rebinding before declaration.

Call on the compiler's actual apply_resources outputs, before declare_policy.
No factory is loaded, no tensor is decoded, and no learning field is editable.
The caller copies the emitted successor module into its new owned controller
before importing its retention policy. Existing controllers remain untouched.
"""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path

RESOURCE_KEYS={'reservation_mib','estimated_seconds'}
MIB=1048576

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def fingerprint(value):return hashlib.sha256(canonical(value)).hexdigest()
def file_receipt(root,relative):
    root=Path(root).resolve();p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Source receipt must be project-relative')
    target=root/p
    if target.is_symlink() or not target.resolve().is_relative_to(root):raise ValueError('Source receipt escapes explicit project')
    return {'path':relative,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}

def _namespace_maps(root,amendment):
    if amendment is None:return {},[]
    if (amendment.get('numerical_ast_changes')!=0 or amendment.get('old_sources_modified') is not False
            or amendment.get('scientific_fits')!=0):raise ValueError('Namespace amendment is not the reviewed import-only successor')
    pairs=amendment['all_runtime_source_files'];changed={p['predecessor']['path']:p for p in amendment['changed_files']}
    mapping={};proof=[]
    def normalize(raw):
        tree=ast.parse(raw)
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
                node.module='adapters.'+node.module;node.level=0
        return ast.dump(tree,include_attributes=False)
    for pair in pairs:
        old,new=pair['predecessor'],pair['successor']
        if file_receipt(root,old['path'])!=old or file_receipt(root,new['path'])!=new:
            raise ValueError('Namespace source bytes differ from reviewed custody')
        if pair['byte_identical']:
            if old['sha256']!=new['sha256'] or old['bytes']!=new['bytes']:raise ValueError('Claimed unchanged namespace source differs')
        else:
            item=changed.get(old['path'])
            if item is None or item['successor']!=new or item.get('AST_equal_after_import_namespace_normalization') is not True:
                raise ValueError('Namespace AST change lacks exact predecessor/successor proof')
            before=(Path(root)/old['path']).read_text();after=(Path(root)/new['path']).read_text()
            if normalize(before)!=normalize(after):raise ValueError('Namespace source contains a numerical/API AST change')
        mapping[old['path']]={'predecessor':old,'successor':new};proof.append(pair)
    return mapping,proof

def _namespace_value(value,mapping,changes):
    if isinstance(value,dict):
        if {'path','bytes','sha256'}.issubset(value) and value['path'] in mapping:
            entry=mapping[value['path']]
            if any(value[k]!=entry['predecessor'][k] for k in ('path','bytes','sha256')):
                raise ValueError('Mapped run source receipt differs from original namespace custody')
            result=copy.deepcopy(value);result.update(entry['successor']);changes.append(entry);return result
        return {k:_namespace_value(v,mapping,changes) for k,v in value.items()}
    if isinstance(value,list):return [_namespace_value(v,mapping,changes) for v in value]
    if isinstance(value,str):
        path,sep,function=value.partition(':')
        if path in mapping:
            entry=mapping[path];changes.append(entry);return entry['successor']['path']+(sep+function if sep else '')
    return copy.deepcopy(value)

def _resource_expected(study,original):
    scope=study.get('resources',{}).get('master_resource_scope')
    if scope is None or 'override' not in scope:raise ValueError('Actual apply_resources output with its registered scope required')
    override=scope['override'];change={} if override is None else override.get('runs',{}).get(original['run_id'],{})
    if set(change)-RESOURCE_KEYS:raise ValueError('Registered resource override attempts to change a learning/source field')
    expected=copy.deepcopy(original);before=copy.deepcopy(expected['config']);expected['config'].update(change)
    reserve=expected['config'].get('reservation_mib');seconds=expected['config'].get('estimated_seconds')
    if type(reserve)!=int or reserve<1:raise ValueError('Positive integer GPU reservation required')
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('Finite nonnegative registered timing weight required')
    if 'maximum_cuda_bytes' in before:expected['config']['maximum_cuda_bytes']=reserve*MIB
    changed=[{'field':'config.'+k,'before':before.get(k),'after':expected['config'].get(k),
              'registered_override':k in change,'derived_existing_CUDA_ceiling':k=='maximum_cuda_bytes'}
             for k in sorted(RESOURCE_KEYS|{'maximum_cuda_bytes'}) if canonical(before.get(k))!=canonical(expected['config'].get(k))]
    return expected,changed,change

def rebind(final_priced_studies,previous_protected76,namespace_amendment,*,project_root):
    """Return new module text/roster/proof; performs source reads, no mutation."""
    rows=previous_protected76['rows']
    if len(rows)!=76 or previous_protected76.get('row_count')!=76:
        raise ValueError('Exactly the reviewed76protected keys and full predecessor runs required')
    previous_keys={r['study_id']+'/'+r['run_id'] for r in rows}
    if len(previous_keys)!=76:raise ValueError('Duplicate protected predecessor key')
    studies={s['study_id']:s for s in final_priced_studies}
    if len(studies)!=len(final_priced_studies):raise ValueError('Duplicate final study identity')
    namespace,namespace_proof=_namespace_maps(project_root,namespace_amendment)
    rebound={};diff=[]
    for entry in rows:
        key=entry['study_id']+'/'+entry['run_id'];study=studies.get(entry['study_id'])
        if study is None:raise ValueError('Final study omits protected key '+key)
        lookup={r['run_id']:r for r in study['runs']};actual=lookup.get(entry['run_id'])
        if actual is None:raise ValueError('Final study omits protected source run '+key)
        original=entry['original_run']
        if fingerprint(original)!=entry['contract_sha256'] or original['parents']!=entry['parents'] or original['adapter']!=entry['adapter']:
            raise ValueError('Protected predecessor contract custody differs')
        if file_receipt(project_root,entry['source']['path'])!=entry['source']:
            raise ValueError('Protected predecessor source bytes changed')
        namespace_changes=[];mapped=_namespace_value(original,namespace,namespace_changes)
        expected,resource_changes,override=_resource_expected(study,mapped)
        if canonical(expected)!=canonical(actual):
            differing=[k for k in set(expected)|set(actual) if canonical(expected.get(k))!=canonical(actual.get(k))]
            raise ValueError('Protected source learning/selector/budget/parents/initialization changed: '+key+' fields '+repr(sorted(differing)))
        # No implicit consumer removal or addition is permitted by this helper.
        direct=[{'run_id':r['run_id'],'kind':r['kind'],'stage':r['config'].get('stage'),'adapter':r['adapter'],'parents':r['parents']}
                for r in study['runs'] if actual['run_id'] in r['parents']]
        expected_direct=_namespace_value(entry['direct_children'],namespace,namespace_changes)
        if canonical(sorted(direct,key=lambda x:x['run_id']))!=canonical(sorted(expected_direct,key=lambda x:x['run_id'])):
            raise ValueError('Protected direct state-consumer roster changed: '+key)
        source=file_receipt(project_root,actual['adapter'].split(':',1)[0])
        expected_source=_namespace_value(entry['source'],namespace,namespace_changes)
        if source!=expected_source:raise ValueError('Protected source learning bytes changed outside reviewed namespace substitution')
        record={'adapter':actual['adapter'],'source':source,'contract_sha256':fingerprint(actual),
                'parents':actual['parents'],'direct_children':direct};rebound[key]=record
        stripped_old=copy.deepcopy(original);stripped_new=copy.deepcopy(actual)
        for config in (stripped_old['config'],stripped_new['config']):
            for k in RESOURCE_KEYS|{'maximum_cuda_bytes'}:config.pop(k,None)
        normalized_old=_namespace_value(stripped_old,namespace,[])
        if canonical(normalized_old)!=canonical(stripped_new):raise AssertionError('Nonresource source contract differs after exact namespace normalization')
        diff.append({'key':key,'predecessor_contract_sha256':entry['contract_sha256'],'successor_contract_sha256':record['contract_sha256'],
                     'resource_changes':resource_changes,'registered_final_override':override,'namespace_changes':namespace_changes,
                     'unchanged_nonresource_contract_sha256':fingerprint(stripped_new),'all_learning_budget_selection_parent_initialization_fields_unchanged':True,
                     'direct_state_consumer_roster_unchanged_after_namespace_mapping':True,'predecessor_source':entry['source'],'successor_source':source})
    if set(rebound)!=previous_keys:raise AssertionError('Protected source key roster changed')
    module='"""Reviewed resource/namespace-only protected76 successor; no releasable source state."""\nPROTECTED='+repr(rebound)+'\n'
    proof={'schema':'protected76-resource-and-import-namespace-only-rebinding-proof-v1','predecessor_plan':previous_protected76['plan_path'],
           'predecessor_plan_canonical_sha256':previous_protected76['plan_canonical_sha256'],'protected_key_count':76,
           'rows':diff,'namespace_source_custody_proofs':namespace_proof,'successor_roster_module_sha256':hashlib.sha256(module.encode()).hexdigest(),
           'field_changes_allowed':['registered config.reservation_mib','registered config.estimated_seconds','existing config.maximum_cuda_bytes derived by actual apply_resources','exact reviewed namespace-only source/adapter substitutions'],
           'source_arrays_decoded':0,'scientific_fits':0,'TRAIN_callbacks':0,'optimizer_maps':0,'existing_controller_modified':False,'GO':False}
    return {'module_source':module,'roster':rebound,'proof':proof}

def write_successor(result,destination):
    """Exclusive output in an owned fresh directory; never rewrites a controller."""
    destination=Path(destination);destination.mkdir()
    for name,value in [('protected_search_roster.py',result['module_source']),('PROTECTED_ROSTER_REBINDING_PROOF.json',json.dumps(result['proof'],indent=2,sort_keys=True)+'\n')]:
        p=destination/name
        with p.open('x') as stream:stream.write(value)
        p.chmod(0o444)
    return {'directory':str(destination),'module_sha256':result['proof']['successor_roster_module_sha256'],'protected_keys':76,'scientific_fits':0,'GO':False}
