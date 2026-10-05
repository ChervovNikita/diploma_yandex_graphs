#!/usr/bin/env python3
"""Focused source/metadata AST checks only; never import or run prepared sources."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def call_name(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):return call_name(node.value)+'.'+node.attr
    return ast.unparse(node)


def call_sites(function):
    return sorted([{'line':n.lineno,'call':call_name(n.func),'arguments':ast.unparse(n)}
                   for n in ast.walk(function) if isinstance(n,ast.Call)],key=lambda n:n['line'])


def closure(modules, start):
    visited, pending = set(), [start]
    while pending:
        key = pending.pop()
        if key in visited:continue
        visited.add(key);module,name = key
        for node in ast.walk(modules[module]['functions'][name]):
            if not isinstance(node,ast.Call):continue
            called = call_name(node.func)
            target_module,target_name = ('collector',called[2:]) if called.startswith('c.') else (module,called)
            if target_name in modules[target_module]['functions']:
                pending.append((target_module,target_name))
    return sorted('.'.join(key) for key in visited)


def main():
    modules = {}
    allowed = {'argparse','datetime','hashlib','json','pathlib','collect_companion'}
    for name,file in (('collector','collect_companion.py'),('inventory','inventory_full39_histories.py')):
        path = HERE/file;source = path.read_text();tree = ast.parse(source,filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(alias.name for alias in node.names)
            if isinstance(node,ast.ImportFrom):imports.append(node.module)
        require(set(imports) <= allowed,'Non-stdlib/nonlocal prepared import')
        require(not any(isinstance(n,ast.Call) and call_name(n.func) in ('eval','exec','compile','__import__') for n in ast.walk(tree)),
                'Dynamic source extraction/execution is not admitted')
        values = {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                  and isinstance(n.value,(ast.Constant,ast.Dict,ast.Set))}
        functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        modules[name] = dict(file=file,source=source,tree=tree,imports=imports,values=values,functions=functions,
                             sha256=digest(path),lines=len(source.splitlines()))
    collector, inventory = modules['collector'],modules['inventory']
    require(collector['values']['COLLECTOR_SOURCE_REVIEWED'] is False
            and inventory['values']['INVENTORY_SOURCE_REVIEWED'] is False,'Hard source guards must remain false')
    mains = {name:call_sites(module['functions']['main']) for name,module in modules.items()}
    guards = {}
    for name,module in modules.items():
        token = 'COLLECTOR_SOURCE_REVIEWED' if name=='collector' else 'INVENTORY_SOURCE_REVIEWED'
        guards[name] = next(row['line'] for row in mains[name] if token in row['arguments'] and row['call'] in ('require','c.require'))
        first_read = min(row['line'] for row in mains[name] if row['call'] in ('read_metadata','c.read_metadata','phase_file','c.phase_file'))
        require(guards[name] < first_read,'Disabled guard must precede release/path reads')
    cmain = collector['functions']['main'];imain = inventory['functions']['main']
    ccall = call_sites(cmain);icall = call_sites(imain)
    line = lambda rows,name:min(row['line'] for row in rows if row['call']==name)
    require(line(ccall,'authenticate_nine') < line(ccall,'descriptor') < line(ccall,'fresh_output'),
            'Complete all9 metadata pass must precede optional CONFIG observation and outputs')
    require(line(icall,'authenticate_all39') < line(icall,'observe_histories') < line(icall,'c.fresh_output'),
            'Complete all39 pass must precede first history hash and output')
    all39_closure = closure(modules,('inventory','authenticate_all39'))
    require('collector.descriptor' not in all39_closure and 'inventory.observe_histories' not in all39_closure,
            'All39 authentication may not parse CONFIG/outcomes or observe histories')
    history_sites = call_sites(inventory['functions']['observe_histories'])
    require(all(row['call'] not in ('c.read_metadata','json.loads','json.load') for row in history_sites)
            and "c.relative(row['freeze_relative']).parent / 'VALID_HISTORY.jsonl'" in ast.unparse(inventory['functions']['observe_histories']),
            'History path must be source fixed and history bytes never parsed')
    meta_reader = ast.unparse(collector['functions']['read_metadata'])
    require("('FREEZE.json', 'VALID_HISTORY.jsonl')" in meta_reader and "path.suffix == '.json'" in meta_reader,
            'Metadata helper must reject source FREEZE/history/tensor deserialization')
    forbidden = {'torch','numpy','subprocess','socket','requests','importlib'}
    require(not forbidden.intersection(collector['imports']+inventory['imports']),'No numerical/network/process imports admitted')
    for function in collector['functions'].values():
        for row in call_sites(function):
            if row['call']=='read_metadata':
                require('FREEZE.json' not in row['arguments'] and 'VALID_HISTORY.jsonl' not in row['arguments'],'Outcome passed to metadata reader')
    source_constants = {'SPEC':'774f45e1778ab7f0708aaa5f5a3a7aa6970968c63594218993673541856f72e4',
                        'ORIGINAL_PLAN':'ae4b0f5c77cf48a86ccdbe51179fc95881157ac225f593c3992a5a5014d5346b',
                        'PROMOTION':'f638ee0d768cabb5efb999befa9bc688322c6086b983497fa5c49c8fe44d34e3'}
    for key,expected in source_constants.items():
        ref = collector['values'][key]
        require(ref['sha256']==expected and digest(PHASE/ref['path'])==expected,'Immutable scientific binding changed')
    bound = read_json(HERE/'INPUT_BINDINGS.json')
    for row in bound['files']:
        path = PHASE/row['path'];require(digest(path)==row['sha256'] and path.stat().st_size==row['bytes'],'Source/metadata input bytes changed')
    for name,expected in (('MANIFEST.json',bound['D2_v2_preserved_manifest_sha256']),('SEAL.json',bound['D2_v2_preserved_seal_sha256'])):
        require(digest(PHASE/'shared_private_transfer_fixed39_d2_analysis_preparation_20261005_v2'/name)==expected,'D2 v2 modified')
    release = read_json(HERE/'COLLECTION_RELEASE_DISABLED.json');history = read_json(HERE/'HISTORY_INVENTORY_RELEASE_DISABLED.json')
    require(release['root_collection_approved'] is False and release['collection_contract_frozen_before_companion_score_analysis'] is False
            and release['collector_source_manifest_sha256'] is None and history['root_history_inventory_approved'] is False,
            'No release approval may be fabricated')
    b0,b1,b2 = release['donors']
    auth = read_json(PHASE/b0['job_freeze_authentication']['path'])
    launch = read_json(PHASE/b0['launch_receipt']['path'])
    review = read_json(PHASE/b0['root_job_review']['path'])
    require(b0['queue_binding']['sha256']==auth['blocks']['b0']['queue_sha256']==launch['queue_sha256']==review['queue_sha256']
            and b0['root_job_review']==launch['provider_job_review']
            and b0['launch_source_manifest']['sha256']==auth['activation_manifest_sha256']==review['activation_manifest_sha256']
            and b0['root_source_review']==auth['root_source_review']
            and b0['block_freeze_binding']['sha256'] is None and b0['replica_directory_relative'] is None,
            'Actual b0 pre-fit/launch links or unresolved completion changed')
    for donor in (b1,b2):
        require(all(donor[key]['sha256'] is None for key in ('root_source_review','root_job_review','job_freeze_authentication','launch_receipt','block_freeze_binding'))
                and donor['replica_directory_relative'] is None and donor['historical_binding_is_access_or_launch_permission'] is False,
                'Unreleased GPU77 authority/completion must remain unresolved')
    require(all(ref['path'] is None and ref['sha256'] is None for ref in history['registries'].values()),'No real complete39 registry may be fabricated')
    require(not list(HERE.glob('provider_custody_*.json')) and not (HERE/'COLLECTION_FREEZE.json').exists()
            and not (HERE/'VALID_HISTORY_INVENTORY.json').exists(),'Preparation must emit no actual completion')
    metadata_reads = []
    for name,module in modules.items():
        for function_name,function in module['functions'].items():
            for row in call_sites(function):
                if row['call'] in ('read_metadata','c.read_metadata'):
                    metadata_reads.append(dict(row,module=name,function=function_name))
    result = {'schema':'focused_static_exact9_full39_source_check_results_v1','UTC':datetime.now(timezone.utc).isoformat(),
              'status':'PASS_SOURCE_ONLY_NOT_EXECUTION_OR_NUMERICAL_VERIFICATION',
              'claims':['Both prepared modules parse as Python AST and import only stdlib plus ordinary local metadata helper.',
                        'Both hard-disabled guards precede release/path reads; disabled recipes contain no actual approvals/completion.',
                        'Exact source/spec/plan/promotion fingerprints and all selected source/metadata bindings agree; D2 v2 preserved.',
                        'All9 authentication returns before optional CONFIG.runtime observation and output creation.',
                        'All39 metadata/source/job/owned terminal/artifact helper closure completes before any history byte observation.',
                        'Source-fixed history path uses registry FREEZE parent; observed history bytes are hashed without JSON parsing.',
                        'Metadata reader rejects source FREEZE, VALID history and tensor paths; no numerical/network/process import or source extraction.',
                        'B0 disabled recipe uses actual immutable pre-fit/launch hashes; actual terminal custody and all b1/b2 authority remain unresolved.'],
              'prepared_modules':{name:{k:module[k] for k in ('file','sha256','imports','lines')} for name,module in modules.items()},
              'source_guards':guards,'main_call_sites':mains,'all39_authentication_helper_closure':all39_closure,
              'collector_authentication_helper_closure':closure(modules,('collector','authenticate_nine')),
              'metadata_JSON_read_sites':metadata_reads,'history_observation_call_sites':history_sites,
              'verification_scope':{'target_imports':0,'prepared_function_or_fixture_execution':0,'numerical_execution':0,
                                    'outcome_HISTORY_FREEZE_checkpoint_feature_payload_reads':0,'server_MacLink_calls':0},
              'limits':'Focused static/control-flow/binding evidence, not independent source/custody review or actual numerical proof.'}
    evidence_path = HERE/'CONTROL_FLOW_EVIDENCE.json'
    if evidence_path.exists():
        saved = read_json(evidence_path)
        require(saved['prepared_modules'] == result['prepared_modules'] and saved['claims'] == result['claims']
                and saved['all39_authentication_helper_closure'] == result['all39_authentication_helper_closure'],
                'Saved static evidence differs; preserve the packet and prepare a separately sealed successor')
    else:
        with evidence_path.open('x') as stream:
            stream.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(result['status'])


if __name__=='__main__':main()
