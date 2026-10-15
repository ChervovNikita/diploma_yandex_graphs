"""Freeze the explicit BUDDY seed0 interaction handoff as a compact overlay."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXPANSION=HERE.parent
PHASE=EXPANSION.parent


def pin(path, expected=None):
    path=Path(path).resolve();assert path.is_relative_to(PHASE)
    assert path.suffix in {'.py','.json','.md','.txt','.xml','.html'}
    raw=path.read_bytes()
    if path.suffix=='.py':ast.parse(raw)
    value={'path_relative_to_phase':str(path.relative_to(PHASE)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    if expected:
        for k in ('sha256','bytes'):
            if k in expected:assert value[k]==expected[k], str(path)
    return value


def main():
    target=HERE/'OVERLAY_v2.json';assert not target.exists()
    path=EXPANSION/'qualification/interaction_BUDDY_v1_COMPILER_BINDING.json'
    handoff_pin=pin(path,{'sha256':'91b8e73d2de761ffbad2a2b416199defe6f37341d6a5bcfd58c8f9206c941185'})
    h=json.loads(path.read_text());assert h['current_source_supported_seeds']==[0] and h['scientific_launch_enabled'] is False
    assert set(h['method_settings'])=={'manifest_sha256','context_manifest','context_manifest_sha256'}
    mf=pin(EXPANSION/'interaction_BUDDY_v1/MANIFEST.json',{'sha256':h['method_settings']['manifest_sha256']})
    callback=pin(PHASE/h['callback_module'],{'sha256':h['callback_sha256']})
    names={n.name for n in ast.parse((PHASE/h['callback_module']).read_text()).body if isinstance(n,ast.FunctionDef)}
    assert 'create_adapter' in names
    sources={handoff_pin['path_relative_to_phase']:handoff_pin}
    occurrences=[]
    def remember(p, expected=None):
        value=pin(p,expected);sources[value['path_relative_to_phase']]=value;occurrences.append(value)
    packages=[{'path':'interaction_BUDDY_v1','manifest_sha256':mf['sha256']}]+h['ingredient_and_native_source_pins']['packages']
    for package in packages:
        folder=EXPANSION/package['path'];remember(folder/'MANIFEST.json',{'sha256':package['manifest_sha256']})
        rows=json.loads((folder/'MANIFEST.json').read_text())['files']
        if isinstance(rows,dict):rows=[{'path':n,'sha256':s} for n,s in rows.items()]
        for row in rows:
            p=(folder/row['path']).resolve();assert p.is_relative_to(folder)
            remember(p,row)
    for row in h['ingredient_and_native_source_pins']['native_sources']:
        remember(PHASE/row['path_relative_to_phase'],row)
    portable=EXPANSION/'portable_private_input_v2'
    for row in json.loads((portable/'SOURCES.json').read_text())['source_files']:
        if 'link' in row['families']:remember(PHASE/row['path_relative_to_phase'],row)
    registry=json.loads((HERE/'REGISTRY_v1.json').read_text())
    entries=[]
    for method in h['methods']:
        old=next(r for r in registry['entries'] if (r['cell_id'],r['method_id'])==('collab-buddy',method))
        r=copy.deepcopy(old)
        r.update(callback={**callback,'function':'create_adapter'}, session_transform=None,
                 runtime_procedure=method,base_procedure=h['base_procedure'],source_manifest=mf,
                 method_settings=h['method_settings'],status='source_prepared_not_runtime_admitted',
                 supported_seeds=[0],unsupported_seed_reason='authentic_BUDDY_native_year2007_interaction_context_only_seed0',
                 context_manifest_receipt={'path':h['method_settings']['context_manifest'],'sha256':h['method_settings']['context_manifest_sha256']},
                 native_CPU_gate='pending_at_handoff;source_binding_only')
        r.pop('missing_reason',None)
        entries.append(r)
    value={'schema':'expanded-fit-source-overlay-v2','parent_registry':pin(HERE/'REGISTRY_v1.json'),
           'handoff':handoff_pin,'entries':entries,'source_files':sorted(sources.values(),key=lambda x:x['path_relative_to_phase']),
           'source_pin_occurrences':len(occurrences),'source_only':True,'runtime_v4_link_successor_still_required':True,
           'scientific_launch_enabled':False,'native_CPU_gate':'pending_at_handoff','runnable':False}
    raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    with target.open('xb') as out:out.write(raw)
    print(json.dumps({'path':str(target),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'source_files':len(sources),'source_pin_occurrences':len(occurrences),'bindings':len(entries)}))


if __name__=='__main__':main()
