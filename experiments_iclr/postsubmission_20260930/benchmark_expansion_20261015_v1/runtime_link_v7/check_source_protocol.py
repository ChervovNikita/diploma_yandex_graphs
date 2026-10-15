"""Check exact published source metadata; no Torch import or native calls."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
E=HERE.parent
from link_authority import REGISTRY, authenticate_extension


def pin(path):
    p=Path(path).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def check():
    checks=[]
    for name in ('run_fit.py','bookkeeping.py','common.py','controllers.py','source_rng.py','engineering_link_policy.py'):
        assert (HERE/name).read_bytes()==(E/'runtime_link_v6'/name).read_bytes()
        checks.append('unchanged:'+name)
    old=ast.parse((E/'runtime_link_v6/adapters.py').read_text());new=ast.parse((HERE/'adapters.py').read_text())
    for name in ('LinkSameCallMeanAdapter','BaseAdapter','NodeAdapter','OGBAdapter'):
        a=next(n for n in old.body if isinstance(n,ast.ClassDef)and n.name==name)
        b=next(n for n in new.body if isinstance(n,ast.ClassDef)and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
        checks.append('unchanged_AST:'+name)
    callbacks=json.loads((E/'expanded_dispatcher_v2/LINK_CALLBACK_INPUTS.json').read_text())['link_cases']
    compiler_cases=0
    for row in callbacks:
        package=Path(row['callback']).parent.name
        if package=='link_native_integration_v1':continue
        entry=REGISTRY[package]
        assert row['callback_sha256']==entry['callback_sha256']
        assert row['method_settings']['manifest_sha256']==entry['manifest_sha256']
        assert pin(E/package/Path(row['callback']).name)['sha256']==entry['callback_sha256']
        assert pin(E/package/'MANIFEST.json')['sha256']==entry['manifest_sha256']
        tree=ast.parse((E/package/entry['extension_name']).read_text())
        assert entry['schema'] in (E/package/entry['extension_name']).read_text()
        compiler_cases+=1
    remote_E='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
    remote_root=remote_E.split('/experiments_iclr/')[0]
    local_root=E.parent  # Local research workspace maps to remote PHASE for source files.
    remote_phase=remote_root+'/experiments_iclr/postsubmission_20260930'
    def localize(path):
        p=str(path)
        if p.startswith(remote_phase+'/'):return local_root/p[len(remote_phase)+1:]
        raise ValueError('Fixture source outside known phase: '+p)
    def source_pin(path):
        # Preserve authentic allocation identity in returned pins while reading only local source bytes.
        result=pin(localize(path));result['path']=str(path);return result
    def inside(path,root):
        p=Path(path)
        if not p.is_absolute()or not p.is_relative_to(Path(root)):raise ValueError('Source escapes fixed repository')
        return p
    fixture_file=HERE/'AUTHENTIC_DECLARATIONS.json';fixtures=json.loads(fixture_file.read_text())['fixtures']
    outcomes=[];mutation_refusals=0
    for f in fixtures:
        declaration=f['declaration'];schema=declaration['schema'];method=declaration['method_id']
        package,entry=next((k,v)for k,v in REGISTRY.items()if v['schema']==schema)
        compatible=next((r for r in callbacks if Path(r['callback']).parent.name==package and r['procedure']==method),None)
        if compatible is None:continue
        callback=Path(compatible['callback']);extension=source_pin(callback.with_name(entry['extension_name']))
        session=SimpleNamespace();owner=session
        for attr in entry['declaration_path'][:-1]:
            child=SimpleNamespace();setattr(owner,attr,child);owner=child
        setattr(owner,entry['declaration_path'][-1],copy.deepcopy(declaration))
        job={'callback_module':str(callback),'callback_sha256':entry['callback_sha256'],
             'procedure':method,'method_settings':compatible['method_settings']}
        value=authenticate_extension(session,job,extension,remote_root,source_pin,inside)
        assert value['declaration']==declaration
        # Exact source/schema authority refuses a genuine fixture with one field altered.
        bad=copy.deepcopy(job);bad['callback_sha256']='0'*64
        try:authenticate_extension(session,bad,extension,remote_root,source_pin,inside)
        except ValueError:mutation_refusals+=1
        else:raise AssertionError('Changed source accepted')
        outcomes.append({'package':package,'method':method,'mean_protocol':value['mean_protocol'],'source':f['source']})
    assert len(outcomes)>=20 and mutation_refusals==len(outcomes)
    return {'schema':'runtime-link-v7-source-protocol-checks-v1','status':'passed-source-metadata-only',
        'checks':checks,'compiler_extended_callback_cases':compiler_cases,'authentic_metadata_cases':len(outcomes),
        'exact_source_mutation_refusals':mutation_refusals,'cases':outcomes,
        'source_only_schema_branches':['native-BUDDY-expanded-risk-production-v1','native-BUDDY-private-BN-production-source-v1'],
        'training_semantics_modified':False,'native_models':0,'native_forwards':0,'optimizer_updates':0,
        'Torch_imported':False,'TEST_accessed':False,'native_qualification':False,'campaign_GO':False}


if __name__=='__main__':print(json.dumps(check(),sort_keys=True))
