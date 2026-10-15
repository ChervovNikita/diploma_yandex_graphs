"""Execute the actual receipt dispatcher with the authentic completed NCNC annotation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent


def main():
    source=(HERE/'adapters.py').read_text();tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_link_source_dependencies')
    fixture=json.loads((HERE/'ANNOTATED_NATIVE_RECEIPT_FIXTURE.json').read_text())
    annotated=fixture['authentic_annotated_receipt'];canonical={k:annotated[k] for k in ('path','bytes','sha256')}
    records={canonical['path']:canonical}
    namespace={'copy':copy,'receipt':lambda path:copy.deepcopy(records[str(path)]),'inside':lambda path,root:path}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(HERE/'adapters.py'),'exec'),namespace)
    dispatch=namespace['_link_source_dependencies'];checks=[]
    def adapter(manifest=None,rows=()):
        meta={'native_integration':{'integration_sources':list(copy.deepcopy(rows)),'finalized_helper_sources':[],
            'native_binding':{'adapter_source_receipts':{},'adapter_manifest_receipt':copy.deepcopy(annotated if manifest is None else manifest)}}}
        return SimpleNamespace(extension_receipt=None,metadata=lambda:copy.deepcopy(meta))
    def passed(name,owner):
        old=owner.metadata();result=dispatch(owner,canonical,'unused-in-fixture')
        assert result.dependency_receipts==[canonical]
        assert result.dependency_annotations=={canonical['path']:{'adapter_hashes_verified':True}}
        assert result.metadata()==old
        checks.append({'name':name,'passed':True})
    def refused(name,owner):
        try:dispatch(owner,canonical,'unused-in-fixture')
        except (ValueError,KeyError):checks.append({'name':name,'expected_refusal':True})
        else:raise AssertionError('Expected refusal: '+name)
    passed('authentic-completed-NCNC-annotated-manifest',adapter())
    passed('canonical-and-annotated-duplicate-identities',adapter(rows=[canonical]))
    passed('repeated-authentic-annotation',adapter(rows=[annotated]))
    for key,value in [('bytes',canonical['bytes']+1),('sha256','0'*64)]:
        modified=dict(annotated,**{key:value});refused('changed-'+key,adapter(modified))
    refused('missing-verification-annotation',adapter(canonical))
    refused('false-verification-annotation',adapter(dict(annotated,adapter_hashes_verified=False)))
    refused('integer-not-true-verification-annotation',adapter(dict(annotated,adapter_hashes_verified=1)))
    refused('unknown-manifest-annotation',adapter(dict(annotated,unreviewed=True)))
    refused('conflicting-duplicate-byte-identity',adapter(rows=[dict(canonical,bytes=0)]))
    other=dict(canonical,path=canonical['path']+'.other')
    records[other['path']]=other
    refused('annotation-on-nonmanifest-dependency',adapter(rows=[dict(other,adapter_hashes_verified=True)]))
    parent=HERE.parent/'runtime_link_v5'
    declaration=json.loads((HERE/'PARENT_SOURCE.json').read_text())
    assert hashlib.sha256((parent/'MANIFEST.json').read_bytes()).hexdigest()==declaration['manifest_sha256']
    for pin in declaration['files']:
        raw=(parent/pin['path']).read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256']
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    unchanged=[pin['path'] for pin in declaration['files'] if pin['path']!='adapters.py']
    assert all((HERE/name).read_bytes()==(parent/name).read_bytes() for name in unchanged)
    old_tree=ast.parse((parent/'adapters.py').read_text())
    old_function=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_link_source_dependencies')
    old_lines=(parent/'adapters.py').read_text().splitlines(True);new_lines=source.splitlines(True)
    assert ''.join(new_lines[:function.lineno-1])==''.join(old_lines[:old_function.lineno-1])
    assert ''.join(new_lines[function.end_lineno:])==''.join(old_lines[old_function.end_lineno:])
    checks.append({'name':'only-dependency-receipt-dispatch-producer-text-changed','passed':True})
    return {'schema':'runtime-link-v6-authentic-annotation-contract-v1','status':'passed','checks':checks,
        'checks_count':len(checks),'unchanged_producers_byte_identical':unchanged,
        'annotation_preserved_in_native_metadata_and_separate_dependency_annotations':True,
        'required_path_bytes_sha256_identity_exact':True,'parent29_tensor_checks':'Inherited unchanged mechanisms; not re-executed here',
        'native_models':0,'native_forward_calls':0,'native_optimizer_updates':0,'TEST_accessed':False}


if __name__=='__main__':print(json.dumps(main(),indent=2))
