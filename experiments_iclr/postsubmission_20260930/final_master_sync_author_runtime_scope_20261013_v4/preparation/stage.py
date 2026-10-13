import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib
OUT = Path(__file__).resolve().parents[1]
PHASE = OUT.parent
prior = PHASE / 'final_master_sync_author_runtime_scope_20261013_v3/MANIFEST.json'
prior_rows = json.loads(prior.read_text())['files']
proof = {}
for row in prior_rows:
    name = Path(row['path']).name
    before = (PHASE / 'native_structural_cardinality_source_ports_20261013_v1/OWNER_SCOPE_REFERENCE/runtime' / name).read_bytes()
    after = (OUT / 'runtime' / name).read_bytes()
    assert len(before) == row['bytes'] and hashlib.sha256(before).hexdigest() == row['sha256']
    if name == 'build_final_master_sync.py':
        def funcs(raw):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}
        left, right = funcs(before), funcs(after)
        assert set(left) == set(right) and [name for name in left if left[name] != right[name]] == ['main']
    else:
        assert before == after
    proof[name] = {'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': hashlib.sha256(after).hexdigest(), 'changed': before != after}
spec = importlib.util.spec_from_file_location('guarded_transport', PHASE / 'training_pipeline_v2_20261012/preparation/prepare_molecular_packets_allocation.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
transport = OUT / 'transport'
transport.mkdir(exist_ok=True)
files = []
for folder in ('runtime', 'tests', 'preparation'):
    for q in sorted((OUT / folder).rglob('*.py')):
        raw = q.read_bytes()
        files.append({'path': str(q.relative_to(OUT)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'data': base64.b64encode(raw).decode()})
code = helper.GUARD + r'''
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];q=stage/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');tick=time.monotonic();r=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(stage/'tests/check_cache.py')],cwd=stage,env=env,capture_output=True,text=True,timeout=30);qa={'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'wall_seconds':time.monotonic()-tick,'models':0,'fits':0,'GO':False};(stage/'ENGINEERING_QA.json').write_text(json.dumps(qa,indent=2,sort_keys=True)+'\n');assert r.returncode==0,qa
def receipt(q):
 raw=q.read_bytes();return {'path':str(q.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
proof=payload['proof'];(stage/'NARROW_CACHE_AUTHORITY_PROOF.json').write_text(json.dumps({'files':proof,'only_changed_top_function':'build_final_master_sync.main','only_external_authority':'exact author_cli_group execution_repository and runtime_provider_files under .venv/lib','arbitrary_roots_paths_and_nonvenv_receipts_rejected':True,'scientific_fits':0,'GO':False},indent=2,sort_keys=True)+'\n')
manifest={'schema':'declared-author-runtime-receipt-cache-successor-v4','prior_manifest':receipt(phase/'final_master_sync_author_runtime_scope_20261013_v3/MANIFEST.json'),'files':[receipt(q) for q in sorted((stage/'runtime').rglob('*.py'))],'proof':receipt(stage/'NARROW_CACHE_AUTHORITY_PROOF.json'),'QA':receipt(stage/'ENGINEERING_QA.json'),'author_runtime_providers_not_phase_files_and_not_bundled':True,'frozen_v29_controller_PLAN_SUMMARY_unchanged':True,'scientific_fits':0,'GO':False};(stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
for q in stage.rglob('*'):
 if q.is_file():q.chmod(0o444)
artifacts=[]
for name in ('MANIFEST.json','ENGINEERING_QA.json','NARROW_CACHE_AUTHORITY_PROOF.json'):
 q=stage/name;raw=q.read_bytes();artifacts.append({**receipt(q),'name':name,'data':base64.b64encode(raw).decode()})
print(json.dumps({'manifest':receipt(stage/'MANIFEST.json'),'builder':receipt(stage/'runtime/build_final_master_sync.py'),'artifacts':artifacts}))
'''
payload = base64.b64encode(zlib.compress(json.dumps({'destination': OUT.name, 'files': files, 'proof': proof}).encode(), 9)).decode()
result = helper.invoke(code, [payload], transport, 'STAGE_DECLARED_AUTHOR_RECEIPT_CACHE_V4', 60)
assert result['exit_code'] == 0, result['stdout']
value = json.loads(result['stdout'].strip())
for row in value.pop('artifacts'):
    raw = base64.b64decode(row['data'])
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    path = OUT / row['name']
    assert not path.exists()
    path.write_bytes(raw)
    path.chmod(0o444)
print(json.dumps(value, sort_keys=True))
