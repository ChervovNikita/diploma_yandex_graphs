import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib
OUT = Path(__file__).resolve().parents[1]
PHASE = OUT.parent
spec = importlib.util.spec_from_file_location('guarded_transport', PHASE / 'training_pipeline_v2_20261012/preparation/prepare_molecular_packets_allocation.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
transport = OUT / 'transport'
transport.mkdir(exist_ok=True)
files = []
for name in ('runtime/hpo_frontier.py', 'amend_controller.py', 'tests/check_frontier.py', 'preparation/stage.py'):
    raw = (OUT / name).read_bytes()
    files.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'data': base64.b64encode(raw).decode()})
code = helper.GUARD + r'''
import ast
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir();base=phase/'bounded_hpo_frontier_scheduler_20261013_v3';shutil.copytree(base/'runtime',stage/'runtime')
before=(base/'runtime/hpo_frontier.py').read_bytes()
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];q=stage/rel;q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():q.chmod(q.stat().st_mode|0o200)
 q.write_bytes(raw)
after=(stage/'runtime/hpo_frontier.py').read_bytes()
def functions(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
left,right=functions(before),functions(after);assert set(left)==set(right);changed=[n for n in left if left[n]!=right[n]];assert changed==['attention']
for name in ('launcher.py','compile_master_plan.py'):assert (base/'runtime'/name).read_bytes()==(stage/'runtime'/name).read_bytes()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');tick=time.monotonic();r=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(stage/'tests/check_frontier.py'),'--controller',str(stage/'runtime')],cwd=stage,env=env,capture_output=True,text=True,timeout=30);qa={'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'wall_seconds':time.monotonic()-tick,'models':0,'fits':0,'GO':False};(stage/'ENGINEERING_QA.json').write_text(json.dumps(qa,indent=2,sort_keys=True)+'\n');assert r.returncode==0,qa
def receipt(q):
 raw=q.read_bytes();return {'path':str(q.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
proof={'only_changed_helper_function':'attention','every_other_function_AST_identical':True,'launcher_and_compiler_bytes_identical_to_v3':True,'cohort_declaration_admission_inspection_validation_unchanged':True,'prior_actual_v24_metadata_proof':receipt(base/'ACTUAL_V24_METADATA_DAG_AUDIT.json'),'models':0,'fits':0,'GO':False};(stage/'AST_DELTA_PROOF.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
manifest={'schema':'bounded-HPO-headroom-wait-correction-v4','prior_manifest':receipt(base/'MANIFEST.json'),'files':[receipt(q) for q in sorted((stage/'runtime').rglob('*')) if q.is_file()],'helper':receipt(stage/'runtime/hpo_frontier.py'),'amender':receipt(stage/'amend_controller.py'),'QA':receipt(stage/'ENGINEERING_QA.json'),'AST_proof':receipt(stage/'AST_DELTA_PROOF.json'),'scientific_fits':0,'GO':False};(stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
for q in stage.rglob('*'):
 if q.is_file():q.chmod(0o444)
artifacts=[]
for name in ('MANIFEST.json','ENGINEERING_QA.json','AST_DELTA_PROOF.json'):
 q=stage/name;raw=q.read_bytes();artifacts.append({**receipt(q),'name':name,'data':base64.b64encode(raw).decode()})
print(json.dumps({'manifest':receipt(stage/'MANIFEST.json'),'helper':manifest['helper'],'amender':manifest['amender'],'artifacts':artifacts}))
'''
payload = base64.b64encode(zlib.compress(json.dumps({'destination': OUT.name, 'files': files}).encode(), 9)).decode()
result = helper.invoke(code, [payload], transport, 'STAGE_HPO_HEADROOM_WAIT_CORRECTION_V4', 60)
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
