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
for name in ('runtime/build_run_census.py', 'tests/check_census.py', 'preparation/stage.py'):
    raw = (OUT / name).read_bytes()
    files.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'data': base64.b64encode(raw).decode()})
code = helper.GUARD + r'''
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];q=stage/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');tick=time.monotonic();r=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(stage/'tests/check_census.py')],cwd=stage,env=env,capture_output=True,text=True,timeout=30);qa={'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'wall_seconds':time.monotonic()-tick,'fits':0,'GO':False};(stage/'SYNTHETIC_QA.json').write_text(json.dumps(qa,indent=2,sort_keys=True)+'\n');assert r.returncode==0,qa
def receipt(q):
 raw=q.read_bytes();return {'path':str(q.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
manifest={'schema':'final-exact-census-generator-sealed-preparation-v1','files':[receipt(q) for q in sorted(stage.rglob('*')) if q.is_file()],'final_master_runs':0,'scientific_fits':0,'GO':False};(stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
for q in stage.rglob('*'):
 if q.is_file():q.chmod(0o444)
artifacts=[]
for name in ('MANIFEST.json','SYNTHETIC_QA.json'):
 q=stage/name;raw=q.read_bytes();artifacts.append({**receipt(q),'name':name,'data':base64.b64encode(raw).decode()})
print(json.dumps({'manifest':receipt(stage/'MANIFEST.json'),'artifacts':artifacts}))
'''
payload = base64.b64encode(zlib.compress(json.dumps({'destination': OUT.name, 'files': files}).encode(), 9)).decode()
result = helper.invoke(code, [payload], transport, 'STAGE_EXACT_FINAL_RUN_CENSUS_GENERATOR', 60)
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
