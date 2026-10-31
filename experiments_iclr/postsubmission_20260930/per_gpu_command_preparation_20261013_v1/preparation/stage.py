"""Guarded allocation staging; real-launcher QA uses fabricated metadata only."""
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
rows = []
for name in ('runtime/gpu_command.py', 'amend_controller.py', 'tests/check_gpu_commands.py', 'preparation/stage.py'):
    raw = (OUT / name).read_bytes()
    rows.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'data': base64.b64encode(raw).decode()})
code = helper.GUARD + r'''
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
base=phase/'native_structural_cardinality_controller_delta_20261013_v2/runtime';shutil.copytree(base,stage/'runtime')
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];q=stage/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');records=[]
for label,args in [('SOURCE_INVENTORY_ONLY_AMEND',[str(stage/'amend_controller.py'),str(stage/'runtime')]),('GPU_COMMAND_ACTUAL_LAUNCHER_SYNTHETIC_QA',[str(stage/'tests/check_gpu_commands.py'),'--controller',str(stage/'runtime')])]:
 tick=time.monotonic();r=subprocess.run([str(repo/'.venv/bin/python'),'-B',*args],cwd=stage,env=env,capture_output=True,text=True,timeout=30);record={'label':label,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'wall_seconds':time.monotonic()-tick,'fits':0,'GO':False};(stage/(label+'_RESULT.json')).write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');records.append(record)
 if r.returncode:print(json.dumps({'records':records}));sys.exit(1)
assert (base/'launcher.py').read_bytes()==(stage/'runtime/launcher.py').read_bytes()
def receipt(q):
 raw=q.read_bytes();return {'path':str(q.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
manifest={'schema':'per-GPU-companion-preparation-packet-v1','base_controller_manifest':receipt(base.parent/'MANIFEST.json'),'files':[receipt(q) for q in sorted((stage/'runtime').rglob('*')) if q.is_file()],'helper':receipt(stage/'runtime/gpu_command.py'),'amender':receipt(stage/'amend_controller.py'),'QA':receipt(stage/'GPU_COMMAND_ACTUAL_LAUNCHER_SYNTHETIC_QA_RESULT.json'),'source_inventory_amendment':receipt(stage/'SOURCE_INVENTORY_ONLY_AMEND_RESULT.json'),'actual_launcher_unchanged':True,'models':0,'fits':0,'GO':False}
(stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
for q in stage.rglob('*'):
 if q.is_file():q.chmod(0o444)
artifacts=[]
for name in ('MANIFEST.json','SOURCE_INVENTORY_ONLY_AMEND_RESULT.json','GPU_COMMAND_ACTUAL_LAUNCHER_SYNTHETIC_QA_RESULT.json'):
 q=stage/name;raw=q.read_bytes();artifacts.append({**receipt(q),'name':name,'data':base64.b64encode(raw).decode()})
print(json.dumps({'manifest':receipt(stage/'MANIFEST.json'),'helper':manifest['helper'],'amender':manifest['amender'],'artifacts':artifacts}))
'''
payload = base64.b64encode(zlib.compress(json.dumps({'destination': OUT.name, 'files': rows}).encode(), 9)).decode()
result = helper.invoke(code, [payload], transport, 'STAGE_PER_GPU_COMPANION_ACTUAL_LAUNCHER_QA', 120)
assert result['exit_code'] == 0, result['stdout']
value = json.loads(result['stdout'].strip())
for row in value.pop('artifacts'):
    raw = base64.b64decode(row['data'])
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    target = OUT / row['name']
    assert not target.exists()
    target.write_bytes(raw)
    target.chmod(0o444)
(OUT / 'ALLOCATION_STAGE_RESULT.json').write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
print(json.dumps(value, sort_keys=True))
