"""Guarded fresh final packet, finite census launch and exact owner status."""
import argparse
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
STAGE = r'''
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])));bindings=payload['bindings']
for name in ('PLAN','SUMMARY','INPUT_INDEX','CONTROLLER_MANIFEST'):
 row=bindings[name];raw=(phase/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
stage=phase/'final_sync_master_v29_20261013_v1';assert not stage.exists();assert not (phase/'final_census_master_v29_20261013_v1').exists();stage.mkdir()
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];q=stage/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw);q.chmod(0o444)
syntax=subprocess.run(['bash','-n',str(stage/'run_gpu.sh')],capture_output=True,text=True,timeout=10);assert syntax.returncode==0,syntax.stderr
launch=stage/'census_launch';launch.mkdir();command=[str(repo/'.venv/bin/python'),'-B',str(stage/'preparation/census_owner.py')];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (launch/'STDOUT.log').open('x') as stdout,(launch/'STDERR.log').open('x') as stderr:child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split();owner={'pid':child.pid,'start_ticks':int(fields[19]),'pgid':int(fields[2])};value={'owner':owner,'command':command,'bindings':bindings,'scientific_fits':0,'GO':False};(launch/'LAUNCH_RECEIPT.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps({'census_launch':value,'wrapper_bytes':(stage/'run_gpu.sh').stat().st_size,'wrapper_sha256':hashlib.sha256((stage/'run_gpu.sh').read_bytes()).hexdigest(),'scientific_fits':0,'GO':False}))
'''
STATUS = r'''
stage=phase/'final_sync_master_v29_20261013_v1';launch=stage/'census_launch';receipt=json.loads((launch/'LAUNCH_RECEIPT.json').read_text());owner=receipt['owner'];stat=Path('/proc')/str(owner['pid'])/'stat';alive=False
if stat.exists():
 fields=stat.read_text().rsplit(')',1)[1].split();alive=int(fields[19])==owner['start_ticks'] and int(fields[2])==owner['pgid'] and fields[0]!='Z'
result=stage/'census_owner_attempt1/OWNER_RESULT.json';value={'owner_alive':alive,'owner':owner,'result':json.loads(result.read_text()) if result.is_file() else None,'scientific_fits':0,'GO':False}
if not alive and not result.is_file():value['startup_failure_logs']={name:(launch/name).read_text()[-6000:] for name in ('STDOUT.log','STDERR.log')}
if result.is_file():
 value['child_stdout_tail']=(stage/'census_owner_attempt1/EXACT_FINAL_CENSUS_STDOUT.log').read_text()[-3000:];value['child_stderr_tail']=(stage/'census_owner_attempt1/EXACT_FINAL_CENSUS_STDERR.log').read_text()[-6000:]
print(json.dumps(value))
'''
parser = argparse.ArgumentParser()
parser.add_argument('operation', choices=('stage_census', 'census_status'))
parser.add_argument('--label', required=True)
args = parser.parse_args()
if args.operation == 'stage_census':
    rows = []
    for name in ('EXACT_FINAL_BINDINGS.json', 'run_gpu.sh', 'preparation/census_owner.py', 'preparation/operations.py'):
        raw = (OUT / name).read_bytes()
        rows.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'data': base64.b64encode(raw).decode()})
    payload = {'bindings': json.loads((OUT / 'EXACT_FINAL_BINDINGS.json').read_text()), 'files': rows}
    result = helper.invoke(helper.GUARD + STAGE, [base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()], transport, args.label, 60)
else:
    result = helper.invoke(helper.GUARD + STATUS, [], transport, args.label, 60)
assert result['exit_code'] == 0, result['stdout']
value = json.loads(result['stdout'].strip())
path = OUT / (args.label + '_RESULT.json')
with path.open('x') as stream:
    stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
print(json.dumps(value, sort_keys=True))
