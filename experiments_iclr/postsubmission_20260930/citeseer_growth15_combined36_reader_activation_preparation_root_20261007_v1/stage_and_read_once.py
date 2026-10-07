"""Approved one-shot singleton staging/CPU read; no scientific job operations."""
import base64
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
PAYLOAD = json.loads((HERE / 'STAGE_PAYLOAD.json').read_text())
REMOTE = r'''
import base64,datetime,gzip,hashlib,json,os,resource,socket,subprocess,time
from pathlib import Path
payload = json.loads(PAYLOAD_JSON)
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
gpu='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as stream:stream.write(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[gpu]
os.chdir(repo);assert Path.cwd().resolve()==repo
# Metadata-only preflight before staging/readout; actual reader repeats its full custody gate.
G=phase/'citeseer_growth_full15_flat_execution_root_20261007_v1'
assert not Path('/proc/495546').exists(), 'Original growth owner is still present; do not read partial cohort'
assert (G/'owner/COMPLETE.json').is_file(), 'No authoritative complete growth owner receipt'
assert not (G/'owner/FAILURE.json').exists()
assert len(list((G/'runs').iterdir()))==15
assert all((q/'FREEZE.json').is_file() and not (q/'FAILURE.json').exists() for q in (G/'runs').iterdir())
source=phase/payload['source_name'];activation=phase/payload['activation_name']
assert source.parent==activation.parent==phase and not activation.exists()
source.mkdir(exist_ok=True)
for row in payload['files']:
 assert Path(row['name']).name==row['name']
 raw=base64.b64decode(row['b64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 p=source/row['name'];assert p.resolve().is_relative_to(source.resolve())
 if p.exists():assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 else:
  with p.open('xb') as stream:stream.write(raw)
 p.chmod(0o444)
assert sha(source/'MANIFEST.json')==payload['source_manifest_sha256']
manifest=json.loads((source/'MANIFEST.json').read_text())
for row in manifest['files']:assert sha(source/row['path'])==row['sha256'] and (source/row['path']).stat().st_size==row['bytes']
assert sha(source/'read_closed_cohort.py')==payload['reader_sha256']
activation.mkdir();config=activation/'CONFIG_ACTIVATED.json'
raw=base64.b64decode(payload['config_b64']);assert hashlib.sha256(raw).hexdigest()==payload['config_sha256']
with config.open('xb') as stream:stream.write(raw)
config.chmod(0o444);cfg=json.loads(raw)
assert cfg['requested_families']==['initialization','growth'] and cfg['TEST_access'] is False
assert cfg['output_relative']=='citeseer_initialization21_growth15_combined36_closed_cohort_readout_root_20261007_v2'
write(activation/'STAGE_RECEIPT.json',{'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'physical_gpu_uuid':gpu,'source_manifest_sha256':payload['source_manifest_sha256'],'reader_sha256':payload['reader_sha256'],'config_sha256':payload['config_sha256'],'source_files_verified':len(payload['files']),'requested_families':['initialization','growth'],'TEST_access':False,'scientific_jobs_launched_or_signalled':False})
python=str(phase/'native_ncn_runtime_20261005_v1/.venv/bin/python')
argv=[python,'-B',str(source/'read_closed_cohort.py'),'--config',str(config),'--config-sha256',payload['config_sha256']]
environment=dict(os.environ);environment.pop('PYTHONHOME',None)
environment.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(phase/'native_ncn_dependency_overlay_20261005_v1')+':'+str(repo/'.venv/lib/python3.11/site-packages'))
write(activation/'INVOCATION.json',{'argv':argv,'CPU_environment':{k:environment[k] for k in ('CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH')},'requested_families':['initialization','growth'],'TEST_access':False,'existing_whole_family_gate_required':True})
started=time.monotonic()
with (activation/'cpu_reader.stdout.log').open('xb') as out,(activation/'cpu_reader.stderr.log').open('xb') as err:
 completed=subprocess.run(argv,cwd=repo,env=environment,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
elapsed=time.monotonic()-started
output=phase/cfg['output_relative']
record={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':completed.returncode,'elapsed_reader_wall_seconds':elapsed,'maximum_child_RSS_bytes_including_inventory_cli':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,'reader_stdout_sha256':sha(activation/'cpu_reader.stdout.log'),'reader_stderr_sha256':sha(activation/'cpu_reader.stderr.log'),'output_manifest_present':(output/'MANIFEST.json').is_file(),'requested_families':['initialization','growth'],'TEST_access':False,'scientific_jobs_launched_or_signalled':False,'models_loaded':False,'new_predictions':False,'warm1_process_exit_status':'unknown'}
write(activation/'READ_RECEIPT.json',record)
if completed.returncode!=0:
 write(activation/'FAILURE_RECEIPT.json',{'read_receipt':record,'stderr':(activation/'cpu_reader.stderr.log').read_text(),'preserve_all_source_and_outputs':True,'automatic_retries':0,'engineering_loops_started':False})
names=['STAGE_RECEIPT.json','CONFIG_ACTIVATED.json','INVOCATION.json','READ_RECEIPT.json','cpu_reader.stdout.log','cpu_reader.stderr.log']
if completed.returncode!=0:names.append('FAILURE_RECEIPT.json')
exports=[]
for name in names:
 p=activation/name;raw=p.read_bytes();exports.append({'group':'activation','name':name,'sha256':sha(p),'bytes':len(raw),'gzip_b64':base64.b64encode(gzip.compress(raw,mtime=0)).decode()})
if completed.returncode==0:
 assert (output/'MANIFEST.json').is_file()
 for p in sorted(output.iterdir()):
  if p.name=='QUERY_RANKS_AND_COMMON_ERRORS.json':continue  # full query banks remain on server
  assert p.is_file() and p.suffix in ('.json','.csv')
  raw=p.read_bytes();exports.append({'group':'readout','name':p.name,'sha256':sha(p),'bytes':len(raw),'gzip_b64':base64.b64encode(gzip.compress(raw,mtime=0)).decode()})
print(json.dumps({'receipt':record,'exports':exports,'failure_stderr':(activation/'cpu_reader.stderr.log').read_text() if completed.returncode else None}))
'''


def main():
    script = 'PAYLOAD_JSON = ' + repr(json.dumps(PAYLOAD, sort_keys=True)) + '\n' + REMOTE
    (HERE / 'REMOTE_ONE_SHOT.py').write_text(script)
    remote_command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c',
                                 'import sys;exec(sys.stdin.read())'])
    ssh = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
           '-o', 'StrictHostKeyChecking=yes',
           'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru', remote_command]
    (HERE / 'TRANSPORT_COMMAND.json').write_text(json.dumps({'argv': ssh,
        'remote_command_is_one_line': '\n' not in remote_command,
        'script_stdin_sha256': hashlib.sha256(script.encode()).hexdigest()}, indent=2) + '\n')
    started = time.monotonic()
    result = subprocess.run(ssh, input=script.encode(), capture_output=True)
    (HERE / 'TRANSPORT_STDOUT.json').write_bytes(result.stdout)
    (HERE / 'TRANSPORT_STDERR.txt').write_bytes(result.stderr)
    transport = {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'exit_code': result.returncode, 'transport_wall_seconds': time.monotonic() - started,
        'stdout_bytes': len(result.stdout), 'stderr_bytes': len(result.stderr),
        'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(),
        'stderr_sha256': hashlib.sha256(result.stderr).hexdigest(), 'attempts': 1,
        'automatic_retries': 0, 'TEST_access': False, 'scientific_jobs_launched_or_signalled': False}
    (HERE / 'TRANSPORT_RECEIPT.json').write_text(json.dumps(transport, indent=2) + '\n')
    if result.returncode:
        print(json.dumps({'transport': transport, 'transport_stderr': result.stderr.decode(errors='replace')}))
        return
    packet = json.loads(result.stdout)
    for row in packet['exports']:
        assert Path(row['name']).name == row['name'] and row['group'] in ('activation', 'readout')
        raw = gzip.decompress(base64.b64decode(row['gzip_b64']))
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        folder = HERE / ('remote_activation' if row['group'] == 'activation' else 'readout')
        folder.mkdir(exist_ok=True)
        with (folder / row['name']).open('xb') as stream:
            stream.write(raw)
    (HERE / 'RETRIEVAL_RECEIPT.json').write_text(json.dumps({'files': [
        {k: row[k] for k in ('group', 'name', 'sha256', 'bytes')} for row in packet['exports']],
        'reader_exit_code': packet['receipt']['exit_code'], 'TEST_access': False}, indent=2) + '\n')
    print(json.dumps({'transport': transport, 'reader_receipt': packet['receipt'],
                      'failure_stderr': packet['failure_stderr'], 'exported_files': len(packet['exports'])}))


if __name__ == '__main__':
    main()
