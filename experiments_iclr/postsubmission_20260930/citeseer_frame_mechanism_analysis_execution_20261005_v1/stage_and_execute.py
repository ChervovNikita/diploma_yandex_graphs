"""Stage exact root-reviewed v2 and execute it once on the authorized host."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
P = HERE.parent
SOURCE = P / 'citeseer_frame_mechanism_analysis_preparation_20261005_v2'
SCRIPT_SHA = 'd28babdc92a0b001effd55ae4707bad023a6431a79e603aafa046d3fa0d6490e'
MANIFEST_SHA = 'f4c59f4715f833d151a586c0660fabc1101b408dcb976f15aa6009484d5089c4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    assert not (HERE / 'TRANSPORT_RESULT.json').exists()
    assert sha(SOURCE / 'analyze_mechanism.py') == SCRIPT_SHA
    assert sha(SOURCE / 'SOURCE_MANIFEST.json') == MANIFEST_SHA
    seal = json.loads((SOURCE / 'SEAL.json').read_text())
    assert seal['source_manifest_sha256'] == MANIFEST_SHA
    for row in json.loads((SOURCE / 'SOURCE_MANIFEST.json').read_text())['files']:
        raw = (SOURCE / row['path']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    admission = dict(source_review_approved=True, script_sha256=SCRIPT_SHA,
        source_manifest_sha256=MANIFEST_SHA,
        plan_sha256='fd9fec451a81d0512cd8431ba2a990c580592ff6a6e8e12d784386a99c5e9daa',
        mechanism_protocol_sha256='71d42989be1c16c4eef423eb1b66abdc321fc759783054faa2b34c9b62b52d28',
        cohort_freeze_sha256='120f57e53377494a055a994405581fd9fb1fcfe0c5636bdf4446dd8735bd94f8',
        analysis_RESULTS_sha256='d675f6ccd3d4d3eef1da88c445c82bc28c4ba2b664710b057e6630a491127ec6',
        analysis_START_sha256='3affba6d2e63280da7a3c4730119a5b1ad06068f26d9622096c864812becb4d4',
        served_VALID_predictions_sha256='681e8a93d10baba356876100a47452b6388236d8bc9992c091ce2fa942661b09',
        output_relative='citeseer_frame_mechanism_analysis_execution_20261005_v1',
        hard_child_wall_limit_seconds=120, single_execution_no_retry=True,
        root_authorization='Explicit parent root authorization for one bounded CPU execution; both completion gates verified.')
    save('ROOT_ADMISSION.json', admission)
    files = {path.name: base64.b64encode(path.read_bytes()).decode()
             for path in sorted(SOURCE.iterdir()) if path.is_file()}
    payload = json.dumps(dict(files=files, admission=admission))
    code = r'''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,signal,socket,subprocess,sys,time
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
data=json.loads(sys.stdin.read());admission=data['admission']
source=phase/'citeseer_frame_mechanism_analysis_preparation_20261005_v2'
stage=phase/'citeseer_frame_mechanism_analysis_execution_stage_20261005_v1'
assert not stage.exists();stage.mkdir()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):
 with (stage/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
save('STAGING_START.json',dict(UTC=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),source_manifest_sha256=admission['source_manifest_sha256'],scientific_executions=0))
try:
 if source.exists():
  assert source.is_dir() and set(p.name for p in source.iterdir())==set(data['files'])
 else:source.mkdir()
 for name,encoded in data['files'].items():
  assert Path(name).name==name
  raw=base64.b64decode(encoded)
  path=source/name
  if path.exists():assert path.read_bytes()==raw
  else:
   with path.open('xb') as stream:stream.write(raw)
   path.chmod(0o444)
 assert sha(source/'analyze_mechanism.py')==admission['script_sha256']
 assert sha(source/'SOURCE_MANIFEST.json')==admission['source_manifest_sha256']
 for item in json.loads((source/'SOURCE_MANIFEST.json').read_text())['files']:
  assert sha(source/item['path'])==item['sha256'] and (source/item['path']).stat().st_size==item['bytes']
 admission_path=stage/'ROOT_ADMISSION.json'
 save('ROOT_ADMISSION.json',admission)
 interpreter=phase/'native_ncn_runtime_20261005_v1/.venv/bin/python'
 assert interpreter.is_file()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
          PYTHONPATH=str(phase/'native_ncn_dependency_overlay_20261005_v1')+os.pathsep+str(repo/'.venv/lib/python3.11/site-packages'))
 command=[str(interpreter),'-B',str(source/'analyze_mechanism.py'),'--admission',str(admission_path)]
 started=time.monotonic()
 with (stage/'stdout.log').open('xb') as out,(stage/'stderr.log').open('xb') as err:
  child=subprocess.Popen(command,cwd=repo,env=env,stdout=out,stderr=err,start_new_session=True)
  save('EXECUTION_START.json',dict(UTC=datetime.now(timezone.utc).isoformat(),command=command,pid=child.pid,
       CUDA_VISIBLE_DEVICES='',threads=2,hard_wall_seconds=120,scientific_executions=1,no_retry=True))
  timeout=False
  try:exit_code=child.wait(timeout=120)
  except subprocess.TimeoutExpired:
   timeout=True;os.killpg(child.pid,signal.SIGKILL);exit_code=child.wait(timeout=10)
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=exit_code,timeout=timeout,
             inclusive_seconds=time.monotonic()-started,scientific_executions=1,no_retry=True,TEST_access=False,
             fits=0,optimizer_updates=0,all_partial_artifacts_preserved=True,
             stdout_sha256=sha(stage/'stdout.log'),stderr_sha256=sha(stage/'stderr.log'),stage_path=str(stage),
             output_relative=admission['output_relative'])
 output=phase/admission['output_relative']
 if exit_code==0 and not timeout:
  assert (output/'RESULTS.json').is_file()
  result['mechanism_RESULTS_sha256']=sha(output/'RESULTS.json')
  result['status']='COMPLETE'
 else:result['status']='FAILED_OR_TIMEOUT_NO_RETRY'
 save('EXECUTION_RESULT.json',result)
 print(json.dumps(result))
 if exit_code!=0:sys.exit(1)
except (Exception,KeyboardInterrupt) as error:
 failure=dict(UTC=datetime.now(timezone.utc).isoformat(),error=type(error).__name__+': '+str(error),partial_artifacts_preserved=True,no_retry=True)
 save('STAGING_OR_EXECUTION_FAILURE.json',failure)
 print(json.dumps(failure));raise
'''
    ssh = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=15',
           'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
    save('TRANSPORT_START.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
         target='authorized anogena-2 only', hard_child_seconds=120, no_retry=True,
         remote_code_sha256=hashlib.sha256(code.encode()).hexdigest(), payload_bytes=len(payload.encode())))
    try:
        completed = subprocess.run([*ssh, shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', code])],
                                   input=payload, capture_output=True, text=True, timeout=160)
        value = dict(UTC=datetime.now(timezone.utc).isoformat(), transport_exit_code=completed.returncode,
                     stdout=completed.stdout, stderr=completed.stderr, no_retry=True)
        save('TRANSPORT_RESULT.json', value)
        print(json.dumps(value))
    except (Exception, KeyboardInterrupt) as error:
        save('TRANSPORT_FAILURE.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
             error=type(error).__name__ + ': ' + str(error), no_retry=True,
             disposition='Do not relaunch. Preserve server attempt and inspect owned terminal metadata only.'))
        raise


if __name__ == '__main__':
    main()
