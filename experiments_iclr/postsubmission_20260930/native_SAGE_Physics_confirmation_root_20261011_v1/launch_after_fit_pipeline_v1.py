"""Start the committed, prospectively authorized one-shot Physics watcher."""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H = Path(__file__).resolve().parent
P = H.parent
R = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REMOTE = r'''
import base64,zlib,json,sys,socket,subprocess,pathlib,hashlib,os,datetime
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[sys.argv[2]]
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
repo=pathlib.Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930';H=phase/'native_SAGE_Physics_confirmation_root_20261011_v1'
source=phase/'native_SAGE_Physics_after_fit_pipeline_source_20261011_v1/pipeline.py'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==payload['commit']
for name,digest in payload['pinned_files'].items():assert hashlib.sha256((H/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256(source.read_bytes()).hexdigest()==payload['source_sha256']
assert not (H/'AFTER_FIT_PIPELINE_LAUNCH_V1.json').exists() and not (H/'after_fit_pipeline_v1').exists()
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
os.chdir(repo)
with (H/'after_fit_watcher_outer.log').open('x') as log:
    owner=subprocess.Popen([str(repo/'.venv/bin/python'),'-B','-u',str(source),'--config',str(H/'AFTER_FIT_PIPELINE_CONFIG_V1.json'),'--admission',str(H/'AFTER_FIT_PIPELINE_ADMISSION_V1.json')],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='AFTER_FIT_FIXED_PIPELINE_WATCHER_LAUNCHED',watcher_PID=owner.pid,commit=payload['commit'],pinned_files=payload['pinned_files'],source_sha256=payload['source_sha256'],complete_roster_required=True,extra_native_fit_updates=0,current_CAL_TEST_scoring=False,paper_review=False)
(H/'AFTER_FIT_PIPELINE_LAUNCH_V1.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''


def main():
    commit = json.loads((P / 'publication/Physics_running_and_complete_pipeline_20261011_v1/COMMIT_RECEIPT.json').read_text())['commit']
    payload = dict(commit=commit, pinned_files={name: hashlib.sha256((H / name).read_bytes()).hexdigest() for name in ('AFTER_FIT_PIPELINE_CONFIG_V1.json', 'AFTER_FIT_PIPELINE_ADMISSION_V1.json')},
                   source_sha256=hashlib.sha256((P / 'native_SAGE_Physics_after_fit_pipeline_source_20261011_v1/pipeline.py').read_bytes()).hexdigest())
    spec = importlib.util.spec_from_file_location('watcher_launch_transport', P / 'publication/publish_exact_inventory_v19.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    args = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
            shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, R, UUID, 'Physics-one-shot-watcher-launch'])]
    r = m.terminal_transport(args, input=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode(), capture_output=True, text=True, timeout=45)
    (H / 'AFTER_FIT_PIPELINE_LAUNCH_TRANSPORT_V1.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr), indent=2) + '\n')
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1000:]
    value = json.loads(r.stdout)
    (H / 'AFTER_FIT_PIPELINE_LAUNCH_V1.json').write_text(json.dumps(value, indent=2) + '\n'); print(json.dumps(value))


if __name__ == '__main__': main()
