"""Observe fixed Physics lifecycle metadata; no prediction/target reads."""
import argparse
import base64
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H = Path(__file__).resolve().parent
P = H.parent
REMOTE = r'''
import base64,zlib,json,sys,socket,subprocess,pathlib,datetime
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
H=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_SAGE_Physics_confirmation_root_20261011_v1')
def get(rel):
    p=H/rel
    return json.loads(p.read_text()) if p.exists() else None
v=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),launch=get('SCIENTIFIC_LAUNCH_V1.json'),external_start=get('external_cohort_owner_v1/OWNER_START.json'),external_end=get('external_cohort_owner_v1/OWNER_END.json'),cohort_start=get('cohort_owner_v1/OWNER_START.json'),cohort_end=get('cohort_owner_v1/OWNER_END.json'),core_progress=get('core_acquisition_v1/PROGRESS.json'),core_failure=get('core_acquisition_v1/FAILURE.json'),CAL_CONFIRM_truth_or_prediction_read=False)
author=H/'author_acquisition_v1'
v['author_progress']={p.parent.name:json.loads(p.read_text()) for p in sorted(author.glob('*/PROGRESS.json'))}
ac=get('author_acquisition_v1/COMPLETE_AUTHOR_CONTEXT.json')
v['author_complete_metadata']={k:ac.get(k) for k in ('terminal','complete','banks','fits','failure')} if ac else None
v['after_fit_pipeline']={key:get(rel) for key,rel in {'launch':'AFTER_FIT_PIPELINE_LAUNCH_V1.json','start':'after_fit_pipeline_v1/PIPELINE_START.json','failure':'after_fit_pipeline_v1/PIPELINE_FAILURE.json','end':'after_fit_pipeline_v1/PIPELINE_END.json'}.items()}
launch=v['after_fit_pipeline']['launch']
if launch:
    pid=launch['watcher_PID'];stat=pathlib.Path(f'/proc/{pid}/stat')
    v['after_fit_pipeline']['process_present']=stat.exists()
    if stat.exists():v['after_fit_pipeline']['process_state']=stat.read_text().rsplit(')',1)[1].split()[0]
    if not stat.exists() and not v['after_fit_pipeline']['end']:
        log=H/'after_fit_watcher_outer.log';v['after_fit_pipeline']['failure_log']=log.read_text()[-5000:] if log.exists() else None
v['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).splitlines()
if v['external_end'] and v['external_end']['exit_code']!=0:
    v['failure_logs']={str(p.relative_to(H)):p.read_text()[-5000:] for p in (H/'external_cohort_owner_v1/cohort.log',H/'cohort_owner_v1/core.log',H/'cohort_owner_v1/author.log') if p.exists()}
print(json.dumps(v))
'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--version', type=int, required=True); a = ap.parse_args()
    path = H / f'COHORT_OBSERVATION_V{a.version}.json'; assert not path.exists()
    spec = importlib.util.spec_from_file_location('metadata_transport', P / 'publication/publish_exact_inventory_v19.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    args = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
            shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, 'Physics-metadata', 'literal-route', 'existing-owned-processes'])]
    r = m.terminal_transport(args, input=base64.b64encode(zlib.compress(b'{}')).decode(), capture_output=True, text=True, timeout=60)
    if r.returncode:
        value = dict(UTC=datetime.now(timezone.utc).isoformat(), transport_failed=True, stdout=r.stdout, stderr=r.stderr, exit_code=r.returncode)
    else: value = json.loads(r.stdout)
    path.write_text(json.dumps(value, indent=2) + '\n'); print(json.dumps(value))


if __name__ == '__main__': main()
