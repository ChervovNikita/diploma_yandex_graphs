"""Collect original evaluation output only after its successful source-bound freeze."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import hashlib,json,socket,subprocess,sys
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'amazon_polynormer_complete15_evaluation_root_preparation_20261005_v1'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
a=json.loads((root/'ROOT_EVALUATION_RELEASE.json').read_text());assert a==json.load(sys.stdin)
output=phase/a['output'];assert output.resolve().is_relative_to(phase) and not output.is_symlink()
out={'scientific_results_opened':False,'TEST_access':False,'new_launches':0,'files':[]}
def copy(p):
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<2000000
 b=p.read_bytes();v=json.loads(b) if p.suffix=='.json' else None
 out['files'].append({'path':str(p.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'utf8':b.decode()});return v
for name in ('DETACHED_LAUNCH.json','RUNNER_STARTED.json','SUPERVISOR_LAUNCH.json','SUPERVISOR_TERMINAL.json'):
 p=root/'detached_launch_v1'/name
 if p.exists():copy(p)
for name in ('LAUNCH.json','TERMINAL.json','FAILURE.json'):
 p=output/name
 if p.exists():copy(p)
if (output/'FREEZE.json').exists():
 freeze=copy(output/'FREEZE.json')
 assert freeze['schema']=='amazon_polynormer_success_freeze_v2' and freeze['status']=='success' and freeze['source']==a['source']
 for row in freeze['files']:
  p=output/row['relative'];assert p.resolve().is_relative_to(output) and not p.is_symlink()
  b=p.read_bytes();assert {'path':str(p.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}==row['descriptor']
 actual=sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name!='FREEZE.json')
 assert actual==sorted(row['relative'] for row in freeze['files'])
 terminal=json.loads((output/'TERMINAL.json').read_text());assert terminal['physical_exit_code']==0 and terminal['status']=='success'
 ready=json.loads((output/'READY.json').read_text());assert ready['source']==a['source'] and ready['kind']=='evaluate'
 result=copy(output/'RESULT.json')
 assert result['schema']=='amazon_polynormer_comparison_v2' and result['status']=='complete' and result['source']==a['source'] and result['closure_freeze']==a['closure_freeze']
 assert result['unique_physical_fits']==15 and result['family_records']==9 and result['TEST_labels_used'] is False
 assert result['unique_scientific_optimizer_steps']==40500 and result['complete_member_training_trajectory_updates']==64800
 copy(output/'READY.json');out['scientific_results_opened']=True;out['status']='success_frozen'
elif (output/'FAILURE.json').exists():
 out['status']='failed'
 for name in ('stderr.txt','OUTER_TRACEBACK.txt'):
  p=output/name
  if p.exists() and p.stat().st_size<2000000:copy(p)
else:out['status']='pending'
print(json.dumps(out))
'''

def main():
    out=HERE/('evaluation_observation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
             '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=20',
             'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
             'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec python3 -I -B -c '+shlex.quote(REMOTE)]
    r=subprocess.run(command,input=(HERE/'ROOT_EVALUATION_RELEASE.json').read_text(),capture_output=True,text=True,timeout=50)
    receipt={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stderr':r.stderr,'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest()}
    if r.returncode:receipt['stdout']=r.stdout
    else:
        v=json.loads(r.stdout)
        for row in v['files']:
            rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
            b=row.pop('utf8').encode();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
            p=out/rel;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(b)
        (out/'OBSERVATION.json').write_text(json.dumps(v,indent=2)+'\n')
        receipt['observation_sha256']=hashlib.sha256((out/'OBSERVATION.json').read_bytes()).hexdigest()
        receipt['scientific_results_opened']=v['scientific_results_opened']
    with (out/'TRANSPORT.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'directory':str(out),'exit_code':r.returncode,'status':v['status'] if r.returncode==0 else 'transport_failed','scientific_results_opened':v['scientific_results_opened'] if r.returncode==0 else False}))
    if r.returncode:raise SystemExit(r.returncode)

if __name__=='__main__':main()
