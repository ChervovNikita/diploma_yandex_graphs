"""Observe this fixed owner without reading partial predictive outcomes."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = r'''
import json,subprocess,socket
from pathlib import Path
from datetime import datetime,timezone
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
H=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_neighborhood_distribution_pilot_root_20261010_v1')
s=json.loads((H/'OWNER_START.json').read_text())
assert s['owner']['PID']==621814 and s['child']['PID']==621816
assert s['boot_id']=='24c315a7-3c08-471f-b550-b9a3e1faf75d'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==s['boot_id']
processes={}
for name in ['owner','child']:
 q=s[name]; f=Path('/proc',str(q['PID']),'stat')
 if f.exists():
  v=f.read_text().rsplit(')',1)[1].split();processes[name]={'PID':q['PID'],'present':True,'same_start_ticks':int(v[19])==q['start_ticks'],'state':v[0]}
 else:processes[name]={'PID':q['PID'],'present':False}
pr=H/'actual_family_v1/PROGRESS.json'
progress=json.loads(pr.read_text()) if pr.is_file() else None
if progress is not None:progress={k:progress[k] for k in ['arm','seed','member','update','completed_units'] if k in progress}
end=H/'OWNER_END.json'
v={'UTC':datetime.now(timezone.utc).isoformat(),'processes':processes,'progress':progress,'owner_end':json.loads(end.read_text()) if end.is_file() else None,'stderr_tail':(H/'worker.stderr.log').read_text()[-1800:] if (H/'worker.stderr.log').is_file() else None}
print('GNNM_OBSERVATION='+json.dumps(v))
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True, type=int)
    args = parser.parse_args()
    observation = HERE / f'LIVE_OBSERVATION_V{args.version}.json'
    transport = HERE / f'OBSERVATION_TRANSPORT_V{args.version}.json'
    assert args.version >= 3 and not observation.exists() and not transport.exists()
    command = ['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
               '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
               '-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
               'python3 -c ' + shlex.quote(REMOTE)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=55)
    transport.write_text(json.dumps(dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2)+'\n')
    assert result.returncode == 0, result.stderr[-1800:]
    lines = [x.split('=',1)[1] for x in result.stdout.splitlines() if x.startswith('GNNM_OBSERVATION=')]
    assert len(lines) == 1
    value = json.loads(lines[0])
    observation.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value, indent=2))

if __name__ == '__main__':
    main()
