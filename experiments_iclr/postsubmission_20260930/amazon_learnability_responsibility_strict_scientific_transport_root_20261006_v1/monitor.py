"""Read only owned process and fit-progress metadata; no checkpoints or scores."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
out=phase/'amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v1'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
result={'UTC':datetime.now(timezone.utc).isoformat(),'files':[],'owned_processes':[],'scores_or_checkpoints_read':False}
def metadata(relative):
 p=out/relative
 if not p.exists():return None
 assert p.resolve().is_relative_to(out) and p.is_file() and not p.is_symlink() and p.stat().st_size<2_000_000
 b=p.read_bytes();result['files'].append({'path':relative,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'utf8':b.decode()})
 try:return json.loads(b)
 except json.JSONDecodeError:return None
def handle(owner):
 proc=Path('/proc')/str(owner['PID'])
 try:raw=(proc/'stat').read_text()
 except FileNotFoundError:return {'PID':owner['PID'],'start_ticks':owner['start_ticks'],'known_handle_absent':True}
 f=raw[raw.rfind(')')+2:].split()
 if int(f[19])!=owner['start_ticks']:return {'PID':owner['PID'],'start_ticks':owner['start_ticks'],'known_handle_absent':True,'foreign_process_not_inspected':True}
 argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
 return {'PID':owner['PID'],'start_ticks':int(f[19]),'state':f[0],'known_handle_absent':False,'owned_argv_matches':argv==owner['argv'],'owned_cwd_matches':str((proc/'cwd').resolve())==owner['cwd']}
for relative in ('SUPERVISOR_LAUNCH.json','CHILD_LAUNCH.json','TERMINAL.json','output/RUNNER_RESULT.json','output/scientific_run/COMPLETE.json','output/scientific_run/FAILURE.json'):
 value=metadata(relative)
 if relative in ('SUPERVISOR_LAUNCH.json','CHILD_LAUNCH.json') and value:result['owned_processes'].append({'role':relative,**handle(value)})
p=out/'output/scientific_run/ATTEMPTED_OPERATIONS.jsonl'
if p.exists():
 assert p.resolve().is_relative_to(out) and not p.is_symlink()
 with p.open('rb') as stream:
  length=p.stat().st_size;stream.seek(max(0,length-131072));tail=stream.read()
 allowed={'acquisition_update_completed','episode_started','episode_completed','all_six_endpoints_completed','phase_started'}
 rows=[]
 for line in tail.splitlines():
  try:item=json.loads(line)
  except json.JSONDecodeError:continue
  if item.get('kind') in allowed:rows.append(item)
 result['progress_events']=rows[-8:]
 result['event_stream_observed_bytes']=length
print(json.dumps(result))
'''


def main():
    command = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
               '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=20',
               'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
               'cd ' + shlex.quote(REPO) + ' && exec /usr/bin/python3 -I -S -B -c ' + shlex.quote(REMOTE)]
    snapshot = PHASE / 'amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v1' / ('monitor_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    snapshot.mkdir()
    r = subprocess.run(command, capture_output=True, text=True, timeout=50)
    receipt = {'UTC': datetime.now(timezone.utc).isoformat(), 'exit_code': r.returncode,
               'stderr': r.stderr, 'remote_source_sha256': hashlib.sha256(REMOTE.encode()).hexdigest(),
               'scientific_fit_or_score_called_here': False}
    if r.returncode == 0:
        receipt['result'] = json.loads(r.stdout)
        for row in receipt['result']['files']:
            b = row['utf8'].encode()
            assert len(b) == row['bytes'] and hashlib.sha256(b).hexdigest() == row['sha256']
            path = snapshot / row['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b)
    else:
        receipt['stdout'] = r.stdout
    (snapshot / 'MONITOR_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    result = receipt.get('result', {})
    print(json.dumps({'snapshot': str(snapshot), 'exit_code': r.returncode,
        'owned_processes': result.get('owned_processes'), 'progress_events': result.get('progress_events'),
        'status_files': [{'path':row['path'],'sha256':row['sha256']} for row in result.get('files', [])]}))


if __name__ == '__main__':
    main()
