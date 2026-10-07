"""Observe only owned CPU work and retrieve compact, nonpredictive receipts."""
import base64
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=15',
       '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-p', '2222', LOGIN]
REMOTE = r'''
import base64,datetime,hashlib,json,socket
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
A=P/'allocation_controls_molhiv_CPU_representative_activation_root_20261007_v1'
E=P/'allocation_controls_molhiv_CPU_representative_execution_root_20261007_v1/run01'
launch=json.loads((A/'LAUNCH.json').read_text())
def ident(pid):
 try:
  f=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(') ',1)[1].split()
  return {'PID':pid,'start_ticks':int(f[19]),'state':f[0]}
 except FileNotFoundError:return None
parent=ident(launch['parent_PID'])
assert parent is None or parent['start_ticks']==launch['parent_start_ticks']
owner=json.loads((E/'OWNER.json').read_text()) if (E/'OWNER.json').is_file() else None
worker=ident(owner['worker_PID']) if owner else None
assert worker is None or worker['start_ticks']==owner['worker_start_ticks']
terminal=json.loads((E/'TERMINAL.json').read_text()) if (E/'TERMINAL.json').is_file() else None
progress=json.loads((E/'work/PROGRESS.json').read_text()) if (E/'work/PROGRESS.json').is_file() else None
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent':parent,'worker':worker,
 'terminal':terminal,'progress':progress,'quality_values_requested':False,'files':[]}
if terminal is not None and parent is None and worker is None:
 for relative in ['INVOCATION.json','OWNER.json','TERMINAL.json','WORK_RECEIPT.json','PARENT_FAILURE.json','CLEANUP_FAILURE.json',
  'work/PARAMETER_CATALOG.json','work/PROGRESS.json','work/CANDIDATE.json','work/FAILURE.json',
  'work/P/UPDATE_CHECK.json','work/P/CASE_CHECK.json','work/G/UPDATE_CHECK.json','work/G/CASE_CHECK.json']:
  path=E/relative
  if path.is_file():
   data=path.read_bytes();assert len(data)<500000
   out['files'].append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()})
print(json.dumps(out))
'''


def main():
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE])
    result = subprocess.run([*SSH, command], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    files = value.pop('files')
    value['retrieved_files'] = [{k: row[k] for k in ('path', 'bytes', 'sha256')} for row in files]
    number = len(list(HERE.glob('OBSERVATION*.json'))) + 1
    path = HERE / ('OBSERVATION%02d.json' % number)
    assert not path.exists()
    path.write_text(json.dumps(value, indent=2) + '\n')
    for row in files:
        data = base64.b64decode(row['data'])
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        target = HERE / 'receipts/run01' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == data
        else:
            target.write_bytes(data)
    if files:
        receipt = HERE / 'RETRIEVAL_RECEIPT.json'
        assert not receipt.exists()
        receipt.write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), files=value['retrieved_files'],
                                          selected_weights_or_scores_transferred=False), indent=2) + '\n')
    text = dict(UTC=value['UTC'], parent=value['parent'], worker=value['worker'], progress=value['progress'],
                terminal=value['terminal'], retrieved_files=len(files))
    print(json.dumps(text))


if __name__ == '__main__':
    main()
