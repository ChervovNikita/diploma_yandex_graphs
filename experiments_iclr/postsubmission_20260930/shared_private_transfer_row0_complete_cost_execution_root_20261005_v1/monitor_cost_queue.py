"""Read bounded TRAIN-cost receipts and verify known owned process identities."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
CODE=r'''
from datetime import datetime,timezone
import hashlib,json,os,socket,subprocess
from pathlib import Path
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'shared_private_transfer_row0_complete_cost_execution_root_20261005_v1'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
g=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).splitlines()
assert g==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def checked_read(path):
 if not path.exists():return None
 assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root) and path.stat().st_size<2_000_000
 b=path.read_bytes();return dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),utf8=b.decode())
def physical(expected):
 p=Path('/proc')/str(expected['PID'])
 try:
  raw=(p/'stat').read_text();f=raw[raw.rfind(')')+2:].split();argv=[s.decode() for s in (p/'cmdline').read_bytes().split(bytes([0])) if s]
 except FileNotFoundError:return None
 assert int(f[19])==expected['start_ticks']
 if f[0]!='Z':
  assert argv==expected['argv'] and int(f[2])==expected['pgid'] and int(f[3])==expected['sid'] and str((p/'cwd').resolve())==str(repo)
 return dict(PID=expected['PID'],start_ticks=int(f[19]),state=f[0],physically_live=f[0]!='Z',argv=argv)
launch=json.loads((root/'LAUNCH_RECEIPT.json').read_text())
result=dict(UTC=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),queue=physical(launch['queue_identity']),cells=[],files={},fits=0,VALID_TEST_values_access=False,signals_sent=False)
for name in ['QUEUE_STARTED.json','QUEUE_PROGRESS.json','QUEUE_RESULT.json','QUEUE_FAILURE.json']:
 row=checked_read(root/'queue'/name)
 if row is not None:result['files']['queue/'+name]=row
plan=json.loads((phase/'shared_private_transfer_row0_complete_cost_queue_released_20261005_v1/PLAN.json').read_text())
for cell in plan['cells']:
 folder=root/'queue'/cell['cell_id'];child=checked_read(folder/'CHILD_STARTED.json')
 row=dict(cell_id=cell['cell_id'],child=None,files={})
 if child is not None:
  row['child']=physical(json.loads(child['utf8'])['identity']);row['files']['CHILD_STARTED.json']=child
 for name in ['EXECUTION_RECEIPT.json','result/PROGRESS.json','result/COST_RESULT.json','result/FAILURE.json']:
  record=checked_read(folder/name)
  if record is not None:row['files'][name]=record
 if child or row['files']:result['cells'].append(row)
failure=result['files'].get('queue/QUEUE_FAILURE.json')
if failure:
 for cell in result['cells']:
  path=root/'queue'/cell['cell_id']/'child.stderr.log'
  if path.exists():
   with path.open('rb') as stream:stream.seek(max(0,path.stat().st_size-8192));cell['failure_stderr_tail']=stream.read().decode(errors='replace')
print(json.dumps(result))
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',required=True,type=int);args=parser.parse_args()
    dest=HERE/('observation_%03d'%args.sequence);dest.mkdir()
    argv=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
      'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec python3 -I -c '+shlex.quote(CODE)]
    run=subprocess.run(argv,capture_output=True,text=True,timeout=50)
    (dest/'TRANSPORT.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr},indent=2)+'\n')
    if run.returncode:raise RuntimeError(run.stderr)
    result=json.loads(run.stdout);(dest/'OBSERVATION.json').write_text(json.dumps(result,indent=2)+'\n')
    for cell in result['cells']:
        for name,row in cell['files'].items():
            assert hashlib.sha256(row['utf8'].encode()).hexdigest()==row['sha256']
            path=dest/cell['cell_id']/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(row['utf8'])
    compact={'UTC':result['UTC'],'queue_physically_live':bool(result['queue'] and result['queue']['physically_live']),'cells':[],'failure':result['files'].get('queue/QUEUE_FAILURE.json') is not None,'complete':result['files'].get('queue/QUEUE_RESULT.json') is not None}
    for cell in result['cells']:
        row={'cell_id':cell['cell_id'],'child_physically_live':bool(cell['child'] and cell['child']['physically_live'])}
        for name,key in [('result/PROGRESS.json','progress'),('EXECUTION_RECEIPT.json','receipt'),('result/FAILURE.json','failure')]:
            if name in cell['files']:
                value=json.loads(cell['files'][name]['utf8'])
                row[key]={k:value[k] for k in ['cycle','episode','episodes_in_cycle','inclusive_seconds','passed','reason','error','complete_cycle_costs'] if k in value}
        if 'failure_stderr_tail' in cell:row['failure_stderr_tail']=cell['failure_stderr_tail']
        compact['cells'].append(row)
    (dest/'SUMMARY.json').write_text(json.dumps(compact,indent=2)+'\n');print(json.dumps(compact))

if __name__=='__main__':main()
