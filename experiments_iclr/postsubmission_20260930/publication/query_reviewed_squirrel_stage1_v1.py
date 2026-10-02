"""Record source-bound Squirrel progress; expose comparisons only at closure."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

PHASE = Path(__file__).resolve().parents[1]
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REMOTE = r'''
import json,os,pathlib,subprocess,sys
repo=pathlib.Path(sys.argv[1]);uuid=sys.argv[2]
assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()==str(repo)
inventory=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()
assert inventory==[uuid], 'Wrong allocation'
packet=repo/'experiments_iclr/postsubmission_20260930/graph_paired_squirrel_continuation_preparation_v2'
start=json.loads((packet/'launch_root01/START.json').read_text());pid=start['PID']
assert start['route_uuid']==uuid and start['prepared_manifest_sha256']=='3f2d8faa81039fd6e5ceb5fecf7ea4b46bf725b516dbb040827926a92ddde452'
proc=pathlib.Path('/proc')/str(pid)
alive=proc.exists()
if alive:
 assert pathlib.Path(os.readlink(proc/'cwd'))==repo
 assert str(packet/'prototype/continue_squirrel.py') in (proc/'cmdline').read_bytes().decode().split('\0')
run=packet/'runs/root01';terminal=run/'STAGE1.json'
progress=[]
for trace in sorted(run.glob('seed*_split*/*/continuation_trace.jsonl')):
 lines=trace.read_text().splitlines()
 last=json.loads(lines[-1]) if lines else {}
 progress.append(dict(cell=str(trace.parent.relative_to(run)),trace_records=len(lines),continuation_epoch=last.get('continuation_epoch'),actual_update=last.get('actual_update')))
receipt=dict(alive=alive,PID=pid,progress_only=progress,selected_receipts=len(list(run.glob('seed*_split*/*/SELECTION.json'))),failure_receipts=len(list(run.glob('seed*_split*/*/FAILURE.json'))),terminal_exists=terminal.exists(),validation_comparison_disclosed=False,final_labels_read=False)
if terminal.exists():
 result=json.loads(terminal.read_text());receipt['status']=result['status']
 receipt['error']=result.get('error_message')
 receipt['block_statuses']=[dict(seed=b['seed'],status=b['status'],error=b.get('error_message'),fits={a:r['status'] for a,r in b['fits'].items()}) for b in result['blocks']]
 development=result.get('development')
 if development and development.get('all12_selected_fits') is True and result['status']=='development_comparison_complete':
  receipt['development']=development;receipt['validation_comparison_disclosed']=True
  receipt['whole_study_wall_seconds']=result['wall_seconds']
 else:
  receipt['development_status']=development.get('status') if development else None
elif not alive:
 receipt['unexpected_missing_terminal_log_tail']=(packet/'launch_root01/execution.log').read_text()[-2500:]
receipt['GPU_utilization']=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
print(json.dumps(receipt))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.snapshot)
    out = PHASE/'coordination_snapshots'/args.snapshot
    out.mkdir(exist_ok=False)
    (out/'REMOTE_CODE.txt').write_text(REMOTE)
    command = shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO,UUID])
    ssh = ['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no',
           '-o','StrictHostKeyChecking=yes',LOGIN]
    result = subprocess.run([*ssh,command],capture_output=True,text=True,timeout=45)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,
                   destination=LOGIN,source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),
                   stdout=result.stdout,stderr=result.stderr)
    if result.returncode == 0:
        receipt['observation'] = json.loads(result.stdout)
    (out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt.get('observation',receipt)))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
