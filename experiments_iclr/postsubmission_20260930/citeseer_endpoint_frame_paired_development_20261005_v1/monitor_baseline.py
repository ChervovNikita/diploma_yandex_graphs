"""Read the owned baseline's resource/progress metadata, without comparative scores."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shlex
import subprocess

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',type=int,required=True)
    args=parser.parse_args()
    dest=ROOT/('MONITOR_%03d.json'%args.sequence)
    assert args.sequence>0 and not dest.exists()
    code='''from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=phase/'citeseer_endpoint_frame_paired_development_20261005_v1';owned=json.loads((root/'OWNED_BASELINE_PROCESS.json').read_text());fit=phase/owned['output']
proc=Path('/proc')/str(owned['pid']);identity=None
if proc.exists():
 raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 assert int(fields[19])==owned['start_ticks']
 argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(bytes([0])) if v]
 assert argv==owned['command'] or fields[0]=='Z'
 identity=dict(pid=owned['pid'],state=fields[0],start_ticks=int(fields[19]))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),owned_identity=identity,status='pending_start',TEST_access=False,comparative_quality_read=False,signals_sent=False)
progress=fit/'PROGRESS.json'
if progress.exists():v['progress']=json.loads(progress.read_text());v['status']='running'
failure=fit/'FAILURE.json'
if failure.exists():v['failure']=json.loads(failure.read_text());v['status']='failed'
freeze=fit/'FREEZE.json'
if freeze.exists():
 d=json.loads(freeze.read_text());v['status']='complete'
 v['resources']={k:d[k] for k in ['last_epoch','updates','inclusive_seconds','peak_CUDA_allocated_bytes','peak_CUDA_reserved_bytes','checkpoint_sha256','VALID_logits_sha256','TEST_access']}
if identity is None and v['status'] not in ['complete','failed']:
 v['status']='process_missing_without_terminal'
stderr=root/'baseline.stderr.log'
if v['status'] in ['failed','process_missing_without_terminal'] and stderr.exists():v['stderr_tail']=stderr.read_text()[-12000:]
print(json.dumps(v))
'''
    ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
         '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
         '-o','UpdateHostKeys=no','-o','ConnectTimeout=15',
         'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
    r=subprocess.run([*ssh,shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code])],
                     capture_output=True,text=True,timeout=35)
    value=dict(UTC=datetime.now(timezone.utc).isoformat(),transport_exit_code=r.returncode,stderr=r.stderr)
    if r.returncode==0:value['observation']=json.loads(r.stdout)
    else:value['stdout']=r.stdout
    with dest.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value))


if __name__=='__main__':
    main()
