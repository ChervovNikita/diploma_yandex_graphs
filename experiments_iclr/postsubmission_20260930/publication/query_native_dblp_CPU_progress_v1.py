"""Read progress, process ownership and failures without disclosing model scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

CODE = r'''from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,time
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()==str(repo)
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
phase=repo/'experiments_iclr/postsubmission_20260930'
supervisor=phase/'graph_heterogeneous_dblp_native_cpu_supervisor_preparation_20261003_v1'
s=supervisor/'runs/root_native_serial_CPU_run01'
native=phase/'graph_heterogeneous_dblp_native_challengers_preparation_20261003_v2'
r=native/'runs/root_native_serial_CPU_run01'
owned=json.loads((s/'OWNED_CHILD.json').read_text()) if (s/'OWNED_CHILD.json').exists() else None
processes=[]
ps=subprocess.run(['ps','-eo','pid=,ppid=,stat=,etime=,pcpu=,rss=,args='],capture_output=True,text=True,check=True).stdout
for line in ps.splitlines():
 if str(supervisor/'cpu_supervisor.py') not in line and not (owned and line.split()[0]==str(owned['pid'])):continue
 parts=line.split(None,6);pid=int(parts[0]);cwd=Path(os.readlink(Path('/proc')/str(pid)/'cwd'));assert cwd.is_relative_to(repo)
 processes.append(dict(pid=pid,ppid=int(parts[1]),state=parts[2],elapsed=parts[3],CPU_percent=parts[4],RSS_KiB=parts[5],cwd=str(cwd)))
progress=[]
for trace in sorted(r.rglob('TRACE.jsonl')):
 lines=trace.read_text().splitlines();last=json.loads(lines[-1]) if lines else {}
 progress.append(dict(case=str(trace.parent.relative_to(r)),epoch_records=len(lines),last_epoch=last.get('epoch'),trace_age_seconds=time.time()-trace.stat().st_mtime))
terminal=json.loads((s/'SUPERVISOR_RECEIPT.json').read_text()) if (s/'SUPERVISOR_RECEIPT.json').exists() else None
errors={}
for path in [s/'stderr.log',supervisor/'transport/root_native_serial_CPU_run01/supervisor.stderr.log']:
 if path.exists():
  value=path.read_text()
  if value:errors[str(path.relative_to(supervisor))]=value[-1800:]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),native_study_started=(r/'STUDY_STARTED.json').exists(),owned_child_pid=owned['pid'] if owned else None,processes=processes,progress_only=progress,terminal_status=terminal['status'] if terminal else None,exit_code=terminal['exit_code'] if terminal else None,canonical_study_exists=(r/'STUDY.json').exists(),errors=errors,outcome_metrics_requested=False)))
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    assert args.snapshot and all(c.isalnum() or c in '_-' for c in args.snapshot)
    phase = Path(__file__).resolve().parents[1]
    output = phase / 'coordination_snapshots' / args.snapshot
    output.mkdir(exist_ok=False)
    (output / 'REMOTE_CODE.txt').write_text(CODE)
    destination = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
    ssh = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
           '-o', 'StrictHostKeyChecking=yes', destination]
    result = subprocess.run([*ssh, shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', CODE])],
                            capture_output=True, text=True, timeout=45)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), destination=destination,
                   exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                   source_sha256=hashlib.sha256(CODE.encode()).hexdigest(), outcome_metrics_requested=False)
    (output / 'RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return result.returncode

if __name__ == '__main__':
    raise SystemExit(main())
