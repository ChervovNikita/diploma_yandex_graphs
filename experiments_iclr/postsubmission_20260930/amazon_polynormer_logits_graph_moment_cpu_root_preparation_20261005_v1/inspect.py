"""Read only this CPU run's own qualification/ownership/completion records."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import json
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,socket,subprocess,sys
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'amazon_polynormer_logits_graph_moment_cpu_root_preparation_20261005_v1'
output=phase/'amazon_polynormer_logits_graph_moment_retrospective_cpu_execution_root_20261005_v1'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
mode=sys.argv[1];files=[]
def add(p):
 assert p.resolve().is_relative_to(phase) and not p.is_symlink()
 if p.is_file():
  assert p.stat().st_size<2_000_000
  data=p.read_bytes();files.append(dict(path=str(p.relative_to(root)) if p.is_relative_to(root) else str(p.relative_to(phase)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),data=base64.b64encode(data).decode()))
def physical(owner):
 proc=Path('/proc')/str(owner['pid'])
 try:
  raw=(proc/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
  if int(v[19])!=owner['start_ticks']:return dict(identity_matches=False,PID_reused=True)
  argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
  return dict(identity_matches=argv==owner['argv'],pid=owner['pid'],start_ticks=int(v[19]),state=v[0],cwd=str((proc/'cwd').resolve()))
 except FileNotFoundError:return None
value=dict(UTC=datetime.now(timezone.utc).isoformat(),mode=mode,TEST_access=False,new_backbone_fits=0)
if mode=='qualification':
 for name in ['QUALIFICATION_TERMINAL.json','QUALIFICATION_STDOUT.txt','QUALIFICATION_STDERR.txt']:
  add(root/name)
 for name in ['REPORT.json','QUALIFICATION_RESULT.json','qp_against_scipy.json','tiny_spd.json','degenerate_projection.json','gram_reconstruction.json','stable_skip.json','tiny_gap_single.json','head_counts.json','tiny_discarded_fit.json','sparse_and_exclusion.json']:
  add(root/'NUMERICAL_QUALIFICATION'/name)
else:
 assert mode=='development'
 for name in ['DETACHED_LAUNCH.json','CHILD_IDENTITY.json','DEVELOPMENT_TERMINAL.json','ROOT_DEVELOPMENT_RELEASE.json']:
  p=root/name;add(p)
  if p.is_file() and name in ['DETACHED_LAUNCH.json','CHILD_IDENTITY.json']:
   v=json.loads(p.read_text());value[name+'_physical']=physical(v.get('runner_identity',v.get('identity')))
 progress=output/'PROGRESS.jsonl'
 if progress.is_file():
  with progress.open('rb') as f:f.seek(max(0,progress.stat().st_size-8192));value['progress_tail']=f.read().decode()
 terminal=root/'DEVELOPMENT_TERMINAL.json'
 if terminal.is_file():
  t=json.loads(terminal.read_text());value['exit_code']=t['exit_code']
  if t['exit_code']==0:
   add(output/'RESULT.json');add(output/'ALL_CONFIGURATION_METRICS.csv');add(output/'ENVIRONMENT.json');add(output/'INPUTS.json')
  else:
   for name in ['DEVELOPMENT_STDOUT.txt','DEVELOPMENT_STDERR.txt','RUNNER_STDERR.txt']:add(root/name)
value['files']=files
print(json.dumps(value))
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('qualification','development'));args=parser.parse_args()
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=20','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
        'cd '+REPO+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)+' '+shlex.quote(args.mode)]
    r=subprocess.run(command,capture_output=True,text=True,timeout=45)
    out=HERE/(args.mode+'_observation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
    with (out/'TRANSPORT.json').open('x') as f:json.dump(dict(exit_code=r.returncode,stderr=r.stderr,remote_source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest()),f,indent=2);f.write('\n')
    assert r.returncode==0,r.stderr
    value=json.loads(r.stdout)
    descriptors=[]
    for row in value.pop('files'):
        data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
        path=out/rel;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(data)
        descriptors.append({k:row[k] for k in ('path','sha256','bytes')})
    value['files']=descriptors
    with (out/'OBSERVATION.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    compact={k:v for k,v in value.items() if k!='files'}
    compact['files']=len(descriptors);compact['local_observation']=str(out)
    if args.mode=='qualification':
        report=json.loads((out/'NUMERICAL_QUALIFICATION/REPORT.json').read_text())
        compact.update(status=report['status'],tests=[{'test':v['test'],'status':v['status']} for v in report['tests']],cost=report['cost'])
    print(json.dumps(compact))

if __name__=='__main__':main()
