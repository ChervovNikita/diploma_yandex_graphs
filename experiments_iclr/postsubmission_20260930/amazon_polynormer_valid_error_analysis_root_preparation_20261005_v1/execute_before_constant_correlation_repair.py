"""Run reviewed descriptive VALID-only analysis once; preserve all paired errors."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import json
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
SOURCE='amazon_polynormer_valid_error_analysis_source_preparation_20261005_v1'
OUTPUT='amazon_polynormer_valid_error_analysis_execution_root_20261005_v1'
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,csv,hashlib,json,os,socket,subprocess,sys,time
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
packet=json.load(sys.stdin)
def path(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 p=phase/p;assert p.resolve().is_relative_to(phase)
 assert not any(q.is_symlink() for q in (p,*p.parents) if q.is_relative_to(phase))
 return p
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
for row in packet['files']:
 p=path(row['path']);b=base64.b64decode(row['data'])
 assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 if p.exists():assert p.read_bytes()==b
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
root=path(packet['root']);root.mkdir(exist_ok=False)
out=path(packet['output']);assert not out.exists()
write(root/'RELEASE.json',packet['release'])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',BLIS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
start=time.perf_counter()
with (root/'STDOUT.txt').open('x') as stdout,(root/'STDERR.txt').open('x') as stderr:
 r=subprocess.run(packet['release']['argv'],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=120)
write(root/'TERMINAL.json',dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,wall_seconds=time.perf_counter()-start,automatic_retry=False))
rows=[]
collection=[]
if r.returncode==0:
 original=out/'AGGREGATE.json';data=json.loads(original.read_text());assert data['status']=='complete_retrospective_descriptive_VALID'
 metric_rows=data.pop('metric_rows');groups={}
 for row in metric_rows:
  key=(str(row['split']),row['classifier'])
  assert key[0] in ['0','1','2','three_split_summary'] and key[1] in ['single','shared4','independent4','paired_native','shared_minus_independent']
  groups.setdefault(key,[]).append(row)
 folder=root/'EXACT_METRIC_COLLECTION';folder.mkdir()
 columns=list(metric_rows[0]);descriptors=[]
 for (split,classifier),values in sorted(groups.items()):
  p=folder/('split_'+split+'_'+classifier+'.csv')
  with p.open('x',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(values)
  b=p.read_bytes();assert len(b)<2_000_000
  descriptors.append(dict(path=str(p.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),metric_rows=len(values)))
  collection.append(p)
 assert sum(d['metric_rows'] for d in descriptors)==len(metric_rows)
 data['full_original_aggregate']=dict(path=str(original.relative_to(phase)),sha256=hashlib.sha256(original.read_bytes()).hexdigest(),bytes=original.stat().st_size)
 data['complete_metric_shards']=descriptors;data['metric_row_count']=len(metric_rows);data['collection_omits_no_metric_rows']=True
 write(folder/'AGGREGATE_METADATA.json',data);collection.append(folder/'AGGREGATE_METADATA.json')
for p in [root/'RELEASE.json',root/'TERMINAL.json',root/'STDOUT.txt',root/'STDERR.txt',*collection,*[out/n for n in ['TERMINAL.json','ENVIRONMENT.json','INPUTS.json','RUN_BINDINGS.json','PROGRESS.jsonl','FAILURE.json']]]:
 if p.is_file():
  assert p.stat().st_size<2_000_000
  b=p.read_bytes();rows.append(dict(path=str(p.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=base64.b64encode(b).decode()))
print(json.dumps(dict(exit_code=r.returncode,files=rows)))
'''

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seal-sha256',required=True)
    parser.add_argument('--review',required=True)
    args=parser.parse_args()
    source=PHASE/SOURCE
    assert hashlib.sha256((source/'SEAL.json').read_bytes()).hexdigest()==args.seal_sha256
    review_path=PHASE/args.review
    assert review_path.resolve().is_relative_to(PHASE)
    rows=[]
    manifest=json.loads((source/'MANIFEST.json').read_text())
    paths=[source/'MANIFEST.json',source/'SEAL.json',*[source/r['path'] for r in manifest['payload']],review_path]
    for path in paths:
        assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
        data=path.read_bytes();assert len(data)<2_000_000
        rows.append(dict(path=str(path.relative_to(PHASE)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),data=base64.b64encode(data).decode()))
    assert len(rows)==len({r['path'] for r in rows})
    remote_phase=REPO+'/experiments_iclr/postsubmission_20260930'
    argv=[REPO+'/.venv/bin/python','-B',remote_phase+'/'+SOURCE+'/run_error_analysis.py',
          '--phase',remote_phase,'--closure-freeze','amazon_polynormer_paired_family_execution_root_20261003_v3/v6_closure_v1/FREEZE.json',
          '--closure-sha256','37145c431daaf74dbd622f287285ec4947f7b4dbdb6b88a19c723376d1ca29fc',
          '--protocol','amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v2/PROTOCOL.json',
          '--protocol-sha256','5d667f15640102995fa998ca6932c53c5b71554273c84d4938a331c0a4eae7ba',
          '--source-seal-sha256',args.seal_sha256,'--output',remote_phase+'/'+OUTPUT,'--execute-authorized-valid-error-analysis']
    release=dict(UTC=datetime.now(timezone.utc).isoformat(),root_authorized=True,purpose='Diagnose complete selected VALID banks descriptively',
                 retrospective_development_only=True,head_fits=0,optimizer_updates=0,backbone_forwards=0,TEST_access=False,
                 TRAIN_control_access=False,automatic_retry=False,argv=argv,
                 source_descriptors=[{k:r[k] for k in ('path','sha256','bytes')} for r in rows])
    with (HERE/'RELEASE.json').open('x') as f:json.dump(release,f,indent=2);f.write('\n')
    packet=dict(files=rows,root=HERE.name,output=OUTPUT,release=release)
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes',
             '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=20',
             'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','cd '+REPO+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)]
    r=subprocess.run(command,input=json.dumps(packet),capture_output=True,text=True,timeout=155)
    with (HERE/'TRANSPORT.json').open('x') as f:json.dump(dict(exit_code=r.returncode,stderr=r.stderr,remote_source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest()),f,indent=2);f.write('\n')
    assert r.returncode==0,r.stderr
    received=json.loads(r.stdout)
    observation=HERE/'observation';observation.mkdir()
    for row in received['files']:
        rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
        data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        path=observation/rel;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(data)
    terminal=observation/OUTPUT/'TERMINAL.json'
    print(json.dumps(dict(exit_code=received['exit_code'],analysis_terminal=json.loads(terminal.read_text()) if terminal.is_file() else None,local_observation=str(observation))))

if __name__=='__main__':main()
