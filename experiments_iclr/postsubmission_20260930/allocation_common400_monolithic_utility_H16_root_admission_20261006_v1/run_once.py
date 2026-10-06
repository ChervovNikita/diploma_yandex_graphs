"""Stage exact reviewed H16 sources and metadata, then launch once on allocation."""
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,shlex,subprocess,zlib
D=Path(__file__).resolve().parent;P=D.parent
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess,sys,zlib
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
p=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
def path(rel):
 r=Path(rel);assert not r.is_absolute() and '..' not in r.parts
 q=phase/r;assert q.resolve().is_relative_to(phase) and not any(x.is_symlink() for x in (q,*q.parents) if x.is_relative_to(phase))
 return q
def verify(row):
 q=path(row['path']);assert q.is_file() and q.stat().st_size==row['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256']
 q.chmod(0o444);return q
launch=path('allocation_common400_monolithic_utility_H16_launch_root_20261006_v1');assert not launch.exists()
for row in p['stage']:
 q=path(row['path']);b=base64.b64decode(row['base64']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 if q.exists():assert q.read_bytes()==b
 else:q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
 q.chmod(0o444)
for row in p['verify']:verify(row)
fit=json.loads(verify(p['fit_scope']).read_text());sup=json.loads(verify(fit['external_fit_supervision_admission']).read_text())
assert fit['root_fit_authorized'] and fit['fixed_before_fit'] and sup['root_supervision_fit_authorized'] and sup['fixed_before_launch']
for key in ('caller_review','native_utility_terminal','native_utility_worker_result','native_utility_worker_scope','native_utility_supervision_scope','origin_run'):verify(fit[key])
verify(fit['common400'])
terminal=json.loads(path(fit['native_utility_terminal']['path']).read_text())
assert terminal['wrapper_resource_closure_PASS'] and terminal['owned_child']['exit_code']==0 and not terminal['cleanup_errors']
for owner in p['closed_owners']:
 proc=Path('/proc')/str(owner['PID'])
 try:
  raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split();assert int(f[19])!=owner['start_ticks'],'Preceding owned process is still present'
 except FileNotFoundError:pass
python=Path(fit['python_executable']);assert hashlib.sha256(python.read_bytes()).hexdigest()==fit['python_executable_sha256']
a=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip().splitlines()
assert len(a)==1;uuid,free=[x.strip() for x in a[0].split(',')]
assert uuid==fit['GPU_UUID'] and int(free)*1048576>=sup['minimum_initial_cuda_free_bytes']
execution=path('allocation_common400_monolithic_utility_H16_execution_root_20261006_v1');assert not execution.exists();execution.mkdir()
launch.mkdir()
argv=[str(python),'-B',str(path(sup['supervisor_source']['path'])),'--execute-authorized','--source-root',str(phase),'--fit-scope',str(path(p['fit_scope']['path'])),'--fit-scope-sha256',p['fit_scope']['sha256']]
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'argv':argv,'cwd':str(repo),'GPU_UUID':uuid,'fresh_free_MiB':int(free),'fit_invocation_id':fit['fit_invocation_id'],'launch_attempts':1,'automatic_retry':False,'held_scoring':False}
with (launch/'stdout.log').open('xb') as out,(launch/'stderr.log').open('xb') as err:
 child=subprocess.Popen(argv,cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
receipt['PID']=child.pid
try:
 proc=Path('/proc')/str(child.pid);raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split();receipt['start_ticks']=int(f[19]);receipt['actual_argv']=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x];receipt['actual_cwd']=str((proc/'cwd').resolve());assert receipt['actual_argv']==argv and receipt['actual_cwd']==str(repo)
except FileNotFoundError:receipt['immediate_handle_absent']=True
(launch/'LAUNCH.json').write_text(json.dumps(receipt,indent=2)+chr(10));(launch/'LAUNCH.json').chmod(0o444)
print(json.dumps(receipt))
'''
def desc(rel):
 b=(P/rel).read_bytes();return {'path':rel,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def main():
 out=D/'LAUNCH_TRANSPORT_RECEIPT.json';assert not out.exists()
 fit=json.loads((D/'FIT_SCOPE.json').read_text());sup=json.loads((D/'FIT_SUPERVISION_ADMISSION.json').read_text())
 stage=[str(f.relative_to(P)) for f in D.iterdir() if f.is_file() and f.suffix in ('.json','.md')]
 stage += [fit['caller_review']['path'],sup['source_review']['path'],sup['supervisor_source']['path'],'matched_first_order_private_gradient_utility_common400_H16_preparation_20261006_v4/run_utility_H16.py']
 rows=[]
 for rel in sorted(set(stage)):
  row=desc(rel);row['base64']=base64.b64encode((P/rel).read_bytes()).decode();rows.append(row)
 source=ast.parse((P/'matched_first_order_private_gradient_utility_common400_H16_preparation_20261006_v4/run_utility_H16.py').read_text())
 pins=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets))
 verify=[]
 for rel,sha in pins.values():
  row=desc(rel);assert row['sha256']==sha;verify.append(row)
 parent=json.loads((P/'allocation_monolithic_utility_native_root_admission_20261006_v1/LAUNCH_RECEIPT.json').read_text())
 terminal=json.loads((P/'allocation_monolithic_utility_native_terminal_handoff_root_20261006_v1/TERMINAL.json').read_text())
 p={'stage':rows,'verify':verify,'fit_scope':desc(str((D/'FIT_SCOPE.json').relative_to(P))),'closed_owners':[{'PID':parent['PID'],'start_ticks':parent['start_ticks']},{'PID':terminal['owned_child']['PID'],'start_ticks':terminal['owned_child']['physical_identity']['start_ticks']}]}
 command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','cd '+shlex.quote(REPO)+' && /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)]
 r=subprocess.run(command,input=base64.b64encode(zlib.compress(json.dumps(p).encode(),9)).decode(),capture_output=True,text=True,timeout=50)
 v={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest(),'wrapper_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'automatic_retry':False}
 out.write_text(json.dumps(v,indent=2)+'\n')
 if r.returncode:print(json.dumps(v));raise SystemExit(r.returncode)
 owner=json.loads(r.stdout);(D/'LAUNCH_RECEIPT.json').write_text(json.dumps(owner,indent=2)+'\n');print(json.dumps(owner))

if __name__=='__main__':main()
