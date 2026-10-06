"""Stage pinned CPU fixture sources, then start one reviewed owned invocation."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess,zlib
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
 q=phase/r;assert q.resolve().is_relative_to(phase) and not any(v.is_symlink() for v in (q,*q.parents) if v.is_relative_to(phase));return q
parent=path('cmcl_CPU_fixture_execution_root_20261006_v2');assert not parent.exists()
for row in p['stage']:
 q=path(row['path']);raw=base64.b64decode(row['base64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if q.exists():assert q.read_bytes()==raw
 else:q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
 q.chmod(0o444)
s=json.loads(path(p['scope']['path']).read_text());assert hashlib.sha256(path(p['scope']['path']).read_bytes()).hexdigest()==p['scope']['sha256']
binary=Path(s['python_executable']);assert binary.resolve().is_relative_to(repo) and hashlib.sha256(binary.read_bytes()).hexdigest()==s['python_binary_sha256']
helper=path('matched_first_order_private_gradient_utility_native_supervisor_preparation_20261006_v4/supervise.py');assert hashlib.sha256(helper.read_bytes()).hexdigest()=='80611cae8ab2e557a11db72c99321bf5f5d550992dbf16f63374364f48f8f808';helper.chmod(0o444)
parent.mkdir()
argv=['/usr/bin/python3','-I','-S','-B',str(path(p['supervisor']['path'])),'--execute-authorized','--scope',str(path(p['scope']['path'])),'--scope-sha256',p['scope']['sha256']]
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'argv':argv,'cwd':str(repo),'CPU_only':True,'scope_sha256':p['scope']['sha256'],'one_launch':True,'retry':False,'fits_or_scoring':False}
claim=parent/'ROOT_LAUNCH_CLAIM.json'
with claim.open('x') as f:json.dump(receipt,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
claim.chmod(0o444)
with (parent/'ROOT.stdout.log').open('xb') as out,(parent/'ROOT.stderr.log').open('xb') as err:
 child=subprocess.Popen(argv,cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONOPTIMIZE='',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
proc=Path('/proc')/str(child.pid);f=(proc/'stat').read_text().rsplit(')',1)[1].split();receipt.update(PID=child.pid,start_ticks=int(f[19]),actual_argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(bytes([0])) if v],actual_cwd=str((proc/'cwd').resolve()))
assert receipt['actual_argv']==argv and receipt['actual_cwd']==str(repo)
q=parent/'ROOT_LAUNCH.json';q.write_text(json.dumps(receipt,indent=2)+chr(10));q.chmod(0o444)
print(json.dumps(receipt))
'''
def desc(rel):
 b=(P/rel).read_bytes();return {'path':rel,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
 target=D/'LAUNCH_TRANSPORT_RECEIPT.json';assert not target.exists()
 scope=desc(str((D/'SCOPE.json').relative_to(P)));s=json.loads((D/'SCOPE.json').read_text());supervisor=desc('cmcl_CPU_fixture_root_execution_20261006_v1/supervise_once_v2.py')
 rows=[scope,supervisor,desc(str((D/'ROOT_ADMISSION.json').relative_to(P)))]+[s[k] for k in ('executor','executor_review','helper','helper_review','fixture_plan','external_supervisor_review')]
 for row in rows:raw=(P/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];row['base64']=base64.b64encode(raw).decode()
 payload={'stage':rows,'scope':{k:v for k,v in scope.items() if k!='base64'},'supervisor':{k:v for k,v in supervisor.items() if k!='base64'}}
 cmd=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','cd '+shlex.quote(REPO)+' && /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)]
 r=subprocess.run(cmd,input=base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode(),capture_output=True,text=True,timeout=45)
 receipt={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest(),'local_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};target.write_text(json.dumps(receipt,indent=2)+'\n')
 if r.returncode:print(json.dumps(receipt));raise SystemExit(r.returncode)
 owner=json.loads(r.stdout);(D/'LAUNCH_RECEIPT.json').write_text(json.dumps(owner,indent=2)+'\n');print(json.dumps(owner))
if __name__=='__main__':main()
