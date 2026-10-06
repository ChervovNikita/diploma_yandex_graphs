"""Inspect Torch spec without import, deploy one stdlib supervisor, dispatch it detached once."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess,zlib
HERE=Path(__file__).resolve().parent;P=HERE.parent;NAME=HERE.name
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs';PHASE=REPO+'/experiments_iclr/postsubmission_20260930'
NATIVE=PHASE+'/native_ncn_runtime_20261005_v1/.venv/bin/python'
OUT='shared_private_transfer_complete39_D2_analysis_execution_root_20261006_v2'
RELEASE='shared_private_transfer_complete39_D2_operation_successor_20261006_v1/ROOT_D2_ANALYSIS_RELEASE.json'
SOURCE='shared_private_transfer_amended39_d2_activation_preparation_20261006_v1/analyze_d2_accounting_proposal.py'
code=(HERE/'supervise_once.py').read_bytes();code_sha=hashlib.sha256(code).hexdigest()
remote=r'''
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
cfg=CFG_TEXT;target=phase/cfg['supervisor_directory'];output=phase/cfg['output'];assert not target.exists() and not output.exists()
for rel,expected in [(cfg['source'],'0fdd061baaa71bb525c7122ff8bdf98c8dfe450074c41819cc3ab659e4ab648a'),(cfg['release'],'edd23d75f60810d941f50eee1d052ad88cf151e8f5e0e53730643b7e497cbe93')]:assert hashlib.sha256((phase/rel).read_bytes()).hexdigest()==expected
old=Path('/proc/488425/stat')
if old.exists():
 raw=old.read_text();assert int(raw[raw.rfind(')')+2:].split()[19])!=6010099178
env=dict(os.environ);env.update({'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
spec_code="import importlib.util,json; s=importlib.util.find_spec('torch'); print(json.dumps({'present':s is not None,'origin':s.origin if s else None,'locations':list(s.submodule_search_locations) if s and s.submodule_search_locations else []}))"
def probe(environment):
 r=subprocess.run([cfg['native'],'-B','-c',spec_code],cwd=str(repo),env=environment,capture_output=True,text=True,timeout=20)
 assert r.returncode==0,(r.returncode,r.stderr);return json.loads(r.stdout)
bare=probe(env);site=str(repo/'.venv/lib/python3.11/site-packages');used=None
if not bare['present']:
 assert Path(site).is_dir();env['PYTHONPATH']=site;used=site
selected=probe(env);assert selected['present'] and Path(selected['origin']).resolve().is_relative_to(repo)
version=Path(selected['origin']).parent/'version.py';v=None
for n in ast.parse(version.read_text()).body:
 if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='__version__' for x in n.targets) and isinstance(n.value,ast.Constant):v=n.value.value
assert isinstance(v,str) and v.startswith('2.1.2'),v
runtime={'native_executable':cfg['native'],'bare_spec':bare,'selected_spec':selected,'torch_imported':False,'existing_reviewed_site_added_only_if_absent':used,'torch_version_source':v,'version_source_sha256':hashlib.sha256(version.read_bytes()).hexdigest()}
target.mkdir();b=base64.b64decode(cfg['supervisor_base64']);assert hashlib.sha256(b).hexdigest()==cfg['supervisor_sha256']
with (target/'supervise_once.py').open('xb') as f:f.write(b)
(target/'supervise_once.py').chmod(0o444)
child_env={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
if used:child_env['PYTHONPATH']=used
configuration={'argv':[cfg['native'],'-B',str(phase/cfg['source']),'--release',str(phase/cfg['release']),'--output',str(output)],'child_environment':child_env,'runtime_spec':runtime,'source_sha256':'0fdd061baaa71bb525c7122ff8bdf98c8dfe450074c41819cc3ab659e4ab648a','release_sha256':'edd23d75f60810d941f50eee1d052ad88cf151e8f5e0e53730643b7e497cbe93'}
with (target/'CONFIG.json').open('x') as f:json.dump(configuration,f,indent=2,sort_keys=True);f.write('\n')
(target/'CONFIG.json').chmod(0o444)
with (target/'supervisor.stdout').open('xb') as out,(target/'supervisor.stderr').open('xb') as err:
 proc=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(target/'supervise_once.py')],cwd=str(repo),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 raw=(Path('/proc')/str(proc.pid)/'stat').read_text();identity={'PID':proc.pid,'start_ticks':int(raw[raw.rfind(')')+2:].split()[19]),'pgid':os.getpgid(proc.pid),'sid':os.getsid(proc.pid)}
print(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'supervisor_identity':identity,'supervisor_directory':cfg['supervisor_directory'],'output':cfg['output'],'runtime_spec':runtime,'child_argv':configuration['argv'],'limits_seconds':600,'RSS_limit_bytes':8*1024**3,'D2_attempt_limit':1,'retry_or_resume':False}))
'''
cfg={'supervisor_directory':NAME,'output':OUT,'source':SOURCE,'release':RELEASE,'native':NATIVE,'supervisor_sha256':code_sha,'supervisor_base64':base64.b64encode(code).decode()}
remote=remote.replace('CFG_TEXT',repr(cfg));payload=base64.b64encode(zlib.compress(remote.encode(),9)).decode()+'\n'
request=json.loads((P/'shared_private_transfer_complete39_history_inventory_root_request_preparation_20261006_v1/ROOT_APPROVAL_REQUEST.json').read_text());bootstrap="import sys,base64,zlib;exec(compile(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())), '<owned-D2-launch>', 'exec'))"
argv=request['remote_invocation_argv'][:-1]+['cd '+shlex.quote(REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(bootstrap)]
with (HERE/'ROOT_AUTHORIZATION_RECEIPT.json').open('x') as f:json.dump({'sender':'/root','authorization':'Distinct owned detached D2 successor once under same release/source/actual registry/history; inspect Torch spec without import; add exact existing reviewedTorch2.1.2 site only if absent;600s/8GiB RSS; CPU only; preserve all logs/results; no auto retry,TEST,fit,model,new selection or77 access.'},f,indent=2);f.write('\n')
with (HERE/'LAUNCH_INTENT.json').open('x') as f:json.dump({'argv':argv,'remote_source_sha256':hashlib.sha256(remote.encode()).hexdigest(),'supervisor_source_sha256':code_sha,'once_only':True,'declared_output':OUT},f,indent=2,sort_keys=True);f.write('\n')
r=subprocess.run(argv,input=payload,capture_output=True,text=True,timeout=55);receipt={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stderr':r.stderr,'stdout':r.stdout}
if r.returncode==0:receipt['result']=json.loads(r.stdout)
with (HERE/'LAUNCH_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(receipt,indent=2));assert r.returncode==0,'Owned successor not dispatched; preserve failure.'
