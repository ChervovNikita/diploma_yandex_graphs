import base64,importlib.util,json,shlex,zlib
from pathlib import Path
P=Path('postsubmission_research_20260930').resolve();H=P/'native_SAGE_fixed_context_key_pilot_root_20261011_v1';rows=json.loads((H/'READER_DEPLOY_PAYLOAD_V1.json').read_text())
REMOTE=r"""import base64,hashlib,json,os,socket,subprocess,sys,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';H=P/'native_SAGE_fixed_context_key_pilot_root_20261011_v1'
for row in payload:
 q=P/row['path'];assert q.resolve().is_relative_to(P);b=base64.b64decode(row['data']);assert hashlib.sha256(b).hexdigest()==row['sha256'];q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==b,q
 else:q.write_bytes(b)
assert not (H/'READER_JOB_START_V1.json').exists() and not (H/'complete_analysis_v1').exists()
os.chdir(R);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
with (H/'watcher.stdout.log').open('x') as out,(H/'watcher.stderr.log').open('x') as err:
 child=subprocess.Popen([str(R/'.venv/bin/python'),'-B',str(H/'reader_after_closure_v1.py')],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 for _ in range(30):
  if (H/'READER_JOB_START_V1.json').exists():break
  if child.poll() is not None:break
  time.sleep(.1)
q=H/'READER_JOB_START_V1.json';v={'watcher_PID':child.pid,'start':json.loads(q.read_text()) if q.exists() else None,'source_files_deployed':len(payload),'CPU_only':True,'TEST_access':False};print(json.dumps(v));assert q.exists(),(H/'watcher.stderr.log').read_text()
"""
spec=importlib.util.spec_from_file_location('stream',P/'publication/publish_exact_inventory_v19.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
args=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'deploy','literal','reader'])]
r=m.terminal_transport(args,input=base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode(),capture_output=True,text=True,timeout=70);(H/'READER_DEPLOY_TRANSPORT_V1.json').write_text(json.dumps({'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n');assert r.returncode==0,r.stderr[-1000:];v=json.loads(r.stdout);(H/'ACTUAL_READER_LAUNCH_V1.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
