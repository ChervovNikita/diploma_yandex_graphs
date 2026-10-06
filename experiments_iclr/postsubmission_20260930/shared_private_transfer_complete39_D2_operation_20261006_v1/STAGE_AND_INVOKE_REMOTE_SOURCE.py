
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess,sys,zlib
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split();assert uuids==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
value=json.loads(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())));rows=value['files'];pins=value['actual_pins'];argv=value['D2_argv'];output=phase/value['output']
assert len(rows)==23 and not output.exists() and Path(argv[0]).is_file()
for r in pins:
 p=phase/r['path'];assert p.resolve(strict=True).is_relative_to(phase) and not p.is_symlink() and p.stat().st_mode&0o222==0 and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
for r in rows:
 p=phase/r['path'];assert not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts and p.resolve().is_relative_to(phase)
 for q in [p,*p.parents]:
  if q==phase.parent:break
  assert not q.is_symlink()
 b=base64.b64decode(r['base64']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 if p.exists():assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'] and p.stat().st_size==r['bytes']
result=[]
for r in rows:
 p=phase/r['path'];existed=p.exists()
 if not existed:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(base64.b64decode(r['base64']))
  p.chmod(0o444)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 result.append({k:r[k] for k in ('path','bytes','sha256')}|{'mode':oct(p.stat().st_mode&0o777),'existing_exact_bytes_reused_without_chmod':existed})
assert not output.exists()
stat=Path('/proc/self/stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
print(json.dumps({'schema':'exact_D2_staging_and_once_only_invocation_started_v1','UTC':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'GPU_UUIDs':uuids,'files':result,
 'D2_process_identity':{'PID':os.getpid(),'start_ticks':ticks},'argv':argv,'fresh_output_absent':True,'staging_read_scientific_payload_semantics':False,'no77_connection':True}),flush=True)
env=dict(os.environ);env.update({'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
os.execve(argv[0],argv,env)
