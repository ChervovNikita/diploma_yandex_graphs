import socket,json,subprocess,datetime,os
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
os.chdir(R)
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
gpu=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();assert gpu.split(',')[0].strip()==GPU
processes=[]
for q in Path('/proc').iterdir():
 if not q.name.isdecimal():continue
 try:
  cmd=(q/'cmdline').read_bytes().split(b'\0');s=(q/'stat').read_text();f=s[s.rfind(')')+2:].split()
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 args=[x.decode(errors='replace') for x in cmd if x]
 owned=any(x.startswith(str(P)+'/') for x in args)
 if owned or int(q.name)==506790:processes.append({'PID':int(q.name),'start_ticks':int(f[19]),'state':f[0],'ppid':int(f[1]),'args':args})
E=P/'learnable_internal_be_WikiCS_resource_execution_root_20261007_v2'
rows=[]
for f in (E/'receipts').glob('*_RESOURCE.json'):
 a=json.loads(f.read_text());rows.append({'cell':f.name,'passed':a.get('passed'),'peak_GPU_bytes':a.get('peak_GPU_bytes'),'inclusive_seconds':a.get('inclusive_seconds')})
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU':gpu,'processes':processes,'resources':rows,'resource_closure':(E/'COMPLETE.json').exists(),'resource_failure':(E/'FAILURE.json').exists(),'memory_wait_failure':(E/'MEMORY_WAIT_FAILURE.json').exists(),'scores_read':False}))
