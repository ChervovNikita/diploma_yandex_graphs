import os,socket,json,subprocess,datetime
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';os.chdir(R);assert socket.gethostname()=='anogena-2-0'
D=P/'learnable_internal_be_WikiCS_scientific_family_execution_root_20261007_v1';A=P/'learnable_internal_be_WikiCS_scientific_family_activation_20261007_v1'
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'processes':[],'metadata':{},'scores_read':False}
for q in Path('/proc').iterdir():
 if not q.name.isdecimal():continue
 try:
  args=[x.decode(errors='replace') for x in (q/'cmdline').read_bytes().split(b'\0') if x];stat=(q/'stat').read_text();f=stat[stat.rfind(')')+2:].split()
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if any(x.startswith(str(P)+'/learnable_internal_be_') for x in args):out['processes'].append({'PID':int(q.name),'start_ticks':int(f[19]),'state':f[0],'ppid':int(f[1]),'args':args})
for rel in ['OWNER.json','RUNNING_CELL.json','MEMORY_WAIT.json','DRIVER_FAILURE.json','FAMILY_CLOSURE.json','fits/receipts/single_6101_LIVE.json','fits/outputs/single_6101/PROGRESS.json']:
 q=D/rel
 if q.is_file():out['metadata'][rel]=json.loads(q.read_text())
log=A/'DRIVER.log'
if log.is_file() and log.stat().st_size:out['controller_error_log']=log.read_text()[-5000:]
out['GPU']=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
assert out['GPU'].split(',')[0].strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
print(json.dumps(out))
